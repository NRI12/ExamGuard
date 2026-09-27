# ExamGuard on Windows 10/11 or Windows Server with an NVIDIA GPU (RTX 3060), without Docker or admin rights.
# Everything is placed inside this folder: portable PostgreSQL 16 (port 5433), ffmpeg, uv + Python 3.11,
# torch 2.7.1 CUDA 12.6 (NVIDIA driver >= 560). On the first install the database is restored from
# data\examguard.dump (sessions, candidates, events, reviews).
#
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1            (or double-click install.cmd)
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1 -Firewall  (admin: open the port on the LAN)
param([switch]$Firewall)
. (Join-Path $PSScriptRoot 'common.ps1')

Write-Step 'GPU'
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) { & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader }
else { Write-Warning 'nvidia-smi not found: install the NVIDIA driver (>= 560) before starting ExamGuard.' }

Write-Step 'tools (uv, PostgreSQL, ffmpeg)'
$uv = Join-Path $Tools 'uv\uv.exe'
if (-not (Test-Path $uv)) {
    Get-File 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' (Join-Path $Tools 'uv.zip')
    Expand-Archive (Join-Path $Tools 'uv.zip') (Join-Path $Tools 'uv') -Force
}
if (-not (Test-Path (Join-Path $PgBin 'pg_ctl.exe'))) {
    Get-File 'https://get.enterprisedb.com/postgresql/postgresql-16.15-1-windows-x64-binaries.zip' (Join-Path $Tools 'pgsql.zip')
    Expand-Archive (Join-Path $Tools 'pgsql.zip') $Tools -Force      # -> tools\pgsql
}
if (-not (Test-Path (Join-Path $FfmpegBin 'ffprobe.exe'))) {
    Get-File 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' (Join-Path $Tools 'ffmpeg.zip')
    $unpacked = Join-Path $Tools 'ffmpeg-unpacked'
    Expand-Archive (Join-Path $Tools 'ffmpeg.zip') $unpacked -Force
    $inner = Get-ChildItem $unpacked -Directory | Select-Object -First 1
    if (Test-Path (Join-Path $Tools 'ffmpeg')) { Remove-Item -Recurse -Force (Join-Path $Tools 'ffmpeg') }
    Move-Item $inner.FullName (Join-Path $Tools 'ffmpeg')
    Remove-Item -Recurse -Force $unpacked
}

Write-Step 'Python 3.11 environment (backend\.venv)'
if (-not (Test-Path $Python)) { & $uv venv (Join-Path $Root 'backend\.venv') --python 3.11 }
& $uv pip install --python $Python -r (Join-Path $Root 'deploy\requirements-gpu.txt')
if ($LASTEXITCODE -ne 0) { throw 'Python package installation failed' }
# Triton for torch.compile (about 3x faster X3D); without it the model runs in eager mode
& $uv pip install --python $Python 'triton-windows<3.4' *> $null
& $Python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

Write-Step 'configuration (.env)'
if (-not (Test-Path $EnvFile)) {
    $dbPassword = New-Secret 16
    $adminPassword = (New-Secret 7) + 'Aa1'
    $rootFwd = $Root -replace '\\', '/'
    $content = @"
APP_ENV=production
APP_PROFILE=rtx3060
PGPORT=5433
POSTGRES_DB=examguard
POSTGRES_USER=examguard
POSTGRES_PASSWORD=$dbPassword
POSTGRES_SUPERUSER_PASSWORD=$(New-Secret 16)
DATABASE_URL=postgresql+psycopg://examguard@127.0.0.1:5433/examguard
DATABASE_PASSWORD=$dbPassword
JWT_SECRET=$(New-Secret 32)
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480
UPLOAD_ROOT=$rootFwd/data/uploads
EVIDENCE_ROOT=$rootFwd/data/evidence
MODEL_ROOT=$rootFwd/model_artifacts
CONFIG_ROOT=$rootFwd/backend/configs
FRONTEND_DIST=$rootFwd/frontend/dist
MAX_UPLOAD_BYTES=4294967296
FFPROBE_TIMEOUT_SECONDS=30
# HTTP on a LAN needs COOKIE_SECURE=false; set true behind HTTPS (deploy\windows\tunnel.ps1)
COOKIE_SECURE=false
PORT=8000
ADMIN_USERNAME=admin
ADMIN_PASSWORD=$adminPassword
ADMIN_FULL_NAME="Quản trị viên"
"@
    Write-Utf8NoBom $EnvFile $content
}
Import-DotEnv
New-Item -ItemType Directory -Force $env:UPLOAD_ROOT, $env:EVIDENCE_ROOT | Out-Null

Write-Step 'PostgreSQL (data\pgdata, port 5433)'
if (-not (Test-Path (Join-Path $PgData 'PG_VERSION'))) {
    $pw = Join-Path $Tools 'pgpass.tmp'
    Write-Utf8NoBom $pw $env:POSTGRES_SUPERUSER_PASSWORD
    & (Join-Path $PgBin 'initdb.exe') -D $PgData -U postgres --pwfile=$pw -E UTF8 --locale=C --auth=scram-sha-256 | Out-Null
    Remove-Item $pw
    if ($LASTEXITCODE -ne 0) { throw 'initdb failed' }
}
Start-Postgres
$psql = Join-Path $PgBin 'psql.exe'
$env:PGPASSWORD = $env:POSTGRES_SUPERUSER_PASSWORD
$hasRole = & $psql -h 127.0.0.1 -p $env:PGPORT -U postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='$env:POSTGRES_USER'"
if ($hasRole -ne '1') { & $psql -h 127.0.0.1 -p $env:PGPORT -U postgres -qc "CREATE ROLE $env:POSTGRES_USER LOGIN PASSWORD '$env:POSTGRES_PASSWORD'" }
$hasDb = & $psql -h 127.0.0.1 -p $env:PGPORT -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$env:POSTGRES_DB'"
if ($hasDb -ne '1') {
    & (Join-Path $PgBin 'createdb.exe') -h 127.0.0.1 -p $env:PGPORT -U postgres -O $env:POSTGRES_USER $env:POSTGRES_DB
    $dump = Join-Path $Root 'data\examguard.dump'
    if (Test-Path $dump) {
        Write-Host '   restoring data\examguard.dump'
        $env:PGPASSWORD = $env:POSTGRES_PASSWORD
        & (Join-Path $PgBin 'pg_restore.exe') -h 127.0.0.1 -p $env:PGPORT -U $env:POSTGRES_USER -d $env:POSTGRES_DB --no-owner --no-privileges $dump
        if ($LASTEXITCODE -ne 0) { throw 'database restore failed' }
    }
}
Remove-Item Env:\PGPASSWORD

Write-Step 'migrations + admin'
Push-Location (Join-Path $Root 'backend')
try {
    & $Python -m alembic upgrade head
    if ($LASTEXITCODE -ne 0) { throw 'alembic upgrade failed' }
    & $Python -m app.cli.bootstrap_admin
} finally { Pop-Location }

foreach ($f in 'model_artifacts\detection\yolo11n.pt', 'model_artifacts\cheating\x3d_l_v1_best.pt', 'frontend\dist\index.html') {
    if (-not (Test-Path (Join-Path $Root $f))) { throw "missing $f" }
}
if ($Firewall) {
    New-NetFirewallRule -DisplayName "ExamGuard $env:PORT" -Direction Inbound -Protocol TCP -LocalPort $env:PORT -Action Allow | Out-Null
    Write-Host "   firewall: TCP $env:PORT open"
}
Write-Host ''
Write-Host "Installed. Admin login: $env:ADMIN_USERNAME / $env:ADMIN_PASSWORD (a restored database keeps its own accounts)" -ForegroundColor Green
Write-Host 'Start: deploy\windows\start.cmd'
