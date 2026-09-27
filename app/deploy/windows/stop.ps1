# Stop ExamGuard and its PostgreSQL.
. (Join-Path $PSScriptRoot 'common.ps1')
Import-DotEnv
$pidFile = Join-Path $Logs 'backend.pid'
if (Test-Path $pidFile) {
    Stop-Process -Id (Get-Content $pidFile) -Force -ErrorAction SilentlyContinue
    Remove-Item $pidFile
    Write-Host 'backend stopped'
}
if (Test-Postgres) { & (Join-Path $PgBin 'pg_ctl.exe') -D $PgData -m fast stop | Out-Null; Write-Host 'PostgreSQL stopped' }
