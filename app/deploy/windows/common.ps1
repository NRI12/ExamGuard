# Shared helpers of the Windows scripts (dot-sourced).
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is much faster without the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Tools = Join-Path $PSScriptRoot 'tools'
$EnvFile = Join-Path $Root '.env'
$Python = Join-Path $Root 'backend\.venv\Scripts\python.exe'
$PgBin = Join-Path $Tools 'pgsql\bin'
$PgData = Join-Path $Root 'data\pgdata'
$FfmpegBin = Join-Path $Tools 'ffmpeg\bin'
$Logs = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force $Tools, $Logs | Out-Null

function Write-Step([string]$Text) { Write-Host "== $Text" -ForegroundColor Cyan }

function Get-File([string]$Url, [string]$Destination) {
    if (-not (Test-Path $Destination)) {
        Write-Host "   downloading $Url"
        Invoke-WebRequest -Uri $Url -OutFile "$Destination.part" -UseBasicParsing
        Move-Item "$Destination.part" $Destination
    }
}

function New-Secret([int]$Bytes = 32) {
    $buffer = New-Object byte[] $Bytes
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($buffer)
    return (($buffer | ForEach-Object { $_.ToString('x2') }) -join '')
}

function Write-Utf8NoBom([string]$Path, [string]$Content) {
    # PowerShell 5 writes a BOM with -Encoding UTF8, which would corrupt the first .env key
    [IO.File]::WriteAllText($Path, $Content, (New-Object Text.UTF8Encoding $false))
}

function Import-DotEnv {
    # KEY=VALUE lines of .env into this process environment (quotes stripped)
    foreach ($line in [IO.File]::ReadAllLines($EnvFile)) {
        if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
        $key, $value = $line -split '=', 2
        $value = $value.Trim().Trim('"')
        [Environment]::SetEnvironmentVariable($key.Trim(), $value, 'Process')
    }
    $env:Path = "$FfmpegBin;$PgBin;$env:Path"
}

function Test-Postgres {
    & (Join-Path $PgBin 'pg_ctl.exe') -D $PgData status *> $null
    return ($LASTEXITCODE -eq 0)
}

function Start-Postgres {
    if (-not (Test-Postgres)) {
        & (Join-Path $PgBin 'pg_ctl.exe') -D $PgData -l (Join-Path $Logs 'postgres.log') `
            -o "-p $env:PGPORT -h 127.0.0.1" -w start | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "PostgreSQL did not start, see logs\postgres.log" }
    }
}
