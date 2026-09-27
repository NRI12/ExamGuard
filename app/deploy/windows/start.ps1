# Start PostgreSQL and ExamGuard in the background: http://<this-machine>:$PORT
# The first start compiles the X3D model on the GPU (up to a few minutes).
. (Join-Path $PSScriptRoot 'common.ps1')
Import-DotEnv
Start-Postgres
$pidFile = Join-Path $Logs 'backend.pid'
if ((Test-Path $pidFile) -and (Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue)) {
    Write-Host "already running (pid $(Get-Content $pidFile))"; exit 0
}
$process = Start-Process -FilePath $Python -WorkingDirectory (Join-Path $Root 'backend') -WindowStyle Hidden -PassThru `
    -ArgumentList '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', $env:PORT, '--proxy-headers' `
    -RedirectStandardOutput (Join-Path $Logs 'backend.log') -RedirectStandardError (Join-Path $Logs 'backend.err.log')
Set-Content $pidFile $process.Id
for ($i = 0; $i -lt 90; $i++) {
    try {
        Invoke-WebRequest "http://127.0.0.1:$env:PORT/api/v1/health" -UseBasicParsing -TimeoutSec 3 | Out-Null
        $ips = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' }).IPAddress
        Write-Host "ExamGuard running: http://localhost:$env:PORT" -ForegroundColor Green
        foreach ($ip in $ips) { Write-Host "  on the LAN: http://${ip}:$env:PORT" }
        exit 0
    } catch {
        if ($process.HasExited) { Get-Content (Join-Path $Logs 'backend.err.log') -Tail 40; throw 'backend exited' }
        Start-Sleep -Seconds 2
    }
}
Get-Content (Join-Path $Logs 'backend.err.log') -Tail 40
throw 'backend not answering after 3 minutes'
