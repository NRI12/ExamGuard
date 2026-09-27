# Optional HTTPS without a domain: Cloudflare quick tunnel to the local ExamGuard. Prints the public
# https://*.trycloudflare.com URL (new URL on every start). With HTTPS, set COOKIE_SECURE=true in .env.
. (Join-Path $PSScriptRoot 'common.ps1')
Import-DotEnv
$cloudflared = Join-Path $Tools 'cloudflared.exe'
Get-File 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' $cloudflared
$log = Join-Path $Logs 'tunnel.log'
Start-Process -FilePath $cloudflared -ArgumentList 'tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$env:PORT" `
    -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput (Join-Path $Logs 'tunnel.out.log') | Out-Null
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 1
    $match = Select-String -Path $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($match) { Write-Host $match.Matches[0].Value -ForegroundColor Green; exit 0 }
}
Get-Content $log -Tail 20
