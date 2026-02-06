
# 0. Fix encoding issues (Force UTF-8)
chcp 65001 > $null
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 1. Start Edge browser with remote debugging enabled (in background)
# Note: Start-Process ensures it doesn't block the current terminal
Start-Process "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" -ArgumentList "--remote-debugging-port=9222", "--user-data-dir=C:\temp\edge-cdp-profile"

# 2. Wait a few seconds for the browser to fully start
Write-Host "Starting Edge browser, please wait..."
Start-Sleep -Seconds 3

# 3. Set necessary environment variables
$env:BROWSER_CDP_URL = "http://127.0.0.1:9222"
$env:VISION_DETAIL_LEVEL = "high"
$env:MANUAL_LOGIN = "1"
$env:MANUAL_LOGIN_NAVIGATE = "1"
$env:MANUAL_LOGIN_URL = "https://marsprod.service-now.com/kb_view.do?sys_kb_id=31a761fc93f23e9c4ea774f86cba10ae"

# 4. Run the Python download script
Write-Host "Starting download task..."
python .\KBDownload.py