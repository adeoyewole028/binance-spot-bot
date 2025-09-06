param(
    [int]$IntervalSeconds = 120
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Python venv not found at $python. Activate or create the venv first."
    exit 1
}
Write-Host "Starting loop. Interval: $IntervalSeconds seconds. Press Ctrl+C to stop." -ForegroundColor Green
while ($true) {
    try {
        & $python "$root\bot.py"
    }
    catch {
        Write-Warning "Run failed: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
