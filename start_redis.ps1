# PowerShell script to install and start Redis on Windows

Write-Host "Setting up Redis for Windows..." -ForegroundColor Green

# Check if Chocolatey is installed
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Install Redis
Write-Host "Installing Redis..." -ForegroundColor Yellow
choco install redis-64 -y

# Start Redis service
Write-Host "Starting Redis service..." -ForegroundColor Yellow
redis-server --service-start

Write-Host "Redis should now be running on localhost:6379" -ForegroundColor Green
Write-Host "You can test it by running: redis-cli ping" -ForegroundColor Cyan
