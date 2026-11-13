[CmdletBinding()]
param([switch]$SkipGpuTest)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
Set-Location -LiteralPath $PSScriptRoot
if ($env:OS -ne 'Windows_NT') { throw 'Windows 10/11 is required.' }

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    $python = $null
    $pythonArgs = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @('-3.11', '-3.12')) {
            & py $version -c 'import sys; print(sys.version)' 2>$null
            if ($LASTEXITCODE -eq 0) { $python = 'py'; $pythonArgs = @($version); break }
        }
    }
    if (-not $python -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)'
        if ($LASTEXITCODE -eq 0) { $python = 'python' }
    }
    if (-not $python) { throw 'Install 64-bit Python 3.11 or 3.12, then run install.ps1 again.' }
    & $python @pythonArgs -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating the virtual environment failed.' }
}
& $venvPython -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) and sys.maxsize>2**32 else 1)'
if ($LASTEXITCODE -ne 0) { throw 'The virtual environment must use 64-bit Python 3.11 or 3.12.' }

foreach ($directory in @('data', 'data\uploads', 'data\recordings', 'data\tmp', 'models', 'output', 'logs')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot $directory) | Out-Null
}
Write-Host 'Installing application dependencies...'
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Installing the official PyTorch CUDA 12.8 wheel (large download)...'
& $venvPython -m pip install 'torch==2.9.1+cu128' --index-url 'https://download.pytorch.org/whl/cu128'
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. Check the internet connection and run repair.ps1.' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { & nvidia-smi --query-gpu=name,driver_version --format=csv }
else { Write-Warning 'nvidia-smi not found. Install the current NVIDIA driver for CUDA.' }
Write-Host 'Environment diagnostics:'
if ($SkipGpuTest) { & $venvPython -m app.core.gpu_manager }
else { & $venvPython -m app.core.gpu_manager --test --download }
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'CUDA inference failed. See the report above; CPU can still be selected explicitly.'
    exit 1
}
Write-Host 'Installation complete. Run start.bat.'
