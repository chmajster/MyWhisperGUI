[CmdletBinding()]
param([switch]$SkipGpuTest, [switch]$SkipCuda, [switch]$Cuda, [string]$GpuId = '')
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
Set-Location -LiteralPath $PSScriptRoot
if ($env:OS -ne 'Windows_NT') { throw 'Windows 10/11 is required.' }

$venvPython = Join-Path $PSScriptRoot '.venv-py312\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    $python = $null
    $pythonArgs = @()
    if ($env:WHYSPER_PYTHON) {
        $python = $env:WHYSPER_PYTHON
    }
    if (-not $python -and (Get-Command py -ErrorAction SilentlyContinue)) {
        & py -3.12 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { $python = 'py'; $pythonArgs = @('-3.12') }
    }
    if (-not $python) {
        foreach ($candidate in @('python3.12', 'python')) {
            if (Get-Command $candidate -ErrorAction SilentlyContinue) {
                & $candidate -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)' 2>$null
                if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
            }
        }
    }
    if (-not $python -and (Get-Command uv -ErrorAction SilentlyContinue)) {
        $previousUvCache = $env:UV_CACHE_DIR
        try {
            $env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'data\tmp\uv-cache'
            $candidate = @(& uv python find 3.12 --offline 2>$null)
            if ($LASTEXITCODE -eq 0 -and $candidate.Count -eq 1) { $python = $candidate[0] }
        } finally {
            if ($null -eq $previousUvCache) { Remove-Item Env:UV_CACHE_DIR -ErrorAction SilentlyContinue }
            else { $env:UV_CACHE_DIR = $previousUvCache }
        }
    }
    if (-not $python) { throw 'Install 64-bit Python 3.12 or set WHYSPER_PYTHON to its executable, then run install.ps1 again.' }
    & $python @pythonArgs -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'The selected interpreter must be 64-bit Python 3.12.' }
    & $python @pythonArgs -m venv (Join-Path $PSScriptRoot '.venv-py312')
    if ($LASTEXITCODE -ne 0) { throw 'Creating the virtual environment failed.' }
}
& $venvPython -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)'
if ($LASTEXITCODE -ne 0) { throw 'The .venv-py312 environment must use 64-bit Python 3.12.' }

foreach ($directory in @('data', 'data\uploads', 'data\recordings', 'data\tmp', 'models', 'output', 'logs')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot $directory) | Out-Null
}
Write-Host 'Installing application dependencies...'
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
$installCuda = -not $SkipCuda -and ($Cuda -or [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue))
if ($installCuda) {
    Write-Host 'Installing the official PyTorch CUDA 12.8 wheel (large download)...'
    & $venvPython -m pip install 'torch==2.9.1+cu128' --index-url 'https://download.pytorch.org/whl/cu128'
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'PyTorch CUDA installation failed. CPU and Vulkan can still be used; run repair.ps1 -Cuda to retry.'
        $installCuda = $false
    }
} else {
    Write-Host 'Skipping optional NVIDIA CUDA libraries. CPU and Vulkan do not require PyTorch.'
}
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { & nvidia-smi --query-gpu=name,driver_version --format=csv }
Write-Host 'Environment diagnostics:'
$gpuArgs = @()
if ($GpuId) { $gpuArgs = @('--gpu-id', $GpuId) }
if ($SkipGpuTest -or -not $installCuda) { & $venvPython -m app.core.gpu_manager @gpuArgs }
else { & $venvPython -m app.core.gpu_manager --test --download @gpuArgs }
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'CUDA inference failed. See the report above; CPU and Vulkan remain available.'
}
Write-Host 'Installation complete. Run start.bat.'
exit 0
