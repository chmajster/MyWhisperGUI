[CmdletBinding()]
param([switch]$SkipGpuTest, [switch]$Full, [string]$GpuId = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if ($Full -and (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'Reinstalling declared packages and the CUDA PyTorch wheel...'
    & $venvPython -m pip install --force-reinstall -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Application dependency reinstall failed.' }
    & $venvPython -m pip install --force-reinstall 'torch==2.9.1+cu128' --index-url 'https://download.pytorch.org/whl/cu128'
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch reinstall failed.' }
}
& (Join-Path $PSScriptRoot 'install.ps1') -SkipGpuTest:$SkipGpuTest -GpuId $GpuId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed. Recreate .venv and run install.ps1.' }
Write-Host 'Repair and dependency checks completed.'
