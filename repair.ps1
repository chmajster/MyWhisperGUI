[CmdletBinding()]
param([switch]$SkipGpuTest, [switch]$Full, [switch]$SkipCuda, [switch]$Cuda, [string]$GpuId = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot '.venv-py312\Scripts\python.exe'
if ($Full -and (Test-Path -LiteralPath $venvPython)) {
    & $venvPython -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'The .venv-py312 environment must use 64-bit Python 3.12.' }
    Write-Host 'Reinstalling declared application packages...'
    & $venvPython -m pip install --force-reinstall -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Application dependency reinstall failed.' }
}
& (Join-Path $PSScriptRoot 'install.ps1') -SkipGpuTest:$SkipGpuTest -SkipCuda:$SkipCuda -Cuda:$Cuda -GpuId $GpuId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed. Recreate .venv-py312 and run install.ps1.' }
Write-Host 'Repair and dependency checks completed.'
