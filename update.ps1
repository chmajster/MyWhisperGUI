[CmdletBinding()]
param([switch]$SkipGpuTest, [switch]$SkipCuda, [switch]$Cuda, [string]$GpuId = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ((Test-Path -LiteralPath '.git') -and (Get-Command git -ErrorAction SilentlyContinue)) {
    $changes = & git status --porcelain
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Git status.' }
    if ($changes) { Write-Warning 'Local Git changes detected: source update skipped to protect them.' }
    else {
        & git pull --ff-only
        if ($LASTEXITCODE -ne 0) { throw 'Git update failed; local files have not been reset.' }
    }
} else { Write-Host 'No Git checkout: update source files from your distribution separately.' }
& (Join-Path $PSScriptRoot 'install.ps1') -SkipGpuTest:$SkipGpuTest -SkipCuda:$SkipCuda -Cuda:$Cuda -GpuId $GpuId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
