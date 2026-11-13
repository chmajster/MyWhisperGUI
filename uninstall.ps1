[CmdletBinding(SupportsShouldProcess=$true)]
param([switch]$RemoveData)
$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$names = @('.venv')
if ($RemoveData) { $names += @('data', 'models', 'output', 'logs') }
foreach ($name in $names) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $root $name))
    if (-not $target.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe uninstall path.' }
    if ((Test-Path -LiteralPath $target) -and $PSCmdlet.ShouldProcess($target, 'Remove')) {
        $item = Get-Item -Force -LiteralPath $target
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Refusing to remove a linked directory: $target" }
        $links = @(Get-ChildItem -Force -LiteralPath $target -Recurse -Attributes ReparsePoint -ErrorAction Stop)
        if ($links.Count -gt 0) { throw "Refusing recursive removal because linked items were found in $target" }
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}
if ($WhatIfPreference) { Write-Host 'Preview only; no files were removed.' }
else { Write-Host 'Environment removed. Source files and any custom external directories are preserved.' }
if (-not $RemoveData) { Write-Host 'Models, recordings, history, and exports were preserved.' }
