<#
.SYNOPSIS
Sync this LaTeX manuscript folder into the flat Overleaf mirror checkout.

.DESCRIPTION
The source manuscript lives in docs\latex inside the MEA repository. The mirror
checkout is a separate Git repository connected to Overleaf, and its root should
contain the manuscript files directly, not a nested latex folder.

This script intentionally excludes itself so the mirror remains a clean Overleaf
project. Use -WhatIf to preview the sync without writing to the mirror.
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$MirrorRoot = 'C:\Users\Tanner\Documents\git\Publications\MEA-Absorption-Column-LaTeX',
    [string[]]$ExcludedSourceEntries = @('scripts', 'builds'),
    [switch]$CleanBuildFiles
)

$ErrorActionPreference = 'Stop'

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label does not exist: $Path"
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-PortableRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFullPath = [System.IO.Path]::GetFullPath($BasePath)
    $targetFullPath = [System.IO.Path]::GetFullPath($TargetPath)
    if (-not $baseFullPath.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $baseFullPath += [System.IO.Path]::DirectorySeparatorChar
    }

    $baseUri = [System.Uri]::new($baseFullPath)
    $targetUri = [System.Uri]::new($targetFullPath)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

function Get-ExactMirrorChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$MirrorRootPath,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    $parts = $RelativePath -split '[\\/]'
    $currentPath = $MirrorRootPath
    foreach ($part in $parts) {
        $child = Get-ChildItem -LiteralPath $currentPath -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ceq $part } |
            Select-Object -First 1
        if ($null -eq $child) {
            return $null
        }
        $currentPath = $child.FullName
    }

    return $currentPath
}

$latexSourcePath = Resolve-RequiredPath -Path (Join-Path $PSScriptRoot '..') -Label 'LaTeX source folder'
$mirrorRootPath = Resolve-RequiredPath -Path $MirrorRoot -Label 'Mirror checkout root'
$repoRootPath = Resolve-RequiredPath -Path (Join-Path $latexSourcePath '..\..') -Label 'repository root'
$python = Join-Path $repoRootPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = 'python'
}
$bibliographyArgs = @((Join-Path $latexSourcePath 'scripts\latex_workflows.py'), 'sync-bibliography')
if ($WhatIfPreference) {
    $bibliographyArgs += '--check'
}
& $python @bibliographyArgs
if ($LASTEXITCODE -ne 0) {
    throw "Zotero bibliography sync failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath (Join-Path $mirrorRootPath '.git'))) {
    throw "Mirror root is not a Git checkout: $mirrorRootPath"
}

Write-Host "Source LaTeX folder: $latexSourcePath"
Write-Host "Mirror root: $mirrorRootPath"

$sourceEntries = Get-ChildItem -LiteralPath $latexSourcePath -Force |
    Where-Object { $_.Name -notin $ExcludedSourceEntries }
$allowedMirrorNames = @('.git') + @($sourceEntries | ForEach-Object { $_.Name })

$figureSyncScript = Join-Path $latexSourcePath 'scripts\sync_latex_figures.ps1'
if (Test-Path -LiteralPath $figureSyncScript) {
    if ($PSCmdlet.ShouldProcess($figureSyncScript, 'Refresh LaTeX figures from project outputs')) {
        & $figureSyncScript
    }
}

foreach ($sourceEntry in $sourceEntries) {
    $destinationPath = Join-Path $mirrorRootPath $sourceEntry.Name
    if ($sourceEntry.PSIsContainer) {
        if ($PSCmdlet.ShouldProcess($destinationPath, "Sync directory $($sourceEntry.FullName)")) {
            $exactDestination = Get-ExactMirrorChildPath -MirrorRootPath $mirrorRootPath -RelativePath $sourceEntry.Name
            if ($null -ne $exactDestination) {
                Remove-Item -LiteralPath $exactDestination -Recurse -Force
            }
            elseif (Test-Path -LiteralPath $destinationPath) {
                Remove-Item -LiteralPath $destinationPath -Recurse -Force
            }
            New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null

            $assetFiles = Get-ChildItem -LiteralPath $sourceEntry.FullName -Recurse -File -Force
            foreach ($assetFile in $assetFiles) {
                if ($assetFile.Name.EndsWith('.synctex.gz', [System.StringComparison]::OrdinalIgnoreCase)) {
                    continue
                }
                if ($assetFile.Extension -in @('.abs', '.aux', '.bbl', '.blg', '.fdb_latexmk', '.fls', '.log', '.out', '.xdv')) {
                    continue
                }

                $relativePath = Get-PortableRelativePath -BasePath $sourceEntry.FullName -TargetPath $assetFile.FullName
                $assetDestinationPath = Join-Path $destinationPath $relativePath
                $destinationParent = Split-Path -Parent $assetDestinationPath
                if (-not (Test-Path -LiteralPath $destinationParent)) {
                    New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
                }
                Copy-Item -LiteralPath $assetFile.FullName -Destination $assetDestinationPath -Force
            }
        }
    }
    else {
        if ($PSCmdlet.ShouldProcess($destinationPath, "Copy $($sourceEntry.FullName)")) {
            Copy-Item -LiteralPath $sourceEntry.FullName -Destination $destinationPath -Force
        }
    }
}

foreach ($mirrorEntry in Get-ChildItem -LiteralPath $mirrorRootPath -Force) {
    if ($allowedMirrorNames -ccontains $mirrorEntry.Name) {
        continue
    }

    if ($PSCmdlet.ShouldProcess($mirrorEntry.FullName, 'Remove mirror item outside docs\latex projection')) {
        Remove-Item -LiteralPath $mirrorEntry.FullName -Recurse -Force
    }
}

if (-not $WhatIfPreference) {
    foreach ($allowedName in $allowedMirrorNames) {
        if ($allowedName -eq '.git') {
            continue
        }
        $exactPath = Get-ExactMirrorChildPath -MirrorRootPath $mirrorRootPath -RelativePath $allowedName
        if ($null -eq $exactPath) {
            throw "Expected mirror item missing after sync: $allowedName"
        }
    }
}

if (-not $WhatIfPreference) {
    $figureReferences = New-Object System.Collections.Generic.List[string]
    $texFiles = @(Get-ChildItem -LiteralPath $mirrorRootPath -Filter '*.tex' -File)
    $mirrorSectionsPath = Join-Path $mirrorRootPath 'sections'
    if (Test-Path -LiteralPath $mirrorSectionsPath) {
        $texFiles += @(Get-ChildItem -LiteralPath $mirrorSectionsPath -Filter '*.tex' -File)
    }
    $mirrorAppendicesPath = Join-Path $mirrorRootPath 'appendices'
    if (Test-Path -LiteralPath $mirrorAppendicesPath) {
        $texFiles += @(Get-ChildItem -LiteralPath $mirrorAppendicesPath -Filter '*.tex' -File)
    }

    foreach ($texFile in $texFiles) {
        if (-not (Test-Path -LiteralPath $texFile.FullName)) {
            continue
        }

        $content = Get-Content -LiteralPath $texFile.FullName -Raw
        $matches = [regex]::Matches($content, '\\includegraphics(?:\[[^\]]*\])?\{(?<path>figures/[^}]+)\}')
        foreach ($match in $matches) {
            $figureReferences.Add($match.Groups['path'].Value)
        }
    }

    $missingMirrorReferences = New-Object System.Collections.Generic.List[string]
    foreach ($reference in ($figureReferences | Sort-Object -Unique)) {
        $relativePath = $reference.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
        $expectedPath = Join-Path $mirrorRootPath $relativePath
        if (-not (Test-Path -LiteralPath $expectedPath)) {
            $missingMirrorReferences.Add($reference)
        }
    }

    if ($missingMirrorReferences.Count -gt 0) {
        $missing = ($missingMirrorReferences | Sort-Object -Unique) -join ', '
        throw "Missing mirror figure references after sync: $missing"
    }
}

if ($CleanBuildFiles) {
    $buildPatterns = @(
        '*.abs',
        '*.aux',
        '*.bbl',
        '*.blg',
        '*.fdb_latexmk',
        '*.fls',
        '*.log',
        '*.out',
        '*.synctex.gz',
        '*.xdv'
    )

    foreach ($pattern in $buildPatterns) {
        $buildFiles = Get-ChildItem -LiteralPath $mirrorRootPath -Filter $pattern -File -ErrorAction SilentlyContinue
        foreach ($buildFile in $buildFiles) {
            if ($PSCmdlet.ShouldProcess($buildFile.FullName, 'Remove LaTeX build artifact')) {
                Remove-Item -LiteralPath $buildFile.FullName -Force
            }
        }
    }
}

Write-Host 'LaTeX mirror sync complete.'
