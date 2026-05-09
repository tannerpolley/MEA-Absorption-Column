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
    [string]$MirrorRoot = 'C:\Users\Tanner\Documents\git\LaTeX-Projects\MEA-Absorption-Column-LaTeX',
    [string[]]$AssetDirectories = @('figures', 'tables', 'sections'),
    [string[]]$RepoAssetDirectories = @('analyses\nccc_validation\results\final'),
    [string[]]$StaleMirrorDirectories = @('benchmark_figures', 'figs', 'thumbnails'),
    [string[]]$StaleMirrorFiles = @(
        'main.pdf',
        'benchmark_results_section.tex',
        'revised_benchmark_results.tex',
        'Figures\profile_3C_epcsaft_neutral.png',
        'Figures\profile_3C_ideal_henry.png',
        'Figures\profile_7C_epcsaft_neutral.png'
    ),
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

function Invoke-RobocopyChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string[]]$ExtraArgs = @()
    )

    $args = @(
        $Source,
        $Destination,
        '/E',
        '/R:2',
        '/W:1',
        '/NFL',
        '/NDL',
        '/NJH',
        '/NJS',
        '/XD',
        '.git',
        '__pycache__',
        '.pytest_cache',
        '/XF',
        '*.aux',
        '*.bbl',
        '*.blg',
        '*.fdb_latexmk',
        '*.fls',
        '*.log',
        '*.out',
        '*.synctex.gz',
        '*.xdv'
    ) + $ExtraArgs

    if ($WhatIfPreference) {
        $args += '/L'
    }

    & robocopy @args | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -gt 7) {
        throw "robocopy failed with exit code $exitCode while syncing '$Source' to '$Destination'"
    }
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

function Sync-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Destination)) {
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    }

    $excludedNames = @(
        '.git',
        '__pycache__',
        '.pytest_cache'
    )
    $excludedExtensions = @(
        '.abs',
        '.aux',
        '.bbl',
        '.blg',
        '.fdb_latexmk',
        '.fls',
        '.log',
        '.out',
        '.xdv'
    )

    Get-ChildItem -LiteralPath $Destination -Force | ForEach-Object {
        if ($_.Name -in $excludedNames) {
            return
        }
        if ($PSCmdlet.ShouldProcess($_.FullName, 'Remove stale asset mirror item')) {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force
        }
    }

    Get-ChildItem -LiteralPath $Source -Recurse -Force | ForEach-Object {
        if ($_.PSIsContainer) {
            if ($_.Name -in $excludedNames) {
                return
            }
            return
        }

        if ($_.Name -eq (Split-Path -Leaf $PSCommandPath)) {
            return
        }
        if ($_.Name -eq 'sync_latex_figures.ps1') {
            return
        }
        if ($_.Name.EndsWith('.synctex.gz', [System.StringComparison]::OrdinalIgnoreCase)) {
            return
        }
        if ($_.Extension -in $excludedExtensions) {
            return
        }

        $relativePath = Get-PortableRelativePath -BasePath $Source -TargetPath $_.FullName
        $destinationPath = Join-Path $Destination $relativePath
        $destinationParent = Split-Path -Parent $destinationPath
        if (-not (Test-Path -LiteralPath $destinationParent)) {
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        }

        if ($PSCmdlet.ShouldProcess($destinationPath, "Copy $($_.FullName)")) {
            Copy-Item -LiteralPath $_.FullName -Destination $destinationPath -Force
        }
    }
}

$latexSourcePath = Resolve-RequiredPath -Path $PSScriptRoot -Label 'LaTeX source folder'
$docsRootPath = Resolve-RequiredPath -Path (Split-Path -Parent $latexSourcePath) -Label 'Docs root'
$repoRootPath = Resolve-RequiredPath -Path (Split-Path -Parent $docsRootPath) -Label 'Repository root'
$mirrorRootPath = Resolve-RequiredPath -Path $MirrorRoot -Label 'Mirror checkout root'
$scriptName = Split-Path -Leaf $PSCommandPath
$buildFileNames = @(
    'main.abs',
    'main.aux',
    'main.bbl',
    'main.blg',
    'main.fdb_latexmk',
    'main.fls',
    'main.log',
    'main.out',
    'main.pdf',
    'main.synctex.gz',
    'main.xdv'
)

if (-not (Test-Path -LiteralPath (Join-Path $mirrorRootPath '.git'))) {
    throw "Mirror root is not a Git checkout: $mirrorRootPath"
}

Write-Host "Source LaTeX folder: $latexSourcePath"
Write-Host "Mirror root: $mirrorRootPath"

$figureSyncScript = Join-Path $latexSourcePath 'sync_latex_figures.ps1'
if (Test-Path -LiteralPath $figureSyncScript) {
    if ($PSCmdlet.ShouldProcess($figureSyncScript, 'Refresh LaTeX figures from project outputs')) {
        & $figureSyncScript
    }
}

$excludedScriptNames = @($scriptName, 'sync_latex_figures.ps1')
$latexFiles = Get-ChildItem -LiteralPath $latexSourcePath -File |
    Where-Object { $_.Name -notin $excludedScriptNames -and $_.Name -notin $buildFileNames }

foreach ($file in $latexFiles) {
    $destination = Join-Path $mirrorRootPath $file.Name
    if ($PSCmdlet.ShouldProcess($destination, "Copy $($file.FullName)")) {
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }
}

foreach ($assetDirectory in $AssetDirectories) {
    $sourceAssetPath = Join-Path $latexSourcePath $assetDirectory
    if (-not (Test-Path -LiteralPath $sourceAssetPath)) {
        Write-Warning "Skipping missing asset directory: $sourceAssetPath"
        continue
    }

    $destinationAssetPath = Join-Path $mirrorRootPath $assetDirectory
    if ($PSCmdlet.ShouldProcess($destinationAssetPath, "Sync asset directory $sourceAssetPath")) {
        if (Test-Path -LiteralPath $destinationAssetPath) {
            Remove-Item -LiteralPath $destinationAssetPath -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $destinationAssetPath | Out-Null

        $assetFiles = Get-ChildItem -LiteralPath $sourceAssetPath -Recurse -File -Force
        foreach ($assetFile in $assetFiles) {
            if ($assetFile.Name -eq $scriptName -or $assetFile.Name -eq 'sync_latex_figures.ps1') {
                continue
            }
            if ($assetFile.Name.EndsWith('.synctex.gz', [System.StringComparison]::OrdinalIgnoreCase)) {
                continue
            }
            if ($assetFile.Extension -in @('.abs', '.aux', '.bbl', '.blg', '.fdb_latexmk', '.fls', '.log', '.out', '.xdv')) {
                continue
            }

            $relativePath = Get-PortableRelativePath -BasePath $sourceAssetPath -TargetPath $assetFile.FullName
            $destinationPath = Join-Path $destinationAssetPath $relativePath
            $destinationParent = Split-Path -Parent $destinationPath
            if (-not (Test-Path -LiteralPath $destinationParent)) {
                New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
            }
            Copy-Item -LiteralPath $assetFile.FullName -Destination $destinationPath -Force
        }
    }
}

foreach ($assetDirectory in $RepoAssetDirectories) {
    $sourceAssetPath = Join-Path $repoRootPath $assetDirectory
    if (-not (Test-Path -LiteralPath $sourceAssetPath)) {
        Write-Warning "Skipping missing repository asset directory: $sourceAssetPath"
        continue
    }

    $destinationAssetPath = Join-Path $mirrorRootPath $assetDirectory
    if ($PSCmdlet.ShouldProcess($destinationAssetPath, "Sync repository asset directory $sourceAssetPath")) {
        Invoke-RobocopyChecked -Source $sourceAssetPath -Destination $destinationAssetPath
    }
}

foreach ($staleDirectory in $StaleMirrorDirectories) {
    $stalePath = Join-Path $mirrorRootPath $staleDirectory
    if (-not (Test-Path -LiteralPath $stalePath)) {
        continue
    }

    if ($PSCmdlet.ShouldProcess($stalePath, 'Remove stale mirror directory')) {
        Remove-Item -LiteralPath $stalePath -Recurse -Force
    }
}

foreach ($staleFile in $StaleMirrorFiles) {
    $stalePath = Join-Path $mirrorRootPath $staleFile
    if (-not (Test-Path -LiteralPath $stalePath)) {
        continue
    }

    if ($PSCmdlet.ShouldProcess($stalePath, 'Remove stale mirror file')) {
        Remove-Item -LiteralPath $stalePath -Force
    }
}

$figureReferences = New-Object System.Collections.Generic.List[string]
$texFiles = @(Get-ChildItem -LiteralPath $mirrorRootPath -Filter '*.tex' -File)
$mirrorSectionsPath = Join-Path $mirrorRootPath 'sections'
if (Test-Path -LiteralPath $mirrorSectionsPath) {
    $texFiles += @(Get-ChildItem -LiteralPath $mirrorSectionsPath -Filter '*.tex' -File)
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
