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
    [string[]]$AssetDirectories = @('figs', 'Figures', 'thumbnails'),
    [string[]]$RepoAssetDirectories = @('analyses\nccc_validation\results\final'),
    [string[]]$StaleMirrorDirectories = @('benchmark_figures'),
    [string[]]$StaleMirrorFiles = @(
        'main.pdf',
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

$latexFiles = Get-ChildItem -LiteralPath $latexSourcePath -File |
    Where-Object { $_.Name -ne $scriptName -and $_.Name -notin $buildFileNames }

foreach ($file in $latexFiles) {
    $destination = Join-Path $mirrorRootPath $file.Name
    if ($PSCmdlet.ShouldProcess($destination, "Copy $($file.FullName)")) {
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }
}

foreach ($assetDirectory in $AssetDirectories) {
    $sourceAssetPath = Join-Path $docsRootPath $assetDirectory
    if (-not (Test-Path -LiteralPath $sourceAssetPath)) {
        Write-Warning "Skipping missing asset directory: $sourceAssetPath"
        continue
    }

    $destinationAssetPath = Join-Path $mirrorRootPath $assetDirectory
    if ($PSCmdlet.ShouldProcess($destinationAssetPath, "Sync asset directory $sourceAssetPath")) {
        Invoke-RobocopyChecked -Source $sourceAssetPath -Destination $destinationAssetPath
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
