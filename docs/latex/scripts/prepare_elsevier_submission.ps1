<#
.SYNOPSIS
Create a flat Elsevier Editorial Manager LaTeX source package.

.DESCRIPTION
Elsevier's LaTeX instructions state that Editorial Manager cannot process
LaTeX submissions that rely on subfolders. This script keeps the repository
source organized, but writes a flat copy under docs\latex\builds for upload.

The copied TeX files rewrite figures/<name>, tables/<name>, sections/<name>,
and appendices/<name> references to bare filenames. Referenced figure files,
table .tex files, section .tex files, and appendix .tex files are copied to the
same folder as main.tex.
#>

[CmdletBinding()]
param(
    [string]$OutputRoot,
    [switch]$Zip
)

$ErrorActionPreference = 'Stop'

$scriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = Split-Path -Parent $PSCommandPath
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path (Split-Path -Parent $scriptRoot) 'builds\elsevier_submission_flat'
}

function Copy-TextWithFlatFigurePaths {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $text = Get-Content -LiteralPath $Source -Raw
    $text = $text -replace 'figures/', ''
    $text = $text -replace 'tables/', ''
    $text = $text -replace 'sections/', ''
    $text = $text -replace 'appendices/', ''
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Destination, $text, $utf8NoBom)
}

$latexRoot = (Resolve-Path -LiteralPath (Join-Path $scriptRoot '..')).Path
$buildsRootParent = Join-Path $latexRoot 'builds'
$fullOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$fullBuildsParent = [System.IO.Path]::GetFullPath($buildsRootParent)

if (-not $fullOutputRoot.StartsWith($fullBuildsParent, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must be inside docs\latex\builds: $fullOutputRoot"
}

if (Test-Path -LiteralPath $fullOutputRoot) {
    Remove-Item -LiteralPath $fullOutputRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $fullOutputRoot | Out-Null

$texFiles = @(
    Get-Item -LiteralPath (Join-Path $latexRoot 'main.tex')
) + @(Get-ChildItem -LiteralPath (Join-Path $latexRoot 'sections') -Filter '*.tex' -File) +
    @(Get-ChildItem -LiteralPath (Join-Path $latexRoot 'appendices') -Filter '*.tex' -File)

foreach ($texFile in $texFiles) {
    Copy-TextWithFlatFigurePaths `
        -Source $texFile.FullName `
        -Destination (Join-Path $fullOutputRoot $texFile.Name)
}

$tableFiles = Get-ChildItem -LiteralPath (Join-Path $latexRoot 'tables') -Filter '*.tex' -File
foreach ($tableFile in $tableFiles) {
    Copy-TextWithFlatFigurePaths -Source $tableFile.FullName -Destination (Join-Path $fullOutputRoot $tableFile.Name)
}

$plainFiles = @(
    'references.bib',
    'builds\main.pdf'
)

foreach ($plainFile in $plainFiles) {
    Copy-Item -LiteralPath (Join-Path $latexRoot $plainFile) -Destination (Join-Path $fullOutputRoot ([System.IO.Path]::GetFileName($plainFile))) -Force
}

$builtBbl = Join-Path $latexRoot 'builds\main.bbl'
if (Test-Path -LiteralPath $builtBbl) {
    Copy-Item -LiteralPath $builtBbl -Destination (Join-Path $fullOutputRoot 'main.bbl') -Force
}

$sourceText = ($texFiles | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw }) -join "`n"
$figureMatches = [regex]::Matches($sourceText, '\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
$figures = foreach ($match in $figureMatches) {
    $match.Groups[1].Value
}

foreach ($figure in ($figures | Sort-Object -Unique)) {
    $sourceFigure = Join-Path $latexRoot $figure
    if (-not (Test-Path -LiteralPath $sourceFigure)) {
        throw "Referenced figure does not exist: $figure"
    }
    Copy-Item -LiteralPath $sourceFigure -Destination (Join-Path $fullOutputRoot ([System.IO.Path]::GetFileName($figure))) -Force
}

if ($Zip) {
    $zipPath = "$fullOutputRoot.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $fullOutputRoot '*') -DestinationPath $zipPath -Force
    Write-Host "Flat Elsevier source package: $zipPath"
}

Write-Host "Flat Elsevier source folder: $fullOutputRoot"
