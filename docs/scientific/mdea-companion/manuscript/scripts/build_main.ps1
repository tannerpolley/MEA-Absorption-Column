[CmdletBinding()]
param(
    [switch]$CleanBuildFiles,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LatexDir = Resolve-Path (Join-Path $ScriptDir '..')
$RepoRoot = Resolve-Path (Join-Path $LatexDir '..\..')
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$OutDir = Join-Path $LatexDir 'builds'
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = 'python'
}
if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

Push-Location $LatexDir
try {
    & $Python .\scripts\latex_workflows.py sync-bibliography
    if ($LASTEXITCODE -ne 0) {
        throw "Zotero bibliography sync failed with exit code $LASTEXITCODE"
    }

    & latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=builds main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "latexmk failed with exit code $LASTEXITCODE"
    }

    $BuiltPdf = Join-Path $OutDir 'main.pdf'
    if (-not (Test-Path -LiteralPath $BuiltPdf)) {
        throw "Expected built PDF not found: $BuiltPdf"
    }

    & $Python .\scripts\check_main_pdf_fresh.py
    if ($LASTEXITCODE -ne 0) {
        throw "builds\main.pdf freshness check failed with exit code $LASTEXITCODE"
    }

    if ($CleanBuildFiles) {
        & latexmk -c -outdir=builds main.tex
        if ($LASTEXITCODE -ne 0) {
            throw "latexmk cleanup failed with exit code $LASTEXITCODE"
        }
    }

    Write-Host "Fresh PDF: $BuiltPdf"
    if ($Open) {
        Invoke-Item -LiteralPath $BuiltPdf
    }
}
finally {
    Pop-Location
}
