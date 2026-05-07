[CmdletBinding()]
param(
    [switch]$CleanBuildFiles,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'
$LatexDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $LatexDir '..\..')
$Python = Join-Path $env:USERPROFILE '.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = 'python'
}

Push-Location $LatexDir
try {
    & latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "latexmk failed with exit code $LASTEXITCODE"
    }

    & $Python .\check_main_pdf_fresh.py
    if ($LASTEXITCODE -ne 0) {
        throw "main.pdf freshness check failed with exit code $LASTEXITCODE"
    }

    if ($CleanBuildFiles) {
        & latexmk -c main.tex
        if ($LASTEXITCODE -ne 0) {
            throw "latexmk cleanup failed with exit code $LASTEXITCODE"
        }
    }

    $PdfPath = Join-Path $LatexDir 'main.pdf'
    Write-Host "Fresh PDF: $PdfPath"
    if ($Open) {
        Invoke-Item -LiteralPath $PdfPath
    }
}
finally {
    Pop-Location
}
