[CmdletBinding()]
param(
    [switch]$CleanBuildFiles,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'
$LatexDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $LatexDir '..\..')
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$OutDir = Join-Path $LatexDir 'out'
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = 'python'
}
if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

Push-Location $LatexDir
try {
    & latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=out main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "latexmk failed with exit code $LASTEXITCODE"
    }

    $BuiltPdf = Join-Path $OutDir 'main.pdf'
    $PdfPath = Join-Path $LatexDir 'main.pdf'
    if (-not (Test-Path -LiteralPath $BuiltPdf)) {
        throw "Expected built PDF not found: $BuiltPdf"
    }
    Copy-Item -LiteralPath $BuiltPdf -Destination $PdfPath -Force

    & $Python .\check_main_pdf_fresh.py
    if ($LASTEXITCODE -ne 0) {
        throw "main.pdf freshness check failed with exit code $LASTEXITCODE"
    }

    if ($CleanBuildFiles) {
        & latexmk -c -outdir=out main.tex
        if ($LASTEXITCODE -ne 0) {
            throw "latexmk cleanup failed with exit code $LASTEXITCODE"
        }
    }

    Write-Host "Fresh PDF: $PdfPath"
    if ($Open) {
        Invoke-Item -LiteralPath $PdfPath
    }
}
finally {
    Pop-Location
}
