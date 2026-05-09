<#
.SYNOPSIS
Refresh manuscript figures from project-generated outputs.

.DESCRIPTION
Copies generated figure files referenced by docs\latex\main.tex and
docs\latex\sections\*.tex into docs\latex\figures. Static manuscript figures
already live in docs\latex\figures. This keeps the LaTeX source
folder self-contained before it is mirrored to Overleaf.
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param()

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

$latexSourcePath = Resolve-RequiredPath -Path (Join-Path $PSScriptRoot '..') -Label 'LaTeX source folder'
$docsRootPath = Resolve-RequiredPath -Path (Split-Path -Parent $latexSourcePath) -Label 'Docs root'
$repoRootPath = Resolve-RequiredPath -Path (Split-Path -Parent $docsRootPath) -Label 'Repository root'
$figureRootPath = Join-Path $latexSourcePath 'figures'

if (-not (Test-Path -LiteralPath $figureRootPath)) {
    if ($PSCmdlet.ShouldProcess($figureRootPath, 'Create LaTeX figures folder')) {
        New-Item -ItemType Directory -Force -Path $figureRootPath | Out-Null
    }
}

$figureCopies = @(
    @{ Source = 'analyses\nccc_validation\results\final\figures\c_case_thermo_benchmark.pdf'; Destination = 'c-case-thermo-benchmark.pdf' },
    @{ Source = 'analyses\nccc_validation\results\final\figures\c_case_campaign_temperature_overlays\3C_temperature_overlay.png'; Destination = 'case-3c-temperature-validation.png' },
    @{ Source = 'analyses\nccc_validation\results\final\figures\c_case_campaign_temperature_overlays\c_case_temperature_overlay_contact_sheet.png'; Destination = 'case-c-temperature-overlay.png' },
    @{ Source = 'analyses\nccc_validation\results\final\figures\error_regime_capture_error.pdf'; Destination = 'error-regime-capture-error.pdf' },
    @{ Source = 'analyses\nccc_validation\results\final\figures\calibration_uncertainty_band.pdf'; Destination = 'calibration-uncertainty-band.pdf' },
    @{ Source = 'analyses\nccc_validation\results\final\figures\method_case_solver_contrast.pdf'; Destination = 'method-case-solver-contrast.pdf' },
    @{ Source = 'analyses\nccc_validation\results\final\profiles\3C\ideal_henry\temperature_profile.png'; Destination = 'profile-3c-henry.png' },
    @{ Source = 'analyses\nccc_validation\results\final\profiles\3C\epcsaft_neutral\temperature_profile.png'; Destination = 'profile-3c-epcsaft.png' },
    @{ Source = 'analyses\nccc_validation\results\final\profiles\7C\epcsaft_neutral\temperature_profile.png'; Destination = 'profile-7c-epcsaft.png' }
)

foreach ($copy in $figureCopies) {
    $sourcePath = Join-Path $repoRootPath $copy.Source
    $destinationPath = Join-Path $figureRootPath $copy.Destination

    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Required figure source is missing: $sourcePath"
    }

    if ($PSCmdlet.ShouldProcess($destinationPath, "Copy $sourcePath")) {
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
    }
}

$texFiles = @(
    Join-Path $latexSourcePath 'main.tex'
) + @(Get-ChildItem -LiteralPath (Join-Path $latexSourcePath 'sections') -Filter '*.tex' -File |
    ForEach-Object { $_.FullName }) + @(Get-ChildItem -LiteralPath (Join-Path $latexSourcePath 'appendices') -Filter '*.tex' -File |
    ForEach-Object { $_.FullName })

$missingReferences = New-Object System.Collections.Generic.List[string]
foreach ($texFile in $texFiles) {
    if (-not (Test-Path -LiteralPath $texFile)) {
        continue
    }

    $content = Get-Content -LiteralPath $texFile -Raw
    $matches = [regex]::Matches($content, '\\includegraphics(?:\[[^\]]*\])?\{(?<path>figures/[^}]+)\}')
    foreach ($match in $matches) {
        $relativePath = $match.Groups['path'].Value.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
        $expectedPath = Join-Path $latexSourcePath $relativePath
        if (-not (Test-Path -LiteralPath $expectedPath)) {
            $missingReferences.Add($match.Groups['path'].Value)
        }
    }
}

if ($missingReferences.Count -gt 0) {
    $missing = ($missingReferences | Sort-Object -Unique) -join ', '
    throw "Missing LaTeX figure references after refresh: $missing"
}

Write-Host "LaTeX figures refreshed: $figureRootPath"
