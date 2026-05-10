# Reproduce the Manuscript Evidence

This document gives the command chain for rebuilding the curated manuscript artifacts without rerunning slow full-species simulations by default. Run commands from the repository root.

## Environment

```powershell
uv sync --group test
uv pip install /path/to/ePC-SAFT
$env:PYTHONPATH = "src"
$env:MEA_EPCSAFT_DATASET_NAME = "MEA_CO2_H2O_ionic_fit"
$config = Import-Csv analyses\nccc_validation\results\final\tables\epcsaft_electrolyte_config_user_options.csv | Where-Object { $_.config -eq "2025_Figiel_empirical_fitted_Born_SSM_DS" } | Select-Object -First 1
$env:MEA_EPCSAFT_USER_OPTIONS_JSON = $config.user_options_json
```

Henry-law checks do not require the external ePC-SAFT package. ePC-SAFT validation, electrolyte option probes, and activity-coupled runs do require it.

## Refresh Curated Tables, Figures, and Profiles

These commands rebuild plot-ready tables and figures from existing committed results or already completed run folders:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\generate_nccc_one_bed_artifacts.py
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\render_source_backed_temperature_capture_gallery.py
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\generate_data.py
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\render_figures.py
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\collect_clean_profiles.py --collect-existing
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py
docs\latex\scripts\sync_latex_figures.ps1
```

## Routine ePC-SAFT C-Case Campaign

This reruns the 2017 C-case campaign used for temperature overlays. It is slower than table rendering but much faster than the full activity-coupled path.

```powershell
.\.venv\Scripts\python.exe -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_ionic --c-case-dataset campaign --c-case-ids 1C 2C 3C 4C 5C 6C 7C --nccc-case-limit 0 --srp-case-limit 0 --staged-beds false --mesh-points 51 --tol 0.5 --bc-tol 0.001 --max-nodes 1000 --subprocess-timeout-s 120 --profile-csvs --profile-pngs --output-dir analyses\nccc_validation\results\runs\c_case_campaign_temperature_gallery
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\render_c_case_campaign_temperature_gallery.py
```

## Full Activity-Coupled Evidence

The committed evidence for the slow full path is:

```text
analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv
```

Do not rerun this path during a normal manuscript refresh. To intentionally regenerate it, use the run script only in a long-running session and preserve the resulting CSV metadata:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\run_full_species_ionic_2017_c_case_sweep.py --run-cases
```

Expected evidence fields include case id, model labels, capture, capture error, runtime, chemistry-solve time, residuals, guard counts, Python version, platform, package versions, exact command, and relevant environment variables.

## LaTeX Build

```powershell
docs\latex\scripts\build_main.ps1
.\.venv\Scripts\python.exe docs\latex\scripts\check_main_pdf_fresh.py
```

The source of truth is `docs/latex`. Use the strict Overleaf mirror sync only after the local build and freshness check pass.
