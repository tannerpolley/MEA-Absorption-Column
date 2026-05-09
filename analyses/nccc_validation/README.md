# NCCC Validation Analysis

This analysis owns the reviewer-response validation artifacts for the MEA absorber model. It separates disposable benchmark runs from curated final evidence so completed results can be inspected without digging through temporary solver folders.

## Layout

- `results/runs/`: disposable benchmark runs and long diagnostics. This folder is ignored by Git.
- `results/final/tables/`: accepted result CSVs, diagnostic CSVs, plot-ready tables, and profile indexes.
- `results/final/figures/`: paper-ready SVG/PDF benchmark figures.
- `results/final/profiles/<case_id>/<thermo_model>/`: clean temperature-profile PNGs for quick visual review.
- `results/final/reports/`: Markdown or CSV summaries for accepted rows, fallback rows, and unresolved diagnostics.
- `results/runs/<run_id>/profiles/<case_source>/<case_id>/<method>/<thermo_model>/`: requested dense profile CSV exports. These are the CSV replacement for the legacy `Profiles.xlsx` workbook: each old workbook sheet is written as its own CSV, with `Position`, `height_m`, `bed_id`, and `bed_position_m` coordinate columns.

## Commands

When running from a Git worktree with the shared Codex venv, set the source path first so subprocess benchmark workers import this worktree:

```powershell
$env:PYTHONPATH = "src"
```

## Script Inventory

| Script | Role | Runs model? | ePC-SAFT dependency |
| --- | --- | --- | --- |
| `generate_data.py` | Converts curated run/final CSVs into raw, verified, and plot-ready tables. | No. | No direct dependency; may process existing ePC-SAFT rows. |
| `render_figures.py` | Renders final manuscript figures from final tables. | No. | No direct dependency; may plot existing ePC-SAFT rows. |
| `collect_clean_profiles.py` | Refreshes the clean temperature-profile PNG gallery and index. | Sometimes. | Required only when rerunning or collecting `epcsaft_*` lanes. |
| `run_case_profile.py` | Runs one case and exports dense legacy-`Profiles.xlsx`-style CSVs. | Yes. | Required for `epcsaft_neutral`, `epcsaft_ionic`, and experimental reactive lanes. |
| `generate_clean_profile_csvs.py` | Runs accepted clean rows with per-case timeouts and writes dense profile CSV folders. | Yes. | Required only for ePC-SAFT rows in the selected suite. |
| `render_c_case_campaign_temperature_gallery.py` | Renders the corrected one-bed C-case temperature overlay gallery from a completed campaign-input benchmark run. | No. | No direct dependency; the source run may include ePC-SAFT rows. |
| `probe_reactive_epcsaft_speciation.py` | Experimental reactive-speciation probe using the external ePC-SAFT package and the repo-vendored MEA dataset. | Thermodynamic probe only. | Required. Not a default validation script. |
| `probe_epcsaft_electrolyte_options.py` | Exercises neutral, ion-only, Born, SSM, DS, and unsupported fitted-Born user-option paths and writes contribution diagnostics. | Thermodynamic probe only. | Required. Not a default validation script. |
| `run_epcsaft_electrolyte_config_matrix.py` | Runs the NCCC 3C absorber case across dated and mode-coverage ePC-SAFT electrolyte user-option configurations, then exports parameter provenance tables. | Yes. | Required. Not a default validation script. |
| `validate_results.py` | Checks final tables, figures, profile indexes, and stale path regressions. | No. | No direct dependency. |

Henry-only validation does not require the external ePC-SAFT checkout. ePC-SAFT diagnostics should be run only after installing or updating the external package. The MEA parameter datasets are vendored in this repo under `src/mea_absorption_column/data/epcsaft_datasets/`.

Typical ePC-SAFT diagnostic environment:

```powershell
$env:MEA_EPCSAFT_ROOT = "C:\Users\Tanner\Documents\git\ePC-SAFT"
$env:MEA_EPCSAFT_DATASET_NAME = "MEA_CO2_H2O_draft"
```

Run the electrolyte option matrix diagnostic:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\probe_epcsaft_electrolyte_options.py
```

Run the column-level electrolyte configuration matrix and regenerate the pure-component and binary-interaction parameter provenance tables:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\run_epcsaft_electrolyte_config_matrix.py
```

Regenerate plot-ready tables from curated inputs or available run folders:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\generate_data.py
```

Render final figures from the final tables:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\render_figures.py
```

Refresh the clean profile index without rerunning simulations:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\collect_clean_profiles.py --collect-existing
```

Run one specific case and export dense per-variable profile CSVs:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\run_case_profile.py --case-source C_cases_data --case-id 3C --method scipy-bvp --thermo-model ideal_henry --output-dir analyses\nccc_validation\results\runs\manual_case_profiles
```

Full benchmark runs can also request these dense profile CSVs:

```powershell
.\.venv\Scripts\python.exe -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --c-case-ids 3C --nccc-case-limit 0 --profile-csvs --subprocess-timeout-s 60 --output-dir analyses\nccc_validation\results\runs\profile_csv_probe
```

Run the corrected one-bed NCCC C-case campaign table and regenerate the 1C--7C temperature overlay gallery:

```powershell
.\.venv\Scripts\python.exe -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_neutral --c-case-dataset campaign --c-case-ids 1C 2C 3C 4C 5C 6C 7C --nccc-case-limit 0 --srp-case-limit 0 --staged-beds false --mesh-points 51 --tol 0.5 --bc-tol 0.001 --max-nodes 1000 --subprocess-timeout-s 60 --profile-csvs --profile-pngs --output-dir analyses\nccc_validation\results\runs\c_case_campaign_temperature_gallery
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\render_c_case_campaign_temperature_gallery.py
```

The campaign dataset is stored separately as `src/mea_absorption_column/data/C_cases_campaign_inputs.csv`; the legacy `C_cases_data.csv` remains available through the default `--c-case-dataset legacy` path for reproducibility.

Run the favorable SRP-style method-comparison case across shooting, collocation BVP, and finite difference:

```powershell
.\.venv\Scripts\python.exe -m mea_absorption_column.benchmark --methods single scipy-bvp finite --thermo-models ideal_henry --c-case-limit 0 --nccc-case-limit 0 --srp-case-limit 1 --mesh-points 21 --tol 0.5 --bc-tol 0.001 --max-runtime-s 30 --seed-from-shooting --subprocess-timeout-s 60 --output-dir analyses\nccc_validation\results\runs\srp_method_slice
```

The current SRP/NCCC method contrast is summarized in `results/final/reports/solver_method_contrast_srp_3c.md`.

Each generated profile folder includes `profile_manifest.json`, `profile_manifest.csv`, `run_spec.json`, and `rerun_profile.ps1` so a single case can be rerun from that folder without launching the entire benchmark sweep.

Generate dense profile CSVs for the accepted validation rows with 60-second per-case timeouts and an incremental log:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\generate_clean_profile_csvs.py --suite all --output-dir analyses\nccc_validation\results\runs\clean_profile_csvs
```

This script writes each case row as soon as it finishes. If a case exceeds the default 60-second subprocess timeout or errors, the row is logged as a failed diagnostic result and the script continues to the next case. Use `--per-case-timeout-s <seconds>` only when intentionally running a longer diagnostic probe.
It also writes `profile_runtime_index.csv` and refreshes each profile folder's `profile_manifest.json` / `profile_manifest.csv` with `runtime_s` and a human-readable `runtime_label`.

Validate the analysis artifacts used by the manuscript:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py
```

## Result Semantics

The clean profile gallery contains accepted validation rows and explicitly accepted fallback rows used in the manuscript. Diagnostic or unresolved rows stay in final tables and reports, but they are not mixed into the clean profile gallery unless the caveat is explicit in the profile index.

The dense profile CSV folders are run artifacts, not summary tables. Use them to inspect how internal variables change with column position; use `verified_*.csv`, `raw_*.csv`, and `plot_*.csv` for manuscript validation metrics and figure generation.
