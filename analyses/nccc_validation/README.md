# NCCC Validation Analysis

This analysis owns the reviewer-response validation artifacts for the MEA absorber model. It separates disposable benchmark runs from curated final evidence so completed results can be inspected without digging through temporary solver folders.

## NCCC Case Names

Morgan et al. 2018 uses `K1`-`K23` for the NCCC steady-state rows. The Appendix-C-style temperature-profile workflow uses `1A`-`15A`, `1B`-`3B`, `1C`-`7C`, and `1D`-`4D` as visible plot labels. The canonical crosswalk and operating table is `data/reference/nccc_master_cases.csv`, with an analysis-local copy at `analyses/nccc_validation/data/input/nccc_master_cases.csv`. New figures should use Appendix-style names for display only and retain the source K row in metadata.

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
| `build_nccc_master_case_table.py` | Extracts Morgan 2018 and Moore 2021 markdown from the local literature export, builds `nccc_master_cases.csv`, and writes the K-to-Appendix-style crosswalk note. | No. | No. |
| `render_figures.py` | Renders final manuscript figures from final tables. | No. | No direct dependency; may plot existing ePC-SAFT rows. |
| `collect_clean_profiles.py` | Refreshes the clean temperature-profile PNG gallery and index. | Sometimes. | Required only when rerunning or collecting `epcsaft_*` lanes. |
| `run_case_profile.py` | Runs one case and exports dense legacy-`Profiles.xlsx`-style CSVs. | Yes. | Required for `epcsaft_neutral`, `epcsaft_ionic`, and experimental reactive lanes. |
| `generate_clean_profile_csvs.py` | Runs accepted clean rows with per-case timeouts and writes dense profile CSV folders. | Yes. | Required only for ePC-SAFT rows in the selected suite. |
| `generate_accuracy_credibility_artifacts.py` | Builds the primary/recovery/diagnostic evidence registry, calibration screen, error-regime plots, staged ePC-SAFT reliability summary, and intercooled temperature-profile figures. | No. | No direct dependency; uses curated final rows and one literature profile extracted from Morgan et al. Appendix C. |
| `render_appendix_c_temperature_profiles.py` | Builds an Appendix C-style absorber temperature-profile PDF using measured points and exported true model profiles only. | No. | No direct dependency; reads existing model profile CSVs when available. |
| `probe_reactive_epcsaft_speciation.py` | Experimental reactive-speciation probe using the external ePC-SAFT package and the repo-vendored MEA dataset. | Thermodynamic probe only. | Required. Not a default validation script. |
| `validate_results.py` | Checks final tables, figures, profile indexes, and stale path regressions. | No. | No direct dependency. |

Henry-only validation does not require the external ePC-SAFT checkout. ePC-SAFT diagnostics should be run only after installing or updating the external package. The MEA parameter datasets are vendored in this repo under `src/mea_absorption_column/data/epcsaft_datasets/`.

Typical ePC-SAFT diagnostic environment:

```powershell
$env:MEA_EPCSAFT_ROOT = "C:\Users\Tanner\Documents\git\ePC-SAFT"
$env:MEA_EPCSAFT_DATASET_NAME = "MEA_CO2_H2O_draft"
```

Regenerate plot-ready tables from curated inputs or available run folders:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\generate_data.py
```

Refresh the Morgan 2018 NCCC master case table and source markdown notes:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\build_nccc_master_case_table.py
```

Render final figures from the final tables:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\render_figures.py
```

Generate the accuracy-credibility artifacts added for the manuscript revision:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\generate_accuracy_credibility_artifacts.py
```

This script writes the evidence-class registry, the no-case-specific-tuning gate, the small structured holdout calibration screen, the error-regime plot data, the staged ePC-SAFT reliability table, and two intercooled temperature-profile figures. The measured three-bed profile is a curated extraction of Morgan et al. Appendix C, Table C1, stored under `data/input/`.

Generate the Appendix C-style temperature-profile review PDF:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\render_appendix_c_temperature_profiles.py
```

This script writes `results/final/figures/appendix_c_temperature_profiles.pdf`, per-page PNG previews in `results/final/figures/appendix_c_temperature_profile_pages/`, individual case previews in `results/final/figures/appendix_c_temperature_profile_cases/`, the measured Appendix C table under `data/input/`, and `results/final/tables/appendix_c_temperature_profile_index.csv`. Measured points are plotted without connecting lines. Model overlays are exported true model profiles only; if a case has no exported profile or has a failed/diagnostic model profile, the plot labels that broken or missing model output directly. One-bed `C` cases use `C_cases_data.csv`; staged/intercooled rows use the legacy K-row model outputs mapped to Appendix C-style names for plotting.

Refresh the clean profile index without rerunning simulations:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\collect_clean_profiles.py --collect-existing
```

Run one specific case and export dense per-variable profile CSVs:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\run_case_profile.py --case-source C_cases_data --case-id 3C --method scipy-bvp --thermo-model ideal_henry --output-dir analyses\nccc_validation\results\runs\manual_case_profiles
```

Full benchmark runs can also request these dense profile CSVs:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --c-case-ids 3C --nccc-case-limit 0 --profile-csvs --subprocess-timeout-s 60 --output-dir analyses\nccc_validation\results\runs\profile_csv_probe
```

Run the favorable SRP-style method-comparison case across shooting, SciPy BVP, and finite difference:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe -m mea_absorption_column.benchmark --methods single scipy-bvp finite --thermo-models ideal_henry --c-case-limit 0 --nccc-case-limit 0 --srp-case-limit 1 --mesh-points 21 --tol 0.5 --bc-tol 0.001 --max-runtime-s 30 --seed-from-shooting --subprocess-timeout-s 60 --output-dir analyses\nccc_validation\results\runs\srp_method_slice
```

The current SRP/NCCC method contrast is summarized in `results/final/reports/solver_method_contrast_srp_3c.md`.

Each generated profile folder includes `profile_manifest.json`, `profile_manifest.csv`, `run_spec.json`, and `rerun_profile.ps1` so a single case can be rerun from that folder without launching the entire benchmark sweep.

Generate dense profile CSVs for the accepted validation rows with 60-second per-case timeouts and an incremental log:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\generate_clean_profile_csvs.py --suite all --output-dir analyses\nccc_validation\results\runs\clean_profile_csvs
```

This script writes each case row as soon as it finishes. If a case exceeds the default 60-second subprocess timeout or errors, the row is logged as a failed diagnostic result and the script continues to the next case. Use `--per-case-timeout-s <seconds>` only when intentionally running a longer diagnostic probe.
It also writes `profile_runtime_index.csv` and refreshes each profile folder's `profile_manifest.json` / `profile_manifest.csv` with `runtime_s` and a human-readable `runtime_label`.

Validate the analysis artifacts used by the manuscript:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py
```

## Result Semantics

The clean profile gallery contains accepted validation rows and explicitly accepted fallback rows used in the manuscript. Diagnostic or unresolved rows stay in final tables and reports, but they are not mixed into the clean profile gallery unless the caveat is explicit in the profile index.

The dense profile CSV folders are run artifacts, not summary tables. Use them to inspect how internal variables change with column position; use `verified_*.csv`, `raw_*.csv`, and `plot_*.csv` for manuscript validation metrics and figure generation.
