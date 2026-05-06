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

Regenerate plot-ready tables from curated inputs or available run folders:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\generate_data.py
```

Render final figures from the final tables:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\render_figures.py
```

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
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --c-case-ids 3C --nccc-case-limit 0 --profile-csvs --output-dir analyses\nccc_validation\results\runs\profile_csv_probe
```

Each generated profile folder includes `profile_manifest.json`, `profile_manifest.csv`, `run_spec.json`, and `rerun_profile.ps1` so a single case can be rerun from that folder without launching the entire benchmark sweep.

Validate the analysis artifacts used by the manuscript:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py
```

## Result Semantics

The clean profile gallery contains accepted validation rows and explicitly accepted fallback rows used in the manuscript. Diagnostic or unresolved rows stay in final tables and reports, but they are not mixed into the clean profile gallery unless the caveat is explicit in the profile index.

The dense profile CSV folders are run artifacts, not summary tables. Use them to inspect how internal variables change with column position; use `verified_*.csv`, `raw_*.csv`, and `plot_*.csv` for manuscript validation metrics and figure generation.
