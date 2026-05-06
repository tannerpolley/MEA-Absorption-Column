# NCCC Validation Analysis

This analysis owns the reviewer-response validation artifacts for the MEA absorber model. It separates disposable benchmark runs from curated final evidence so completed results can be inspected without digging through temporary solver folders.

## Layout

- `results/runs/`: disposable benchmark runs and long diagnostics. This folder is ignored by Git.
- `results/final/tables/`: accepted result CSVs, diagnostic CSVs, plot-ready tables, and profile indexes.
- `results/final/figures/`: paper-ready SVG/PDF benchmark figures.
- `results/final/profiles/<case_id>/<thermo_model>/`: clean temperature-profile PNGs for quick visual review.
- `results/final/reports/`: Markdown or CSV summaries for accepted rows, fallback rows, and unresolved diagnostics.

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

Validate the analysis artifacts used by the manuscript:

```powershell
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py
```

## Result Semantics

The clean profile gallery contains accepted validation rows and explicitly accepted fallback rows used in the manuscript. Diagnostic or unresolved rows stay in final tables and reports, but they are not mixed into the clean profile gallery unless the caveat is explicit in the profile index.
