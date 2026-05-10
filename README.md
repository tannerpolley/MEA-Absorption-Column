# MEA Absorption Column

## 🧪 Overview
This is a custom Python-based model built to simulate an amine-based absorption column. The model is intended for quick, flexible simulation of post-combustion carbon capture processes.

## 🎯 Purpose
The goal of this model is to enable the development and further the research of Post-Combustion Carbon Capture. It supports rapid simulations using a variety of numerical methods such as:
- Shooting Method  
- Finite Difference  
- Collocation  

All design variables and parameters are customizable, making the tool adaptable to the user's research or engineering needs.

## ⚙️ Status
The model is currently under active development.  
✅ **Functional** – It can already produce consistent results with the current implementation.

## Project organization

This repository follows the local project architecture standard for scientific Python work:

- Package code and reusable model data live under `src/mea_absorption_column/`.
- Fast regression tests live under `tests/`.
- Manuscript validation workflows live under `analyses/nccc_validation/`.
- Manuscript source and LaTeX build scripts live under `docs/latex/`.
- Root `scripts/` is reserved for repo-wide tools or small smoke checks, not benchmark sweeps.

For a handoff map that tells another Codex agent which scripts run the absorber, which scripts only render or validate artifacts, and which workflows require the external ePC-SAFT package, see `docs/workflow_map.md`.

## Usage

This repo is now `uv`-first for reproducible reviewer-response benchmarks.

The Overleaf mirror is a separate Git checkout configured locally. The source of truth remains `docs\latex`; mirror paths are intentionally machine-local and should not be committed.

The manuscript source lives in `docs\latex`. To refresh the flat Overleaf mirror checkout after manuscript or figure updates, run:

```powershell
.\docs\latex\scripts\sync_to_overleaf_mirror.ps1 -CleanBuildFiles
```

Use `-WhatIf` first when you want to preview the files that would be copied. The sync script intentionally excludes itself so it does not appear in the Overleaf mirror project.

To build a fresh local manuscript PDF after editing `docs\latex\main.tex` or included LaTeX inputs, run:

```powershell
.\docs\latex\scripts\build_main.ps1
```

The clickable local artifact is `docs\latex\builds\main.pdf`. The build script also runs a freshness check and can open the PDF directly with `-Open`.

Set up the project-local Python environment once from the repository root:

```powershell
uv sync --group test
uv pip install /path/to/ePC-SAFT
```

The local environment lives at `.venv/` and is ignored by Git. The ePC-SAFT install is optional for Henry-only validation, but it is required before running `epcsaft_ionic` or experimental reactive diagnostics. Use the project-local interpreter directly for normal checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m mea_absorption_column.benchmark --methods single scipy-bvp --thermo-models ideal_henry
```

Benchmark CSV and Markdown outputs are written to `analyses/nccc_validation/results/runs/benchmark` by default. Run-specific files under `results/runs/` are ignored by Git; curated manuscript evidence lives under `analyses/nccc_validation/results/final/`.

### NCCC validation results

The reviewer-response benchmark evidence is organized as a self-contained analysis:

```text
analyses/nccc_validation/
  scripts/
  results/
    runs/
    final/
      tables/
      figures/
      profiles/
      reports/
```

Use these commands to refresh and validate the curated tables, figures, and clean profile index without rerunning long simulations:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\generate_data.py
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\render_figures.py
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\collect_clean_profiles.py --collect-existing
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py
```

Clean temperature-profile PNGs are arranged by case and thermodynamic lane under `analyses/nccc_validation/results/final/profiles/`.

See `analyses/README.md`, `analyses/nccc_validation/README.md`, and `analyses/nccc_validation/analysis.yaml` before adding new validation scripts or result folders.

The benchmark CLI exposes solver settings for reproducibility:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --mesh-points 51 --tol 0.5 --bc-tol 0.001 --max-nodes 1000 --success-boundary-residual-max 1
```

The hand-built central-difference Jacobian is available with `--finite-jacobian`, but it is opt-in because it is slower than the default solver path on stiff nonlinear cases.

Shooting-method experiments can use `--shooting-integrator euler|bdf|radau|rk45`. Stiff IVP integrators are diagnostic only at this stage; pair them with `--max-runtime-s` or `--subprocess-timeout-s` when sweeping cases so a bad shooting branch returns a structured timeout row instead of tying up the workflow.

## Thermodynamics

The default model is `ideal_henry`, matching the Henry-law CO2 driving-force implementation. The selected manuscript ePC-SAFT comparison is `epcsaft_ionic`: it retains the concentration-based chemical-equilibrium calculation and replaces the CO2 driving force with vapor- and liquid-side ePC-SAFT fugacity coefficients. The full activity-coupled path is separate and slower; it routes ePC-SAFT activity-related quantities through a nine-species chemical-equilibrium solve before evaluating the fugacity driving force.

Thermodynamic modes are intentionally explicit:

- `ideal_henry`: default validation baseline; concentration-based chemical equilibrium and Henry-law CO2 driving force.
- `epcsaft_ionic`: selected ePC-SAFT fugacity lane; concentration-based chemistry is retained while the ionic liquid state is passed to ePC-SAFT for CO2 fugacity.
- `epcsaft_neutral`: historical or diagnostic sensitivity lane only; not the selected manuscript comparison.
- `epcsaft_reactive_*`: experimental activity-coupled chemistry lanes; use only with runtime, residual, guard-count, and convergence diagnostics.

The supported ePC-SAFT comparison in the manuscript is a controlled thermodynamic driving-force benchmark, not a claim that every absorber result uses the full activity-coupled chemistry loop. Full reactive rows must be reported with runtime and convergence diagnostics.

The MEA ePC-SAFT parameter datasets are vendored in this repository under `src/mea_absorption_column/data/epcsaft_datasets/`. `MEA_EPCSAFT_DATASET_NAME` can select a different vendored dataset, and `MEA_THERMODYNAMICS_EPCSAFT_DATASET` remains an override for temporary external comparisons only. Normal repo tests and absorber runs must not depend on a sibling MEA-Thermodynamics checkout for parameter files.

`--epcsaft-fugacity-blend <0..1>` is available for continuation diagnostics. A value of `0` returns the Henry-law fugacity values through the ePC-SAFT adapter path, intermediate values linearly blend Henry and ePC-SAFT fugacity values, and `1` is the full ePC-SAFT fugacity endpoint. This is intended for branch diagnosis and warm-start studies, not as a publishable calibrated thermodynamic model by itself.
