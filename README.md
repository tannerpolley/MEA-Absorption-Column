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
- Paper-facing validation workflows live under `analyses/nccc_validation/`.
- Manuscript source and LaTeX build scripts live under `docs/latex/`.
- Root `scripts/` is reserved for repo-wide tools or small smoke checks, not benchmark sweeps.

For a handoff map that tells another Codex agent which scripts run the absorber, which scripts only render or validate artifacts, and which workflows require the external ePC-SAFT package, see `docs/workflow_map.md`.

## Usage

This repo is now `uv`-first for reproducible reviewer-response benchmarks.

The sibling LaTeX manuscript checkout lives at `C:\Users\Tanner\Documents\git\LaTeX-Projects\MEA-Absorption-Column-LaTeX` and is connected to the Overleaf Git remote.

The manuscript source lives in `docs\latex`. To refresh the flat Overleaf mirror checkout after manuscript or figure updates, run:

```powershell
.\docs\latex\sync_to_overleaf_mirror.ps1 -CleanBuildFiles
```

Use `-WhatIf` first when you want to preview the files that would be copied. The sync script intentionally excludes itself so it does not appear in the Overleaf mirror project.

To build a fresh local manuscript PDF after editing `docs\latex\main.tex` or included LaTeX inputs, run:

```powershell
.\docs\latex\build_main.ps1
```

The clickable local artifact is `docs\latex\main.pdf`. The build script also runs a freshness check and can open the PDF directly with `-Open`.

```powershell
uv run --group test python -m pytest
uv run python -m mea_absorption_column.benchmark --methods single scipy-bvp --thermo-models ideal_henry epcsaft_neutral
```

When using the shared Codex Python environment directly from a worktree instead of `uv run`, set `PYTHONPATH=src` first so benchmark subprocesses import the active checkout:

```powershell
$env:PYTHONPATH = "src"
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Benchmark CSV and Markdown outputs are written to `analyses/nccc_validation/results/runs/benchmark` by default. Run-specific files under `results/runs/` are ignored by Git; curated paper-facing evidence lives under `analyses/nccc_validation/results/final/`.

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
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\generate_data.py
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\render_figures.py
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\collect_clean_profiles.py --collect-existing
C:\Users\Tanner\.codex\venvs\MEA-Absorption-Column-py313\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py
```

Clean temperature-profile PNGs are arranged by case and thermodynamic lane under `analyses/nccc_validation/results/final/profiles/`.

See `analyses/README.md`, `analyses/nccc_validation/README.md`, and `analyses/nccc_validation/analysis.yaml` before adding new validation scripts or result folders.

Robust-convergence diagnostics and the current verification snapshot are summarized in `docs/robust_convergence_status.md`. The benchmark CLI exposes solver settings for reproducibility:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --mesh-points 51 --tol 0.5 --bc-tol 0.001 --max-nodes 1000 --success-boundary-residual-max 1
```

The hand-built central-difference Jacobian is available with `--finite-jacobian`, but it is opt-in because stacked/intercooled cases are much slower with it.

Shooting-method experiments can use `--shooting-integrator euler|bdf|radau|rk45`. Stiff IVP integrators are diagnostic only at this stage; pair them with `--max-runtime-s` or `--subprocess-timeout-s` when sweeping cases so a bad shooting branch returns a structured timeout row instead of tying up the workflow.

### Intercooled bed stack benchmark

Multi-bed NCCC cases should be run with staged beds enabled:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --staged-beds auto --transform-mode case_bounded_flow_pressure --output-dir analyses\nccc_validation\results\runs\intercooled_benchmark
```

`--staged-beds auto` uses the stacked solver when a case has `Beds > 1` or `Intercoolers > 0`. Intercoolers are modeled as liquid enthalpy resets between beds. Unless explicit target temperatures are supplied, the first comparison uses the measured liquid feed temperature as the inter-stage target and reports this as `intercooler_assumption=Tl_feed_target`.

`--intercooler-strength` can be used for continuation studies: `0` keeps the staged topology without applying liquid cooling, `0.25` applies a weak reset, and `1` applies the full target-temperature reset.

The staged/intercooled NCCC workflow is still diagnostic rather than fully predictive. Failed rows are expected to stay in the benchmark CSV with `success=False`, `boundary_residual_norm`, `boundary_residual_components`, `jacobian_status`, guard counts, and the first failed domain. `--thermal-state-mode temperature` is available for staged solver experiments, but the current reference path remains the enthalpy-state solver with case-bounded flow/pressure variables. The continuation ladder in `python -m mea_absorption_column.continuation` records optional seed/ramp stages and only treats the full-intercooler stage as the required staged Henry result.

## Thermodynamics

The default model is `ideal_henry`, matching the previous Henry-law driving-force implementation. The `epcsaft_neutral` option imports the external `epcsaft` package read-only and uses only the CO2 fugacity coefficient from a neutral CO2/MEA/H2O parameter set stored in this repo under `src/mea_absorption_column/data/epcsaft_neutral/`.

Thermodynamic modes are intentionally explicit:

- `ideal_henry`: default validation baseline; legacy concentration-based chemical equilibrium and Henry-law CO2 driving force.
- `epcsaft_neutral`: supported opt-in fugacity sensitivity lane; legacy chemistry is retained while neutral CO2/MEA/H2O ePC-SAFT replaces only the CO2 fugacity coefficient calculation.
- `epcsaft_ionic`: opt-in diagnostic fugacity lane; legacy chemistry is retained while the ionic six-species liquid state is passed to ePC-SAFT for CO2 fugacity.
- `epcsaft_reactive_*`: experimental opt-in chemistry lanes; ePC-SAFT reactive speciation replaces the legacy chemistry solve for diagnostic Case 3C-style smoke tests, but it is not yet broadly validated across all NCCC cases.

The supported ePC-SAFT comparison in the paper-facing workflow remains a controlled thermodynamic sensitivity study, not a broad claim that every absorber result is a fully validated electrolyte-reactive ePC-SAFT model. Experimental reactive rows must be reported with runtime and convergence diagnostics.

The MEA ePC-SAFT parameter datasets are vendored in this repository under `src/mea_absorption_column/data/epcsaft_datasets/`. `MEA_EPCSAFT_DATASET_NAME` can select a different vendored dataset, and `MEA_THERMODYNAMICS_EPCSAFT_DATASET` remains an override for temporary external comparisons only. Normal repo tests and absorber runs must not depend on a sibling MEA-Thermodynamics checkout for parameter files.

`--epcsaft-fugacity-blend <0..1>` is available for continuation diagnostics in the `epcsaft_neutral` lane. A value of `0` returns the Henry-law fugacity values through the ePC-SAFT adapter path, intermediate values linearly blend Henry and neutral ePC-SAFT fugacity values, and `1` is the full neutral ePC-SAFT fugacity endpoint. This is intended for branch diagnosis and staged warm-start studies, not as a publishable calibrated thermodynamic model by itself.
