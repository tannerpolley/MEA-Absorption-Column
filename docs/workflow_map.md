# Workflow Map

This document is the repo-facing handoff map for future agents or forks. It follows the local Codex project architecture standard without requiring any machine-specific path.

## Repository Layout

| Path | Owner | Purpose | Generated outputs |
| --- | --- | --- | --- |
| `src/mea_absorption_column/` | Package | Reusable absorber model, thermodynamic adapters, solver wrappers, benchmark CLI, packaged reference CSV data, and MEA-local ePC-SAFT parameter files. | No study run outputs should be written here. |
| `tests/` | Package tests | Fast regression and schema checks for model, benchmark, thermodynamics, and artifact behavior. | Test temp files should use pytest temp folders or ignored `.tmp_local/`. |
| `analyses/nccc_validation/` | Analysis | Canonical reviewer-response validation workflow for one-bed C cases, SRP-style solver comparisons, dense profile CSVs, and manuscript figures. | Disposable runs under `results/runs/`; curated evidence under `results/final/`. |
| `docs/latex/` | Manuscript | LaTeX source, bibliography, source-only helper scripts, Overleaf mirror sync, appendices, and manuscript-local figure copies. | Local LaTeX build products stay under ignored `builds/`; regenerate `builds/main.pdf` with `uv run python docs/latex/scripts/latex_workflows.py build`. |
| `docs/` | Supporting notes | Reviewer-response notes, robust-convergence status, and this workflow map. | Do not put benchmark run artifacts here. |
| `scripts/` | Repo tools | Repository-wide utilities or small smoke checks only. | No manuscript sweep outputs. |

## NCCC Validation Workflow

The canonical analysis folder is `analyses/nccc_validation/`.

| Script | What it does | Runs the absorber model? | ePC-SAFT dependency |
| --- | --- | --- | --- |
| `scripts/generate_data.py` | Normalizes curated benchmark rows into raw, verified, and plot-ready final tables. | No, except it reads existing run/final CSVs. | No direct ePC-SAFT import; may process ePC-SAFT result rows already generated elsewhere. |
| `scripts/render_figures.py` | Renders manuscript figures from final tables. | No. | No direct ePC-SAFT import; plots ePC-SAFT rows when present. |
| `scripts/collect_clean_profiles.py` | Builds or refreshes the clean temperature-profile PNG gallery and index. | Yes when not using existing profile images. | Optional; required only when collecting/rerunning `epcsaft_*` thermodynamic lanes. |
| `scripts/run_case_profile.py` | Runs one case and writes dense per-variable profile CSVs plus a rerun spec. | Yes. | Optional; required for `epcsaft_ionic`. |
| `scripts/generate_clean_profile_csvs.py` | Runs accepted clean rows with per-case timeouts and exports dense profile CSVs. | Yes. | Optional by suite; ePC-SAFT required for ePC-SAFT C-case profile rows. |
| `scripts/probe_reactive_epcsaft_speciation.py` | Archived probe for the superseded reactive interface. | No supported current run. | Retained for provenance; it must be migrated to the typed 0.2 equilibrium API before reuse. |
| `scripts/validate_results.py` | Checks final tables, figures, profile indexes, and stale path regressions. | No. | No direct ePC-SAFT import. |

## ePC-SAFT Dependency Contract

Henry-only tests and benchmarks should run without the external ePC-SAFT checkout. ePC-SAFT workflows are opt-in thermodynamic lanes:

- `ideal_henry`: default validation lane; no external ePC-SAFT dependency.
- `epcsaft_ionic`: selected manuscript ePC-SAFT fugacity lane. It requires the external `epcsaft` package and uses the vendored six-species ePC-SAFT dataset plus the liquid state produced by the concentration-based chemistry model.
- `epcsaft_reactive_*`: intentionally unavailable after the 0.2 cutover because the archived locally rebased constants do not satisfy the new typed standard-state contract.

The selected vendored dataset may be chosen explicitly:

```bash
export MEA_EPCSAFT_DATASET_NAME="MEA_CO2_H2O_ionic_fit"
```

The parameter datasets live in `src/mea_absorption_column/data/epcsaft_datasets/`. `MEA_THERMODYNAMICS_EPCSAFT_DATASET` remains an explicit one-off override. The package dependency is an immutable wheel built by the sibling ePC-SAFT project; the adapter uses `Parameters`, `Mixture`, and `State`, and does not expose a derivative-backend selector because CppAD is the package's sole production authority.

## Common Commands

Use the project-root `.venv` for this repository. Create or refresh it with:

```bash
uv sync --group test
```

If running scripts from a Git worktree or from an unusual shell context, set `PYTHONPATH=src` so Python imports the active checkout:

```bash
export PYTHONPATH="src"
```

Fast package test:

```bash
uv run python -m pytest -q -p no:cacheprovider
```

Validate curated NCCC artifacts without rerunning long simulations:

```bash
uv run python analyses/nccc_validation/scripts/validate_results.py
```

Run one clean Henry profile export:

```bash
uv run python analyses/nccc_validation/scripts/run_case_profile.py --case-source C_cases_data --case-id 3C --method scipy-bvp --thermo-model ideal_henry --output-dir analyses/nccc_validation/results/runs/manual_case_profiles
```

Run one ePC-SAFT smoke profile after installing/updating the external package:

```bash
uv run python analyses/nccc_validation/scripts/run_case_profile.py --case-source C_cases_data --case-id 3C --method scipy-bvp --thermo-model epcsaft_ionic --output-dir analyses/nccc_validation/results/runs/manual_epcsaft_profile
```

The root `.venv/` folder is ignored by Git. Use `uv run python` for portable repository workflows; reserve `.venv/bin/python` for interpreter-specific debugging.

## Runtime Policy

Long or broken sweeps should not run indefinitely. Use benchmark timeout options such as `--subprocess-timeout-s 60` or analysis scripts with per-case timeout support. A timed-out or failed case should write a diagnostic row and continue to the next case instead of blocking the whole workflow.
