# Workflow Map

This document is the repo-facing handoff map for future agents or forks. It follows the project architecture standard in `C:\Users\Tanner\.codex\PROJECT_ARCHITECTURE.md`.

## Repository Layout

| Path | Owner | Purpose | Generated outputs |
| --- | --- | --- | --- |
| `src/mea_absorption_column/` | Package | Reusable absorber model, thermodynamic adapters, solver wrappers, benchmark CLI, packaged reference CSV data, and MEA-local ePC-SAFT parameter files. | No study run outputs should be written here. |
| `tests/` | Package tests | Fast regression and schema checks for model, benchmark, thermodynamics, and artifact behavior. | Test temp files should use pytest temp folders or ignored `.tmp_local/`. |
| `analyses/nccc_validation/` | Analysis | Canonical reviewer-response validation workflow for one-bed C cases, SRP-style solver comparisons, dense profile CSVs, and manuscript figures. | Disposable runs under `results/runs/`; curated evidence under `results/final/`. |
| `docs/latex/` | Manuscript | LaTeX source, bibliography, source-only helper scripts, Overleaf mirror sync, appendices, and manuscript-local figure copies. | Local LaTeX build products stay under ignored `builds/`; `builds/main.pdf` can be regenerated with `scripts/build_main.ps1`. |
| `docs/` | Supporting notes | Reviewer-response notes, robust-convergence status, and this workflow map. | Do not put benchmark run artifacts here. |
| `scripts/` | Repo tools | Repository-wide utilities or small smoke checks only. | No paper-facing sweep outputs. |

## NCCC Validation Workflow

The canonical analysis folder is `analyses/nccc_validation/`.

| Script | What it does | Runs the absorber model? | ePC-SAFT dependency |
| --- | --- | --- | --- |
| `scripts/generate_data.py` | Normalizes curated benchmark rows into raw, verified, and plot-ready final tables. | No, except it reads existing run/final CSVs. | No direct ePC-SAFT import; may process ePC-SAFT result rows already generated elsewhere. |
| `scripts/render_figures.py` | Renders manuscript figures from final tables. | No. | No direct ePC-SAFT import; plots ePC-SAFT rows when present. |
| `scripts/collect_clean_profiles.py` | Builds or refreshes the clean temperature-profile PNG gallery and index. | Yes when not using existing profile images. | Optional; required only when collecting/rerunning `epcsaft_*` thermodynamic lanes. |
| `scripts/run_case_profile.py` | Runs one case and writes dense per-variable profile CSVs plus a rerun spec. | Yes. | Optional; required for `epcsaft_neutral`, `epcsaft_ionic`, or experimental reactive lanes. |
| `scripts/generate_clean_profile_csvs.py` | Runs accepted clean rows with per-case timeouts and exports dense profile CSVs. | Yes. | Optional by suite; ePC-SAFT required for ePC-SAFT C-case profile rows. |
| `scripts/probe_reactive_epcsaft_speciation.py` | Experimental downstream probe of ePC-SAFT reactive speciation against legacy MEA chemistry states. | Yes, but it is a thermodynamic/speciation probe rather than a full column validation sweep. | Required; uses the vendored MEA ePC-SAFT dataset by default. |
| `scripts/validate_results.py` | Checks final tables, figures, profile indexes, and stale path regressions. | No. | No direct ePC-SAFT import. |

## ePC-SAFT Dependency Contract

Henry-only tests and benchmarks should run without the external ePC-SAFT checkout. ePC-SAFT workflows are opt-in thermodynamic lanes:

- `ideal_henry`: default validation lane; no external ePC-SAFT dependency.
- `epcsaft_neutral`: uses the external `epcsaft` package read-only and the MEA-local neutral parameter data under `src/mea_absorption_column/data/epcsaft_neutral/`.
- `epcsaft_ionic`: experimental diagnostic fugacity lane. It requires the external `epcsaft` package and uses the vendored six-species ePC-SAFT dataset plus the liquid state produced by the legacy chemistry model.
- `epcsaft_reactive_*`: experimental diagnostic chemistry lanes. They require the external `epcsaft` package and the vendored six-species ePC-SAFT dataset. Treat these as Case-3C smoke-tested until broader final tables and manuscript claims are updated.

Typical local environment variables for ePC-SAFT diagnostics:

```powershell
$env:MEA_EPCSAFT_ROOT = "C:\Users\Tanner\Documents\git\ePC-SAFT"
$env:MEA_EPCSAFT_DATASET_NAME = "MEA_CO2_H2O_draft"
```

The parameter datasets live in `src/mea_absorption_column/data/epcsaft_datasets/` so a fork can test ionic and reactive lanes without a sibling MEA-Thermodynamics checkout. `MEA_THERMODYNAMICS_EPCSAFT_DATASET` is still available as an explicit override for one-off comparisons, but it must not be required for default tests. Do not edit the external ePC-SAFT package from this repo. If package behavior blocks absorber validation, record a downstream issue or upstream request in the ePC-SAFT repo and keep the MEA repo changes limited to adapters, caching, guards, benchmark settings, and analysis scripts.

## Common Commands

Use the project-root `.venv` for this repository. Create or refresh it with:

```powershell
uv sync --group test
```

If running scripts from a Git worktree or from an unusual shell context, set `PYTHONPATH=src` so Python imports the active checkout:

```powershell
$env:PYTHONPATH = "src"
```

Fast package test:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Validate curated NCCC artifacts without rerunning long simulations:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py
```

Run one clean Henry profile export:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\run_case_profile.py --case-source C_cases_data --case-id 3C --method scipy-bvp --thermo-model ideal_henry --output-dir analyses\nccc_validation\results\runs\manual_case_profiles
```

Run one ePC-SAFT smoke profile after installing/updating the external package:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\run_case_profile.py --case-source C_cases_data --case-id 3C --method scipy-bvp --thermo-model epcsaft_neutral --output-dir analyses\nccc_validation\results\runs\manual_epcsaft_profile
```

The root `.venv/` folder is ignored by Git. Do not hard-code `C:\Users\Tanner\.codex\venvs\...` in repo workflows; use `.\.venv\Scripts\python.exe` so commands work from a normal checkout.

## Runtime Policy

Long or broken sweeps should not run indefinitely. Use benchmark timeout options such as `--subprocess-timeout-s 60` or analysis scripts with per-case timeout support. A timed-out or failed case should write a diagnostic row and continue to the next case instead of blocking the whole workflow.
