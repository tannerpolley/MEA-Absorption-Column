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

## Current scientific and manuscript scope

Read [the scientific context](scientific/CONTEXT.md) before selecting a CSE
workflow. The complete nine-species manuscript and its incoming result
requirements are described in [SOURCE_MAP.md](latex/SOURCE_MAP.md).
The older fixed-chemistry and six-species commands below remain explicitly
identified legacy calculations; they are not the current coupled-column
result source.

This repository consumes one identified immutable Engine wheel. Generic EOS,
equilibrium, and caloric implementations belong to ePC-SAFT-project; adopted
MEA parameters belong to MEA-Thermodynamics. Packaged data remain under
`src/mea_absorption_column/data/epcsaft_datasets/`. Inspect the selected run's
parameter and Engine identities before using its results.

For manuscript writing and read-only validation, use the commands in
[docs/scientific/README.md](scientific/README.md). The HTML checklist reads
current manuscript files without running any scientific calculation.

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

Legacy Henry profile export (executes a model; not needed for manuscript work):

```bash
uv run python analyses/nccc_validation/scripts/run_case_profile.py --case-source C_cases_data --case-id 3C --method scipy-bvp --thermo-model ideal_henry --output-dir analyses/nccc_validation/results/runs/manual_case_profiles
```

Legacy fixed-chemistry ePC-SAFT profile export (requires its identified wheel and explicit model-run scope):

```bash
uv run python analyses/nccc_validation/scripts/run_case_profile.py --case-source C_cases_data --case-id 3C --method scipy-bvp --thermo-model epcsaft_ionic --output-dir analyses/nccc_validation/results/runs/manual_epcsaft_profile
```

The root `.venv/` folder is ignored by Git. Use `uv run python` for portable repository workflows; reserve `.venv/bin/python` for interpreter-specific debugging.

## Runtime Policy

Long or broken sweeps should not run indefinitely. Use benchmark timeout options such as `--subprocess-timeout-s 60` or analysis scripts with per-case timeout support. A timed-out or failed case should write a diagnostic row and continue to the next case instead of blocking the whole workflow.
