# Analyses

This folder contains study-specific workflows. Each analysis should be self-contained, with local `scripts/`, `data/`, and `results/` folders, and should not scatter generated artifacts into the repository root or `docs/`.

## Active Analyses

| Analysis | Purpose | ePC-SAFT dependency | Primary commands |
| --- | --- | --- | --- |
| `nccc_validation` | Reviewer-response validation for one-bed C cases, SRP-style solver comparisons, temperature profiles, and thermodynamic driving-force comparisons. | Optional for `epcsaft_neutral` and required for experimental reactive ePC-SAFT probes. Henry-only validation does not require ePC-SAFT. | See `analyses/nccc_validation/README.md` and `analyses/nccc_validation/analysis.yaml`. |

## Conventions

- Put analysis entrypoints in `analyses/<analysis_id>/scripts/`.
- Put stable analysis-specific inputs in `analyses/<analysis_id>/data/input/`.
- Put disposable run outputs in `analyses/<analysis_id>/results/runs/`; these are ignored by Git.
- Put curated, paper-facing tables, figures, reports, and clean profile galleries in `analyses/<analysis_id>/results/final/`.
- Keep manuscript source under `docs/latex`; LaTeX should consume curated final artifacts, not disposable run folders.
