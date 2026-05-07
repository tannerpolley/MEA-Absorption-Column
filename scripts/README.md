# Repository Scripts

Root-level `scripts/` is reserved for repository-wide tools and small manual smoke checks. Study-specific benchmark, plotting, profile-export, and validation scripts belong under the owning analysis folder.

## Current Scripts

| Script | Role | ePC-SAFT dependency | Preferred alternative |
| --- | --- | --- | --- |
| `legacy_run_model_example.py` | Minimal legacy-style one-case smoke check for `run_model`. It is useful for quick local orientation only. | None when run with the default `ideal_henry` path. | For reproducible validation, use `analyses/nccc_validation/scripts/run_case_profile.py` or `python -m mea_absorption_column.benchmark`. |

Do not add paper-facing sweep scripts here. Add them to `analyses/<analysis_id>/scripts/` and document them in that analysis' `README.md` and `analysis.yaml`.
