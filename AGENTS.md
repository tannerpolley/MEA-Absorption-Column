# Local Codex Instructions

Repository Profile: scientific-computing

## Startup Reads

- Read `docs/.codex-journal/user_preferences.md` when it exists.
- Read `docs/.codex-journal/project_memory.md` when it exists.

## Memory Policy

- Keep user preferences and durable project facts concise, date-stamped, and deduplicated.
- Do not update memory for routine Q&A or small one-off work.
- Do not store secrets, add placeholder entries, or create new memory under `.codex` or `$HOME/.codex/projects`.

## Repository Workflow

- Prefer `Local` for foreground solver inspection and `Worktree` for isolated background implementation.
- Prefer uv-managed commands. Use `.venv/bin/python` only for interpreter-specific debugging.
- Use `.codex/environments/environment.toml` actions when available.
- For LaTeX/manuscript work, apply `$HOME/.codex/LATEX.md` plus repository-local policy.
- Preserve the user's existing dirty worktree and inspect overlapping files before editing.

## Commit Discipline

- Commit, push, or create a PR only when the user requests it or an approved workflow requires it.
- Before committing, verify the current branch, validation results, status, and final commit.

## ePC-SAFT Cross-Repo Integration

- This is an official downstream application under ePC-SAFT Governance D-037.
- Engine governance and source live at `/home/tnnrpolley21/Workspaces/Engineering/ePC-SAFT-project`; do not use the retired `/ePC-SAFT` path or a sibling source import.
- Normal and final work uses one non-editable `epcsaft` wheel identified by Engine commit and wheel SHA-256. Intentional co-development uses an explicitly supplied candidate wheel with the same recorded identity.
- Keep absorber integration, column validation, process analyses, and this repository's manuscript here. Thermodynamic parameter adoption remains owned by MEA-Thermodynamics; generic equations and solvers remain owned by ePC-SAFT-project.
- Do not create nested repositories, submodules, mutable Git package dependencies, or dictionary compatibility copies of Engine behavior.
- Final manuscript, report, or archive results must pass `uv run python scripts/check_epcsaft_integration.py --mode final` without mutable package state.
- Preserve result-critical datasets under `src/mea_absorption_column/data/epcsaft_datasets`.
- Keep reusable `epcsaft` interactions behind explicit thermodynamics/runtime modules.
- Use `epcsaft-cross-repo` for contracts, upstream feedback, and handoffs.
