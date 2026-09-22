# Research workflow map

Updated 2026-09-08: manuscript revisions are submitted. The foreground activity
is alternate formulations and numerical methods, reported in analysis notebooks.

1. Select the formulation, thermodynamics, film and method in a study TOML file.
2. Preview with `python3 src/mea_absorption_column/research.py STUDY.toml`.
3. Run explicitly with `uv run python -m mea_absorption_column.research STUDY.toml --run`.
4. Inspect retained settings, failures, residuals, conservation and profiles.
5. Update the study Quarto notebook and render with execution disabled.
6. Promote a finding only after review appropriate to its claim.

`Run_Model.run_model` owns the seven-state calculation. `research.run_conserved`
selects among independent conservative solver functions with explicit supplied
node and boundary equations. A common solver name does not imply common physics.
The twelve-state experimental source is retrievable at tag
`archive/coupled-solver-2026-09-08`; the original worktree remains untouched.

The submitted PDF and reproduction records live on branch
`codex/fallback-manuscript` at `eef9dce`. Its SHA256SUMS covers that branch.
Main's retained earlier experiments preserve their original identities and
limitations, not updated claims under the newest parameters or code.
