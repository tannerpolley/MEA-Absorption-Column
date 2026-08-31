# SciPy BVP verification

This Issue #18 analysis set retains the executable Issue #19 column tranche. It compares the existing direct SciPy BVP route for `SRP-LG7` and NCCC 2017 Case 3C with concentration-based chemistry fixed and the existing `ideal_henry` and `epcsaft_ionic` driving-force closures.

The retained campaign does not run or interpret the Issue #16 physical 21-state reactive-film calculation. It changes no thermodynamic, chemistry, kinetic, transport, hydraulic, area, or acceptance parameter.

Run the bounded campaign after committing the executable code:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python analyses/bvp_derivative_trials/scripts/run_issue19_column_verification.py --case-timeout-s 100
```

Validate retained rows without rerunning the model:

```bash
uv run python analyses/bvp_derivative_trials/scripts/run_issue19_column_verification.py --validate-only
```

Run the analysis-specific classification and capture-cluster checks explicitly; root-level pytest does not discover this directory:

```bash
uv run pytest analyses/bvp_derivative_trials/tests/test_analysis.py -q
```

Per-attempt subprocess output is disposable under `results/runs/`. The candidate table and summary under `results/final/tables/` retain successes, typed failures, configuration, solver counters, residuals, conservation checks, scalar capture-cluster accounting, the per-case timeout, and immutable Engine identity. Capture clusters use a 0.5 percentage-point rule; they do not establish numerical solution-profile or branch identity because profiles are not retained.

After independent review admits the retained rows, regenerate the summary figure and render the notebook without executing scientific code:

```bash
uv run python analyses/bvp_derivative_trials/scripts/render_issue19_summary.py
bash analyses/bvp_derivative_trials/render.sh
```
