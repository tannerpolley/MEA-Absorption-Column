# SciPy BVP verification

This Issue #18 analysis set retains the executable Issue #19 column tranche. It compares the existing direct SciPy BVP route for `SRP-LG7` and NCCC 2017 Case 3C with concentration-based chemistry fixed and the existing `ideal_henry` and `epcsaft_ionic` driving-force closures.

The retained campaign does not run or interpret the Issue #16 physical 21-state reactive-film calculation. It changes no thermodynamic, chemistry, kinetic, transport, hydraulic, area, or acceptance parameter.

Run the bounded campaign after committing the executable code:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python analyses/bvp_derivative_trials/scripts/run_issue19_column_verification.py
```

Validate retained rows without rerunning the model:

```bash
uv run python analyses/bvp_derivative_trials/scripts/run_issue19_column_verification.py --validate-only
```

Per-attempt subprocess output is disposable under `results/runs/`. The candidate table and summary under `results/final/tables/` retain successes, timeouts, numerical failures, configuration, solver counters, residuals, conservation checks, branch accounting, and immutable Engine identity.
