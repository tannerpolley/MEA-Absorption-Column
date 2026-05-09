# Full Ionic ePC-SAFT Speciation Handoff

Branch: `codex/full-ionic-speciation`

Date: 2026-05-09

## What changed

- Reactive six-species ePC-SAFT chemistry now reads the same runtime `user_options` source as ionic fugacity. This keeps the chemical-equilibrium activity coefficients and the liquid/vapor fugacity calculations on the same electrolyte option set.
- Reactive ePC-SAFT solver controls can be reproduced through environment variables:
  - `MEA_EPCSAFT_REACTIVE_MAX_ITERATIONS`
  - `MEA_EPCSAFT_REACTIVE_TOLERANCE`
  - `MEA_EPCSAFT_REACTIVE_MASS_TOLERANCE`
  - `MEA_EPCSAFT_REACTIVE_CHARGE_TOLERANCE`
  - `MEA_EPCSAFT_REACTIVE_REACTION_TOLERANCE`
  - `MEA_EPCSAFT_REACTIVE_DAMPING`
  - `MEA_EPCSAFT_REACTIVE_ACCEPT_BEST_EFFORT`
  - `MEA_EPCSAFT_REACTIVE_BEST_EFFORT_MASS_MAX`
  - `MEA_EPCSAFT_REACTIVE_BEST_EFFORT_CHARGE_MAX`
  - `MEA_EPCSAFT_REACTIVE_BEST_EFFORT_REACTION_MAX`
- Benchmark CSV output now includes maximum ePC-SAFT chemistry residual diagnostics and best-effort/failure counts.
- `analyses/nccc_validation/scripts/probe_reactive_epcsaft_speciation.py` now calls `ensure_epcsaft_importable()` before importing the sibling/local `epcsaft` package.

## Why this matches the coupled problem

The current column loop is not a standalone reactive flash. It is the column BVP repeatedly evaluating:

1. Apparent liquid state from the absorber unknowns.
2. ePC-SAFT activity-coefficient chemical equilibrium for true species.
3. Ionic-liquid and external-neutral ePC-SAFT fugacity coefficients from the true-species state.
4. Column residuals and collocation updates, which call the chemistry/fugacity sequence again.

This is consistent with the algorithmic idea in Ascani, Sadowski, and Held (2023): solve chemical equilibrium using thermodynamic-model activity or fugacity coefficients, then update the phase-equilibrium calculation through repeated coefficient updates. Their paper emphasizes a nested coefficient-update procedure for simultaneous chemical and phase equilibrium, with PC-SAFT supplying the non-ideal coefficients.

## Key runs

All runs below used:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MEA_EPCSAFT_CHEMISTRY_CACHE_X_DIGITS='4'
$env:MEA_EPCSAFT_CHEMISTRY_CACHE_T_DIGITS='1'
$env:MEA_EPCSAFT_CHEMISTRY_CACHE_P_ROUND_PA='100'
$env:MEA_EPCSAFT_REACTIVE_MAX_ITERATIONS='160'
```

| Mode | Case | Success | Runtime s | CO2 capture % | Capture error pct-pt | Temp RMSE K | Boundary residual | Invalid states | Domain guards | Chemistry cache hit/miss | Chemistry solve s | Max mass residual | Max reaction residual | Max charge residual | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| `epcsaft_reactive_six_activity_rebased` | 3C | true | 129.282 | 89.837 | 0.337 | 4.314 | 5.67e-14 | 0 | none | 123/167 | 90.629 | 9.999e-09 | 9.982e-09 | 4.39e-13 | Clean proof run. Activity coefficients are used, but reaction constants are rebased to the legacy concentration state at each apparent state. |
| `epcsaft_reactive_six_activity_converted` | 3C | true | 307.196 | 69.926 | -19.574 | 7.679 | 8.94e-10 | 47 | `chemical_equilibrium=3` | 375/251 | 297.124 | 3.04e304 | 5.29e1 | 1.01e304 | Column solver converged, but some intermediate collocation states failed chemistry. Treat as diagnostic, not headline proof. |
| `epcsaft_reactive_six_activity_converted` before patch/settings | 3C | false | 565.221 | 85.925 | -3.575 | 5.395 | 38.667 | 35 | `chemical_equilibrium=7` | 507/405 | 549.661 | not recorded | not recorded | not recorded | Previous loose run converged internally but was rejected by the strict boundary gate and profile fallback. |

Repro command template:

```powershell
.\.venv\Scripts\python.exe -m mea_absorption_column.benchmark `
  --methods scipy-bvp `
  --thermo-models epcsaft_reactive_six_activity_rebased `
  --c-case-ids 3C `
  --nccc-case-limit 0 `
  --srp-case-limit 0 `
  --staged-beds false `
  --mesh-points 7 `
  --tol 10 `
  --bc-tol 0.5 `
  --max-nodes 80 `
  --subprocess-timeout-s 900 `
  --output-dir analyses\nccc_validation\results\runs\full_ionic_speciation_rebased_after_patch
```

## Interpretation

The run worthy of building on is `epcsaft_reactive_six_activity_rebased`. It proves the column can converge while using activity-coupled ePC-SAFT speciation and ionic ePC-SAFT fugacity together, with no domain guards and chemistry residuals below `1e-8`.

The stricter `activity_converted` mode is useful evidence that the non-rebased constant conversion can complete a full column solve, but it is not yet a clean convergence claim because failed intermediate collocation states inflate the maximum chemistry residual diagnostics. The next scientific improvement should target this mode by adding continuation/warm-start handoff from one chemistry solve to the next or by moving closer to the double-nested update strategy from Ascani et al.

## Literature guidance

Ascani, Sadowski, and Held (2023) solve simultaneous chemical and phase equilibrium by using PC-SAFT to supply non-ideal coefficients and repeatedly updating chemical and phase-equilibrium calculations. They also determine reaction equilibrium constants from experimental equilibrium compositions plus PC-SAFT-predicted activity coefficients when standard Gibbs data are not used. That directly supports the current adapter direction:

- use ePC-SAFT activity coefficients inside the chemical-equilibrium residuals;
- keep phase/fugacity coefficients from the same parameter set and electrolyte options;
- use robust nested/continuation initialization for the fully coupled problem;
- be explicit whether equilibrium constants are literature/legacy concentration constants, converted to an activity basis, or locally rebased against an existing equilibrium state.

## Validation

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_epcsaft_reactive_chemistry.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_thermodynamics_adapter.py -k "epcsaft or ionic"
```

Results:

- `3 passed in 44.91s`
- `7 passed, 2 deselected in 14.90s`

BLAS thread pinning is important on this Windows machine. Without it, repeated SciPy/ePC-SAFT workers can saturate process resources and make imports or tests appear to hang.
