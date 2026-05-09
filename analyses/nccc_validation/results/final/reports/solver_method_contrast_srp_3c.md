# Solver Method Contrast: SRP-Style Case And NCCC Case 3C

## Purpose

This report records the first curated method-comparison slice after adding the
legacy SRP-style/high-L/G benchmark case to the current validation workflow.
The comparison separates two different questions:

- A favorable one-bed method case can be used to show that shooting, finite
  difference, and collocation are all executable under benign operating
  conditions.
- NCCC Case 3C is the validation case with measured temperature taps and a
  strong thermal bulge. It is the more relevant test for predictive absorber
  simulation.

## Favorable SRP-Style Method Case

Source artifact: `analyses/nccc_validation/results/runs/srp_all_methods_coarse/benchmark_results.csv`

| Case | Method | Thermo | Success | Runtime s | Capture % | Notes |
|---|---:|---:|---:|---:|---:|---|
| SRP-LG7 | shooting | ideal Henry | true | 4.91 | 90.17 | Fast IVP/root comparison run. |
| SRP-LG7 | Collocation BVP | ideal Henry | true | 7.61 | 89.95 | Shooting-seeded collocation, low boundary residual. |
| SRP-LG7 | finite difference | ideal Henry | true | 15.62 | 107.65 | Coarse 21-point algebraic solve; boundary residual is low, but capture overshoots because this row has no measured target gate. |

The legacy figures reported approximately 0.6 s for shooting, 2.7 s for
collocation, and 227 s for finite difference on the same nominal high-L/G row.
The current code is slower for shooting and collocation because it carries the
reviewer-response guard, diagnostics, Python 3.13/SciPy 1.17 stack, and benchmark
metadata path. The current collocation solve can still be made comparable by
using shooting-seeded continuation; without the seed it enters nonphysical
trial states during Newton iterations.

## NCCC Case 3C Validation Contrast

Source artifacts:

- `analyses/nccc_validation/results/runs/nccc_3c_collocation_default/benchmark_results.csv`
- `analyses/nccc_validation/results/runs/nccc_3c_shoot_fd_60s/benchmark_results.csv`

| Case | Method | Thermo | Success | Runtime s | Capture % | Capture error pct-pt | Temperature RMSE K | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3C | Collocation BVP | ideal Henry | true | 9.40 | 89.40 | -0.10 | 3.94 | Converged directly with measured temperature-profile validation. |
| 3C | shooting | ideal Henry | false | 62.73 |  |  |  | Timed out at the bounded 60 s solver gate. |
| 3C | finite difference | ideal Henry | false | 16.61 | 0.00 | -89.50 | 20.04 | Algebraic residual converged to a boundary-satisfying but physically wrong zero-capture branch. |

This is the core numerical-method argument for the manuscript. Shooting can be
fast when the boundary-value problem behaves like a mild IVP continuation, but
the NCCC thermal pinch makes the missing boundary conditions highly sensitive.
Finite difference can satisfy algebraic boundary equations on a coarse mesh, but
it needs stronger physical-state and capture-profile acceptance gates to avoid
false branches. Collocation is the strongest reference method here because it
solves the distributed boundary-value problem directly and can use residual
control across the full temperature profile.

## ePC-SAFT Electrolyte Configuration Evidence

The pressure-state smoke table has been superseded by
`epcsaft_electrolyte_column_config_matrix.csv`. The selected paper-facing
configuration is `2025_Figiel_empirical_fitted_Born_SSM_DS`, which keeps the
six-species MEA absorber state and uses the repo-vendored
`MEA_CO2_H2O_ionic_fit` dataset. In the Case 3C configuration matrix, that row
converged with the collocation BVP calculation, predicted 89.83% capture, and
reported a 7.90 s runtime under the matrix smoke settings.

The current ePC-SAFT calls remain pressure-specified states. The MEA adapter
passes the previous converged molar density as an initial guess for the package's
internal pressure-density solve; it does not use direct density states for the
validation fugacity calculation. The remaining overhead is the expected cost of
the six-species electrolyte fugacity calculation and external-dataset path.
