# ePC-SAFT Pressure-State Timing Diagnostic

## Purpose

This report records the pressure-state timing check after the external ePC-SAFT
package added `rho_guess` support for pressure-specified states. The MEA adapter
still specifies temperature, pressure, composition, and phase. It does not use
direct density states for validation fugacity calculations. Cached molar
density is passed only as the next internal pressure-density initial guess.

## Case 3C Smoke Result

Source artifact: `analyses/nccc_validation/results/final/tables/case3c_epcsaft_pressure_state_smoke.csv`

| Thermo model | Success | Runtime s | Capture % | Capture error pct-pt | Temperature RMSE K | ePC-SAFT state time s | rho-guess hits | rho-guess misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Henry-law baseline | true | 5.95 | 89.40 | -0.10 | 3.94 | 0.00 | 0 | 0 |
| Neutral ePC-SAFT | true | 5.92 | 89.70 | 0.20 | 3.97 | 0.50 | 783 | 2 |
| Ionic ePC-SAFT | true | 6.15 | 89.83 | 0.33 | 3.97 | 0.34 | 786 | 2 |

## Microbenchmark

The state-construction microbenchmark used warmed mixture objects and repeated
nearby pressure-specified states. Median timings were:

| Path | State without rho guess ms | State with rho guess ms | Fugacity coefficient ms |
|---|---:|---:|---:|
| Neutral liquid, local parameters | 0.596 | 0.599 | 0.081 |
| Neutral vapor, dataset path | 0.211 | 0.214 | 0.043 |
| Ionic liquid, dataset path | 0.391 | 0.396 | 0.239 |

## Interpretation

The older 17 s ionic Case 3C run was dominated by pressure-state construction
inside the external package. After the package update and MEA-side rho-guess
reuse, the ionic lane is not a large outlier for this smoke case. Its remaining
overhead is mainly the larger six-species/electrolyte fugacity calculation and
dataset path, not a runaway density solve. The neutral and ionic rows remain
thermodynamic sensitivity evidence because chemistry, enhancement, transport,
hydraulics, and balances are unchanged.
