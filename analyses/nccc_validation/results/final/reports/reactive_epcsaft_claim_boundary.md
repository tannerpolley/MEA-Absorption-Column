# Reactive ePC-SAFT Claim Readiness

Date: 2026-05-07

## Current Finding

The external ePC-SAFT package exposes a native activity-coupled chemical-equilibrium solver through `solve_reactive_speciation(...)` and `ePCSAFTMixture.chemical_equilibrium(...)`. A local probe was added at `analyses/nccc_validation/scripts/probe_reactive_epcsaft_speciation.py` to test that capability against one-bed NCCC C-case liquid inlet states using the six-species dataset vendored in this repository.

All seven C cases converged for three local speciation modes:

| Probe mode | Successful cases | Median runtime per local solve (s) | Interpretation |
| --- | ---: | ---: | --- |
| `legacy_concentration_constants` | 7/7 | 0.166 | Native ePC-SAFT reactive solver reproduces the existing six-species concentration-basis chemistry closely. |
| `legacy_constants_as_activity_basis` | 7/7 | 0.188 | ePC-SAFT activity coupling with unrevised constants converges, but shifts speciation substantially. |
| `activity_constants_calibrated_to_legacy_state` | 7/7 | 0.138 | Activity-basis constants can reproduce a target state when re-based to that state, but this is not predictive calibration. |

The direct absorber smoke case with the current `epcsaft_ionic` fugacity lane also converged for Case 3C, but that run still used the existing chemistry model before calling the ionic ePC-SAFT CO2 fugacity calculation.

## Column Replacement Test

The `full-electrolyte-reactive` branch now includes a selectable `chemical_equilibrium_model` option. The experimental model `epcsaft_reactive_six_concentration` replaces the local legacy chemistry solve inside `abs_column(...)` with the native ePC-SAFT reactive-speciation solver while preserving the six-species shape expected by the existing enhancement-factor, fugacity, and profile-export code.

The bounded Case 3C collocation comparison used `mesh_points=21`, `max_nodes=250`, and a coarse ePC-SAFT chemistry cache (`T` rounded to 0.1 K, composition rounded to three decimals, pressure rounded to 100 Pa). After reinstalling the current local ePC-SAFT checkout into the Python 3.13 MEA environment, the concentration-standard-state replacement improved from 56.97 s to 51.23 s, but chemistry remained the dominant cost.

| Thermodynamic lane | Chemistry model | Success | Runtime (s) | Capture (%) | Capture error (pct-pt) | Temperature RMSE (K) | ePC-SAFT chemistry solve time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ideal_henry` | `legacy` | yes | 5.62 | 89.40 | -0.10 | 3.94 | 0.00 |
| `epcsaft_ionic` | `legacy` | yes | 6.73 | 89.83 | +0.33 | 3.97 | 0.00 |
| `epcsaft_reactive_six_concentration` | `epcsaft_reactive_six_concentration` | yes | 56.97 | 89.73 | +0.23 | 3.97 | 49.92 |
| `epcsaft_reactive_six_concentration` | `epcsaft_reactive_six_concentration` | yes | 51.23 | 89.73 | +0.23 | 3.97 | 44.98 |
| `epcsaft_reactive_six_concentration` | `epcsaft_reactive_six_concentration`, two-decimal composition cache | yes | 42.83 | 89.41 | -0.09 | 3.96 | 36.28 |

The same column run with `epcsaft_reactive_six_activity` timed out at 60 s under the same bounded settings. This is expected from the local speciation probe: using the legacy concentration-basis constants directly as activity-basis constants shifts free CO2 by orders of magnitude and is not yet a calibrated production chemistry lane.

A basis-corrected activity test was added as `epcsaft_reactive_six_activity_converted`. For the two six-species reactions, the stoichiometric sums are both -1, so the concentration-basis constants were converted with

```text
log K_x = log K_c - sum(nu_i) log(C_total)
```

where `C_total` is the apparent liquid molar concentration in mol/m3. This fixes the unit-basis error in the raw activity test and reduces the Case 3C free-CO2 shift, but the full column run still timed out at a 75 s subprocess gate both with ionic ePC-SAFT fugacity and with Henry fugacity. The remaining issue is the repeated activity-coupled reactive-speciation solve inside the collocation residual, not the fugacity calculation.

## Performance Diagnosis

Single-state Case 3C liquid-inlet timings after reinstalling ePC-SAFT show why direct activity-coupled chemistry is too slow for every BVP residual call:

| Local reactive speciation mode | Runtime per solve (s) | Native iterations | Free CO2 mole fraction | Comment |
| --- | ---: | ---: | ---: | --- |
| concentration standard state | 0.15--0.24 | 13 | 1.36e-10 | Reproduces the legacy concentration chemistry. |
| raw mole-fraction activity | 0.24--0.27 | 28 | 4.09e-3 | Wrong basis for the current constants. |
| converted mole-fraction activity | 0.18 | 24 | 8.76e-7 | Unit-consistent first test, but not yet validated. |

The ePC-SAFT diagnostics report `jacobian_backend=finite_difference` even when `auto` or `autodiff` is requested for these activity-coupled native calls. Therefore, each chemical-equilibrium Newton iteration requires repeated native state/activity evaluations. In the full absorber run, ePC-SAFT fugacity density solves accounted for only about 0.32--0.35 s, while reactive chemistry accounted for 36--45 s depending on cache granularity.

The downstream speed path should therefore be a tabulated reactive-speciation layer over local liquid state variables, followed by exact ePC-SAFT re-evaluation on accepted final profiles. Direct full ePC-SAFT reactive speciation at every nonlinear residual point is not fast enough for uncertainty quantification in its present form.

## Claim Boundary

The manuscript should not yet claim a validated full electrolyte-reactive absorber model. The branch now proves that the column can be run with the legacy six-species chemistry solve replaced by native ePC-SAFT reactive speciation, but the only converged column replacement tested so far uses concentration-basis constants selected to reproduce the legacy chemistry. The more rigorous activity-basis replacement now has a unit-consistent prototype, but it still needs validation, acceleration, and likely regression of activity-basis equilibrium constants before it can be used as paper-facing predictive evidence.

The tested claim that is currently supported is narrower:

> The project now includes a verified ePC-SAFT reactive-speciation prototype for MEA absorber-like liquid states, and a six-species ionic ePC-SAFT fugacity lane for the column model. These results show a feasible path toward a fully coupled electrolyte-reactive absorber, but the published absorber benchmark should still describe the present ePC-SAFT column runs as fugacity-driving-force sensitivity studies.

## Required Work Before Claiming Full Electrolyte-Reactive Absorber Modeling

1. Define and document the reaction set, balances, standard states, and equilibrium constants on the same activity basis used by ePC-SAFT.
2. Regress or validate the activity-basis equilibrium constants against MEA thermodynamic data before using them in absorber predictions.
3. Decide whether the production absorber will use the current six-species set or the full nine-species electrolyte set; then update enhancement-factor, transport-property, and reporting code to consume that species basis consistently.
4. Add a downstream tabulated or continuation-warm-started speciation layer because direct reactive speciation costs about 0.15--0.27 s per local solve, which is too expensive for hundreds to thousands of nonlinear BVP residual calls.
5. Validate the coupled reactive absorber separately from the current fugacity-only sensitivity results, including capture error, temperature profiles, residuals, guard counts, runtime, and failure modes.
