# Intercooler Model Critique And Case 3A Shape Benchmark

## Current Implementation

The original staged-bed implementation treats each absorber bed with the same
packed-column differential model, then couples adjacent beds with vapor
continuity upward and liquid continuity downward. If an intercooler is present,
the liquid enthalpy in the inter-bed boundary residual is reset toward the
liquid-feed target temperature. This is a useful first-pass approximation of a
pumparound or external side cooler outlet specification, but it is a zero-length
thermal operation. It can create sharp local dips in the reported temperature
profile and adds stiff algebraic coupling to the segmented BVP.

## Revised Options

An opt-in `distributed_liquid_cooling` intercooler model was added. It keeps the
liquid and vapor continuity constraints between beds, but removes the hard
liquid-enthalpy reset from the boundary condition. The liquid cooling duty is
instead applied as a finite relaxation source term over a configurable fraction
of the receiving bed near the inter-bed return location:

```text
Q_intercooler(z) = s w(z) (H_target - H_liquid) / L_zone
```

where `s` is the intercooler strength, `w(z)` is a raised-cosine window, and
`L_zone` is the physical cooling-zone length. This is still a reduced model; it
does not yet solve a separate heat-exchanger energy balance or cooling-water UA.
It is more numerically forgiving because the BVP no longer has to satisfy an
instantaneous liquid enthalpy jump exactly at the bed boundary.

A second opt-in mode, `pumparound_temperature_approach`, was added after
checking the old temperature-solve branch and the manuscript derivation. The
legacy `Solve_Temperature` branch used the same chain-rule temperature
differentials now documented in the manuscript:

```text
dTl/dz = (Hl_flux + Hl * (Nl_CO2 + Nl_H2O)) / (Fl * dHl/dTl)
dTv/dz = (Hv_flux - Hv * (Nv_CO2 + Nv_H2O)) / (Fv * Cpv)
```

The current implementation keeps the enthalpy-flow mode as the default for
single-bed reproducibility, but `pumparound_temperature_approach` automatically
uses temperature states for staged/intercooled BVPs. This avoids repeated
enthalpy inversion during the collocation iteration and makes the side-cooler
target a direct temperature boundary contract rather than an inferred enthalpy
constraint.

## Case 3A Guideline

The supplied low-resolution Case 3A model curve was digitized into
`analyses/nccc_validation/data/input/case3a_supplied_image_model_guideline.csv`.
The digitized curve is used only as a qualitative shape guideline. Morgan
Appendix C measured points remain the validation data shown as symbols.

## Current Result

The comparison figure is
`analyses/nccc_validation/results/final/figures/intercooler_model_comparison_case3a.png`.
For the K3/3A smoke case, the newer pumparound-temperature option is the best
current tradeoff: it converges cleanly, matches capture well, and gives rounded
in-bed sections. The pure distributed-cooling probe smooths boundary artifacts
but is not yet a clean accepted run for this case under the current timeout.

| Model | Status | Capture error, pct-pt | Runtime, s | Mean section quadratic R2 |
| --- | --- | ---: | ---: | ---: |
| Hard liquid-temperature reset | accepted | +0.24 | 49.0 | 1.000 |
| Distributed liquid-cooling relaxation | diagnostic timeout | +11.43 | 53.5 | 1.000 |
| Pumparound temperature approach | accepted | -0.25 | 26.0 | 0.983 |

This proves that the temperature-state pumparound formulation can produce
smooth, quadratic-like sections without sacrificing the K3 capture prediction.
It does not yet prove general predictive accuracy across all intercooled NCCC
rows. The remaining mismatch to the digitized 3A guideline and measured
Appendix C points should be treated as a calibration and case-mapping question,
not solved by manually reshaping a plot.

## Next Model Step

The more rigorous next step is a finite heat-exchanger pumparound block:
withdraw liquid at a bed boundary, solve a cooler outlet condition from either
measured outlet temperature, duty, or UA/cooling-water specification, then return
the cooled stream as an explicit inter-bed feed with its own residual. That
would align more closely with RadFrac-style side-cooler/pumparound
specifications while keeping the reduced packed-bed physics transparent.
