# C-Case Trend Root Cause

## Bottom Line
The 1C -> 7C capture-error trend is not explained by mesh density or initial capture guess. The bounded probe on `epcsaft_neutral` kept capture error essentially unchanged across coarse and fine mesh settings, while the profile shapes changed only at the sub-K level for mesh/guess perturbations. The trend also does not move under the vapor-basis toggle (`legacy_ratio` vs `input_o2`).

The only global knob that moved capture materially was the gas-velocity area exponent, but it helped 1C and hurt 3C while leaving 7C essentially overcaptured, so it is not a safe global fix.

## Commands Run

```text
.venv\Scripts\python.exe analyses\nccc_validation\scripts\run_c_case_root_cause.py
.venv\Scripts\python.exe -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models epcsaft_neutral --c-case-ids 1C 3C 7C --nccc-case-limit 0 --srp-case-limit 0 --staged-beds false --mesh-points 51 --tol 0.5 --bc-tol 0.001 --max-nodes 1000 --co2-capture-guess-pct 95 --vapor-composition-mode input_o2 --profile-csvs --profile-pngs --subprocess-timeout-s 60
.venv\Scripts\python.exe -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models epcsaft_neutral --c-case-ids 1C 3C 7C --nccc-case-limit 0 --srp-case-limit 0 --staged-beds false --mesh-points 51 --tol 0.5 --bc-tol 0.001 --max-nodes 1000 --co2-capture-guess-pct 95 --gas-velocity-area-exponent 0.5 --gas-velocity-area-reference-m-s 1.0 --profile-csvs --profile-pngs --subprocess-timeout-s 60
```

## References Found

| Source | Why it matters |
|---|---|
| Zotero `C96CLVL4` - *Effects of the temperature bulge in CO2 absorption from flue gas by aqueous monoethanolamine* | Directly addresses temperature bulge behavior versus L/G, packing height, and CO2 concentration. |
| Zotero `KJ688HMG` - *Development of a Rigorous Modeling Framework for Solvent-Based CO2 Capture. 1. Hydraulic and Mass Transfer Models and Their Uncertainty Quantification* | Strong match for hydraulics, interfacial area, and UQ on absorber submodels. |
| Zotero `CSUAWG2U` - *Review on the mass transfer performance of CO2 absorption by amine-based solvents in low- and high-pressure absorption packed columns* | Broad packed-column mass-transfer review. |
| Zotero `62XFSRXZ` - *Uncertainty quantification of property models: Methodology and its application to CO2-loaded aqueous MEA solutions* | Supports the pilot-data/uncertainty framing. |
| Zotero `4S6JGHGJ` - *Learning the properties of a water-lean amine solvent from carbon capture pilot experiments* | Useful pilot-scale uncertainty and property-learning reference. |
| Zotero `R34X6UXM` - *Prediction of mass transfer columns with dumped and arranged packings: Updated summary of the calculation method of Billet and Schultes* | Useful for effective area / flooding / packing correlations. |

## Input Reconstruction Verified

- `C_cases_data.csv` provides the one-bed C-case inputs directly: `L/G`, `G`, `alpha`, `w_MEA`, `y_CO2`, `Tl`, `Tv`, `P`, `beds`, `y_O2`, and the tap temperatures at normalized positions `0.0, 0.2, 0.4, 0.6, 0.8`.
- For 1C / 3C / 7C, the table values are:
  - 1C: `L/G=3.1148`, `G=24.1359`, `y_CO2=0.076`, `y_O2=0.114`, target capture `97.1%`
  - 3C: `L/G=4.7023`, `G=19.4303`, `y_CO2=0.079`, `y_O2=0.097`, target capture `89.5%`
  - 7C: `L/G=6.8586`, `G=9.5947`, `y_CO2=0.109`, `y_O2=0.072`, target capture `76.4%`
- `convert_data(..., vapor_composition_mode="legacy_ratio")` infers `y_H2O = y_CO2 * 0.9626010166` and splits the remaining vapor between N2 and O2 with the legacy O2/N2 ratio.
- `convert_data(..., vapor_composition_mode="input_o2")` uses the table `y_O2` directly. That changed the split, but not the trend.
- Geometry resolved by the model is `diameter_m = 0.64` and `single_bed_height_m = 6.0`, with one bed for these C-cases.

## Bounded Probe Results

### Mesh and initial guess
Script: `analyses/nccc_validation/scripts/run_c_case_root_cause.py`

Configs tested on `epcsaft_neutral`:
- `baseline_51_95`: mesh 51, `tol=0.5`, `bc_tol=0.001`, `max_nodes=1000`, capture guess 95%
- `coarse_21_80`: mesh 21, `tol=1.0`, `bc_tol=0.01`, `max_nodes=400`, capture guess 80%
- `fine_101_99`: mesh 101, `tol=0.2`, `bc_tol=0.0001`, `max_nodes=2000`, capture guess 99%

Result:
- Capture errors stayed essentially unchanged.
  - 1C: `-17.9935%` -> `-18.0444%` -> `-17.9937%`
  - 3C: `+0.1982%` -> `+0.1975%` -> `+0.2001%`
  - 7C: `+22.0908%` -> `+22.0885%` -> `+22.0910%`
- Profile shape deltas versus baseline were tiny:
  - coarse mesh: TL RMSE `0.114-0.636 K`, TV RMSE `0.066-0.597 K`
  - fine mesh: TL RMSE `0.011-0.111 K`, TV RMSE `0.008-0.095 K`
- Conclusion: the profile shape is not being forced by mesh density or the initial capture guess.

### Vapor basis
Command:
`BenchmarkSettings(..., solver_settings={"vapor_composition_mode": "input_o2"})`

Result:
- Capture errors were unchanged to within about `0.001` percentage points.
- Profile deltas versus baseline were negligible:
  - TL RMSE `0.0003-0.0016 K`
  - TV RMSE `0.0003-0.0017 K`
- Conclusion: the trend is not an artifact of the O2/H2O vapor reconstruction choice.

### Gas-velocity area exponent
Command:
`BenchmarkSettings(..., solver_settings={"gas_velocity_area_exponent": 0.5, "gas_velocity_area_reference_m_s": 1.0})`

Result:
- 1C improved from `-17.9935%` to `-6.5626%`.
- 3C moved the wrong way from `+0.1982%` to `+6.7164%`.
- 7C stayed overcaptured at `+22.2584%`.
- Profile deltas were real, not tiny:
  - 1C TL RMSE `1.745 K`, TV RMSE `1.655 K`
  - 3C TL RMSE `1.617 K`, TV RMSE `1.496 K`
  - 7C TL RMSE `0.671 K`, TV RMSE `0.582 K`
- Conclusion: the area exponent is a plausible physics knob, but this particular global form is not the right correction.

## What Improved, What Did Not

- Improved:
  - The new analysis artifact set is complete and reproducible.
  - Mesh and initialization sensitivity are now bounded with profile deltas, not just scalar capture error.
  - The vapor basis question is effectively ruled out as the main driver.
- Did not improve:
  - The 1C -> 7C capture-error trend remains.
  - 7C remains the high-bias endpoint.
  - The gas-velocity area exponent is not a safe global correction.

## Recommended True Next Step

The highest-probability next correction is a physically calibrated hydraulics/contacting update, not another solver-mesh tweak:

1. Keep the benchmark and profile exports fixed.
2. Replace the ad hoc gas-velocity exponent with a correlation tied to effective area or flooding/contacting data.
3. Validate that change against the pilot/UQ literature rather than case-by-case tuning.
4. Re-check 1C, 3C, and 7C together with the existing capture-error table and the profile overlay artifact.

## Artifacts

- `analyses/nccc_validation/results/runs/c_case_trend_root_cause/c_case_trend_root_cause_rows.csv`
- `analyses/nccc_validation/results/runs/c_case_trend_root_cause/c_case_trend_root_cause_summary.csv`
- `analyses/nccc_validation/results/runs/c_case_trend_root_cause/c_case_trend_root_cause_profile_deltas.csv`
- `analyses/nccc_validation/results/runs/c_case_trend_root_cause/baseline_51_95/temperature_profiles/7C_C_cases_data_scipy-bvp_epcsaft_neutral.png`
- `analyses/nccc_validation/results/runs/c_case_trend_root_cause/gasvel_area_exp_0p5/temperature_profiles/7C_C_cases_data_scipy-bvp_epcsaft_neutral.png`
