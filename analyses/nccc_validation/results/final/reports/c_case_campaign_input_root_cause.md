# C-Case Campaign Input Root Cause

## Bottom line

The main 1C-7C capture trend was traced to the C-case input table, not to mesh forcing or a missing global transport multiplier. The legacy `C_cases_data.csv` differs from the campaign table already curated in the project for several primary operating variables, especially lean loading and gas/liquid flow conversion. Using the campaign-derived one-bed inputs makes all seven C cases converge in bounded subprocess runs and removes the severe 1C undercapture / 7C timeout pattern.

## Input differences

| Case | L/G_legacy | G_legacy | alpha_legacy | w_MEA_legacy | y_CO2_legacy | Tl_legacy | Tv_legacy | P_legacy | CO2 %_legacy | L/G_campaign | G_campaign | alpha_campaign | w_MEA_campaign | y_CO2_campaign | Tl_campaign | Tv_campaign | P_campaign | CO2 %_campaign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1C | 3.115 | 24.14 | 0.25 | 0.3 | 0.076 | 318.1 | 316.8 | 109100 | 97.1 | 3.9 | 19.28 | 0.15 | 0.3 | 0.077 | 318.1 | 316.4 | 109100 | 97.1 |
| 2C | 3.898 | 23.98 | 0.2 | 0.31 | 0.108 | 318.1 | 316.2 | 109600 | 92.3 | 3.938 | 23.98 | 0.2 | 0.3 | 0.109 | 318.1 | 316.2 | 109600 | 92.3 |
| 3C | 4.702 | 19.43 | 0.25 | 0.3 | 0.079 | 318.1 | 316.8 | 109500 | 89.5 | 4.717 | 19.37 | 0.25 | 0.3 | 0.093 | 318.1 | 316.8 | 109500 | 89.5 |
| 4C | 5.144 | 14.43 | 0.25 | 0.31 | 0.097 | 318.1 | 316.4 | 107700 | 88.9 | 5.125 | 14.48 | 0.25 | 0.31 | 0.077 | 318.1 | 316.4 | 107700 | 88.9 |
| 5C | 4.505 | 14.37 | 0.26 | 0.3 | 0.109 | 318.1 | 316.4 | 108400 | 86.4 | 3.68 | 17.3 | 0.26 | 0.3 | 0.1 | 318.1 | 318.1 | 108400 | 86.4 |
| 6C | 3.356 | 26.08 | 0.31 | 0.31 | 0.076 | 317.9 | 316.4 | 109700 | 60.2 | 3.575 | 26.07 | 0.31 | 0.3 | 0.077 | 317.9 | 317.4 | 109300 | 60.2 |
| 7C | 6.859 | 9.595 | 0.25 | 0.3 | 0.109 | 317.9 | 316.6 | 107300 | 76.4 | 4.628 | 14.35 | 0.34 | 0.29 | 0.117 | 318.6 | 316.2 | 107000 | 76.4 |

## Solver result comparison

| case_id | success_legacy | runtime_s_legacy | capture_pct_legacy | capture_error_pct_legacy | temperature_rmse_K_legacy | success_campaign | runtime_s_campaign | capture_pct_campaign | capture_error_pct_campaign | temperature_rmse_K_campaign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1C | True | 10.76 | 78.78 | -18.32 | 5.148 | True | 12.43 | 94.69 | -2.41 | 3.477 |
| 2C | True | 10.55 | 86.34 | -5.961 | 4.396 | True | 10.18 | 84.98 | -7.317 | 4.19 |
| 3C | True | 10.05 | 89.4 | -0.1044 | 3.945 | True | 11.59 | 88.33 | -1.168 | 4.176 |
| 4C | True | 10.63 | 94.4 | 5.5 | 7.723 | True | 11.47 | 95.09 | 6.195 | 6.672 |
| 5C | True | 11.02 | 90.18 | 3.783 | 10.2 | True | 11.52 | 83.71 | -2.693 | 7.108 |
| 6C | True | 10.87 | 70.01 | 9.814 | 6.055 | True | 12.32 | 69.52 | 9.318 | 6.44 |
| 7C | False | 45.12 |  |  |  | True | 11.23 | 72.63 | -3.769 | 16.11 |

## ePC-SAFT neutral check

The campaign-input correction also resolves the endpoint behavior in the neutral ePC-SAFT fugacity lane for the key 1C/3C/7C probe. This supports the conclusion that the previous trend was input-driven rather than specific to the ideal-Henry thermodynamic lane.

| case_id | success | runtime_s | capture_pct | capture_error_pct | temperature_rmse_K | epcsaft_cache_hits | epcsaft_cache_misses | epcsaft_direct_density_solve_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1C | True | 16.22 | 95.01 | -2.09 | 3.687 | 1990 | 1950 | 1.484 |
| 3C | True | 11.93 | 88.56 | -0.9362 | 3.682 | 1935 | 1399 | 1.041 |
| 7C | True | 19.78 | 72.44 | -3.962 | 14.35 | 1936 | 2004 | 2.12 |

The complete 1C--7C campaign gallery gives a stronger thermodynamic comparison than the three-case probe. Neutral ePC-SAFT lowers the liquid-temperature RMSE in six of seven cases relative to the ideal-Henry lane, reducing the campaign mean from 6.88 K to 6.13 K. The improvement is modest rather than transformative, but it is consistent evidence that replacing the idealized CO2 driving force with an ePC-SAFT fugacity basis improves the temperature-profile validation while preserving the same absorber model, solver settings, chemistry closure, and transport correlations. This is a useful manuscript result because it supports the ePC-SAFT lane as a rigorous and experimentally visible thermodynamic sensitivity, not just a software embellishment.

## Interpretation

- 1C changes from alpha 0.25 in the legacy CSV to alpha 0.15 in the campaign table; this raises predicted capture from 78.8% to 94.7%.
- 7C changes from alpha 0.25 and a high molar L/G of 6.86 in the legacy CSV to alpha 0.34 and L/G 4.63 in the campaign-derived table; this removes the enthalpy-lane timeout and gives 72.6% capture versus the 76.4% target.
- 3C remains close under both datasets, which explains why a single-case validation around 3C did not expose the input-table issue.
- Residual transport knobs such as constant mass-transfer scaling, heat-transfer scaling, gas-area exponents, or flooding-contacting reductions should not be used as the primary fix because the data correction addresses the dominant trend without per-case tuning.

## Temperature-profile gallery recommendation

The corrected campaign inputs support a much stronger validation figure than the original single-case comparison. The generated 1C--7C temperature overlays show consistent model behavior across the complete one-bed NCCC C-case campaign: all cases converge in the same solver configuration, capture errors remain bounded without case-specific tuning, and the measured liquid-temperature taps can be compared directly against the liquid and vapor model profiles. This should be added to the manuscript as validation evidence, either as a compact multi-panel gallery or as selected panels plus the complete gallery in supporting information.

The gallery should be presented carefully. Cases 1C, 2C, and 3C provide the cleanest coupled capture/temperature evidence. Cases 4C--6C remain useful because they show the same profile family and expose the residual capture-temperature tradeoff. Case 7C is especially valuable as a stress case: the corrected inputs remove the previous timeout and capture-bias failure, but the temperature RMSE remains high. That distinction is useful for the validation claim because it shows convergence and capture agreement are not being confused with perfect thermal prediction.

## Remaining trends after the input fix

The dominant capture trend is fixed by the campaign-input correction, but the temperature-profile residuals still show a physical pattern. The model increasingly overpredicts middle/upper-column liquid temperatures as lean loading and inlet CO2 increase, with 7C the clearest endpoint. Neutral ePC-SAFT reduces the temperature RMSE relative to ideal Henry for most cases, but it does not change the profile family enough to solve the 7C shape error. This points to a remaining heat-release/contacting/thermal-closure issue rather than a fugacity-only issue.

For other NCCC runs, this correction may help only if those runs share the same source-data conversion problem. The broader NCCC staged/intercooled tables should be audited against their original mass-flow, loading, and inlet-composition sources before solver or intercooler changes are tuned. If those broader inputs are already correctly converted, the C-case campaign correction will not automatically fix their remaining convergence or thermal-profile issues.

## Thermal-shape diagnosis

The remaining 7C mismatch is a thermal-shape problem, not a recurrence of the original capture-trend bug. Interpolating the solved liquid profile at the NCCC tap locations shows that the ideal-Henry lane is approximately 14 K hot on average for 7C, with the upper measured tap about 25 K hotter than the data. The neutral ePC-SAFT lane lowers that mean hot bias to about 12 K and lowers the RMSE from 16.11 K to 14.35 K, but the same profile family remains. This is useful because ePC-SAFT improves the thermodynamic driving-force evidence while also showing that a fugacity-only correction cannot close the highest-loading thermal shape.

A targeted 7C sensitivity screen supports the same conclusion. Changing the internal gas-liquid heat-transfer factor from 0.5 to 2.0 changes the 7C RMSE only from 16.22 K to 16.07 K and leaves capture essentially unchanged. Reducing the mass-transfer factor does lower the RMSE to 14.23 K at a 0.5 factor, but it also pushes the capture error to -24.7 percentage points, so it is not a defensible validation fix. The heat-transfer coefficient alone is therefore not the controlling missing correction, and mass-transfer retuning would trade away the already-improved capture validation.

The implementation points to three higher-value model checks:

1. Heat-source closure. The current `enthalpy_flux(...)` implementation transports CO2 and H2O enthalpy and adds interphase sensible heat transfer, but it does not expose a separately auditable heat-of-absorption or heat-of-speciation term. The next model check should write the local absorption heat, water condensation/evaporation heat, and interphase sensible heat terms separately in the dense profile exports.
2. Liquid heat-capacity closure. The temperature derivative diagnostic uses `f_dHl_dT(...)`, which is a custom derivative rather than a direct derivative of the same mixture enthalpy expression used to compute `Hl_T`. This should be replaced or audited against a finite-difference derivative of mixture enthalpy at the local composition before using temperature-state runs as validation evidence.
3. External heat loss/contacting closure. Literature on MEA absorber temperature bulges emphasizes that the bulge magnitude and location depend on L/G, heat of absorption, packing height, inlet CO2, and water vaporization/condensation. This model currently treats heat transfer as internal vapor-liquid exchange and does not include a calibrated wall/ambient heat-loss path. The 7C hot upper-column bias is consistent with a missing heat-removal or heat-source-distribution term, especially at high loading and inlet CO2.

The practical next step is to add a thermal-accounting profile table before changing solver strategy: `q_interphase`, `q_absorption`, `q_water_phase_change`, `q_wall_loss`, `dHmix_dT_fd`, and `dHmix_dT_model`. That table will make it possible to distinguish a wrong heat source from a wrong heat capacity, a missing heat sink, or a mass-transfer/contacting distribution error.

## Artifacts

- Legacy run: `analyses/nccc_validation/results/runs/c_case_trend_baseline_all7_now/benchmark_results.csv`
- Campaign run: `analyses/nccc_validation/results/runs/c_case_campaign_dataset_benchmark_all7/benchmark_results.csv`
- Campaign input dataset: `src/mea_absorption_column/data/C_cases_campaign_inputs.csv`
- Thermal-closure probe: `analyses/nccc_validation/results/runs/c_case_campaign_7c_thermal_closure_probe/`
- Temperature overlay metrics: `analyses/nccc_validation/results/final/tables/c_case_campaign_temperature_overlay_metrics.csv`
- Temperature overlay figures: `analyses/nccc_validation/results/final/figures/c_case_campaign_temperature_overlays/`
