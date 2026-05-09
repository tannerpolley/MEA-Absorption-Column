# Thermal-Closure Improvement Screen

## Scope

This analysis tests whether the remaining one-bed NCCC C-case temperature-profile errors are caused by the thermodynamic driving force, internal gas-liquid heat transfer, or a missing thermal-closure term. The model code now exports a `thermal_accounting.csv` profile with local heat and enthalpy-balance diagnostics for every requested dense profile run.

## Key result

A global wall heat-loss screen with a single coefficient of 75 W m^-1 K^-1 improves the liquid-temperature RMSE while preserving capture accuracy and using the same coefficient for all seven C cases. This is not a final calibrated plant heat-loss model, but it is strong evidence that the remaining profile error is a thermal-closure problem rather than a solver artifact or a fugacity-only problem.

| thermo_model | cases | wall_heat_loss_coeff_W_m_K | capture_mae_baseline_pct | capture_mae_wall_loss_pct | temperature_rmse_mean_baseline_K | temperature_rmse_mean_wall_loss_K | temperature_rmse_mean_improvement_K | temperature_rmse_max_baseline_K | temperature_rmse_max_wall_loss_K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epcsaft_neutral | 7 | 75 | 4.655 | 4.644 | 6.13 | 5.683 | 0.4476 | 14.35 | 13.35 |
| ideal_henry | 7 | 75 | 4.696 | 4.683 | 6.882 | 6.26 | 0.622 | 16.11 | 14.99 |

## Thermal accounting findings

The accounting profiles show that the current liquid heat-capacity derivative diagnostic, `f_dHl_dT(...)`, is about three times larger than a finite-difference derivative of the same mixture enthalpy function. That derivative does not control the default enthalpy-state BVP solution, but it explains why direct temperature-state experiments are sensitive and should not replace the enthalpy-state balance until the derivative closure is repaired.

Internal gas-liquid heat-transfer scaling was also tested separately and did not materially improve 7C. In contrast, a countercurrent-coordinate wall heat-removal term improves the C-case temperature profiles with little capture penalty. That pattern supports adding a limited wall-loss/thermal-closure discussion to the manuscript, while keeping the claim conservative.

## Case-level comparison

| case_id | thermo_model | capture_error_pct_baseline | capture_error_pct_wall_loss | temperature_rmse_K_baseline | temperature_rmse_K_wall_loss | temperature_rmse_improvement_K |
| --- | --- | --- | --- | --- | --- | --- |
| 1C | epcsaft_neutral | -2.09 | -2.078 | 3.687 | 3.771 | -0.08455 |
| 2C | epcsaft_neutral | -7.105 | -7.097 | 3.242 | 2.912 | 0.3299 |
| 3C | epcsaft_neutral | -0.9362 | -0.8777 | 3.682 | 3.47 | 0.2125 |
| 4C | epcsaft_neutral | 6.399 | 6.453 | 6.179 | 5.661 | 0.5178 |
| 5C | epcsaft_neutral | -2.589 | -2.583 | 6.349 | 5.743 | 0.6059 |
| 6C | epcsaft_neutral | 9.506 | 9.452 | 5.419 | 4.868 | 0.5516 |
| 7C | epcsaft_neutral | -3.962 | -3.968 | 14.35 | 13.35 | 1 |
| 1C | ideal_henry | -2.41 | -2.41 | 3.477 | 3.442 | 0.03579 |
| 2C | ideal_henry | -7.317 | -7.336 | 4.19 | 3.732 | 0.4578 |
| 3C | ideal_henry | -1.168 | -1.123 | 4.176 | 3.723 | 0.4528 |
| 4C | ideal_henry | 6.195 | 6.175 | 6.672 | 5.916 | 0.7558 |
| 5C | ideal_henry | -2.693 | -2.703 | 7.108 | 6.313 | 0.7946 |
| 6C | ideal_henry | 9.318 | 9.244 | 6.44 | 5.706 | 0.7338 |
| 7C | ideal_henry | -3.769 | -3.793 | 16.11 | 14.99 | 1.123 |

## Artifacts

- Comparison table: `analyses/nccc_validation/results/final/tables/thermal_closure_wall_loss_comparison.csv`
- Thermal diagnostics table: `analyses/nccc_validation/results/final/tables/thermal_accounting_diagnostics.csv`
- Summary table: `analyses/nccc_validation/results/final/tables/thermal_closure_wall_loss_summary.csv`
- Figure: `analyses/nccc_validation/results/final/figures/thermal_closure/thermal_closure_rmse_epcsaft_neutral.png`
- Figure: `analyses/nccc_validation/results/final/figures/thermal_closure/thermal_closure_rmse_epcsaft_neutral.svg`
- Figure: `analyses/nccc_validation/results/final/figures/thermal_closure/thermal_closure_rmse_ideal_henry.png`
- Figure: `analyses/nccc_validation/results/final/figures/thermal_closure/thermal_closure_rmse_ideal_henry.svg`
- Figure: `analyses/nccc_validation/results/final/figures/thermal_closure/thermal_closure_mean_rmse.png`
- Figure: `analyses/nccc_validation/results/final/figures/thermal_closure/thermal_closure_mean_rmse.svg`
