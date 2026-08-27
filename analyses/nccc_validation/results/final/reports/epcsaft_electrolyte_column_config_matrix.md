# ePC-SAFT Electrolyte Column Configuration Matrix

> Historical pre-0.2 evidence only. This report is not a current validation result; use `epcsaft_v02_contribution_table.csv` and `epcsaft_v02_column_row.csv`.

These runs use the six-species MEA absorber state with the repo-vendored `MEA_CO2_H2O_ionic_fit` ePC-SAFT dataset.
The 3C C-case is intentionally small enough to keep the comparison reproducible while still running the full column solver.

- Dataset path: `src\mea_absorption_column\data\epcsaft_datasets\MEA_CO2_H2O_ionic_fit`
- Run root: `analyses/nccc_validation/results/runs/epcsaft_electrolyte_config_matrix`
- Successful column rows: 14/14
- Primary result table: `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_column_config_matrix.csv`
- Pure parameter table: `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_pure_parameters.csv`
- Binary interaction table: `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_binary_parameters.csv`

## Column Results

| config | success | capture_pct | capture_error_pct | runtime_s | temperature_rmse_K | boundary_residual_norm | epcsaft_cache_hits | epcsaft_cache_misses | diagnostic_phi_co2 | a_ion | a_born | diagnostic_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005_Cameretti_constant_DH_no_Born | True | 89.8245 | 0.324546 | 8.70264 | 3.97418 | 5.93688e-14 | 750 | 724 | 23.4295 | -0.0224925 | 0 |  |
| 2008_Held_constant_DH_no_Born | True | 89.8245 | 0.324546 | 9.41452 | 3.97418 | 5.93688e-14 | 750 | 724 | 23.4295 | -0.0224925 | 0 |  |
| 2014_Held_constant_DH_no_Born | True | 89.8245 | 0.324546 | 10.6791 | 3.97418 | 5.93688e-14 | 750 | 724 | 23.4295 | -0.0224925 | 0 |  |
| 2019_Bulow_linear_DH_no_Born | True | 89.8255 | 0.325516 | 10.6028 | 3.97424 | 6.91544e-14 | 750 | 724 | 21.7084 | -0.0336869 | 0 |  |
| 2020_Bulow_linear_base_Born | True | 89.8251 | 0.325142 | 14.2762 | 3.97421 | 1.22464e-13 | 750 | 724 | 22.29 | -0.0336869 | -5.76616 |  |
| 2025_Figiel_empirical_fitted_Born_SSM_DS | True | 89.8252 | 0.325189 | 8.87908 | 3.97422 | 6.69502e-14 | 750 | 724 | 22.9085 | -0.0336869 | -11.7366 |  |
| mode_relperm_combined_no_Born | True | 89.8254 | 0.325367 | 9.45381 | 3.97423 | 6.91544e-14 | 750 | 724 | 22.0006 | -0.0239428 | 0 |  |
| mode_relperm_linear_saltfraction_no_Born | True | 89.825 | 0.324996 | 7.68239 | 3.97421 | 5.67248e-14 | 750 | 724 | 22.8155 | -0.0219026 | 0 |  |
| mode_relperm_aqueous_organic_no_Born | True | 89.398 | -0.10196 | 16.724 | 3.9486 | 5.67248e-14 | 0 | 737 |  |  |  | SolutionError: fixed-state contribution probe failed |
| mode_sigma_radius_classic_Born | True | 89.8227 | 0.322702 | 7.62166 | 3.97406 | 5.66165e-14 | 751 | 723 | 26.6951 | -0.0336869 | -5.2474 |  |
| mode_fitted_Born_SSM_only | True | 89.8309 | 0.33088 | 8.94539 | 3.97451 | 6.69502e-14 | 752 | 722 | 2.27821 | -0.0336869 | -8.40262 |  |
| mode_fitted_Born_DS_only | True | 89.8227 | 0.322703 | 8.64574 | 3.97406 | 6.69502e-14 | 751 | 723 | 35.0222 | -0.0336869 | -12.1371 |  |
| mode_fitted_Born_SSM_DS_auto_mu | True | 89.8232 | 0.323153 | 9.71073 | 3.9741 | 6.91544e-14 | 751 | 723 | 33.1874 | -0.0336869 | -11.7366 |  |
| mode_DH_sigma_no_Born | True | 89.8272 | 0.3272 | 10.9229 | 3.97434 | 6.69502e-14 | 752 | 722 | 16.3331 | -0.0306399 | 0 |  |

## Pure Component Parameters

| component | m | s | e | e_assoc | vol_a | z | dielc | d_born | source_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CO2 | 2.079 | 2.7852 | 169.21 | 0 | 0 | 0 | 1.4122 | 0 | Neutral component parameters from the repo-vendored MEA ePC-SAFT ionic-fit dataset. |
| MEA | 3.0353 | 3.0435 | 277.174 | 2586.3 | 0.03747 | 0 | 32 | 0 | Neutral component parameters from the repo-vendored MEA ePC-SAFT ionic-fit dataset. |
| H2O | 1.2047 | sigma=2.7927+(10.11*exp(-0.01775*T)-1.417*exp(-0.01146*T)) | 353.95 | 2425.7 | 0.04509 | 0 | 78.09 | 0 | Neutral component parameters from the repo-vendored MEA ePC-SAFT ionic-fit dataset. |
| MEAH+ | 1 | 3.563 | 228.71 | 0 | 0 | 1 | 8 | 3.563 | MEA ionic species values from the repo-vendored ionic-fit dataset; d_born is treated as the fitted Born diameter for SSM/DS runs. |
| MEACOO- | 1 | 3.5605 | 533.11 | 0 | 0 | -1 | 8 | 3.5605 | MEA ionic species values from the repo-vendored ionic-fit dataset; d_born is treated as the fitted Born diameter for SSM/DS runs. |
| HCO3- | 1 | 2.9296 | 70 | 0 | 0 | -1 | 8 | 3 | Auxiliary carbonate/water ion placeholder from the repo-vendored ionic-fit dataset; d_born=3 A is used as a reasonable hydrated-ion-scale assumption. |
| CO3^2- | 1 | 3 | 300 | 0 | 0 | -2 | 8 | 3 | Auxiliary carbonate/water ion placeholder from the repo-vendored ionic-fit dataset; d_born=3 A is used as a reasonable hydrated-ion-scale assumption. |
| H3O+ | 1 | 3 | 300 | 0 | 0 | 1 | 8 | 3 | Auxiliary carbonate/water ion placeholder from the repo-vendored ionic-fit dataset; d_born=3 A is used as a reasonable hydrated-ion-scale assumption. |
| OH- | 1 | 3 | 300 | 0 | 0 | -1 | 8 | 3 | Auxiliary carbonate/water ion placeholder from the repo-vendored ionic-fit dataset; d_born=3 A is used as a reasonable hydrated-ion-scale assumption. |

## Nonzero Binary Interaction Parameters

| parameter | component_i | component_j | value | source_note |
| --- | --- | --- | --- | --- |
| k_ij | MEA | H2O | -0.052 | Repo-vendored MEA ePC-SAFT ionic-fit binary interaction matrix; upper triangle reported once. |

## Notes

- The dated rows reproduce the ePC-SAFT package user-option patterns while holding the MEA component dataset fixed.
- The SSM+DS rows use the `MEA_CO2_H2O_ionic_fit` ion Born diameters. For auxiliary carbonate/water ions, the vendored dataset uses 3 A hydrated-ion-scale assumptions.
- The absorber itself is currently the six-species chemistry state (`CO2`, `MEA`, `H2O`, `MEAH+`, `MEACOO-`, `HCO3-`); the parameter tables still report the full nine-species dataset so the unused ionic species are auditable.
