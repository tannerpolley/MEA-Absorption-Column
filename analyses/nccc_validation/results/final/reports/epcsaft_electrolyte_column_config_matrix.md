# ePC-SAFT Electrolyte Column Configuration Matrix

These runs use the six-species MEA absorber state with the repo-vendored `MEA_CO2_H2O_ionic_fit` ePC-SAFT dataset.
The 3C C-case is intentionally small enough to keep the comparison reproducible while still running the full column solver.

- Dataset path: `src/mea_absorption_column/data/epcsaft_datasets/MEA_CO2_H2O_ionic_fit`
- Run root: `analyses/nccc_validation/results/runs/epcsaft_electrolyte_config_matrix`
- Successful column rows: 14/14
- Primary result table: `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_column_config_matrix.csv`
- Pure parameter table: `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_pure_parameters.csv`
- Binary interaction table: `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_binary_parameters.csv`

## Column Results

| config | success | capture_pct | capture_error_pct | runtime_s | temperature_rmse_K | boundary_residual_norm | epcsaft_cache_hits | epcsaft_cache_misses | diagnostic_phi_co2 | a_ion | a_born | diagnostic_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2005_Cameretti_constant_DH_no_Born | True | 89.2577 | -0.242343 | 12.4875 | 4.85203 | 5.65551e-14 | 767 | 953 | 2307.6 | -0.0285178 | 0 |  |
| 2008_Held_constant_DH_no_Born | True | 89.2577 | -0.242343 | 11.9702 | 4.85203 | 5.65551e-14 | 767 | 953 | 2307.6 | -0.0285178 | 0 |  |
| 2014_Held_constant_DH_no_Born | True | 89.2577 | -0.242343 | 10.3867 | 4.85203 | 5.65551e-14 | 767 | 953 | 2307.6 | -0.0285178 | 0 |  |
| 2019_Bulow_linear_DH_no_Born | True | 89.6429 | 0.14294 | 10.4314 | 4.89175 | 7.79644e-14 | 769 | 951 | 2228.37 | -0.0277435 | 0 |  |
| 2020_Bulow_linear_base_Born | True | 88.7468 | -0.753202 | 11.1501 | 4.8022 | 5.93688e-14 | 766 | 954 | 2530.28 | -0.0277435 | -9.15404 |  |
| source_backed_linear_classic_Born | True | 88.86 | -0.640046 | 10.8503 | 4.81357 | 5.67248e-14 | 768 | 952 | 2491.99 | -0.0277435 | -8.05555 |  |
| mode_relperm_combined_no_Born | True | 89.9593 | 0.459282 | 10.1255 | 4.92668 | 5.92067e-14 | 773 | 947 | 2124.68 | -0.0304244 | 0 |  |
| mode_relperm_linear_saltfraction_no_Born | True | 89.3942 | -0.105797 | 10.0177 | 4.86464 | 5.38375e-14 | 767 | 953 | 2311.19 | -0.0277435 | 0 |  |
| mode_relperm_aqueous_organic_no_Born | True | 89.122 | -0.377969 | 22.3091 | 4.88663 | 5.66165e-14 | 0 | 860 |  |  |  | SolutionError: fixed-state contribution probe failed |
| mode_sigma_radius_classic_Born | True | 87.7132 | -1.78675 | 7.8804 | 4.68688 | 5.67248e-14 | 764 | 956 | 2862.89 | -0.0433335 | -8.00726 |  |
| mode_fitted_Born_SSM_only | True | 95.1427 | 5.64266 | 10.0106 | 5.35943 | 5.66165e-14 | 784 | 936 | 472.276 | -0.0433335 | -5.54067 |  |
| mode_fitted_Born_DS_only | True | 87.7144 | -1.7856 | 7.53307 | 4.68703 | 5.66165e-14 | 761 | 959 | 2862.44 | -0.0433335 | -8.00319 |  |
| mode_fitted_Born_SSM_DS_auto_mu | True | 88.0675 | -1.43249 | 8.90261 | 4.74171 | 5.67248e-14 | 760 | 960 | 2762.65 | -0.0433335 | -7.73907 |  |
| mode_DH_sigma_no_Born | True | 91.2656 | 1.76557 | 9.94109 | 5.05102 | 5.92067e-14 | 779 | 941 | 1710.9 | -0.0259089 | 0 |  |

## Pure Component Parameters

| component | m | s | e | e_assoc | vol_a | z | dielc | source_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CO2 | 2.079 | 2.7852 | 169.21 | 0 | 0 | 0 | 1.4122 | Neutral component parameters from the repo-vendored MEA ePC-SAFT ionic-fit dataset. |
| MEA | 3.0353 | 3.0435 | 277.174 | 2586.3 | 0.03747 | 0 | 32 | Neutral component parameters from the repo-vendored MEA ePC-SAFT ionic-fit dataset. |
| H2O | 1.2047 | sigma=2.7927+(10.11*exp(-0.01775*T)-1.417*exp(-0.01146*T)) | 353.95 | 2425.7 | 0.04509 | 0 | 78.09 | Neutral component parameters from the repo-vendored MEA ePC-SAFT ionic-fit dataset. |
| MEAH+ | 1 | 3.563 | 228.71 | 0 | 0 | 1 | 8 | MEA ionic species analog/estimated values from the repo-vendored ionic-fit dataset; charged-species segment number is fixed at m=1. |
| MEACOO- | 1 | 3.5605 | 533.11 | 0 | 0 | -1 | 8 | MEA ionic species analog/estimated values from the repo-vendored ionic-fit dataset; charged-species segment number is fixed at m=1. |
| HCO3- | 1 | 2.9296 | 70 | 0 | 0 | -1 | 8 | Repo-vendored bicarbonate parameter value; charged-species segment number is fixed at m=1. |
| CO3^2- | 1 | 2.4422 | 249.26 | 0 | 0 | -2 | 8 | Source-backed carbonate/water-ion parameter value in the repo-vendored ionic-fit dataset; charged-species segment number is fixed at m=1. |
| H3O+ | 1 | 3.4654 | 500 | 0 | 0 | 1 | 8 | Source-backed carbonate/water-ion parameter value in the repo-vendored ionic-fit dataset; charged-species segment number is fixed at m=1. |
| OH- | 1 | 2.0177 | 650 | 0 | 0 | -1 | 8 | Source-backed carbonate/water-ion parameter value in the repo-vendored ionic-fit dataset; charged-species segment number is fixed at m=1. |

## Nonzero Binary Interaction Parameters

| parameter | component_i | component_j | value | source_note |
| --- | --- | --- | --- | --- |
| k_ij | MEA | H2O | -0.052 | Repo-vendored MEA ePC-SAFT ionic-fit binary interaction matrix; upper triangle reported once. |

## Notes

- The selected row uses linear relative-permittivity mixing and the classic Born radius mode while holding the MEA component dataset fixed.
- Fitted Born-diameter rows are retained only as diagnostic option-coverage rows. They are not the selected paper configuration.
- The absorber itself is currently the six-species chemistry state (`CO2`, `MEA`, `H2O`, `MEAH+`, `MEACOO-`, `HCO3-`); the parameter tables still report the full nine-species dataset so the unused ionic species are auditable.
