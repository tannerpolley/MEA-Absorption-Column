# ePC-SAFT Electrolyte Option Matrix

This diagnostic exercises the MEA six-species ePC-SAFT dataset at a Case-3C-like liquid state.
It separates the neutral fugacity-coefficient path from electrolyte option paths that activate Debye-Huckel ion and Born terms.

- Dataset: `C:\Users\Tanner\Documents\git\MEA-Absorption-Column\src\mea_absorption_column\data\epcsaft_datasets\MEA_CO2_H2O_ionic_fit`
- Successful configurations: 8/9
- Expected unsupported configurations: 1
- Unexpected outcomes: 0

## Matrix

| config | success | expected_success | a_ion | a_born | lnphi_co2_ion | lnphi_co2_born | message |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral_reference | True | True | 0 | 0 | 0 | 0 |  |
| ionic_dataset_default | True | True | -0.0277435 | -8.05555 | 0.0190556 | 0.111814 |  |
| ionic_dh_only_born_disabled | True | True | -0.0277435 | 0 | 0.0190556 | 0 |  |
| ionic_classic_born_sigma_radius | True | True | -0.0277435 | -8.05555 | 0.0190556 | 0.111814 |  |
| ionic_fitted_born_ssm_only | True | True | -0.0433335 | -5.54067 | -0.0279897 | -1.48649 |  |
| ionic_fitted_born_ds_only | True | True | -0.0433335 | -8.00319 | -0.0279897 | 0.315378 |  |
| ionic_fitted_born_ssm_ds_numerical | True | True | -0.0433335 | -7.73907 | -0.0279897 | 0.0354819 |  |
| ionic_fitted_born_ssm_ds_auto | True | True | -0.0433335 | -7.73907 | -0.0279897 | 0.279895 |  |
| unsupported_fitted_born_without_ssm_ds | False | False |  |  |  |  | ValueError: d_Born_mode="fitted_param" requires SSM/DS Born path (include_born_model=true and SSM or DS true). |

## Interpretation

The neutral reference keeps both ion and Born residual Helmholtz contributions at zero.
The ionic dataset path activates the ion term and, when Born is enabled with a supported radius model, activates the Born contribution as well.
The dataset-default path uses linear relative-permittivity mixing and the classic Born radius mode. The fitted Born-diameter diagnostic rows are retained only as option-coverage checks; the unsupported fitted-without-SSM/DS row is an expected clear failure rather than a silent neutral fallback.
