# ePC-SAFT Electrolyte Option Matrix

This diagnostic exercises the MEA six-species ePC-SAFT dataset at a Case-3C-like liquid state.
It separates the neutral fugacity-coefficient path from electrolyte option paths that activate Debye-Huckel ion and Born terms.

- Dataset: `src\mea_absorption_column\data\epcsaft_datasets\MEA_CO2_H2O_draft`
- Successful configurations: 8/9
- Expected unsupported configurations: 1
- Unexpected outcomes: 0

## Matrix

| config | success | expected_success | a_ion | a_born | lnphi_co2_ion | lnphi_co2_born | message |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral_reference | True | True | 0 | 0 | 0 | 0 |  |
| ionic_dataset_default | True | True | -0.0336869 | -11.7366 | -0.0299364 | 0.0538096 |  |
| ionic_dh_only_born_disabled | True | True | -0.0336869 | 0 | -0.0299364 | 0 |  |
| ionic_classic_born_sigma_radius | True | True | -0.0336869 | -5.2474 | -0.0299364 | 0.206781 |  |
| ionic_fitted_born_ssm_only | True | True | -0.0336869 | -8.40262 | -0.0299364 | -2.25431 |  |
| ionic_fitted_born_ds_only | True | True | -0.0336869 | -12.1371 | -0.0299364 | 0.478281 |  |
| ionic_fitted_born_ssm_ds_numerical | True | True | -0.0336869 | -11.7366 | -0.0299364 | 0.0538096 |  |
| ionic_fitted_born_ssm_ds_auto | True | True | -0.0336869 | -11.7366 | -0.0299364 | 0.424471 |  |
| unsupported_fitted_born_without_ssm_ds | False | False |  |  |  |  | ValueError: d_Born_mode="fitted_param" requires SSM/DS Born path (include_born_model=true and SSM or DS true). |

## Interpretation

The neutral reference keeps both ion and Born residual Helmholtz contributions at zero.
The ionic dataset path activates the ion term and, when Born is enabled with a supported radius model, activates the Born contribution as well.
The fitted Born-diameter option is intentionally guarded: it requires the SSM or DS Born path, so the fitted-without-SSM/DS row is an expected clear failure rather than a silent neutral fallback.
