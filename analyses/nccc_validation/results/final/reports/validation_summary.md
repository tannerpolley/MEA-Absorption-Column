# NCCC Validation Artifact Summary

This report summarizes the curated benchmark artifacts used for the main-branch manuscript revision. Full run dumps and exploratory solver probes remain outside the paper-facing evidence set under `analyses/nccc_validation/results/runs/`.

## Accepted Evidence

- One-bed C-case benchmark: 7 NCCC C cases were retained for both `ideal_henry` and `epcsaft_neutral`, giving 14 accepted rows in `results/final/tables/verified_c_case_thermo_benchmark.csv`.
- One-bed capture-error MAE is 9.36 percentage points for `ideal_henry` and 9.37 percentage points for `epcsaft_neutral`. Mean temperature RMSE is 6.44 K for `ideal_henry` and 8.17 K for `epcsaft_neutral`.
- `validation_evidence_registry.csv` and `primary_validation_gate.csv` separate primary validation from diagnostic method-comparison evidence and require the accepted one-bed C rows to use common benchmark settings.

## Thermodynamic Sensitivity Evidence

The ePC-SAFT comparison is described as a controlled CO2 fugacity-driving-force sensitivity lane. The model changes the CO2 fugacity coefficient while preserving chemistry, transport, enhancement factor, hydraulics, and balances from the baseline model.

## Accuracy-Credibility Screens

`calibration_holdout_metrics.csv`, `calibration_holdout_predictions.csv`, `error_regime_capture_data.csv`, and `uncertainty_band_capture.csv` document a low-order residual-correction and uncertainty-screen workflow. These artifacts are not used to claim broad predictive accuracy. They make the current evidence boundary visible by showing that the training subset can be corrected more easily than the small holdout subset.

## Clean Profile Gallery

The clean profile gallery is indexed by `results/final/tables/clean_temperature_profile_index.csv`. It currently includes accepted 3C profiles for both thermodynamic lanes and a retained 7C ePC-SAFT diagnostic profile with an explicit caveat.

## Solver-Method Contrast

`method_case_contrast.csv` retains a favorable SRP-style one-bed method comparison and an NCCC 3C thermal-pinch comparison. These rows explain where shooting, finite difference, and SciPy BVP methods are useful, but they are diagnostic method evidence rather than primary validation evidence.
