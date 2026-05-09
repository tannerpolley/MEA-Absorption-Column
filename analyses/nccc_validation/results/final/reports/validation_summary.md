# NCCC Validation Artifact Summary

This report summarizes the curated benchmark artifacts used for the main-branch manuscript revision. Full run dumps and exploratory solver probes remain outside the paper-facing evidence set under `analyses/nccc_validation/results/runs/`.

## Accepted Evidence

- One-bed C-case benchmark: 7 NCCC C cases were retained for both `ideal_henry` and `epcsaft_ionic`, giving 14 accepted campaign-input rows in `results/final/tables/c_case_campaign_temperature_overlay_metrics.csv`.
- One-bed capture-error MAE is 4.70 percentage points for `ideal_henry` and 4.54 percentage points for `epcsaft_ionic`. Mean temperature RMSE is 6.88 K for `ideal_henry` and 6.91 K for `epcsaft_ionic`.
- `validation_evidence_registry.csv` and `primary_validation_gate.csv` separate primary validation from diagnostic method-comparison evidence and require the accepted one-bed C rows to use common benchmark settings.

## Thermodynamic Sensitivity Evidence

The ePC-SAFT comparison is described as a controlled CO2 fugacity-driving-force sensitivity lane. The selected paper-facing configuration is `2025_Figiel_empirical_fitted_Born_SSM_DS`, which uses the repo-vendored `MEA_CO2_H2O_ionic_fit` dataset with Debye-Huckel, fitted Born diameters, SSM, DS, empirical dielectric mixing, and numerical `mu_born` derivatives. The absorber still preserves the same chemistry, transport, enhancement factor, hydraulics, and balances as the baseline model.

## Accuracy-Credibility Screens

`calibration_holdout_metrics.csv`, `calibration_holdout_predictions.csv`, `error_regime_capture_data.csv`, and `uncertainty_band_capture.csv` document a low-order residual-correction and uncertainty-screen workflow. These artifacts are not used to claim broad predictive accuracy. They make the current evidence boundary visible by showing that the training subset can be corrected more easily than the small holdout subset.

## Clean Profile Gallery

The clean profile gallery is indexed by `results/final/tables/clean_temperature_profile_index.csv`. It points to the regenerated 1C--7C campaign-input temperature overlays for both thermodynamic lanes.

## Solver-Method Contrast

`method_case_contrast.csv` retains a favorable SRP-style one-bed method comparison and an NCCC 3C thermal-pinch comparison. These rows explain where shooting, finite difference, and collocation BVP methods are useful, but they are diagnostic method evidence rather than primary validation evidence.
