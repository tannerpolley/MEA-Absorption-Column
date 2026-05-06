# NCCC Validation Artifact Summary

This report summarizes the curated benchmark artifacts used for manuscript revision. The full run dumps and exploratory solver probes remain outside the paper-facing evidence set under `analyses/nccc_validation/results/runs/`.

## Accepted Evidence

- One-bed C-case benchmark: 7 NCCC C cases were retained for both `ideal_henry` and `epcsaft_neutral`, giving 14 accepted rows in `results/final/tables/verified_c_case_thermo_benchmark.csv`.
- One-bed capture-error MAE is 9.36 percentage points for `ideal_henry` and 9.34 percentage points for `epcsaft_neutral`. Mean temperature RMSE is 6.44 K for `ideal_henry` and 8.16 K for `epcsaft_neutral`.
- Primary staged/intercooled Henry benchmark: 19 accepted K-case rows were retained in `results/final/tables/verified_staged_kcase_benchmark.csv`.
- Primary staged/intercooled Henry capture-error MAE is 2.81 percentage points; the maximum accepted absolute capture error is 6.40 percentage points.
- The accepted staged Henry set includes one-, two-, and three-bed cases, with intercooled cases represented by the explicit staged-bed solver and liquid-temperature reset assumption.

## Thermodynamic Sensitivity Evidence

The staged ePC-SAFT smoke table retains 19 selected rows in `results/final/tables/staged_epcsaft_smoke.csv`. Fourteen rows are accepted under the current gates, with a successful-row capture-error MAE of 3.06 percentage points. The unresolved staged ePC-SAFT rows are K2, K4, K9, K19, and K23; these remain diagnostic rows rather than primary validation evidence.

The ePC-SAFT comparison should therefore be described as a controlled neutral-fugacity sensitivity lane, not as a complete electrolyte-reactive absorber model. The model changes the CO2 fugacity driving force while preserving chemistry, transport, enhancement factor, hydraulics, and balances from the baseline model.

## Clean Profile Gallery

The clean profile gallery is indexed by `results/final/tables/clean_temperature_profile_index.csv`. It currently includes accepted 3C profiles for both thermodynamic lanes and a retained 7C ePC-SAFT diagnostic profile with an explicit caveat.

## Diagnostic Tables

`kcase_sensitivity_recoveries.csv` and `kcase_unresolved_diagnostics.csv` are retained for auditability. They should not be mixed into the primary accepted validation table unless their caveats are preserved in the manuscript text.
