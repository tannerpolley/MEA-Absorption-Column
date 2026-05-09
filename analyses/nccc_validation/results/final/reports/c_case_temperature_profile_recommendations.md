# C-Case Temperature Profile Recommendation

Conservative inclusion criteria used for profile-based figure selection:
- |capture_error_pct| <= 10
- tap-based liquid-temperature RMSE <= 9 K

Case metrics (best model by tap RMSE):

| case_id   | best_thermo_model   |   capture_error_pct |   tap_rmse_K |   tap_mae_K |   temperature_rmse_K |
|:----------|:--------------------|--------------------:|-------------:|------------:|---------------------:|
| 1C        | ideal_henry         |          -18.3171   |      9.4983  |     8.70478 |              5.14805 |
| 2C        | epcsaft_neutral     |           -5.7647   |     10.1486  |     8.87505 |              3.40379 |
| 3C        | epcsaft_neutral     |            0.198344 |      9.31583 |     8.39415 |              3.96885 |
| 4C        | epcsaft_neutral     |            5.65103  |     11.0123  |     8.85475 |              6.84559 |
| 5C        | epcsaft_neutral     |            3.88148  |     12.2931  |    10.1887  |              9.00242 |
| 6C        | epcsaft_neutral     |            9.97035  |      8.60296 |     7.17415 |              5.10028 |
| 7C        | ideal_henry         |           22.073    |      8.31598 |     8.21104 |              7.43757 |

## Recommended additional cases beyond 3C
- 6C

## Caveats
- 4C, 5C, and 7C show materially larger liquid-tap mismatch and/or capture bias than 3C/6C under conservative thresholds.
- 3C remains the anchor case in existing paper-facing figure flow.
- 7C has high runtime in the `ideal_henry` row and large capture-metric miss in both lanes.