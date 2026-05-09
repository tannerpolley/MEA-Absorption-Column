# C-Case Temperature Profile Recommendation

Conservative inclusion criteria used for profile-based figure selection:
- |capture_error_pct| <= 10
- tap-based liquid-temperature RMSE <= 9 K

Case metrics (best model by tap RMSE):

| case_id   | best_thermo_model   |   capture_error_pct |   tap_rmse_K |   tap_mae_K |   temperature_rmse_K |
|:----------|:--------------------|--------------------:|-------------:|------------:|---------------------:|
| 1C        | ideal_henry         |            -2.40997 |      3.47732 |     2.60065 |              3.47732 |
| 2C        | ideal_henry         |            -7.31668 |      4.1897  |     3.27695 |              4.1897  |
| 3C        | ideal_henry         |            -1.16784 |      4.17567 |     3.60783 |              4.17567 |
| 4C        | ideal_henry         |             6.19464 |      6.67206 |     6.25422 |              6.67206 |
| 5C        | ideal_henry         |            -2.69323 |      7.10759 |     6.85457 |              7.10759 |
| 6C        | ideal_henry         |             9.31761 |      6.43978 |     6.23133 |              6.43978 |
| 7C        | ideal_henry         |            -3.76908 |     16.1143  |    14.1038  |             16.1143  |

## Recommended additional cases beyond 3C
- 1C, 2C, 4C, 5C, 6C

## Caveats
- The campaign overlay figure remains the paper-facing profile summary for the 1C--7C set.
- Case 7C remains the hardest thermal-shape case, while the capture errors stay within the campaign validation gate.