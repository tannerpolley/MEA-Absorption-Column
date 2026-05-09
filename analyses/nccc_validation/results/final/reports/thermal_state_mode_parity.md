# Thermal State Mode Parity

This report compares the legacy enthalpy-state BVP with the direct temperature-state BVP.
Temperature-state runs are warm-started from the converged enthalpy profile and then solved
with temperature as the thermal state variable.

Accepted parity rows: 6 of 8.

| Case source | Case | Capture delta, pct-pt | Temperature RMSE delta, K | Parity |
| --- | --- | ---: | ---: | --- |
| C_cases_data | 1C | 0.0077 | 0.0257 | True |
| C_cases_data | 2C | 0.0346 | -0.0110 | True |
| C_cases_data | 3C | -0.0114 | 0.0395 | True |
| C_cases_data | 4C | -0.0117 | 0.0368 | True |
| C_cases_data | 5C | 0.0008 | -0.0132 | True |
| C_cases_data | 6C | -0.0211 | -0.0503 | True |
| C_cases_data | 7C | 0.0044 | -0.1033 | False |
| SRP_method_cases | SRP-LG7 |  |  | False |

Rows that fail parity should keep the enthalpy formulation as the validation reference
until the direct temperature equations and initialization are improved for that regime.
