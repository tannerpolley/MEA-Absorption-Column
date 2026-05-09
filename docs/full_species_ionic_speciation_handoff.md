# Full Species Ionic ePC-SAFT Speciation Handoff

Branch: `codex/full-species-ionic`

Date: 2026-05-09

## What changed

- Added nine-species reactive ePC-SAFT chemical-equilibrium modes:
  - `epcsaft_reactive_nine_activity`
  - `epcsaft_reactive_nine_activity_converted`
  - `epcsaft_reactive_nine_activity_rebased`
  - `epcsaft_full_species_activity`
  - `epcsaft_full_species_activity_converted`
  - `epcsaft_full_species_activity_rebased`
- The nine-species state order is:
  - `CO2, MEA, H2O, MEAH+, MEACOO-, HCO3-, CO3^2-, H3O+, OH-`
- The nine-species model uses the MEA-Thermodynamics reaction set:
  - `H3O+ + OH- - 2 H2O`
  - `H3O+ + HCO3- - CO2 - 2 H2O`
  - `H3O+ + CO3^2- - HCO3- - H2O`
  - `MEA + HCO3- - MEACOO- - H2O`
  - `H3O+ + MEA - MEAH+ - H2O`
- Ionic ePC-SAFT fugacity now uses all nine liquid species when a nine-species chemistry vector is available.
- Existing column transport/enhancement/profile code keeps the six-species compatibility view it already expects, so the BVP state was not expanded.

## Source Contract

The reaction constants and balances were taken from the local MEA-Thermodynamics ionic workflow:

- `C:\Users\Tanner\Documents\git\MEA-Thermodynamics\src\MEA\epcsaft_runtime.py`
- `C:\Users\Tanner\Documents\git\MEA-Thermodynamics\src\MEA\epcsaft_ionic\model.py`
- `C:\Users\Tanner\Documents\git\MEA-Thermodynamics\docs\ePC-SAFT\mea-ionic-regression-completion-handoff.md`

Equilibrium constants use:

```text
ln K = a + b/T + c*ln(T) + d*T
```

with `T` in K. The local ePC-SAFT native solver also appends charge closure from the species charge vector, so charge is not duplicated as a material balance.

## Proof Run

All run commands used BLAS thread pinning:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MEA_EPCSAFT_CHEMISTRY_CACHE_X_DIGITS='4'
$env:MEA_EPCSAFT_CHEMISTRY_CACHE_T_DIGITS='1'
$env:MEA_EPCSAFT_CHEMISTRY_CACHE_P_ROUND_PA='100'
$env:MEA_EPCSAFT_REACTIVE_MAX_ITERATIONS='160'
```

Command:

```powershell
.\.venv\Scripts\python.exe -m mea_absorption_column.benchmark `
  --methods scipy-bvp `
  --thermo-models epcsaft_reactive_nine_activity_rebased `
  --c-case-ids 3C `
  --nccc-case-limit 0 `
  --srp-case-limit 0 `
  --staged-beds false `
  --mesh-points 7 `
  --tol 10 `
  --bc-tol 0.5 `
  --max-nodes 80 `
  --subprocess-timeout-s 900 `
  --output-dir analyses\nccc_validation\results\runs\full_species_ionic_rebased_probe
```

| Mode | Case | Success | Runtime s | CO2 capture % | Capture error pct-pt | Temp RMSE K | Boundary residual | Invalid states | Domain guards | Chemistry cache hit/miss | Chemistry solve s | Max mass residual | Max reaction residual | Max charge residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `epcsaft_reactive_nine_activity_rebased` | 3C | true | 212.809 | 89.855 | 0.355 | 4.316 | 6.70e-14 | 0 | none | 122/168 | 160.191 | 9.64e-09 | 9.97e-09 | 4.71e-14 |

## All Legacy C-Case Sweep

After the single-case proof, the same nine-species activity-rebased model was run for every legacy C case in `src/mea_absorption_column/data/C_cases_data.csv`.

Command:

```powershell
.\.venv\Scripts\python.exe -m mea_absorption_column.benchmark `
  --methods scipy-bvp `
  --thermo-models epcsaft_reactive_nine_activity_rebased `
  --nccc-case-limit 0 `
  --srp-case-limit 0 `
  --staged-beds false `
  --mesh-points 7 `
  --tol 10 `
  --bc-tol 0.5 `
  --max-nodes 80 `
  --subprocess-timeout-s 900 `
  --output-dir analyses\nccc_validation\results\runs\full_species_ionic_all_c_cases
```

Result CSV:

```text
analyses\nccc_validation\results\runs\full_species_ionic_all_c_cases\benchmark_results.csv
```

| Case | Success | Runtime s | CO2 capture % | Capture error pct-pt | Temp RMSE K | Boundary residual | Invalid states | Domain guards | Chem cache hits | Chem cache misses | Chem solve s | Max mass residual | Max reaction residual | Max charge residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1C | true | 336.655 | 79.258 | -17.842 | 4.903 | 6.46e-14 | 0 | 0 | 321 | 255 | 256.648 | 9.87e-09 | 9.95e-09 | 1.85e-13 |
| 2C | true | 289.707 | 86.651 | -5.649 | 4.749 | 2.52e-13 | 0 | 0 | 125 | 178 | 218.627 | 8.50e-09 | 9.89e-09 | 5.12e-14 |
| 3C | true | 281.664 | 89.855 | 0.355 | 4.316 | 6.70e-14 | 0 | 0 | 122 | 168 | 211.027 | 9.64e-09 | 9.97e-09 | 4.71e-14 |
| 4C | true | 308.746 | 94.678 | 5.778 | 7.665 | 1.48e-13 | 0 | 0 | 215 | 179 | 232.915 | 9.68e-09 | 9.93e-09 | 3.37e-13 |
| 5C | true | 444.874 | 90.525 | 4.125 | 10.335 | 2.00e-13 | 0 | 0 | 235 | 250 | 339.182 | 9.67e-09 | 1.00e-08 | 7.62e-14 |
| 6C | true | 420.357 | 70.549 | 10.349 | 6.102 | 9.48e-14 | 0 | 0 | 226 | 220 | 321.668 | 8.19e-09 | 9.97e-09 | 1.16e-13 |
| 7C | true | 380.387 | 98.515 | 22.115 | 22.310 | 1.88e-13 | 0 | 0 | 307 | 204 | 287.236 | 9.85e-09 | 9.93e-09 | 1.86e-13 |

Sweep summary:

- Success: 7 of 7 cases.
- Total benchmark runtime: 2462.389 s.
- Mean runtime: 351.770 s; range: 281.664-444.874 s.
- Mean absolute CO2 capture error: 9.459 pct-pt; maximum absolute error: 22.115 pct-pt.
- Mean temperature RMSE: 8.626 K; maximum temperature RMSE: 22.310 K.
- Maximum boundary residual: 2.52e-13.
- Maximum chemistry mass residual: 9.87e-09.
- Maximum chemistry reaction residual: 1.00e-08.
- Maximum chemistry charge residual: 3.37e-13.

This is the broadest proof so far that the full nine-species ionic speciation/activity loop can run across the legacy C-case validation set. It also makes the computational tradeoff clearer: the nine-species loop converges cleanly, but all seven loose-proof runs still required about 41 minutes total wall time on this machine.

## Interpretation

The nine-species full ionic activity loop now has a clean column proof run and a complete C-case sweep. Compared with the prior six-species activity-rebased proof, the full species set gives almost the same case-3C capture result but increases runtime from about 129 s to about 213-282 s depending on the rerun context and benchmark overhead. This strengthens the manuscript argument that activity-coupled/full-species speciation is technically feasible, but the extra species and activity loop are not an attractive default for repeated absorber simulations when the accuracy gain is small.

## Validation

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_epcsaft_reactive_chemistry.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_thermodynamics_adapter.py -k "epcsaft or ionic"
```

Current targeted result before final commit:

- `tests/test_epcsaft_reactive_chemistry.py`: `4 passed in 5.20s`
- `tests/test_thermodynamics_adapter.py -k "epcsaft or ionic"`: `8 passed, 2 deselected in 3.05s`
