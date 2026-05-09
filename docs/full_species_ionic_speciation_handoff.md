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

## Interpretation

The nine-species full ionic activity loop now has a clean column proof run. Compared with the prior six-species activity-rebased proof, the full species set gives almost the same case-3C capture result but increases runtime from about 129 s to about 213 s under the same loose proof settings. This strengthens the manuscript argument that activity-coupled/full-species speciation is technically feasible, but the extra species and activity loop are not an attractive default for repeated absorber simulations when the accuracy gain is small.

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
