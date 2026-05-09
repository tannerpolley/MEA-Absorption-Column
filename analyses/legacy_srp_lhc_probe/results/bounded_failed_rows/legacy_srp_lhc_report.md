# Legacy SRP-LHC Probe Results

Rows tested: 5

```text
geometry water_mode         method         status  runs  median_runtime_s  median_wall_runtime_s  median_capture_pct
     srp   explicit single-bounded solver_failure     5           1.28262               2.828683                 NaN
```

Notes:
- `single` is the legacy shooting-method path.
- The `srp` geometry mode patches the legacy converter's hardwired NCCC geometry to the SRP dimensions already present in `Constants.py`.
- The legacy converter reconstructs vapor water from a fixed CO2-water ratio, so `y_H2O_input` and `y_H2O_legacy_converter` are both reported.