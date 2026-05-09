# Legacy SRP-LHC Probe Results

Rows tested: 50

```text
geometry   water_mode method  status  runs  median_runtime_s  median_wall_runtime_s  median_capture_pct
     srp     explicit single   error     5          1.662771               1.662771                 NaN
     srp     explicit single success    20          0.781453               2.294884           99.998168
     srp legacy-ratio single success    25          0.848183               2.315802           99.999646
```

Notes:
- `single` is the legacy shooting-method path.
- The `srp` geometry mode uses the SRP dimensions already present in `Constants.py`.
- `explicit` uses the SRP-LHC `y_H2O` column; `legacy-ratio` reproduces the old converter's fixed CO2-water ratio.
- `y_H2O_input`, `y_H2O_legacy_converter`, and `y_H2O_used` are reported so the inlet-water assumption is visible.
- Interpretation for the manuscript: shooting is fast for smoother SRP-like cases, finite difference can be useful as an intermediate method, and collocation remains the more defensible reference method for NCCC-style validation because it handles coupled boundary conditions and sharper thermal behavior more systematically.
