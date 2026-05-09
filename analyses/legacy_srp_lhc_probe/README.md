# Legacy SRP-LHC Probe

This analysis checks how the `legacy/main-legacy` model behaves on the
25-row SRP Latin-hypercube design without modifying the legacy branch itself.
The committed CSV files are the paper-facing evidence from that probe.

The runner adapts `LHC_design_w_SRP_cases.csv` to the column order expected by
the legacy `run_model(...)` API and records per-row success, capture, runtime,
and timeout status.  The default geometry mode is `srp`, which patches the
legacy converter's `NCCC` geometry dictionary entry at runtime to use the
already-present `SRP` dimensions.  Use `--geometry legacy-nccc` to reproduce
the converter with its original hardwired dimensions.

The legacy code imports `pcsaft` at module import time, but the active fugacity
mode in `Thermodynamics/Fugacity.py` is `ideal`.  The runner therefore installs
a process-local shim for the unused `pcsaft` import so the old ideal-Henry path
can be tested without requiring the broken legacy `pcsaft` build.

The runner has two vapor-water modes:

- `explicit`: use the `y_H2O` values stored in `LHC_design_w_SRP_cases.csv`.
- `legacy-ratio`: reproduce the old converter behavior,
  `y_H2O = 0.9626010166*y_CO2`.

Example:

```powershell
# Validate committed probe artifacts from a current main checkout.
.\.venv\Scripts\python.exe analyses\legacy_srp_lhc_probe\scripts\run_legacy_srp_lhc.py --validate-results-only

# Re-run the probe from a legacy-compatible checkout.
.\.venv\Scripts\python.exe analyses\legacy_srp_lhc_probe\scripts\run_legacy_srp_lhc.py --methods single --geometry srp --water-modes explicit legacy-ratio --timeout-s 60
```

Observed result in this probe:

- `single`/shooting with SRP geometry and `legacy-ratio` vapor water solved all
  25 SRP-LHC rows with median model runtime near 0.85 s.
- `single`/shooting with SRP geometry and explicit SRP-LHC vapor water solved
  20 of 25 rows with median successful model runtime near 0.78 s.
- The five explicit-water failures are runs 1, 9, 10, 17, and 23.  A
  runner-local bounded least-squares shooting probe was also tested on those
  rows and did not recover them; the failures are tied to non-finite physical
  states in the legacy transport/thermal path under the drier SRP-LHC vapor
  inlet, not simply the Krylov root method.
- The manuscript-level interpretation should be that shooting is fast for
  smoother SRP-like cases, finite difference can be useful as an intermediate
  method, and collocation remains the more defensible reference method for
  NCCC-style validation because it handles coupled boundary conditions and
  sharper thermal behavior more systematically.
