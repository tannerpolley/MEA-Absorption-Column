# Reviewer-Response Benchmarking Notes

This project should be presented as a reproducible benchmark framework, not as a claim that the absorber model is fundamentally more accurate or faster than prior MEA models.

## Implemented Benchmark Scope

- `ideal_henry`: previous Henry-law liquid CO2 fugacity and ideal vapor partial pressure.
- `epcsaft_neutral`: neutral CO2/MEA/H2O ePC-SAFT fugacity-coefficient sensitivity using the external `C:\Users\Tanner\Documents\git\ePC-SAFT` package read-only.
- Benchmark rows preserve failures instead of dropping them.
- Result columns include case id, method, thermodynamic model, success flag, runtime, capture error, temperature RMSE when taps are available, boundary residual norm, mesh/tolerance settings, Python version, platform, and package versions.
- Multi-bed and intercooled NCCC rows can be routed through a staged SciPy BVP solver. The staged solver uses one packed-bed BVP per bed, enforces inter-bed vapor continuity, and applies liquid continuity or liquid enthalpy reset between beds.

## Publication Limits

The neutral ePC-SAFT comparison isolates only the CO2 fugacity driving force. Chemistry, enhancement factor, transport properties, hydraulic correlations, and balances are intentionally unchanged for the first comparison. This must be described as a thermodynamic sensitivity study, not as a complete electrolyte-reactive ePC-SAFT absorber model.

The local ePC-SAFT parameter file is provisional. Before using it as a final paper result, audit the pure-component association schemes, binary interaction parameters, source tables, temperature validity range, and units against the cited literature.

The NCCC broad dataset includes multi-bed and intercooled cases. These are no longer treated as one equivalent-height packed section when `--staged-beds auto` is used. The staged benchmark solves one packed-bed BVP per bed and enforces inter-bed vapor continuity plus liquid continuity or liquid cooling. This makes the multi-bed validation structurally closer to the experimental column, but the current intercooler model is still simplified because it uses a liquid temperature reset rather than a detailed heat-exchanger design.

## Reproducible Commands

```powershell
uv venv --python 3.12 --clear
uv run --group test python -m pytest
uv run python -m mea_absorption_column.benchmark --methods single scipy-bvp --thermo-models ideal_henry epcsaft_neutral
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_neutral --staged-beds auto --output-dir benchmark_artifacts\intercooled_benchmark
```

The Python 3.12 environment matters on this Windows machine because the sibling ePC-SAFT checkout currently contains a CPython 3.12 native extension.
