# MEA Absorption Column

## 🧪 Overview
This is a custom Python-based model built to simulate an amine-based absorption column. The model is intended for quick, flexible simulation of post-combustion carbon capture processes.

## 🎯 Purpose
The goal of this model is to enable the development and further the research of Post-Combustion Carbon Capture. It supports rapid simulations using a variety of numerical methods such as:
- Shooting Method  
- Finite Difference  
- Collocation  

All design variables and parameters are customizable, making the tool adaptable to the user's research or engineering needs.

## ⚙️ Status
The model is currently under active development.  
✅ **Functional** – It can already produce consistent results with the current implementation.

## Usage

This repo is now `uv`-first for reproducible reviewer-response benchmarks.

```powershell
uv run --group test python -m pytest
uv run python -m mea_absorption_column.benchmark --methods single scipy-bvp --thermo-models ideal_henry epcsaft_neutral
```

Benchmark CSV and Markdown outputs are written to `benchmark_artifacts/` by default. Those generated files are ignored by Git so the paper tables can be regenerated from the committed scripts and data.

Robust-convergence diagnostics and the current verification snapshot are summarized in `docs/robust_convergence_status.md`. The benchmark CLI exposes solver settings for reproducibility:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --mesh-points 51 --tol 0.5 --bc-tol 0.001 --max-nodes 1000
```

The hand-built central-difference Jacobian is available with `--finite-jacobian`, but it is opt-in because stacked/intercooled cases are much slower with it.

### Intercooled bed stack benchmark

Multi-bed NCCC cases should be run with staged beds enabled:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --staged-beds auto --transform-mode case_bounded_flow_pressure --output-dir benchmark_artifacts\intercooled_benchmark
```

`--staged-beds auto` uses the stacked solver when a case has `Beds > 1` or `Intercoolers > 0`. Intercoolers are modeled as liquid enthalpy resets between beds. Unless explicit target temperatures are supplied, the first comparison uses the measured liquid feed temperature as the inter-stage target and reports this as `intercooler_assumption=Tl_feed_target`.

`--intercooler-strength` can be used for continuation studies: `0` keeps the staged topology without applying liquid cooling, `0.25` applies a weak reset, and `1` applies the full target-temperature reset.

The staged/intercooled NCCC workflow is still diagnostic rather than fully predictive. Failed rows are expected to stay in the benchmark CSV with `success=False`, `jacobian_status`, guard counts, and the first failed domain. `--thermal-state-mode temperature` is available for staged solver experiments, but the current reference path remains the enthalpy-state solver with case-bounded flow/pressure variables. The continuation ladder in `python -m mea_absorption_column.continuation` records optional seed/ramp stages and only treats the full-intercooler stage as the required staged Henry result.

## Thermodynamics

The default model is `ideal_henry`, matching the previous Henry-law driving-force implementation. The new `epcsaft_neutral` option imports the external package at `C:\Users\Tanner\Documents\git\ePC-SAFT` read-only and uses only the CO2 fugacity coefficient from a neutral CO2/MEA/H2O parameter set stored in this repo under `src/mea_absorption_column/data/epcsaft_neutral/`.

The ePC-SAFT lane is a controlled sensitivity study of the CO2 fugacity driving force. It does not replace the chemical-equilibrium, enhancement-factor, transport-property, or mass/energy-balance submodels, and it is not a full electrolyte-reactive ePC-SAFT model.
