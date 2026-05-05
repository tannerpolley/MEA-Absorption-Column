# Intercooled Bed Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit multi-bed and intercooler support to the MEA absorber simulation so NCCC cases with `Beds > 1` and `Intercoolers > 0` are modeled as staged packed sections rather than one equivalent-height packed bed.

**Architecture:** Keep the current single-bed equations in `src/mea_absorption_column/BVP/ABS_Column.py` unchanged as the per-bed residual kernel. Add a stacked SciPy BVP wrapper that solves one 7-variable state vector per packed bed and enforces inter-bed gas continuity plus liquid continuity or liquid enthalpy reset at intercoolers. Use this first for `scipy-bvp`; leave shooting and finite difference as single-bed methods until the staged BVP is validated.

**Tech Stack:** Python 3.12, NumPy, SciPy `solve_bvp`, pandas benchmark artifacts, existing `uv` workflow, existing ideal Henry and `epcsaft_neutral` thermodynamic adapters.

---

## Current Repo Facts

- `convert_data()` currently reads `Beds` but multiplies the NCCC bed height by bed count before returning `H`; later it returns `z = np.linspace(0, 1, n)`, so the column model sees one continuous normalized bed with `H = bed_height * beds`.
- `Intercoolers` exists in `NCCC_Data_mole_based.csv` but is not returned by `convert_data()` and is not used in `run_model()`.
- `abs_column()` computes a single packed-section differential system for seven scaled states: `Fl_CO2`, `Fl_H2O`, `Fv_CO2`, `Fv_H2O`, `Hlf`, `Hvf`, and `P`.
- `scipy_BVP_solve()` currently enforces top liquid inlet and bottom vapor inlet boundary conditions for one bed.
- For NCCC data, many failed/bad cases are multi-bed/intercooled, so a reviewer-facing validation should separate unsupported old behavior from the new staged-bed behavior.

## Physical Model Scope

First implementation:

- Treat each packed bed as one independent call to the existing `abs_column()` equations.
- Bed index order is bottom-to-top: bed `0` is the bottom bed receiving fresh gas, bed `N-1` is the top bed receiving lean solvent.
- Gas flows upward:
  - gas at the top of bed `j` enters the bottom of bed `j+1`.
- Liquid flows downward:
  - liquid at the bottom of bed `j+1` enters the top of bed `j`.
- Intercoolers act only on the liquid stream between packed sections.
- Intercooler mass change is zero.
- Initial intercooler mode is `temperature_target`.
- If a target is unavailable, default to the measured liquid feed temperature column `Tl` for a conservative smoke model and mark the result metadata as `intercooler_assumption=Tl_feed_target`.
- Do not add electrolyte ePC-SAFT, intercooler heat-transfer design, pressure-drop coupling, or flash drums in this first pass.

## Files

- Create: `src/mea_absorption_column/intercooling.py`
  - Dataclasses for `IntercoolerSpec` and `BedStackSpec`.
  - Conversion from a case row to stack metadata.
  - Pure liquid enthalpy reset helper.
- Create: `src/mea_absorption_column/BVP/Methods/Segmented_Scipy_BVP_Solve.py`
  - Stacked multi-bed SciPy BVP solver.
  - Boundary residual function kept importable for unit tests.
- Modify: `src/mea_absorption_column/misc/Convert_Data.py`
  - Preserve existing return shape for old callers.
  - Add optional `return_metadata=True` path with `beds`, `intercoolers`, and `single_bed_height`.
- Modify: `src/mea_absorption_column/Run_Model.py`
  - Add `staged_beds='auto'`, `intercooler_settings=None`, and return metadata fields.
  - Route `scipy-bvp` to segmented solver when `Beds > 1` or `Intercoolers > 0`.
- Modify: `src/mea_absorption_column/benchmark.py`
  - Add columns: `beds`, `intercoolers`, `staged_beds`, `intercooler_model`, `intercooler_assumption`.
  - Add CLI controls for staged bed mode.
- Create: `tests/test_intercooling.py`
- Create: `tests/test_segmented_bvp.py`
- Modify: `tests/test_benchmarking.py`
- Modify: `README.md`
- Modify: `docs/reviewer_response_benchmarking.md`
- Later manuscript edit: `docs/main.tex`

---

### Task 1: Add Case Metadata Without Breaking Existing Conversion

**Files:**
- Modify: `src/mea_absorption_column/misc/Convert_Data.py`
- Test: `tests/test_intercooling.py`

- [ ] **Step 1: Write the failing metadata test**

```python
import pandas as pd

from mea_absorption_column.misc.Convert_Data import convert_data


def test_convert_data_can_return_bed_and_intercooler_metadata():
    df = pd.DataFrame(
        {
            "L": [3.0],
            "G": [20.0],
            "alpha": [0.2],
            "w_MEA": [0.3],
            "y_CO2": [0.1],
            "Tl": [314.0],
            "Tv": [316.0],
            "P": [108000.0],
            "Beds": [3],
            "Intercoolers": [2],
            "CO2  %": [90.0],
        },
        index=["K-test"],
    )

    inputs, x, metadata = convert_data(df, run=0, type="mole", return_metadata=True)

    assert metadata["case_id"] == "K-test"
    assert metadata["beds"] == 3
    assert metadata["intercoolers"] == 2
    assert metadata["single_bed_height_m"] > 0.0
    assert metadata["total_packed_height_m"] == inputs[5]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run --group test python -m pytest tests\test_intercooling.py::test_convert_data_can_return_bed_and_intercooler_metadata -q
```

Expected: failure because `return_metadata` is not accepted.

- [ ] **Step 3: Implement metadata return**

In `convert_data()`, change the signature:

```python
def convert_data(df, run=0, type='mole', return_metadata=False):
```

After `X = df.iloc[run, :].to_numpy()`, add:

```python
case_id = str(df.index[run])
intercoolers = int(df.iloc[run]["Intercoolers"]) if "Intercoolers" in df.columns else 0
```

Near the end, replace:

```python
return [Fl, Fv, Tl_z, Tv_0, z, H, A, P, packing], X
```

with:

```python
inputs = [Fl, Fv, Tl_z, Tv_0, z, H, A, P, packing]
if return_metadata:
    single_bed_height = column_params['NCCC']['H']
    metadata = {
        "case_id": case_id,
        "beds": int(beds),
        "intercoolers": intercoolers,
        "single_bed_height_m": float(single_bed_height),
        "total_packed_height_m": float(H),
    }
    return inputs, X, metadata
return inputs, X
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
uv run --group test python -m pytest tests\test_intercooling.py::test_convert_data_can_return_bed_and_intercooler_metadata -q
```

Expected: `1 passed`.

- [ ] **Step 5: Run full tests**

Run:

```powershell
uv run --group test python -m pytest -q
```

Expected: all existing tests pass.

---

### Task 2: Add Intercooler Specification And Liquid Enthalpy Reset

**Files:**
- Create: `src/mea_absorption_column/intercooling.py`
- Test: `tests/test_intercooling.py`

- [ ] **Step 1: Write failing tests for stack construction and cooling**

Append to `tests/test_intercooling.py`:

```python
import numpy as np

from mea_absorption_column.intercooling import (
    build_bed_stack_spec,
    liquid_enthalpy_after_intercooler,
)


def test_build_bed_stack_spec_places_intercoolers_between_beds():
    spec = build_bed_stack_spec(
        beds=3,
        intercoolers=2,
        single_bed_height_m=6.1,
        liquid_feed_temperature_K=314.0,
    )

    assert spec.beds == 3
    assert spec.single_bed_height_m == 6.1
    assert len(spec.intercoolers) == 2
    assert [cooler.below_upper_bed_index for cooler in spec.intercoolers] == [2, 1]
    assert all(cooler.mode == "temperature_target" for cooler in spec.intercoolers)
    assert all(cooler.target_temperature_K == 314.0 for cooler in spec.intercoolers)


def test_liquid_enthalpy_after_intercooler_preserves_liquid_molar_flows():
    state = np.array([1.5, 40.0, 2.0, 5.0, 1.0e6, 8.0e5, 108000.0])
    fl_mea = 20.0
    cooled = liquid_enthalpy_after_intercooler(state, fl_mea, target_temperature_K=313.15)

    assert cooled.shape == state.shape
    assert cooled[0] == state[0]
    assert cooled[1] == state[1]
    assert cooled[4] != state[4]
    assert np.isfinite(cooled[4])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
uv run --group test python -m pytest tests\test_intercooling.py -q
```

Expected: import failure for `mea_absorption_column.intercooling`.

- [ ] **Step 3: Create implementation**

Create `src/mea_absorption_column/intercooling.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_liquid_enthalpy


@dataclass(frozen=True)
class IntercoolerSpec:
    below_upper_bed_index: int
    mode: str = "temperature_target"
    target_temperature_K: float | None = None
    duty_W: float | None = None


@dataclass(frozen=True)
class BedStackSpec:
    beds: int
    single_bed_height_m: float
    intercoolers: tuple[IntercoolerSpec, ...]
    assumption: str


def build_bed_stack_spec(
    beds: int,
    intercoolers: int,
    single_bed_height_m: float,
    liquid_feed_temperature_K: float,
    target_temperatures_K: list[float] | tuple[float, ...] | None = None,
) -> BedStackSpec:
    beds = int(beds)
    intercoolers = int(intercoolers)
    if beds < 1:
        raise ValueError("beds must be at least 1.")
    if intercoolers < 0:
        raise ValueError("intercoolers must be non-negative.")
    if intercoolers > max(0, beds - 1):
        raise ValueError("intercoolers cannot exceed beds - 1.")

    if target_temperatures_K is None:
        targets = [float(liquid_feed_temperature_K)] * intercoolers
        assumption = "Tl_feed_target"
    else:
        targets = [float(value) for value in target_temperatures_K]
        if len(targets) != intercoolers:
            raise ValueError("target_temperatures_K length must match intercoolers.")
        assumption = "explicit_temperature_targets"

    specs = []
    for offset in range(intercoolers):
        specs.append(
            IntercoolerSpec(
                below_upper_bed_index=beds - 1 - offset,
                mode="temperature_target",
                target_temperature_K=targets[offset],
            )
        )

    return BedStackSpec(
        beds=beds,
        single_bed_height_m=float(single_bed_height_m),
        intercoolers=tuple(specs),
        assumption=assumption,
    )


def liquid_enthalpy_after_intercooler(
    unscaled_state: np.ndarray,
    fl_mea: float,
    target_temperature_K: float,
) -> np.ndarray:
    cooled = np.asarray(unscaled_state, dtype=float).copy()
    fl_co2, fl_h2o = float(cooled[0]), float(cooled[1])
    fl = [fl_co2, float(fl_mea), fl_h2o]
    hlt = get_liquid_enthalpy(fl, float(target_temperature_K))
    cooled[4] = hlt * sum(fl)
    return cooled
```

- [ ] **Step 4: Run tests**

Run:

```powershell
uv run --group test python -m pytest tests\test_intercooling.py -q
```

Expected: all tests in that file pass.

---

### Task 3: Add Pure Boundary Residual For Stacked Beds

**Files:**
- Create: `src/mea_absorption_column/BVP/Methods/Segmented_Scipy_BVP_Solve.py`
- Test: `tests/test_segmented_bvp.py`

- [ ] **Step 1: Write boundary residual tests**

Create `tests/test_segmented_bvp.py`:

```python
import numpy as np

from mea_absorption_column.BVP.Methods.Segmented_Scipy_BVP_Solve import (
    stacked_boundary_conditions,
)
from mea_absorption_column.intercooling import build_bed_stack_spec


def test_stacked_boundary_conditions_returns_7_residuals_for_one_bed():
    scales = np.ones(7)
    spec = build_bed_stack_spec(1, 0, 6.1, 314.0)
    bottom = np.array([9.0, 40.0, 2.0, 1.0, 5.0e5, 9.0e5, 108000.0])
    top = np.array([1.0, 50.0, 8.0, 2.0, 6.0e5, 8.0e5, 108000.0])

    residual = stacked_boundary_conditions(
        bottom_scaled=bottom,
        top_scaled=top,
        y_bottom_target_scaled=bottom,
        y_top_target_scaled=top,
        scales=scales,
        fl_mea=20.0,
        stack_spec=spec,
    )

    assert residual.shape == (7,)
    assert np.allclose(residual, 0.0)


def test_stacked_boundary_conditions_returns_7_per_bed_residuals_for_three_beds():
    scales = np.ones(7)
    spec = build_bed_stack_spec(3, 2, 6.1, 314.0)
    bottom = np.tile(np.array([9.0, 40.0, 2.0, 1.0, 5.0e5, 9.0e5, 108000.0]), 3)
    top = bottom.copy()

    residual = stacked_boundary_conditions(
        bottom_scaled=bottom,
        top_scaled=top,
        y_bottom_target_scaled=bottom[:7],
        y_top_target_scaled=top[-7:],
        scales=scales,
        fl_mea=20.0,
        stack_spec=spec,
    )

    assert residual.shape == (21,)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run --group test python -m pytest tests\test_segmented_bvp.py -q
```

Expected: import failure for segmented solver.

- [ ] **Step 3: Create boundary helper**

Create `src/mea_absorption_column/BVP/Methods/Segmented_Scipy_BVP_Solve.py` with this first helper:

```python
from __future__ import annotations

import numpy as np

from mea_absorption_column.intercooling import (
    BedStackSpec,
    liquid_enthalpy_after_intercooler,
)

STATE_SIZE = 7
LIQUID_IDXS = np.array([0, 1, 4])
VAPOR_IDXS = np.array([2, 3, 5, 6])


def _slice_bed(vector: np.ndarray, bed_index: int) -> np.ndarray:
    start = bed_index * STATE_SIZE
    return vector[start:start + STATE_SIZE]


def _intercooler_for_upper_bed(stack_spec: BedStackSpec, upper_bed_index: int):
    for cooler in stack_spec.intercoolers:
        if cooler.below_upper_bed_index == upper_bed_index:
            return cooler
    return None


def stacked_boundary_conditions(
    bottom_scaled,
    top_scaled,
    y_bottom_target_scaled,
    y_top_target_scaled,
    scales,
    fl_mea,
    stack_spec: BedStackSpec,
):
    bottom_scaled = np.asarray(bottom_scaled, dtype=float)
    top_scaled = np.asarray(top_scaled, dtype=float)
    y_bottom_target_scaled = np.asarray(y_bottom_target_scaled, dtype=float)
    y_top_target_scaled = np.asarray(y_top_target_scaled, dtype=float)
    scales = np.asarray(scales, dtype=float)

    residuals = []

    bottom_bed_bottom = _slice_bed(bottom_scaled, 0)
    top_bed_top = _slice_bed(top_scaled, stack_spec.beds - 1)
    residuals.extend(bottom_bed_bottom[VAPOR_IDXS] - y_bottom_target_scaled[VAPOR_IDXS])
    residuals.extend(top_bed_top[LIQUID_IDXS] - y_top_target_scaled[LIQUID_IDXS])

    for lower_bed in range(stack_spec.beds - 1):
        upper_bed = lower_bed + 1
        lower_top = _slice_bed(top_scaled, lower_bed)
        upper_bottom = _slice_bed(bottom_scaled, upper_bed)

        residuals.extend(lower_top[VAPOR_IDXS] - upper_bottom[VAPOR_IDXS])

        liquid_from_upper = upper_bottom.copy() * scales
        cooler = _intercooler_for_upper_bed(stack_spec, upper_bed)
        if cooler is not None:
            liquid_from_upper = liquid_enthalpy_after_intercooler(
                liquid_from_upper,
                fl_mea=fl_mea,
                target_temperature_K=cooler.target_temperature_K,
            )
        liquid_from_upper_scaled = liquid_from_upper / scales
        residuals.extend(lower_top[LIQUID_IDXS] - liquid_from_upper_scaled[LIQUID_IDXS])

    return np.asarray(residuals, dtype=float)
```

- [ ] **Step 4: Run boundary tests**

Run:

```powershell
uv run --group test python -m pytest tests\test_segmented_bvp.py -q
```

Expected: tests pass.

---

### Task 4: Implement Segmented SciPy BVP Solver

**Files:**
- Modify: `src/mea_absorption_column/BVP/Methods/Segmented_Scipy_BVP_Solve.py`
- Test: `tests/test_segmented_bvp.py`

- [ ] **Step 1: Write one-bed equivalence integration test**

Append:

```python
import pandas as pd

from mea_absorption_column.Run_Model import run_model


def test_segmented_scipy_bvp_matches_single_bed_scipy_bvp_for_case_3c():
    df = pd.read_csv(
        "src/mea_absorption_column/data/C_cases_data.csv",
        index_col=0,
    )
    run = list(df.index).index("3C")

    baseline = run_model(
        df,
        method="scipy-bvp",
        run=run,
        thermo_model="ideal_henry",
        return_details=True,
        staged_beds=False,
    )
    segmented = run_model(
        df,
        method="scipy-bvp",
        run=run,
        thermo_model="ideal_henry",
        return_details=True,
        staged_beds=True,
    )

    assert segmented["success"] is True
    assert abs(segmented["capture_pct"] - baseline["capture_pct"]) < 0.25
    assert abs(segmented["temperature_rmse_K"] - baseline["temperature_rmse_K"]) < 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run --group test python -m pytest tests\test_segmented_bvp.py::test_segmented_scipy_bvp_matches_single_bed_scipy_bvp_for_case_3c -q
```

Expected: `run_model()` does not accept `staged_beds`.

- [ ] **Step 3: Implement `segmented_scipy_BVP_solve()`**

Add to `Segmented_Scipy_BVP_Solve.py`:

```python
from scipy.integrate import solve_bvp

from mea_absorption_column.BVP.ABS_Column import abs_column
from mea_absorption_column.BVP.Methods.Scipy_BVP_Solve import DEFAULT_SCIPY_BVP_SETTINGS
from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium


def _stack_initial_guess(Y_a_scaled, Y_b_scaled, mesh_points, beds):
    base = np.vstack([
        np.linspace(Y_a_scaled[i], Y_b_scaled[i], mesh_points)
        for i in range(STATE_SIZE)
    ])
    return np.vstack([base.copy() for _ in range(beds)])


def segmented_scipy_BVP_solve(
    Y_a_scaled,
    Y_b_scaled,
    z,
    parameters,
    stack_spec: BedStackSpec,
    settings=None,
):
    settings = {**DEFAULT_SCIPY_BVP_SETTINGS, **(settings or {})}
    scales, eq_scales, const_flow, H, A, packing, model_options = parameters
    fl_mea = const_flow[0]
    bed_parameters = (
        scales,
        eq_scales,
        const_flow,
        stack_spec.single_bed_height_m,
        A,
        packing,
        model_options,
    )

    mesh_points = int(settings["mesh_points"])
    z_mesh = np.linspace(z[0], z[-1], mesh_points)
    y_guess = _stack_initial_guess(Y_a_scaled, Y_b_scaled, mesh_points, stack_spec.beds)

    def column_odes(z_values, stacked_y):
        blocks = []
        for bed_index in range(stack_spec.beds):
            bed_y = stacked_y[bed_index * STATE_SIZE:(bed_index + 1) * STATE_SIZE, :]
            differentials = [
                abs_column(z_values[i], bed_y[:, i], bed_parameters)
                for i in range(bed_y.shape[1])
            ]
            blocks.append(np.asarray(differentials).T)
        if hasattr(chemical_equilibrium, "cache"):
            del chemical_equilibrium.cache
        return np.vstack(blocks)

    def boundary(bottom, top):
        return stacked_boundary_conditions(
            bottom_scaled=bottom,
            top_scaled=top,
            y_bottom_target_scaled=np.asarray(Y_a_scaled, dtype=float),
            y_top_target_scaled=np.asarray(Y_b_scaled, dtype=float),
            scales=scales,
            fl_mea=fl_mea,
            stack_spec=stack_spec,
        )

    sol = solve_bvp(
        column_odes,
        boundary,
        z_mesh,
        y_guess,
        max_nodes=int(settings["max_nodes"]),
        tol=float(settings["tol"]),
        bc_tol=float(settings["bc_tol"]),
        verbose=int(settings["verbose"]),
    )

    return sol.sol(z), sol.x, "Segmented SciPy collocation-style BVP", sol.success, sol.message
```

- [ ] **Step 4: Wire `run_model()` staged option**

In `Run_Model.py`, import:

```python
from mea_absorption_column.BVP.Methods.Segmented_Scipy_BVP_Solve import segmented_scipy_BVP_solve
from mea_absorption_column.intercooling import build_bed_stack_spec
```

Change `run_model()` signature:

```python
staged_beds='auto',
intercooler_settings=None,
```

Change conversion call:

```python
inputs, X, case_metadata = convert_data(df, run=run, type=data_type, return_metadata=True)
```

After `L_G, Fv_T, alpha...` add:

```python
beds_count = case_metadata["beds"]
intercoolers_count = case_metadata["intercoolers"]
use_staged_beds = (
    bool(staged_beds)
    if staged_beds != "auto"
    else method in {"scipy-bvp", "collocation"} and (beds_count > 1 or intercoolers_count > 0)
)
stack_spec = build_bed_stack_spec(
    beds=beds_count if use_staged_beds else 1,
    intercoolers=intercoolers_count if use_staged_beds else 0,
    single_bed_height_m=case_metadata["single_bed_height_m"],
    liquid_feed_temperature_K=Tl_z,
    target_temperatures_K=None if intercooler_settings is None else intercooler_settings.get("target_temperatures_K"),
)
```

When solving:

```python
if method == "scipy-bvp" and use_staged_beds:
    Y_scaled, z_new, solving_type, success, message = segmented_scipy_BVP_solve(
        Y_a_scaled,
        Y_b_scaled,
        z,
        parameters,
        stack_spec=stack_spec,
        settings=solver_settings,
    )
else:
    Y_scaled, z_new, solving_type, success, message = solving_function(
        Y_a_scaled, Y_b_scaled, z, parameters, settings=solver_settings
    )
```

For first pass, output profiles and capture should use the top bed/bottom bed external profiles. If `Y_scaled.shape[0] == 7 * stack_spec.beds`, create a helper in `Segmented_Scipy_BVP_Solve.py`:

```python
def external_profile_from_stacked_solution(stacked_y, beds):
    if beds == 1:
        return stacked_y
    bottom_bed = stacked_y[:STATE_SIZE, :]
    top_bed = stacked_y[(beds - 1) * STATE_SIZE:beds * STATE_SIZE, :]
    external = bottom_bed.copy()
    external[0, :] = top_bed[0, :]
    external[1, :] = top_bed[1, :]
    external[4, :] = top_bed[4, :]
    return external
```

Use that external profile before existing post-processing:

```python
if use_staged_beds and Y_scaled.shape[0] > 7:
    from mea_absorption_column.BVP.Methods.Segmented_Scipy_BVP_Solve import external_profile_from_stacked_solution
    Y_scaled_for_outputs = external_profile_from_stacked_solution(Y_scaled, stack_spec.beds)
else:
    Y_scaled_for_outputs = Y_scaled
```

Then replace downstream `Y_scaled` references used for output/capture with `Y_scaled_for_outputs`.

- [ ] **Step 5: Run equivalence test**

Run:

```powershell
uv run --group test python -m pytest tests\test_segmented_bvp.py::test_segmented_scipy_bvp_matches_single_bed_scipy_bvp_for_case_3c -q
```

Expected: pass within the specified tolerances.

---

### Task 5: Add Benchmark Metadata And CLI Controls

**Files:**
- Modify: `src/mea_absorption_column/benchmark.py`
- Modify: `tests/test_benchmarking.py`

- [ ] **Step 1: Write benchmark schema test**

Add to `tests/test_benchmarking.py`:

```python
from mea_absorption_column.benchmark import BENCHMARK_COLUMNS


def test_benchmark_schema_includes_staged_bed_metadata():
    for column in [
        "beds",
        "intercoolers",
        "staged_beds",
        "intercooler_model",
        "intercooler_assumption",
    ]:
        assert column in BENCHMARK_COLUMNS
```

- [ ] **Step 2: Run schema test to verify it fails**

Run:

```powershell
uv run --group test python -m pytest tests\test_benchmarking.py::test_benchmark_schema_includes_staged_bed_metadata -q
```

Expected: missing columns.

- [ ] **Step 3: Add benchmark fields**

In `benchmark.py`, add to `BENCHMARK_COLUMNS`:

```python
"beds",
"intercoolers",
"staged_beds",
"intercooler_model",
"intercooler_assumption",
```

Add `staged_beds: str = "auto"` to `BenchmarkSettings`.

Pass to `run_model()`:

```python
staged_beds=settings.staged_beds,
```

Add parser option:

```python
parser.add_argument("--staged-beds", choices=["auto", "true", "false"], default="auto")
```

Normalize in `main()`:

```python
staged_beds = args.staged_beds
if staged_beds == "true":
    staged_beds = True
elif staged_beds == "false":
    staged_beds = False
```

In `run_model(return_details=True)`, include:

```python
"beds": beds_count,
"intercoolers": intercoolers_count,
"staged_beds": bool(use_staged_beds),
"intercooler_model": "liquid_temperature_reset" if stack_spec.intercoolers else "none",
"intercooler_assumption": stack_spec.assumption if stack_spec.intercoolers else "none",
```

- [ ] **Step 4: Run benchmark tests**

Run:

```powershell
uv run --group test python -m pytest tests\test_benchmarking.py -q
```

Expected: pass.

---

### Task 6: Validate NCCC Intercooled Smoke Cases

**Files:**
- No code unless tests expose a defect.
- Artifacts: `benchmark_artifacts/intercooled_smoke/`

- [ ] **Step 1: Run one-bed baseline**

Run:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_neutral --staged-beds false --output-dir benchmark_artifacts\single_bed_baseline --c-case-limit 7 --nccc-case-limit 0
```

Expected: C cases reproduce the current single-bed baseline.

- [ ] **Step 2: Run staged C-case equivalence**

Run:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_neutral --staged-beds true --output-dir benchmark_artifacts\staged_c_equivalence --c-case-limit 7 --nccc-case-limit 0
```

Expected: C-case capture and temperature RMSE remain close to baseline; this proves the stacked solver does not break one-bed cases.

- [ ] **Step 3: Run first three intercooled NCCC cases with ideal Henry**

Run:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --staged-beds auto --output-dir benchmark_artifacts\intercooled_smoke --c-case-limit 0 --nccc-case-limit 3
```

Expected: artifacts are written even if some cases fail. Failures must include `beds=3`, `intercoolers=2`, `staged_beds=True`, and explicit messages.

- [ ] **Step 4: Run first three intercooled NCCC cases with ePC-SAFT using Henry initialization if needed**

If the ePC-SAFT run fails due nonphysical states, defer full ePC-SAFT intercooled validation until a Henry-seeded initial guess is added. Use this acceptance rule:

```text
ideal_henry must be the first solver target for intercooled model validation.
epcsaft_neutral is secondary until the staged Henry profile can seed the ePC-SAFT solve.
```

---

### Task 7: Add Henry-Seeded ePC-SAFT Initialization For Staged Beds

**Files:**
- Modify: `src/mea_absorption_column/Run_Model.py`
- Modify: `src/mea_absorption_column/BVP/Methods/Segmented_Scipy_BVP_Solve.py`
- Test: `tests/test_segmented_bvp.py`

- [ ] **Step 1: Add test for explicit initial guess acceptance**

Append:

```python
from mea_absorption_column.BVP.Methods.Segmented_Scipy_BVP_Solve import _stack_initial_guess


def test_stack_initial_guess_accepts_explicit_profile():
    explicit = np.ones((14, 11))
    guess = _stack_initial_guess(
        Y_a_scaled=np.ones(7),
        Y_b_scaled=np.ones(7) * 2,
        mesh_points=11,
        beds=2,
        explicit_initial_guess=explicit,
    )

    assert guess.shape == (14, 11)
    assert np.allclose(guess, explicit)
```

- [ ] **Step 2: Update `_stack_initial_guess()`**

Change signature:

```python
def _stack_initial_guess(Y_a_scaled, Y_b_scaled, mesh_points, beds, explicit_initial_guess=None):
    if explicit_initial_guess is not None:
        guess = np.asarray(explicit_initial_guess, dtype=float)
        if guess.shape != (STATE_SIZE * beds, mesh_points):
            raise ValueError("explicit_initial_guess has the wrong shape.")
        return guess
    ...
```

- [ ] **Step 3: Add `initial_guess_scaled` solver setting**

In `segmented_scipy_BVP_solve()`:

```python
y_guess = _stack_initial_guess(
    Y_a_scaled,
    Y_b_scaled,
    mesh_points,
    stack_spec.beds,
    explicit_initial_guess=settings.get("initial_guess_scaled"),
)
```

- [ ] **Step 4: Add optional Henry seed path**

In `run_model()`, before an ePC-SAFT staged run, run the same case with `thermo_model="ideal_henry"` and `staged_beds=use_staged_beds` only when:

```python
thermo_model == "epcsaft_neutral"
and use_staged_beds
and solver_settings is not None
and solver_settings.get("seed_from_henry", True)
```

Store the returned stacked profile internally as `initial_guess_scaled`. Do not write this seed run to benchmark artifacts as a separate row.

- [ ] **Step 5: Run smoke benchmark**

Run:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_neutral --staged-beds auto --output-dir benchmark_artifacts\intercooled_epcsaft_seeded --c-case-limit 0 --nccc-case-limit 3
```

Expected: ePC-SAFT failures should shift from invalid-state crashes toward normal solver convergence/failure messages. If they still fail, record the failures and continue with ideal Henry as the publishable intercooled baseline.

---

### Task 8: Documentation And Paper Framing

**Files:**
- Modify: `README.md`
- Modify: `docs/reviewer_response_benchmarking.md`
- Modify: `docs/main.tex`

- [ ] **Step 1: Add README command**

Add:

```markdown
### Intercooled bed stack benchmark

Multi-bed NCCC cases should be run with staged beds enabled:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_neutral --staged-beds auto --output-dir benchmark_artifacts\intercooled_benchmark
```

`--staged-beds auto` uses the stacked solver when a case has `Beds > 1` or `Intercoolers > 0`. Intercoolers are modeled as liquid enthalpy resets between beds. Unless explicit target temperatures are supplied, the first comparison uses the measured liquid feed temperature as the inter-stage target and reports this as `intercooler_assumption=Tl_feed_target`.
```

- [ ] **Step 2: Update reviewer response notes**

Add:

```markdown
The NCCC broad dataset includes multi-bed and intercooled cases. These are no longer treated as one equivalent-height packed section. The staged benchmark solves one packed-bed BVP per bed and enforces inter-bed vapor continuity plus liquid continuity or liquid cooling. This makes the multi-bed validation structurally closer to the experimental column, but the current intercooler model is still simplified because it uses a liquid temperature reset rather than a detailed heat-exchanger design.
```

- [ ] **Step 3: Update manuscript limitations**

Add a limitation:

```latex
For intercooled NCCC cases, the intercoolers are represented as inter-stage liquid enthalpy resets. This captures the first-order effect of solvent cooling on the downstream packed section but does not model detailed heat-exchanger approach temperatures, pressure losses, holdup in collection trays, or maldistribution after redistribution.
```

---

### Task 9: Final Verification

**Files:**
- No code unless verification fails.

- [ ] **Step 1: Run full tests**

Run:

```powershell
uv run --group test python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run deterministic artifact regeneration**

Run:

```powershell
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_neutral --staged-beds auto --output-dir benchmark_artifacts\reviewer_response_staged_beds
```

Expected:

- C cases remain comparable to the pre-staged baseline.
- NCCC multi-bed/intercooled rows include `staged_beds=True`.
- Failures are recorded in CSV, not hidden.
- `benchmark_summary.csv` separates success counts, runtime, capture errors, and temperature errors.

- [ ] **Step 3: Compare old vs staged broad NCCC failures**

Run a short analysis script:

```powershell
uv run python -c "import pandas as pd; p='benchmark_artifacts/reviewer_response_staged_beds/benchmark_results.csv'; df=pd.read_csv(p); print(df.groupby(['case_source','thermo_model','beds','intercoolers'])['success'].agg(['sum','count']))"
```

Expected: output clearly shows whether staged modeling improved the multi-bed/intercooled success rate.

---

## Acceptance Criteria

- `uv run --group test python -m pytest -q` passes.
- `scipy-bvp` one-bed staged mode reproduces current one-bed behavior within tight capture and temperature tolerances.
- NCCC rows with `Beds > 1` or `Intercoolers > 0` are routed to the stacked solver under `--staged-beds auto`.
- Intercooler assumptions are explicit in benchmark artifacts.
- ePC-SAFT remains a thermodynamic sensitivity lane; ideal Henry remains the first validation target for staged/intercooled hydraulics and energy behavior.
- The paper does not claim detailed intercooler design unless target temperatures or duties are later curated from experimental operating records.

## Open Data Questions

- Does the NCCC data source provide measured inter-stage liquid outlet temperatures, coolant inlet/outlet temperatures, or intercooler duties for `K1-K16` and `K22-K23`?
- Are `Beds=3, Intercoolers=2` always cooled between every adjacent bed, or are there cases where an intercooler is bypassed?
- Does the measured `Tl` column represent the top lean-solvent feed only, or a post-cooler solvent temperature reused by the NCCC operating summary?

If those data are not available, keep the first manuscript version honest: staged beds with assumed liquid reset temperature are a structural correction and sensitivity study, not a fully specified intercooler model.
