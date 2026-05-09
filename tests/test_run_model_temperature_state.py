from __future__ import annotations

import numpy as np
import pandas as pd

import mea_absorption_column.Run_Model as run_model_module


def test_temperature_state_mode_does_not_preconvert_boundary_temperatures_to_enthalpy(monkeypatch):
    df = pd.read_csv("src/mea_absorption_column/data/C_cases_data.csv", index_col=0)
    run = list(df.index).index("3C")

    def fail_enthalpy(*_args, **_kwargs):
        raise AssertionError("temperature-state solve should not precompute enthalpy boundary values")

    def fake_solver(y_a_scaled, y_b_scaled, z, _parameters, settings=None):
        y = np.column_stack([y_a_scaled, y_b_scaled])
        return y, np.array([z[0], z[-1]]), "fake-temperature-state", True, "fake success"

    monkeypatch.setattr(run_model_module, "get_liquid_enthalpy", fail_enthalpy)
    monkeypatch.setattr(run_model_module, "get_vapor_enthalpy", fail_enthalpy)
    monkeypatch.setattr(run_model_module, "scipy_BVP_solve", fake_solver)

    result = run_model_module.run_model(
        df,
        method="scipy-bvp",
        run=run,
        thermo_model="ideal_henry",
        return_details=True,
        staged_beds=False,
        solver_settings={"thermal_state_mode": "temperature"},
    )

    assert result["success"] is True
    assert result["thermal_state_mode"] == "temperature"


def test_temperature_state_scaling_does_not_zero_small_nccc_k_case_flows(monkeypatch):
    df = pd.read_csv("src/mea_absorption_column/data/NCCC_Data.csv", index_col=0)
    run = list(df.index).index("K3")

    def fake_segmented_solver(y_a_scaled, y_b_scaled, z, _parameters, stack_spec, settings=None):
        single_bed = np.column_stack([y_a_scaled, y_b_scaled])
        y = np.vstack([single_bed for _ in range(stack_spec.beds)])
        assert np.isfinite(y).all()
        return y, np.array([z[0], z[-1]]), "fake-segmented-temperature-state", True, "fake success"

    monkeypatch.setattr(run_model_module, "segmented_scipy_BVP_solve", fake_segmented_solver)

    result = run_model_module.run_model(
        df,
        method="scipy-bvp",
        run=run,
        thermo_model="ideal_henry",
        return_details=True,
        staged_beds="auto",
        solver_settings={
            "intercooler_model": "pumparound_temperature_approach",
            "thermal_state_mode": "temperature",
        },
    )

    assert result["success"] is True
    assert result["case_id"] == "K3"
    assert result["thermal_state_mode"] == "temperature"
