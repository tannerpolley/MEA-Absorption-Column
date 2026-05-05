import numpy as np
import pandas as pd

from mea_absorption_column.BVP.Methods.Segmented_Scipy_BVP_Solve import (
    external_profile_from_stacked_solution,
    _stack_initial_guess,
    _case_state_bounds_scaled,
    _bounded_scaled_physical_to_solver,
    _bounded_solver_to_scaled_physical,
    segmented_scipy_BVP_solve,
    stacked_boundary_conditions,
)
from mea_absorption_column.Run_Model import run_model
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


def test_segmented_scipy_bvp_returns_structured_timeout():
    scales = np.ones(7)
    parameters = (
        scales,
        scales,
        (1.0, 1.0, 1.0),
        6.1,
        1.0,
        (250.0, 0.97, 1.0, 1.0, 1.0, 1.0, 1.0),
        {},
    )
    y_a = np.ones(7)
    y_b = np.ones(7)
    y, z, _, success, message = segmented_scipy_BVP_solve(
        y_a,
        y_b,
        np.linspace(0.0, 1.0, 5),
        parameters,
        stack_spec=build_bed_stack_spec(1, 0, 6.1, 314.0),
        settings={"mesh_points": 5, "max_runtime_s": -1.0},
    )

    assert success is False
    assert "max_runtime_s" in message
    assert y.shape == (7, 5)
    assert z.shape == (5,)


def test_stack_initial_guess_expands_single_bed_seed_profile_across_beds():
    seed = np.vstack([np.linspace(i, i + 1, 9) for i in range(7)])

    guess = _stack_initial_guess(
        Y_a_scaled=np.zeros(7),
        Y_b_scaled=np.ones(7),
        mesh_points=5,
        beds=3,
        explicit_initial_guess=seed,
    )

    assert guess.shape == (21, 5)
    assert np.allclose(guess[:7, 0], seed[:, 0])
    assert np.allclose(guess[14:21, -1], seed[:, -1])
    assert guess[0, -1] < guess[7, -1] < guess[14, -1]


def test_stack_initial_guess_splits_global_profile_across_beds():
    guess = _stack_initial_guess(
        Y_a_scaled=np.zeros(7),
        Y_b_scaled=np.ones(7),
        mesh_points=5,
        beds=2,
    )

    first_bed = guess[:7]
    second_bed = guess[7:]
    assert first_bed[0, -1] < second_bed[0, -1]
    assert np.allclose(first_bed[:, 0], 0.0)
    assert np.allclose(second_bed[:, -1], 1.0)


def test_stack_initial_guess_with_scales_preserves_positive_endpoints():
    guess = _stack_initial_guess(
        Y_a_scaled=np.ones(7) * 0.1,
        Y_b_scaled=np.ones(7) * 0.01,
        mesh_points=5,
        beds=3,
        scales=np.array([10.0, 20.0, 5.0, 5.0, 1.0e6, 1.0e6, 1.0e5]),
    )

    assert guess.shape == (21, 5)
    assert np.all(guess[[0, 1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 20], :] > 0.0)


def test_external_profile_from_stacked_solution_concatenates_beds_bottom_to_top():
    stacked = np.zeros((21, 2))
    stacked[2, :] = [10.0, 7.0]
    stacked[9, :] = [7.0, 3.0]
    stacked[16, :] = [3.0, 1.0]

    external = external_profile_from_stacked_solution(stacked, beds=3)

    assert external.shape == (7, 6)
    assert np.allclose(external[2, :], [10.0, 7.0, 7.0, 3.0, 3.0, 1.0])


def test_case_bounded_transform_caps_vapor_co2_by_inlet_flow():
    y_a = np.array([2.0, 5.0, 1.0, 0.8, 10.0, 20.0, 1.0])
    y_b = np.array([0.5, 6.0, 0.05, 1.5, 8.0, 18.0, 1.0])
    scales = np.ones(7)

    bounds = _case_state_bounds_scaled(y_a, y_b, scales)
    solver = np.array([0.0, 0.0, 1000.0, 0.0, 10.0, 20.0, 0.0])
    physical = _bounded_solver_to_scaled_physical(solver, bounds)

    assert physical[2] <= y_a[2] * 1.05
    assert physical[2] > 0.0
    round_trip = _bounded_solver_to_scaled_physical(
        _bounded_scaled_physical_to_solver(y_a, bounds),
        bounds,
    )
    assert np.allclose(round_trip[[0, 1, 2, 3, 6]], y_a[[0, 1, 2, 3, 6]])


def test_segmented_scipy_bvp_matches_single_bed_scipy_bvp_for_case_3c():
    df = pd.read_csv(
        "src/mea_absorption_column/data/C_cases_data.csv",
        index_col=0,
    )
    run = list(df.index).index("3C")
    settings = {"mesh_points": 11, "tol": 1.0, "bc_tol": 0.01, "max_nodes": 120}

    baseline = run_model(
        df,
        method="scipy-bvp",
        run=run,
        thermo_model="ideal_henry",
        return_details=True,
        staged_beds=False,
        solver_settings=settings,
    )
    segmented = run_model(
        df,
        method="scipy-bvp",
        run=run,
        thermo_model="ideal_henry",
        return_details=True,
        staged_beds=True,
        solver_settings=settings,
    )

    assert segmented["success"] is True
    assert abs(segmented["capture_pct"] - baseline["capture_pct"]) < 2.5


def test_positive_flow_pressure_transform_reproduces_case_3c_scipy_bvp():
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
    transformed = run_model(
        df,
        method="scipy-bvp",
        run=run,
        thermo_model="ideal_henry",
        return_details=True,
        staged_beds=False,
        solver_settings={"transform_mode": "positive_flow_pressure"},
    )

    assert transformed["success"] is True
    assert transformed["transform_mode"] == "positive_flow_pressure"
    assert abs(transformed["capture_pct"] - baseline["capture_pct"]) < 0.5
