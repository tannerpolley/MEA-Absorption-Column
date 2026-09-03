import math

import numpy as np
import pandas as pd
import pytest

from mea_absorption_column.benchmark import BENCHMARK_COLUMNS
from mea_absorption_column.Run_Model import (
    _apply_method_success_gates,
)
from mea_absorption_column.BVP.ABS_Column import _smooth_absorption_only_vapor_flux
from mea_absorption_column.BVP.robust_core import (
    BoundedStateSettings,
    bounded_to_unbounded_positive,
    guard_column_rhs,
    make_solver_diagnostics,
    scaled_physical_to_solver,
    solver_to_scaled_physical_derivative,
    solver_to_scaled_physical,
    validate_scaled_state,
    unbounded_to_positive,
)
from mea_absorption_column.calibration import (
    CalibrationSettings,
    build_structured_holdout_split,
    calibration_artifact_rows,
    nccc_linear_capture_prediction,
    write_calibration_artifacts,
)
from mea_absorption_column.Thermodynamics.thermo_models import guarded_compute_fugacity
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    compute_fugacity,
    ensure_epcsaft_importable,
    epcsaft_phi_co2,
)
from mea_absorption_column.BVP.Methods import Scipy_BVP_Solve as scipy_bvp_module


def test_positive_transform_round_trips_and_rejects_nonpositive_values():
    values = np.array([1.0e-8, 0.25, 1.0])

    unbounded = bounded_to_unbounded_positive(values)
    recovered = unbounded_to_positive(unbounded)

    assert np.allclose(recovered, values)
    with np.testing.assert_raises(ValueError):
        bounded_to_unbounded_positive([0.0, 1.0])


def test_positive_flow_pressure_solver_transform_round_trips_scaled_state():
    state = np.array([0.1, 2.0, 0.03, 0.2, 1.0e5, 2.0e5, 1.0])

    solver_state = scaled_physical_to_solver(state, transform_mode="positive_flow_pressure")
    recovered = solver_to_scaled_physical(solver_state, transform_mode="positive_flow_pressure")

    assert np.allclose(recovered, state)
    assert not np.allclose(solver_state[[0, 1, 2, 3, 6]], state[[0, 1, 2, 3, 6]])


def test_positive_flow_pressure_transform_stays_finite_for_large_newton_trials():
    solver_state = np.array([1000.0, -1000.0, 800.0, -800.0, 1.0e5, 2.0e5, 700.0])

    recovered = solver_to_scaled_physical(solver_state, transform_mode="positive_flow_pressure")
    derivative = solver_to_scaled_physical_derivative(solver_state, transform_mode="positive_flow_pressure")

    assert np.all(np.isfinite(recovered))
    assert np.all(np.isfinite(derivative))
    assert np.all(recovered[[0, 1, 2, 3, 6]] > 0.0)
    assert np.all(derivative[[0, 1, 2, 3, 6]] > 0.0)


def test_positive_flow_pressure_transform_rejects_nonpositive_flow():
    with np.testing.assert_raises(ValueError):
        scaled_physical_to_solver([0.0, 1.0, 1.0, 1.0, 5.0, 6.0, 1.0], transform_mode="positive_flow_pressure")


def test_scipy_bvp_positive_transform_clips_initial_profile_before_solving(monkeypatch):
    captured = {}

    def fake_polynomial_fit(_z, _y_int, _i):
        return [-1.0, 0.0, 1.0]

    class FakeSolution:
        success = True
        message = "ok"
        status = 0
        niter = 1
        rms_residuals = np.zeros(2)

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self._y = y

        def sol(self, z, derivative=0):
            if derivative:
                return np.zeros((self._y.shape[0], len(z)))
            return np.repeat(self._y[:, :1], len(z), axis=1)

    def fake_solve_bvp(_fun, _bc, z, y, **_kwargs):
        captured["initial_guess"] = y
        assert np.all(np.isfinite(y))
        return FakeSolution(z, y)

    monkeypatch.setattr(scipy_bvp_module, "polynomial_fit", fake_polynomial_fit)
    monkeypatch.setattr(scipy_bvp_module, "solve_bvp", fake_solve_bvp)
    monkeypatch.setattr(scipy_bvp_module, "_column_rhs", lambda *_args, **_kwargs: np.zeros(7))

    y_a_scaled = np.ones(7)
    y_b_scaled = np.ones(7)
    diagnostics = {}
    parameters = (np.ones(7), None, None, None, None, None, {"solver_diagnostics": diagnostics})

    scipy_bvp_module.scipy_BVP_solve(
        y_a_scaled,
        y_b_scaled,
        np.linspace(0.0, 1.0, 3),
        parameters,
        settings={"transform_mode": "positive_flow_pressure", "mesh_points": 3},
    )

    assert "initial_guess" in captured
    assert diagnostics["dense_grid_points"] == 12
    assert diagnostics["solver_final_nodes"] == 3


@pytest.mark.parametrize("index,value", [(0, -1.0), (4, 1.0e11), (6, 100.0), (1, np.nan)])
def test_column_guard_rejects_invalid_state_without_mutation(index, value):
    state = np.array([1., 1., 1., 1., 1., 1., 101325.])
    state[index] = value
    original = state.copy()
    with pytest.raises(ValueError):
        validate_scaled_state(state, np.ones(7), BoundedStateSettings())
    np.testing.assert_array_equal(state, original)


def test_column_guard_preserves_rhs_and_original_typed_failure(monkeypatch):
    from mea_absorption_column.Thermodynamics import thermo_models

    class PhysicalRejection(ValueError):
        pass

    rejection = PhysicalRejection("injected rejected thermodynamic state")
    diagnostics = make_solver_diagnostics()
    parameters = (np.ones(7), None, None, None, None, None, {"solver_diagnostics": diagnostics})
    state = np.array([1., 1., 1., 1., 1., 1., 101325.])
    original = state.copy()
    expected = np.arange(7, dtype=float)
    assert guard_column_rhs(0., state, parameters, lambda *a, **k: expected) is expected

    def reject(*args, **kwargs):
        raise rejection

    monkeypatch.setattr(thermo_models, "compute_fugacity", reject)

    def evaluate(*args, **kwargs):
        return guarded_compute_fugacity(
            "epcsaft_neutral", [.1, .05, .8, .05], [.02, .24, .74],
            [900., 9000., 30000.], 313.15, 320., 2.5e6, 101325., 12000.,
        )

    with pytest.raises(PhysicalRejection) as caught:
        guard_column_rhs(0., state, parameters, evaluate)
    assert caught.value is rejection
    assert diagnostics["invalid_state_count"] == 1
    assert str(rejection) == diagnostics["last_invalid_state"]
    np.testing.assert_array_equal(state, original)
    with pytest.raises(FloatingPointError, match="non-finite column RHS"):
        guard_column_rhs(0., state, parameters, lambda *a, **k: np.full(7, np.nan))


@pytest.mark.parametrize("values", [(np.nan, 1., 1., 1.), (-1., 1., 1., 1.)])
def test_fugacity_guard_rejects_invalid_outputs(monkeypatch, values):
    from mea_absorption_column.Thermodynamics import thermo_models
    monkeypatch.setattr(thermo_models, "compute_fugacity", lambda *a, **k: values)
    with pytest.raises(ValueError, match="finite nonnegative"):
        guarded_compute_fugacity(
            "epcsaft_neutral", [.1, .05, .8, .05], [.02, .24, .74],
            [900., 9000., 30000.], 313.15, 320., 2.5e6, 101325., 12000.,
        )


def test_benchmark_schema_contains_convergence_diagnostic_columns():
    for column in [
        "continuation_stage",
        "invalid_state_count",
        "jacobian_status",
        "scaling_mode",
        "transform_mode",
        "continuation_path",
        "domain_guard_counts",
        "first_failed_domain",
        "continuation_success",
        "profile_png",
    ]:
        assert column in BENCHMARK_COLUMNS


def test_structured_holdout_split_separates_bed_and_intercooler_patterns():
    df = pd.DataFrame(
        {
            "Beds": [1, 1, 2, 2, 3, 3],
            "Intercoolers": [0, 0, 1, 1, 2, 2],
            "CO2  %": [88.0, 89.0, 90.0, 91.0, 92.0, 93.0],
        },
        index=["A", "B", "C", "D", "E", "F"],
    )

    split = build_structured_holdout_split(df, holdout_fraction=0.5)

    assert set(split.train_case_ids).isdisjoint(split.holdout_case_ids)
    assert split.holdout_case_ids == ("B", "D", "F")


def test_calibration_artifact_rows_include_train_and_holdout_metrics():
    settings = CalibrationSettings(fit_factors={"mass_transfer": 1.0, "heat_transfer": 1.0})
    rows = calibration_artifact_rows(
        settings=settings,
        train_metrics={"capture_mae_pct": 1.2},
        holdout_metrics={"capture_mae_pct": 2.4},
    )

    assert {row["split"] for row in rows} == {"train", "holdout"}
    assert rows[0]["mass_transfer_factor"] == 1.0
    assert rows[1]["capture_mae_pct"] == 2.4


def test_write_calibration_artifacts_creates_split_and_metrics_files(tmp_path):
    results = pd.DataFrame(
        {
            "case_id": ["A", "B", "C", "D"],
            "beds": [1, 1, 2, 2],
            "intercoolers": [0, 0, 1, 1],
            "capture_error_pct": [1.0, 2.0, -3.0, -4.0],
            "temperature_rmse_K": [1.0, 2.0, 3.0, 4.0],
        }
    )

    paths = write_calibration_artifacts(results, tmp_path)

    assert paths["split"].exists()
    assert paths["metrics"].exists()
    metrics = pd.read_csv(paths["metrics"])
    assert set(metrics["split"]) == {"train", "holdout"}


def test_nccc_linear_capture_prediction_returns_finite_percent():
    df = pd.DataFrame(
        {
            "L": [3.8, 6.6, 1.8, 1.78],
            "G": [21.5, 21.5, 21.5, 21.5],
            "alpha": [0.145, 0.247, 0.091, 0.083],
            "w_MEA": [0.298, 0.312, 0.295, 0.310],
            "y_CO2": [0.1155, 0.1140, 0.1058, 0.1145],
            "Tl": [314.12, 313.67, 318.83, 319.87],
            "Tv": [315.63, 318.09, 316.82, 317.88],
            "CO2  %": [99.91, 99.49, 83.57, 78.10],
        },
        index=["K1", "K2", "K3", "K4"],
    )

    prediction = nccc_linear_capture_prediction(df, 0)

    assert 0.0 <= prediction <= 100.0


def test_epcsaft_ionic_fugacity_uses_six_species_liquid_state():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    assert MEA_THERMODYNAMICS_EPCSAFT_DATASET.exists()
    y = np.array([0.10, 0.08])
    x_true = np.array([1.0e-8, 0.055, 0.888, 0.028, 0.027, 0.001])
    Cl_true = np.array([1.0e-4, 2400.0, 39000.0, 1200.0, 1180.0, 20.0])

    values = compute_fugacity(
        "epcsaft_ionic",
        y,
        x_true,
        Cl_true,
        Tl=323.15,
        Tv=323.15,
        H_CO2_mix=1.0,
        P=109500.0,
        P_sat_H2O=12000.0,
    )

    assert len(values) == 4
    assert all(np.isfinite(value) and value > 0.0 for value in values)


def test_finite_and_shooting_success_gate_rejects_bad_boundary_residual():
    success, message = _apply_method_success_gates(
        method="finite",
        solver_success=True,
        message="The solution converged.",
        boundary_residual_norm=25.0,
        capture_error_pct=0.2,
        settings={},
    )

    assert success is False
    assert "strict boundary residual gate" in message


def test_collocation_success_gate_preserves_solver_success_with_acceptable_boundary_residual():
    success, message = _apply_method_success_gates(
        method="scipy-bvp",
        solver_success=True,
        message="ok",
        boundary_residual_norm=0.25,
        capture_error_pct=50.0,
        settings={},
    )

    assert success is True
    assert message == "ok"


def test_collocation_success_gate_rejects_bad_boundary_residual():
    success, message = _apply_method_success_gates(
        method="scipy-bvp",
        solver_success=True,
        message="ok",
        boundary_residual_norm=25.0,
        capture_error_pct=0.2,
        settings={},
    )

    assert success is False
    assert "strict boundary residual gate" in message


def test_collocation_low_residual_fallback_cannot_override_strict_boundary_rejection():
    success, message = _apply_method_success_gates(
        method="scipy-bvp",
        solver_success=False,
        message="A singular Jacobian encountered when solving the collocation system.",
        boundary_residual_norm=8.0,
        capture_error_pct=0.2,
        settings={"success_boundary_residual_max": 1.0, "accept_boundary_residual_max": 10.0},
    )

    assert success is False
    assert "strict boundary residual gate" in message
    assert "Accepted low-residual" not in message


def test_collocation_failure_stays_failed_at_low_residual():
    success, message = _apply_method_success_gates(
        method="scipy-bvp",
        solver_success=False,
        message="A singular Jacobian encountered when solving the collocation system.",
        boundary_residual_norm=0.04,
        capture_error_pct=0.1,
        settings={},
    )

    assert success is False
    assert "Accepted" not in message


def test_collocation_success_gate_rejects_large_capture_error_even_with_low_boundary_residual():
    success, message = _apply_method_success_gates(
        method="scipy-bvp",
        solver_success=False,
        message="A singular Jacobian encountered when solving the collocation system.",
        boundary_residual_norm=0.04,
        capture_error_pct=7.0,
        settings={},
    )

    assert success is False
    assert "Accepted low-residual" not in message


def test_collocation_success_gate_can_reject_converged_bad_capture():
    success, message = _apply_method_success_gates(
        method="scipy-bvp",
        solver_success=True,
        message="The algorithm converged.",
        boundary_residual_norm=0.0,
        capture_error_pct=12.0,
        settings={"success_capture_error_max_pct": 8.0},
    )

    assert success is False
    assert "collocation capture gate" in message


def test_collocation_success_gate_does_not_accept_timeout_final_iterate():
    success, message = _apply_method_success_gates(
        method="scipy-bvp",
        solver_success=False,
        message="Segmented SciPy BVP exceeded max_runtime_s=45",
        boundary_residual_norm=0.0,
        capture_error_pct=0.0,
        settings={"accept_low_residual_final_iterate": True},
    )

    assert success is False
    assert "Accepted low-residual" not in message


def test_absorption_only_flux_preserves_absorption_and_smoothly_caps_desorption():
    assert _smooth_absorption_only_vapor_flux(-1.0e-4) < -0.999e-4
    assert -1.0e-9 < _smooth_absorption_only_vapor_flux(1.0e-4) <= 0.0
