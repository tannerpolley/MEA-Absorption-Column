import math

import numpy as np
import pandas as pd
import pytest

from mea_absorption_column.benchmark import BENCHMARK_COLUMNS
from mea_absorption_column.Run_Model import (
    _apply_method_success_gates,
    _fallback_temperature_profile,
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
    sanitize_scaled_state,
    unbounded_to_positive,
)
from mea_absorption_column.continuation import ContinuationStep, run_absorber_continuation, run_continuation_ladder
from mea_absorption_column.calibration import (
    CalibrationSettings,
    build_structured_holdout_split,
    calibration_artifact_rows,
    nccc_linear_capture_prediction,
    write_calibration_artifacts,
)
from mea_absorption_column.uq import UQPlan, estimate_two_tier_throughput
from mea_absorption_column.Thermodynamics.thermo_models import guarded_compute_fugacity
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    clear_epcsaft_phi_cache,
    compute_fugacity,
    ensure_epcsaft_importable,
    epcsaft_cache_stats,
    epcsaft_phi_co2,
    epcsaft_phi_co2_batch,
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


def test_sanitize_scaled_state_clips_flows_and_pressure_without_mutating_input():
    original = np.array([-1.0, -0.5, 0.0, -0.25, 1.0e5, 2.0e5, -10.0])
    scales = np.array([10.0, 10.0, 2.0, 2.0, 1.0e6, 1.0e6, 1.0e5])

    sanitized, report = sanitize_scaled_state(original, scales, BoundedStateSettings())

    assert np.array_equal(original, np.array([-1.0, -0.5, 0.0, -0.25, 1.0e5, 2.0e5, -10.0]))
    assert report.invalid is True
    assert np.all((sanitized * scales)[[0, 1, 2, 3]] > 0.0)
    assert (sanitized * scales)[6] > 0.0


def test_guard_column_rhs_records_penalty_instead_of_raising():
    diagnostics = make_solver_diagnostics()
    parameters = (
        np.ones(7),
        np.ones(7),
        (1.0, 1.0, 0.1),
        1.0,
        1.0,
        (250.0, 0.97, 1.0, 1.0, 1.0, 1.0, 1.0),
        {"solver_diagnostics": diagnostics},
    )

    rhs = guard_column_rhs(
        zi=0.0,
        y_scaled=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000.0]),
        parameters=parameters,
        evaluator=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic invalid state")),
    )

    assert rhs.shape == (7,)
    assert np.all(np.isfinite(rhs))
    assert diagnostics["invalid_state_count"] == 1
    assert diagnostics["guard_penalty_count"] == 1
    assert "synthetic invalid state" in diagnostics["last_invalid_state"]


def test_guarded_epcsaft_invalid_state_returns_structured_penalty():
    diagnostics = make_solver_diagnostics()

    result = guarded_compute_fugacity(
        "epcsaft_neutral",
        y=[np.nan, 0.05, 0.8, 0.1],
        x_true=[0.02, 0.24, 0.62, 0.06, 0.05, 0.01],
        Cl_true=[900.0, 9000.0, 30000.0, 1800.0, 1500.0, 200.0],
        Tl=50.0,
        Tv=320.0,
        H_CO2_mix=2.5e6,
        P=109500.0,
        P_sat_H2O=12000.0,
        diagnostics=diagnostics,
    )

    assert len(result) == 4
    assert all(math.isfinite(value) for value in result)
    assert diagnostics["invalid_state_count"] == 1
    assert diagnostics["guard_penalty_count"] == 1


def test_benchmark_schema_contains_convergence_diagnostic_columns():
    for column in [
        "continuation_stage",
        "invalid_state_count",
        "guard_penalty_count",
        "jacobian_status",
        "scaling_mode",
        "transform_mode",
        "continuation_path",
        "domain_guard_counts",
        "first_failed_domain",
        "continuation_success",
        "epcsaft_cache_hits",
        "epcsaft_cache_misses",
        "epcsaft_direct_density_solve_s",
        "epcsaft_rho_guess_hits",
        "epcsaft_rho_guess_misses",
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


def test_two_tier_uq_throughput_estimate_uses_cache_and_surrogate_fractions():
    plan = UQPlan(reference_runtime_s=10.0, cached_runtime_s=2.0, surrogate_runtime_s=0.05)

    estimate = estimate_two_tier_throughput(plan, samples=100, reference_fraction=0.1, cache_fraction=0.4)

    assert estimate["samples"] == 100
    assert estimate["estimated_total_runtime_s"] < 100 * plan.reference_runtime_s
    assert estimate["reference_samples"] == 10


def test_epcsaft_phi_cache_records_hit_after_repeated_call():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    clear_epcsaft_phi_cache()
    composition = np.array([0.02, 0.24, 0.74])

    first = epcsaft_phi_co2(323.15, 109500.0, composition, phase="liq", cache=True)
    second = epcsaft_phi_co2(323.15, 109500.0, composition, phase="liq", cache=True)
    stats = epcsaft_cache_stats()

    assert first == second
    assert stats["epcsaft_cache_hits"] >= 1
    assert stats["epcsaft_cache_misses"] == 1


def test_epcsaft_phi_cache_uses_coarse_bvp_quantization():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    clear_epcsaft_phi_cache()
    first_composition = np.array([0.0200001, 0.2400001, 0.7399998])
    second_composition = np.array([0.0200002, 0.2400002, 0.7399996])

    first = epcsaft_phi_co2(323.151, 109501.0, first_composition, phase="liq", cache=True)
    second = epcsaft_phi_co2(323.152, 109504.0, second_composition, phase="liq", cache=True)
    stats = epcsaft_cache_stats()

    assert first == second
    assert stats["epcsaft_cache_hits"] >= 1
    assert stats["epcsaft_cache_misses"] == 1


def test_epcsaft_phi_batch_deduplicates_near_duplicate_states():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    clear_epcsaft_phi_cache()
    records = [
        {
            "T": 323.151,
            "P": 109501.0,
            "composition": np.array([0.02, 0.24, 0.74]),
            "phase": "liq",
        },
        {
            "T": 323.152,
            "P": 109504.0,
            "composition": np.array([0.0200001, 0.2400001, 0.7399998]),
            "phase": "liq",
        },
    ]

    values = epcsaft_phi_co2_batch(records)
    stats = epcsaft_cache_stats()

    assert len(values) == 2
    assert values[0] == values[1]
    assert stats["epcsaft_cache_misses"] == 1


def test_epcsaft_pressure_state_reuses_density_guess_after_first_miss():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    clear_epcsaft_phi_cache()

    first = epcsaft_phi_co2(323.15, 109500.0, np.array([0.02, 0.24, 0.74]), phase="liq", cache=True)
    second = epcsaft_phi_co2(324.35, 109500.0, np.array([0.021, 0.239, 0.74]), phase="liq", cache=True)
    stats = epcsaft_cache_stats()

    assert math.isfinite(first)
    assert math.isfinite(second)
    assert stats["epcsaft_cache_misses"] == 2
    assert stats["epcsaft_rho_guess_misses"] == 1
    assert stats["epcsaft_rho_guess_hits"] == 1


def test_epcsaft_ionic_fugacity_uses_six_species_liquid_state():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    assert MEA_THERMODYNAMICS_EPCSAFT_DATASET.exists()
    clear_epcsaft_phi_cache()
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


def test_collocation_success_gate_accepts_low_residual_final_iterate():
    success, message = _apply_method_success_gates(
        method="scipy-bvp",
        solver_success=False,
        message="A singular Jacobian encountered when solving the collocation system.",
        boundary_residual_norm=0.04,
        capture_error_pct=0.1,
        settings={},
    )

    assert success is True
    assert "Accepted low-residual collocation final iterate" in message


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


def test_collocation_low_residual_acceptance_uses_explicit_capture_gate():
    success, message = _apply_method_success_gates(
        method="scipy-bvp",
        solver_success=False,
        message="A singular Jacobian encountered when solving the collocation system.",
        boundary_residual_norm=0.04,
        capture_error_pct=6.5,
        settings={"success_capture_error_max_pct": 8.0},
    )

    assert success is True
    assert "Accepted low-residual collocation final iterate" in message


def test_temperature_profile_fallback_uses_solved_temperature_states():
    Y = np.zeros((7, 3))
    Y[4] = [314.0, 318.0, 321.0]
    Y[5] = [316.0, 319.0, 322.0]

    profiles = _fallback_temperature_profile(
        Y,
        np.array([0.0, 1.0, 2.0]),
        thermal_state_mode="temperature",
        Fl_MEA=1.0,
        Fv_N2=1.0,
        Fv_O2=0.1,
    )

    assert list(profiles) == ["T"]
    assert profiles["T"]["Tl"].tolist() == [314.0, 318.0, 321.0]
    assert profiles["T"]["Tv"].tolist() == [316.0, 319.0, 322.0]


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


def test_continuation_ladder_stops_after_failed_stage():
    calls = []

    def runner(step):
        calls.append(step.name)
        return {"success": step.name == "one_bed_henry", "message": step.name}

    result = run_continuation_ladder(
        [
            ContinuationStep(name="one_bed_henry", thermo_model="ideal_henry", staged_beds=False),
            ContinuationStep(name="staged_henry", thermo_model="ideal_henry", staged_beds=True),
            ContinuationStep(name="epcsaft", thermo_model="epcsaft_neutral", staged_beds=True),
        ],
        runner=runner,
    )

    assert calls == ["one_bed_henry", "staged_henry"]
    assert result.success is False
    assert result.failed_stage == "staged_henry"


def test_continuation_ladder_continues_after_optional_failed_stage():
    calls = []

    def runner(step):
        calls.append(step.name)
        return {"success": step.name != "optional_seed", "message": step.name}

    result = run_continuation_ladder(
        [
            ContinuationStep(
                name="optional_seed",
                thermo_model="ideal_henry",
                staged_beds=False,
                required=False,
            ),
            ContinuationStep(name="required_stage", thermo_model="ideal_henry", staged_beds=True),
        ],
        runner=runner,
    )

    assert calls == ["optional_seed", "required_stage"]
    assert result.success is True
    assert result.failed_stage == ""


def test_run_absorber_continuation_passes_step_settings_to_run_model():
    calls = []

    def fake_run_model(df, **kwargs):
        calls.append(kwargs)
        return {"success": True, "case_id": "synthetic", "message": "ok"}

    result = run_absorber_continuation(
        df=object(),
        run=0,
        method="scipy-bvp",
        steps=[
            ContinuationStep(
                name="one_bed_henry",
                thermo_model="ideal_henry",
                staged_beds=False,
                solver_settings={"transform_mode": "positive_flow_pressure"},
            )
        ],
        run_model_func=fake_run_model,
    )

    assert result.success is True
    assert calls[0]["thermo_model"] == "ideal_henry"
    assert calls[0]["staged_beds"] is False
    assert calls[0]["solver_settings"]["continuation_stage"] == "one_bed_henry"
    assert calls[0]["solver_settings"]["transform_mode"] == "positive_flow_pressure"


def test_run_absorber_continuation_warm_starts_next_stage():
    calls = []
    seed_profile = np.ones((7, 5))

    def fake_run_model(df, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "success": True,
                "case_id": "synthetic",
                "message": "ok",
                "_raw_solution_scaled": seed_profile,
            }
        return {"success": True, "case_id": "synthetic", "message": "ok"}

    result = run_absorber_continuation(
        df=object(),
        run=0,
        method="scipy-bvp",
        steps=[
            ContinuationStep(name="one_bed_henry", thermo_model="ideal_henry", staged_beds=False),
            ContinuationStep(name="staged_henry", thermo_model="ideal_henry", staged_beds=True),
        ],
        run_model_func=fake_run_model,
    )

    assert result.success is True
    assert calls[0]["solver_settings"]["return_internal_profile"] is True
    assert calls[1]["solver_settings"]["initial_guess_scaled"] is seed_profile
    assert "_raw_solution_scaled" not in result.rows[0]


def test_run_absorber_continuation_can_seed_from_capture_close_failed_stage():
    calls = []
    seed_profile = np.ones((7, 5))

    def fake_run_model(df, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "success": False,
                "case_id": "synthetic",
                "message": "capture close but boundary failed",
                "capture_error_pct": 1.0,
                "_raw_solution_scaled": seed_profile,
            }
        return {"success": True, "case_id": "synthetic", "message": "ok"}

    result = run_absorber_continuation(
        df=object(),
        run=0,
        method="scipy-bvp",
        steps=[
            ContinuationStep(
                name="optional_seed",
                thermo_model="ideal_henry",
                staged_beds=False,
                required=False,
            ),
            ContinuationStep(name="required_stage", thermo_model="ideal_henry", staged_beds=True),
        ],
        run_model_func=fake_run_model,
    )

    assert result.success is True
    assert calls[1]["solver_settings"]["initial_guess_scaled"] is seed_profile


def test_run_model_accepts_documented_calibration_factor_settings():
    df = pd.DataFrame(
        {
            "L": [3.8],
            "G": [21.5],
            "alpha": [0.145],
            "w_MEA": [0.298],
            "y_CO2": [0.1155],
            "Tl": [314.12],
            "Tv": [315.63],
            "P": [108820],
            "Beds": [1],
            "Intercoolers": [0],
            "CO2  %": [99.91],
        },
        index=["synthetic"],
    )

    calls = []

    def fake_run_model(df, **kwargs):
        calls.append(kwargs)
        return {"success": True, "case_id": "synthetic", "message": "ok"}

    result = run_absorber_continuation(
        df=df,
        run=0,
        method="scipy-bvp",
        steps=[
            ContinuationStep(
                name="calibrated",
                thermo_model="ideal_henry",
                staged_beds=False,
                solver_settings={"mass_transfer_factor": 0.5, "heat_transfer_factor": 0.8},
            )
        ],
        run_model_func=fake_run_model,
    )

    assert result.success is True
    assert calls[0]["solver_settings"]["mass_transfer_factor"] == 0.5
    assert calls[0]["solver_settings"]["heat_transfer_factor"] == 0.8
