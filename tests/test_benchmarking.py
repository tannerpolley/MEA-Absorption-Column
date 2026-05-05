import pandas as pd
import json
from pathlib import Path

from mea_absorption_column.benchmark import (
    BENCHMARK_COLUMNS,
    BenchmarkSettings,
    load_case_data,
    parse_args,
    run_benchmark,
    _settings_to_payload,
    settings_from_payload,
)
from mea_absorption_column import benchmark_worker


def test_case_data_loads_from_packaged_csvs():
    c_cases, nccc_cases = load_case_data()

    assert len(c_cases) == 7
    assert len(nccc_cases) >= 20
    assert "CO2 %" in c_cases.columns
    assert "CO2  %" in nccc_cases.columns
    assert float(nccc_cases.iloc[0]["L"]) > 3.0


def test_benchmark_failure_rows_keep_stable_schema(monkeypatch):
    def failing_run_model(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr("mea_absorption_column.benchmark.run_model", failing_run_model)

    settings = BenchmarkSettings(
        methods=("single",),
        thermo_models=("ideal_henry",),
        c_case_limit=1,
        nccc_case_limit=0,
        write_artifacts=False,
    )
    results = run_benchmark(settings)

    assert list(results.columns) == BENCHMARK_COLUMNS
    assert len(results) == 1
    row = results.iloc[0]
    assert row["success"] is False or row["success"] == False
    assert "synthetic solver failure" in row["message"]
    assert row["beds"] == 1
    assert row["intercoolers"] == 0


def test_benchmark_result_columns_round_trip_to_csv(tmp_path, monkeypatch):
    def fake_run_model(df, method, thermo_model, run, **kwargs):
        return {
            "case_id": str(df.iloc[run, 0]),
            "method": method,
            "thermo_model": thermo_model,
            "success": True,
            "message": "ok",
            "runtime_s": 0.01,
            "capture_pct": 89.0,
            "capture_error_pct": -0.5,
            "temperature_rmse_K": 1.2,
            "boundary_residual_norm": 0.001,
            "mesh_points": 101,
            "tol": 0.1,
            "bc_tol": 0.001,
            "beds": 1,
            "intercoolers": 0,
            "staged_beds": False,
            "intercooler_model": "none",
            "intercooler_assumption": "none",
            "python_version": "test",
            "platform": "test",
            "package_versions": "numpy=test",
            "case_source": "C_cases_data",
        }

    monkeypatch.setattr("mea_absorption_column.benchmark.run_model", fake_run_model)
    settings = BenchmarkSettings(
        methods=("single",),
        thermo_models=("ideal_henry",),
        c_case_limit=1,
        nccc_case_limit=0,
        output_dir=tmp_path,
    )

    results = run_benchmark(settings)
    loaded = pd.read_csv(tmp_path / "benchmark_results.csv")

    assert list(results.columns) == BENCHMARK_COLUMNS
    assert list(loaded.columns) == BENCHMARK_COLUMNS


def test_benchmark_schema_includes_staged_bed_metadata():
    for column in [
        "beds",
        "intercoolers",
        "staged_beds",
        "intercooler_model",
        "intercooler_assumption",
        "mass_transfer_factor",
        "heat_transfer_factor",
        "intercooler_strength",
        "co2_flux_mode",
    ]:
        assert column in BENCHMARK_COLUMNS


def test_benchmark_cli_accepts_solver_settings():
    args = parse_args([
        "--mesh-points",
        "21",
        "--tol",
        "0.2",
        "--bc-tol",
        "0.01",
        "--max-nodes",
        "200",
        "--finite-jacobian",
        "--profile-pngs",
        "--subprocess-timeout-s",
        "30",
        "--nccc-case-ids",
        "K4",
        "K5",
        "--transform-mode",
        "positive_flow_pressure",
        "--co2-capture-guess-pct",
        "85",
        "--mass-transfer-factor",
        "0.5",
        "--heat-transfer-factor",
        "0.8",
        "--success-capture-error-max-pct",
        "8",
        "--capture-correction-model",
        "nccc_linear",
        "--multistart-capture-guesses",
        "60",
        "85",
        "--multistart-mass-transfer-factors",
        "0.26",
        "1.0",
        "--multistart-intercooler-strengths",
        "0.25",
        "1.0",
        "--multistart-co2-flux-modes",
        "bidirectional",
        "absorption_only",
    ])

    assert args.mesh_points == 21
    assert args.tol == 0.2
    assert args.bc_tol == 0.01
    assert args.max_nodes == 200
    assert args.finite_jacobian is True
    assert args.profile_pngs is True
    assert args.subprocess_timeout_s == 30
    assert args.nccc_case_ids == ["K4", "K5"]
    assert args.transform_mode == "positive_flow_pressure"
    assert args.co2_capture_guess_pct == 85
    assert args.mass_transfer_factor == 0.5
    assert args.heat_transfer_factor == 0.8
    assert args.success_capture_error_max_pct == 8
    assert args.capture_correction_model == "nccc_linear"
    assert args.multistart_capture_guesses == [60, 85]
    assert args.multistart_mass_transfer_factors == [0.26, 1.0]
    assert args.multistart_intercooler_strengths == [0.25, 1.0]
    assert args.multistart_co2_flux_modes == ["bidirectional", "absorption_only"]


def test_benchmark_multistart_selects_lowest_capture_error(tmp_path, monkeypatch):
    def fake_run_model(df, method, thermo_model, run, solver_settings, **kwargs):
        factor = solver_settings["mass_transfer_factor"]
        capture = 78.0 if factor == 0.26 else 100.0
        return {
            "case_id": "K4",
            "method": method,
            "thermo_model": thermo_model,
            "success": True,
            "message": "ok",
            "runtime_s": 1.0,
            "capture_pct": capture,
            "capture_error_pct": capture - 78.1,
            "temperature_rmse_K": None,
            "boundary_residual_norm": 0.0,
            "mesh_points": 7,
            "tol": 2,
            "bc_tol": 0.05,
            "beds": 3,
            "intercoolers": 2,
            "staged_beds": True,
            "intercooler_model": "liquid_temperature_reset",
            "intercooler_assumption": "Tl_feed_target",
            "python_version": "test",
            "platform": "test",
            "package_versions": "test",
        }

    monkeypatch.setattr("mea_absorption_column.benchmark.run_model", fake_run_model)
    settings = BenchmarkSettings(
        methods=("scipy-bvp",),
        thermo_models=("ideal_henry",),
        c_case_limit=0,
        nccc_case_limit=1,
        output_dir=tmp_path,
        solver_settings={
            "multistart_capture_guesses": (85,),
            "multistart_mass_transfer_factors": (0.26, 1.0),
            "multistart_intercooler_strengths": (0.25, 1.0),
            "multistart_co2_flux_modes": ("bidirectional",),
        },
    )

    results = run_benchmark(settings)

    assert results.loc[0, "capture_pct"] == 78.0
    assert results.loc[0, "mass_transfer_factor"] == 0.26
    assert results.loc[0, "intercooler_strength"] == 0.25
    assert results.loc[0, "co2_flux_mode"] == "bidirectional"
    assert "mass_transfer_factor=0.26" in results.loc[0, "continuation_path"]
    assert "intercooler_strength=0.25" in results.loc[0, "continuation_path"]
    assert "co2_flux_mode=bidirectional" in results.loc[0, "continuation_path"]


def test_benchmark_multistart_uses_candidate_subprocess_timeout(tmp_path, monkeypatch):
    calls = []

    def fake_candidate_subprocess(**kwargs):
        calls.append(kwargs["solver_settings_override"])
        factor = kwargs["solver_settings_override"]["mass_transfer_factor"]
        return {
            "case_id": "K4",
            "case_source": "NCCC_Data",
            "method": kwargs["method"],
            "thermo_model": kwargs["thermo_model"],
            "success": factor == 0.26,
            "message": "ok",
            "runtime_s": 1.0,
            "capture_pct": 78.0 if factor == 0.26 else 100.0,
            "capture_error_pct": -0.1 if factor == 0.26 else 21.9,
            "boundary_residual_norm": 0.0,
            "beds": 3,
            "intercoolers": 2,
            "staged_beds": True,
        }

    monkeypatch.setattr("mea_absorption_column.benchmark._run_solver_settings_subprocess", fake_candidate_subprocess)
    settings = BenchmarkSettings(
        methods=("scipy-bvp",),
        thermo_models=("ideal_henry",),
        c_case_limit=0,
        nccc_case_limit=1,
        output_dir=tmp_path,
        subprocess_timeout_s=30,
        solver_settings={
            "multistart_capture_guesses": (85,),
            "multistart_mass_transfer_factors": (0.26, 1.0),
            "multistart_intercooler_strengths": (0.25,),
            "multistart_co2_flux_modes": ("bidirectional",),
        },
    )

    results = run_benchmark(settings)

    assert len(calls) == 2
    assert results.loc[0, "success"] is True or results.loc[0, "success"] == True
    assert "mass_transfer_factor=0.26" in results.loc[0, "continuation_path"]
    assert calls[0]["co2_flux_mode"] == "bidirectional"


def test_benchmark_can_select_nccc_case_ids(monkeypatch):
    seen_runs = []

    def fake_run_model(df, method, thermo_model, run, **kwargs):
        seen_runs.append(str(df.index[run]))
        return {
            "case_id": str(df.index[run]),
            "method": method,
            "thermo_model": thermo_model,
            "success": True,
            "message": "ok",
            "runtime_s": 0.01,
            "capture_pct": 80.0,
            "capture_error_pct": 0.0,
            "boundary_residual_norm": 0.0,
            "beds": 3,
            "intercoolers": 2,
            "staged_beds": True,
            "intercooler_model": "liquid_temperature_reset",
            "intercooler_assumption": "Tl_feed_target",
            "python_version": "test",
            "platform": "test",
            "package_versions": "test",
        }

    monkeypatch.setattr("mea_absorption_column.benchmark.run_model", fake_run_model)
    settings = BenchmarkSettings(
        methods=("scipy-bvp",),
        thermo_models=("ideal_henry",),
        c_case_limit=0,
        nccc_case_limit=None,
        nccc_case_ids=("K4", "K5"),
        write_artifacts=False,
    )

    results = run_benchmark(settings)

    assert list(results["case_id"]) == ["K4", "K5"]
    assert seen_runs == ["K4", "K5"]


def test_benchmark_worker_preserves_filtered_case_ids(tmp_path, monkeypatch):
    seen = {}

    def fake_run_one_case_in_process(df, run, **kwargs):
        seen["case_id"] = str(df.index[run])
        return {
            "case_id": str(df.index[run]),
            "case_source": kwargs["case_source"],
            "method": kwargs["method"],
            "thermo_model": kwargs["thermo_model"],
            "success": True,
        }

    monkeypatch.setattr(benchmark_worker, "_run_one_case_in_process", fake_run_one_case_in_process)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    payload = {
        "case_source": "NCCC_Data",
        "run": 0,
        "method": "scipy-bvp",
        "thermo_model": "ideal_henry",
        "settings": _settings_to_payload(
            BenchmarkSettings(
                methods=("scipy-bvp",),
                thermo_models=("ideal_henry",),
                c_case_limit=0,
                nccc_case_ids=("K4",),
                write_artifacts=False,
            )
        ),
        "output_path": str(output_path),
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    benchmark_worker.main([str(input_path)])

    assert seen["case_id"] == "K4"


def test_benchmark_schema_includes_capture_correction_columns():
    assert "raw_capture_pct" in BENCHMARK_COLUMNS
    assert "capture_correction_model" in BENCHMARK_COLUMNS


def test_benchmark_settings_round_trip_for_worker_payload(tmp_path):
    settings = BenchmarkSettings(
        methods=("scipy-bvp",),
        thermo_models=("ideal_henry",),
        output_dir=tmp_path,
        c_case_limit=0,
        nccc_case_limit=2,
        staged_beds="auto",
        solver_settings={"mesh_points": 5, "co2_flux_mode": "absorption_only"},
        profile_pngs=True,
        subprocess_timeout_s=30,
    )

    restored = settings_from_payload(_settings_to_payload(settings))

    assert restored.methods == ("scipy-bvp",)
    assert restored.thermo_models == ("ideal_henry",)
    assert restored.output_dir == Path(tmp_path)
    assert restored.solver_settings["co2_flux_mode"] == "absorption_only"
    assert restored.profile_pngs is True
    assert restored.subprocess_timeout_s is None
