import pandas as pd
import json
from pathlib import Path

import pytest

from mea_absorption_column.BVP.ABS_Column import _gas_velocity_area_factor
from mea_absorption_column.config.Constants import MWs_v
from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.Properties.Thermophysical_Properties import vapor_pressure
from mea_absorption_column.benchmark import (
    BENCHMARK_COLUMNS,
    BenchmarkSettings,
    load_case_data,
    parse_args,
    run_benchmark,
    _solver_settings_from_args,
    _settings_to_payload,
    settings_from_payload,
)
from mea_absorption_column import benchmark_worker


def test_case_data_loads_from_packaged_csvs():
    c_cases, nccc_cases, srp_cases = load_case_data()

    assert len(c_cases) == 7
    assert len(nccc_cases) >= 20
    assert len(srp_cases) >= 1
    assert "CO2 %" in c_cases.columns
    assert "CO2  %" in nccc_cases.columns
    assert "case_note" in srp_cases.columns
    assert float(nccc_cases.iloc[0]["L"]) > 3.0
    assert float(srp_cases.iloc[0]["L_G"]) == 7.0


def test_benchmark_defaults_preserve_henry_only_baseline():
    settings = BenchmarkSettings()

    assert settings.thermo_models == ("ideal_henry",)


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


def test_benchmark_can_run_srp_method_case_source(monkeypatch):
    seen = {}

    def fake_run_model(df, method, thermo_model, run, **kwargs):
        seen["case_id"] = str(df.index[run])
        seen["columns"] = tuple(df.columns)
        return {
            "case_id": str(df.index[run]),
            "method": method,
            "thermo_model": thermo_model,
            "success": True,
            "message": "ok",
            "runtime_s": 0.02,
            "capture_pct": 90.0,
            "capture_error_pct": None,
            "temperature_rmse_K": None,
            "boundary_residual_norm": 0.0,
            "boundary_residual_components": "{}",
            "beds": 1,
            "intercoolers": 0,
            "staged_beds": False,
            "intercooler_model": "none",
            "intercooler_assumption": "none",
            "python_version": "test",
            "platform": "test",
            "package_versions": "numpy=test",
        }

    monkeypatch.setattr("mea_absorption_column.benchmark.run_model", fake_run_model)
    settings = BenchmarkSettings(
        methods=("single",),
        thermo_models=("ideal_henry",),
        c_case_limit=0,
        nccc_case_limit=0,
        srp_case_limit=None,
        write_artifacts=False,
    )

    results = run_benchmark(settings)

    assert len(results) == 1
    assert results.loc[0, "case_source"] == "SRP_method_cases"
    assert seen["case_id"] == "SRP-LG7"
    assert "L_G" in seen["columns"]


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
            "boundary_residual_components": '{"Fv_CO2_pct": 0.0}',
            "mesh_points": 101,
            "tol": 0.1,
            "bc_tol": 0.001,
            "max_nodes": 250,
            "co2_capture_guess_pct": 88.0,
            "h2o_capture_guess_pct": -90.0,
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
        solver_settings={
            "co2_vapor_upper_factor": 1.05,
            "success_boundary_residual_max": 0.5,
            "eta_psi": 0.5,
            "max_nodes": 300,
            "co2_capture_guess_pct": 91.0,
            "h2o_capture_guess_pct": -80.0,
        },
    )

    results = run_benchmark(settings)
    loaded = pd.read_csv(tmp_path / "benchmark_results.csv")

    assert list(results.columns) == BENCHMARK_COLUMNS
    assert list(loaded.columns) == BENCHMARK_COLUMNS
    assert loaded.loc[0, "co2_vapor_upper_factor"] == 1.05
    assert loaded.loc[0, "success_boundary_residual_max"] == 0.5
    assert loaded.loc[0, "eta_psi"] == 0.5
    assert loaded.loc[0, "max_nodes"] == 300
    assert loaded.loc[0, "co2_capture_guess_pct"] == 91.0
    assert loaded.loc[0, "h2o_capture_guess_pct"] == -80.0


def test_benchmark_schema_includes_staged_bed_metadata():
    for column in [
        "beds",
        "intercoolers",
        "staged_beds",
        "intercooler_model",
        "thermal_state_mode",
        "intercooler_assumption",
        "mass_transfer_factor",
        "heat_transfer_factor",
        "intercooler_strength",
        "co2_flux_mode",
    ]:
        assert column in BENCHMARK_COLUMNS


def test_benchmark_schema_includes_boundary_residual_components():
    assert "boundary_residual_components" in BENCHMARK_COLUMNS
    assert "success_boundary_residual_max" in BENCHMARK_COLUMNS
    assert "max_nodes" in BENCHMARK_COLUMNS
    assert "co2_capture_guess_pct" in BENCHMARK_COLUMNS
    assert "h2o_capture_guess_pct" in BENCHMARK_COLUMNS


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
        "--profile-csvs",
        "--subprocess-timeout-s",
        "30",
        "--c-case-dataset",
        "campaign",
        "--nccc-case-ids",
        "K4",
        "K5",
        "--srp-case-limit",
        "1",
        "--srp-case-ids",
        "SRP-LG7",
        "--transform-mode",
        "positive_flow_pressure",
        "--co2-vapor-upper-factor",
        "1.05",
        "--vapor-composition-mode",
        "input_o2",
        "--gas-flow-basis",
        "reported_dry_mass",
        "--gas-velocity-area-exponent",
        "1.5",
        "--gas-velocity-area-reference-m-s",
        "1.88",
        "--shooting-integrator",
        "bdf",
        "--shooting-root-method",
        "hybr",
        "--co2-capture-guess-pct",
        "85",
        "--h2o-capture-guess-pct",
        "-90",
        "--eta-psi",
        "1.0",
        "--mass-transfer-factor",
        "0.5",
        "--heat-transfer-factor",
        "0.8",
        "--intercooler-model",
        "pumparound_temperature_approach",
        "--success-boundary-residual-max",
        "0.5",
        "--success-capture-error-max-pct",
        "8",
    ])

    assert args.mesh_points == 21
    assert args.tol == 0.2
    assert args.bc_tol == 0.01
    assert args.max_nodes == 200
    assert args.finite_jacobian is True
    assert args.profile_pngs is True
    assert args.profile_csvs is True
    assert args.subprocess_timeout_s == 30
    assert args.c_case_dataset == "campaign"
    assert args.nccc_case_ids == ["K4", "K5"]
    assert args.srp_case_limit == 1
    assert args.srp_case_ids == ["SRP-LG7"]
    assert args.transform_mode == "positive_flow_pressure"
    assert args.co2_vapor_upper_factor == 1.05
    assert args.vapor_composition_mode == "input_o2"
    assert args.gas_flow_basis == "reported_dry_mass"
    assert args.gas_velocity_area_exponent == 1.5
    assert args.gas_velocity_area_reference_m_s == 1.88
    assert args.shooting_integrator == "bdf"
    assert args.shooting_root_method == "hybr"
    assert args.co2_capture_guess_pct == 85
    assert args.h2o_capture_guess_pct == -90
    assert args.eta_psi == 1.0
    assert args.mass_transfer_factor == 0.5
    assert args.heat_transfer_factor == 0.8
    assert args.intercooler_model == "pumparound_temperature_approach"
    assert args.success_boundary_residual_max == 0.5
    assert args.success_capture_error_max_pct == 8

    solver_settings = _solver_settings_from_args(args)
    assert solver_settings["integrator"] == "bdf"
    assert solver_settings["ivp_method"] == "BDF"
    assert solver_settings["root_method"] == "hybr"
    assert solver_settings["co2_capture_guess_pct"] == 85
    assert solver_settings["h2o_capture_guess_pct"] == -90
    assert solver_settings["eta_psi"] == 1.0
    assert solver_settings["intercooler_model"] == "pumparound_temperature_approach"
    assert solver_settings["vapor_composition_mode"] == "input_o2"
    assert solver_settings["gas_flow_basis"] == "reported_dry_mass"
    assert solver_settings["gas_velocity_area_exponent"] == 1.5
    assert solver_settings["gas_velocity_area_reference_m_s"] == 1.88


def test_convert_data_defaults_to_legacy_vapor_reconstruction():
    c_cases, _, _ = load_case_data()

    inputs, _, metadata = convert_data(c_cases, run=0, return_metadata=True)
    _, vapor_flows, *_ = inputs
    y = [flow / sum(vapor_flows) for flow in vapor_flows]

    assert metadata["vapor_composition_mode"] == "legacy_ratio"
    assert round(y[3], 6) == round(0.06655280890171553, 6)


def test_load_case_data_can_use_campaign_c_case_inputs():
    legacy_cases, _, _ = load_case_data()
    campaign_cases, _, _ = load_case_data(c_case_dataset="campaign")

    assert legacy_cases.loc["1C", "alpha"] == pytest.approx(0.25)
    assert campaign_cases.loc["1C", "alpha"] == pytest.approx(0.15)
    assert campaign_cases.loc["7C", "alpha"] == pytest.approx(0.34)
    assert campaign_cases.loc["7C", "L/G"] == pytest.approx(4.628230060332862)


def test_load_case_data_can_use_year_split_nccc_inputs():
    _, nccc_2014, _ = load_case_data(nccc_dataset="2014")
    _, nccc_2017, _ = load_case_data(nccc_dataset="2017")

    assert {"K13", "K17", "K18", "K19", "K20", "K21"}.issubset(nccc_2014.index)
    assert {"1C", "2C", "3C", "7C", "1D", "4D"}.issubset(nccc_2017.index)
    assert nccc_2014.loc["K18", "Intercoolers"] == 0
    assert nccc_2017.loc["1C", "Intercoolers"] == 0
    assert nccc_2017.loc["1C", "Tl"] == pytest.approx(318.15)
    assert nccc_2017.loc["2C", "Tl"] == pytest.approx(318.15)
    assert nccc_2017.loc["3C", "Tl"] == pytest.approx(318.15)
    assert nccc_2017.loc["3D", "Tl"] == pytest.approx(318.15)
    assert bool(nccc_2017.loc["1C", "lean_solvent_temp_imputed"]) is True
    assert bool(nccc_2017.loc["3D", "lean_solvent_temp_imputed"]) is True
    assert nccc_2017.loc["2C", "y_CO2"] == pytest.approx(0.077)


def test_convert_data_can_use_input_o2_column():
    c_cases, _, _ = load_case_data()

    inputs, _, metadata = convert_data(
        c_cases,
        run=0,
        return_metadata=True,
        vapor_composition_mode="input_o2",
    )
    _, vapor_flows, *_ = inputs
    y = [flow / sum(vapor_flows) for flow in vapor_flows]

    assert metadata["vapor_composition_mode"] == "input_o2"
    assert round(y[3], 6) == round(float(c_cases.iloc[0]["y_O2"]), 6)


def test_convert_data_can_convert_dry_gas_to_saturated_wet_basis():
    _, nccc_2017, _ = load_case_data(nccc_dataset="2017")

    inputs, _, metadata = convert_data(
        nccc_2017,
        run=nccc_2017.index.get_loc("1C"),
        type="mass",
        return_metadata=True,
        vapor_composition_mode="dry_saturated",
    )
    _, vapor_flows, *_ = inputs
    y = [flow / sum(vapor_flows) for flow in vapor_flows]
    row = nccc_2017.loc["1C"]
    y_h2o = vapor_pressure(row["Tv"]) / row["P"]

    assert metadata["vapor_composition_mode"] == "dry_saturated"
    assert y[0] == pytest.approx(row["y_CO2"] * (1.0 - y_h2o))
    assert y[1] == pytest.approx(y_h2o)
    assert y[3] == pytest.approx(row["y_O2"] * (1.0 - y_h2o))


def test_convert_data_can_treat_reported_gas_flow_as_dry_mass():
    _, nccc_2017, _ = load_case_data(nccc_dataset="2017")

    inputs, _, metadata = convert_data(
        nccc_2017,
        run=nccc_2017.index.get_loc("1C"),
        type="mass",
        return_metadata=True,
        vapor_composition_mode="dry_saturated",
        gas_flow_basis="reported_dry_mass",
    )
    _, vapor_flows, *_ = inputs
    row = nccc_2017.loc["1C"]
    dry_mass = (
        vapor_flows[0] * MWs_v[0]
        + vapor_flows[2] * MWs_v[2]
        + vapor_flows[3] * MWs_v[3]
    )
    wet_mass = dry_mass + vapor_flows[1] * MWs_v[1]

    assert metadata["gas_flow_basis"] == "reported_dry_mass"
    assert dry_mass == pytest.approx(row["G"])
    assert wet_mass > row["G"]


def test_convert_data_mass_inputs_preserve_loaded_lean_solvent_mass():
    _, nccc_2017, _ = load_case_data(nccc_dataset="2017")

    inputs, _, _ = convert_data(
        nccc_2017,
        run=nccc_2017.index.get_loc("6C"),
        type="mass",
        return_metadata=True,
        vapor_composition_mode="dry_saturated",
    )
    liquid_flows, *_ = inputs
    loaded_mass = (
        liquid_flows[0] * 0.04401
        + liquid_flows[1] * 0.06108
        + liquid_flows[2] * 0.01802
    )

    assert loaded_mass == pytest.approx(nccc_2017.loc["6C", "L"])


def test_gas_velocity_area_factor_is_bounded_and_reference_normalized():
    assert _gas_velocity_area_factor(1.88, 1.88, 2.0, (0.1, 3.0)) == pytest.approx(1.0)
    assert _gas_velocity_area_factor(3.76, 1.88, 2.0, (0.1, 3.0)) == pytest.approx(3.0)
    assert _gas_velocity_area_factor(0.188, 1.88, 2.0, (0.1, 3.0)) == pytest.approx(0.1)


def test_gas_velocity_area_factor_rejects_nonpositive_reference():
    with pytest.raises(ValueError, match="reference"):
        _gas_velocity_area_factor(1.0, 0.0, 1.0, (0.1, 3.0))


def test_benchmark_writes_profile_csvs_when_requested(tmp_path, monkeypatch):
    def fake_run_model(df, method, thermo_model, run, solver_settings, **kwargs):
        assert solver_settings["return_profiles"] is True
        return {
            "case_id": "3C",
            "method": method,
            "thermo_model": thermo_model,
            "success": True,
            "message": "ok",
            "runtime_s": 0.01,
            "capture_pct": 89.0,
            "capture_error_pct": -0.5,
            "temperature_rmse_K": 1.2,
            "boundary_residual_norm": 0.001,
            "boundary_residual_components": '{"Fv_CO2_pct": 0.0}',
            "mesh_points": 101,
            "tol": 0.1,
            "bc_tol": 0.001,
            "max_nodes": 250,
            "co2_capture_guess_pct": 88.0,
            "h2o_capture_guess_pct": -90.0,
            "beds": 1,
            "intercoolers": 0,
            "staged_beds": False,
            "intercooler_model": "none",
            "intercooler_assumption": "none",
            "python_version": "test",
            "platform": "test",
            "package_versions": "numpy=test",
            "_case_metadata": {
                "case_id": "3C",
                "beds": 1,
                "intercoolers": 0,
                "single_bed_height_m": 10.0,
                "total_packed_height_m": 10.0,
            },
            "_profiles": {
                "T": pd.DataFrame({"Tl": [320.0, 315.0], "Tv": [318.0, 316.0]}, index=[1.0, 0.0]),
                "CO2": pd.DataFrame({"DF_CO2": [1.2, 0.8]}, index=[1.0, 0.0]),
            },
        }

    monkeypatch.setattr("mea_absorption_column.benchmark.run_model", fake_run_model)
    settings = BenchmarkSettings(
        methods=("scipy-bvp",),
        thermo_models=("ideal_henry",),
        c_case_ids=("3C",),
        nccc_case_limit=0,
        output_dir=tmp_path,
        profile_csvs=True,
    )

    results = run_benchmark(settings)
    row = results.iloc[0]
    profile_dir = Path(row["profile_csv_dir"])

    assert row["profile_csv_status"] == "written"
    assert (profile_dir / "T.csv").exists()
    assert (profile_dir / "CO2.csv").exists()
    assert (profile_dir / "profile_manifest.json").exists()
    assert (profile_dir / "run_spec.json").exists()
    manifest = json.loads((profile_dir / "profile_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_s"] == 0.01
    assert manifest["runtime_label"] == "0.01 s"
    t_csv = pd.read_csv(profile_dir / "T.csv")
    assert t_csv.columns.tolist()[:4] == [
        "Position",
        "height_m",
        "bed_id",
        "bed_position_m",
    ]
    assert t_csv.loc[0, "Position"] == 0.0
    assert t_csv.loc[0, "Tl"] == 315.0
    assert t_csv.loc[1, "Position"] == 1.0
    assert t_csv.loc[1, "Tl"] == 320.0


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


def test_benchmark_writes_profile_png_for_failed_last_iterate(monkeypatch):
    written = []

    def fake_run_model(*args, **kwargs):
        return {
            "case_id": "3C",
            "method": "scipy-bvp",
            "thermo_model": "ideal_henry",
            "success": False,
            "message": "diagnostic last iterate",
            "capture_pct": 50.0,
            "capture_error_pct": -40.0,
            "_profiles": {"T": pd.DataFrame({"Tl": [315.0], "Tv": [316.0]})},
        }

    def fake_write_profile_png(result, output_dir, case_source):
        written.append((result["case_id"], case_source))
        return str(Path(output_dir) / "diagnostic.png")

    monkeypatch.setattr("mea_absorption_column.benchmark.run_model", fake_run_model)
    monkeypatch.setattr("mea_absorption_column.benchmark._write_profile_png", fake_write_profile_png)

    settings = BenchmarkSettings(
        methods=("scipy-bvp",),
        thermo_models=("ideal_henry",),
        output_dir=Path("unused-output"),
        c_case_limit=1,
        nccc_case_limit=0,
        profile_pngs=True,
        write_artifacts=False,
    )

    results = run_benchmark(settings)

    assert written == [("3C", "C_cases_data")]
    assert results.loc[0, "profile_png"].endswith("diagnostic.png")


def test_benchmark_settings_round_trip_for_worker_payload(tmp_path):
    settings = BenchmarkSettings(
        methods=("scipy-bvp",),
        thermo_models=("ideal_henry",),
        output_dir=tmp_path,
        c_case_limit=0,
        nccc_case_limit=2,
        srp_case_limit=1,
        srp_case_ids=("SRP-LG7",),
        staged_beds="auto",
        solver_settings={"mesh_points": 5, "co2_flux_mode": "absorption_only"},
        profile_pngs=True,
        profile_csvs=True,
        subprocess_timeout_s=30,
    )

    restored = settings_from_payload(_settings_to_payload(settings))

    assert restored.methods == ("scipy-bvp",)
    assert restored.thermo_models == ("ideal_henry",)
    assert restored.output_dir == Path(tmp_path)
    assert restored.srp_case_limit == 1
    assert restored.srp_case_ids == ("SRP-LG7",)
    assert restored.solver_settings["co2_flux_mode"] == "absorption_only"
    assert restored.profile_pngs is True
    assert restored.profile_csvs is True
    assert restored.subprocess_timeout_s is None
