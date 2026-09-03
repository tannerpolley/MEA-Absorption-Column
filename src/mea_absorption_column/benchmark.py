from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from importlib import resources
from pathlib import Path

import pandas as pd

from mea_absorption_column.Run_Model import run_model
from mea_absorption_column.calibration import write_calibration_artifacts
from mea_absorption_column.misc.Save_Run_Outputs import build_profile_coordinate_frame, write_profile_csvs


BENCHMARK_COLUMNS = [
    "case_id",
    "case_source",
    "method",
    "thermo_model",
    "chemical_equilibrium_model",
    "co2_mass_transfer_model",
    "success",
    "message",
    "runtime_s",
    "capture_pct",
    "capture_error_pct",
    "temperature_rmse_K",
    "boundary_residual_norm",
    "boundary_residual_components",
    "co2_conservation_relative_residual",
    "h2o_conservation_relative_residual",
    "mesh_points",
    "tol",
    "bc_tol",
    "max_nodes",
    "co2_capture_guess_pct",
    "h2o_capture_guess_pct",
    "epcsaft_dataset",
    "eta_psi",
    "mass_transfer_factor",
    "heat_transfer_factor",
    "intercooler_strength",
    "co2_flux_mode",
    "vapor_composition_mode",
    "gas_flow_basis",
    "gas_velocity_area_exponent",
    "gas_velocity_area_reference_m_s",
    "co2_vapor_upper_factor",
    "success_boundary_residual_max",
    "integrator",
    "root_method",
    "ivp_method",
    "beds",
    "intercoolers",
    "staged_beds",
    "intercooler_model",
    "thermal_state_mode",
    "intercooler_assumption",
    "continuation_stage",
    "continuation_success",
    "invalid_state_count",
    "domain_guard_counts",
    "first_failed_domain",
    "jacobian_status",
    "solver_rhs_calls",
    "solver_rhs_node_evaluations",
    "solver_boundary_calls",
    "solver_jacobian_calls",
    "solver_iterations",
    "solver_final_nodes",
    "solver_mesh_nodes_added",
    "solver_max_rms_residual",
    "dense_grid_points",
    "dense_ode_residual_max",
    "dense_boundary_residual_max",
    "scaling_mode",
    "transform_mode",
    "continuation_path",
    "epcsaft_chemistry_solve_s",
    "epcsaft_chemistry_max_mass_residual",
    "epcsaft_chemistry_max_reaction_residual",
    "epcsaft_chemistry_max_charge_residual",
    "epcsaft_chemistry_failed_count",
    "epcsaft_chemistry_last_iterations",
    "epcsaft_chemistry_last_native_success",
    "epcsaft_chemistry_last_message",
    "epcsaft_chemistry_table_hits",
    "epcsaft_chemistry_interpolation_fallback_count",
    "epcsaft_chemistry_max_mea_mass_fraction_deviation",
    "profile_png",
    "profile_csv_dir",
    "profile_csv_status",
    "profile_csv_files",
    "python_version",
    "platform",
    "package_versions",
]


@dataclass(frozen=True)
class BenchmarkSettings:
    methods: tuple[str, ...] = ("single", "scipy-bvp", "finite")
    thermo_models: tuple[str, ...] = ("ideal_henry",)
    output_dir: Path = Path("analyses/nccc_validation/results/runs/benchmark")
    c_case_limit: int | None = None
    nccc_case_limit: int | None = None
    srp_case_limit: int | None = 0
    c_case_ids: tuple[str, ...] | None = None
    nccc_case_ids: tuple[str, ...] | None = None
    srp_case_ids: tuple[str, ...] | None = None
    write_artifacts: bool = True
    data_type: str = "mole"
    staged_beds: str | bool = "auto"
    solver_settings: dict | None = None
    profile_pngs: bool = False
    profile_csvs: bool = False
    subprocess_timeout_s: float | None = None
    c_case_dataset: str = "legacy"
    nccc_dataset: str = "legacy"


def _data_path(filename: str):
    return resources.files("mea_absorption_column").joinpath(f"data/{filename}")


def load_case_data(c_case_dataset: str = "legacy", nccc_dataset: str = "legacy"):
    c_case_files = {
        "legacy": "C_cases_data.csv",
        "campaign": "C_cases_campaign_inputs.csv",
    }
    nccc_case_files = {
        "legacy": "NCCC_Data_mole_based.csv",
        "2014": "NCCC_2014_model_inputs_mass.csv",
        "2017": "NCCC_2017_model_inputs_mass.csv",
    }
    if c_case_dataset not in c_case_files:
        raise ValueError(f"Unknown c_case_dataset: {c_case_dataset!r}")
    if nccc_dataset not in nccc_case_files:
        raise ValueError(f"Unknown nccc_dataset: {nccc_dataset!r}")
    c_cases = pd.read_csv(_data_path(c_case_files[c_case_dataset]), index_col=0)
    nccc_cases = pd.read_csv(_data_path(nccc_case_files[nccc_dataset]), index_col=0)
    srp_cases = pd.read_csv(_data_path("SRP_method_cases.csv"), index_col=0)
    return c_cases, nccc_cases, srp_cases


def _nccc_case_source(nccc_dataset: str) -> str:
    return {
        "legacy": "NCCC_Data",
        "2014": "NCCC_2014_cases",
        "2017": "NCCC_2017_cases",
    }[nccc_dataset]


def run_benchmark(settings: BenchmarkSettings = BenchmarkSettings()) -> pd.DataFrame:
    c_cases, nccc_cases, srp_cases = load_case_data(settings.c_case_dataset, settings.nccc_dataset)
    case_groups = []
    if settings.c_case_limit != 0:
        c_subset = c_cases.iloc[: settings.c_case_limit] if settings.c_case_limit is not None else c_cases
        c_subset = _filter_case_ids(c_subset, settings.c_case_ids, "C_cases_data")
        case_source = "C_cases_campaign_inputs" if settings.c_case_dataset == "campaign" else "C_cases_data"
        case_groups.append((case_source, c_subset))
    if settings.nccc_case_limit != 0:
        nccc_subset = nccc_cases.iloc[: settings.nccc_case_limit] if settings.nccc_case_limit is not None else nccc_cases
        nccc_case_source = _nccc_case_source(settings.nccc_dataset)
        nccc_subset = _filter_case_ids(nccc_subset, settings.nccc_case_ids, nccc_case_source)
        case_groups.append((nccc_case_source, nccc_subset))
    if settings.srp_case_limit != 0:
        srp_subset = srp_cases.iloc[: settings.srp_case_limit] if settings.srp_case_limit is not None else srp_cases
        srp_subset = _filter_case_ids(srp_subset, settings.srp_case_ids, "SRP_method_cases")
        case_groups.append(("SRP_method_cases", srp_subset))

    rows = []
    for case_source, df in case_groups:
        for run in range(len(df)):
            for method in settings.methods:
                for thermo_model in settings.thermo_models:
                    rows.append(_run_one_case(df, run, case_source, method, thermo_model, settings))

    results = pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)
    if settings.write_artifacts:
        write_benchmark_artifacts(results, settings.output_dir)
    return results


def _filter_case_ids(df, case_ids, label):
    if not case_ids:
        return df
    case_ids = [str(case_id) for case_id in case_ids]
    missing = [case_id for case_id in case_ids if case_id not in df.index]
    if missing:
        raise ValueError(f"Unknown {label} case id(s): {', '.join(missing)}")
    return df.loc[case_ids]


def _run_one_case(df, run, case_source, method, thermo_model, settings):
    if settings.subprocess_timeout_s is not None:
        return _run_one_case_subprocess(df, run, case_source, method, thermo_model, settings)
    return _run_one_case_in_process(df, run, case_source, method, thermo_model, settings)


def _run_one_case_in_process(df, run, case_source, method, thermo_model, settings):
    start = time.time()
    try:
        solver_settings = dict(settings.solver_settings or {})
        if "vapor_composition_mode" not in solver_settings and case_source == "NCCC_2017_cases":
            solver_settings["vapor_composition_mode"] = "dry_saturated"
        if settings.profile_pngs or settings.profile_csvs:
            solver_settings["return_profiles"] = True
        if settings.profile_csvs:
            solver_settings["case_source"] = case_source
        result = run_model(
            df,
            method=method,
            data_type=settings.data_type,
            run=run,
            show_info=False,
            save_run_results=False,
            plot_temperature=False,
            thermo_model=thermo_model,
            return_details=True,
            staged_beds=settings.staged_beds,
            solver_settings=solver_settings or None,
        )
        result = _annotate_solver_settings(result, solver_settings)
        if settings.profile_pngs and result.get("_profiles"):
            result["profile_png"] = _write_profile_png(result, settings.output_dir, case_source)
        if settings.profile_csvs:
            result.update(_write_profile_csvs_for_result(result, settings.output_dir, case_source))
        result["case_source"] = case_source
        return _coerce_row(result)
    except Exception as exc:
        metadata = _failure_metadata(df, run, method, settings.staged_beds)
        failure = {
            "case_id": str(df.index[run]),
            "case_source": case_source,
            "method": method,
            "thermo_model": thermo_model,
            "success": False,
            "message": str(exc),
            "runtime_s": float(time.time() - start),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": _package_versions(),
            **metadata,
        }
        return _coerce_row(_annotate_solver_settings(failure, dict(settings.solver_settings or {})))


def _run_one_case_subprocess(df, run, case_source, method, thermo_model, settings):
    start = time.time()
    timeout_s = float(settings.subprocess_timeout_s)
    effective_solver_settings = settings.solver_settings
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_root = Path(".tmp_local") / "benchmark_workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    tmp_path = worker_root / f"benchmark_worker_{os.getpid()}_{time.time_ns()}"
    tmp_path.mkdir(parents=True, exist_ok=False)
    try:
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        input_payload = {
            "case_source": case_source,
            "run": int(run),
            "method": method,
            "thermo_model": thermo_model,
            "settings": _settings_to_payload(settings),
            "output_path": str(output_path),
        }
        input_path.write_text(json.dumps(input_payload), encoding="utf-8")
        cmd = [sys.executable, "-m", "mea_absorption_column.benchmark_worker", str(input_path)]
        process = subprocess.Popen(
            cmd,
            cwd=str(Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = "", ""
            failure = {
                "case_id": str(df.index[run]),
                "case_source": case_source,
                "method": method,
                "thermo_model": thermo_model,
                "success": False,
                "message": f"Benchmark subprocess exceeded subprocess_timeout_s={timeout_s:g}",
                "runtime_s": float(time.time() - start),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "package_versions": _package_versions(),
                **_failure_metadata(df, run, method, settings.staged_beds),
            }
            failure["jacobian_status"] = "subprocess_timeout"
            return _coerce_row(_annotate_solver_settings(failure, effective_solver_settings or {}))
        if process.returncode != 0:
            failure = {
                "case_id": str(df.index[run]),
                "case_source": case_source,
                "method": method,
                "thermo_model": thermo_model,
                "success": False,
                "message": _worker_error_message(process.returncode, stdout, stderr),
                "runtime_s": float(time.time() - start),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "package_versions": _package_versions(),
                **_failure_metadata(df, run, method, settings.staged_beds),
            }
            return _coerce_row(_annotate_solver_settings(failure, effective_solver_settings or {}))
        if not output_path.exists():
            failure = {
                "case_id": str(df.index[run]),
                "case_source": case_source,
                "method": method,
                "thermo_model": thermo_model,
                "success": False,
                "message": "Benchmark subprocess completed without writing output row.",
                "runtime_s": float(time.time() - start),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "package_versions": _package_versions(),
                **_failure_metadata(df, run, method, settings.staged_beds),
            }
            return _coerce_row(_annotate_solver_settings(failure, effective_solver_settings or {}))
        row = json.loads(output_path.read_text(encoding="utf-8"))
        row["runtime_s"] = float(time.time() - start)
        return _coerce_row(row)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def _settings_to_payload(settings: BenchmarkSettings):
    return {
        "methods": list(settings.methods),
        "thermo_models": list(settings.thermo_models),
        "output_dir": str(settings.output_dir),
        "c_case_limit": settings.c_case_limit,
        "nccc_case_limit": settings.nccc_case_limit,
        "srp_case_limit": settings.srp_case_limit,
        "c_case_ids": list(settings.c_case_ids) if settings.c_case_ids is not None else None,
        "nccc_case_ids": list(settings.nccc_case_ids) if settings.nccc_case_ids is not None else None,
        "srp_case_ids": list(settings.srp_case_ids) if settings.srp_case_ids is not None else None,
        "write_artifacts": False,
        "data_type": settings.data_type,
        "staged_beds": settings.staged_beds,
        "solver_settings": settings.solver_settings,
        "profile_pngs": settings.profile_pngs,
        "profile_csvs": settings.profile_csvs,
        "subprocess_timeout_s": None,
        "c_case_dataset": settings.c_case_dataset,
        "nccc_dataset": settings.nccc_dataset,
    }


def settings_from_payload(payload: dict) -> BenchmarkSettings:
    defaults = BenchmarkSettings()
    return BenchmarkSettings(
        methods=tuple(payload.get("methods") or defaults.methods),
        thermo_models=tuple(payload.get("thermo_models") or defaults.thermo_models),
        output_dir=Path(payload.get("output_dir") or defaults.output_dir),
        c_case_limit=payload.get("c_case_limit"),
        nccc_case_limit=payload.get("nccc_case_limit"),
        srp_case_limit=payload.get("srp_case_limit", 0),
        c_case_ids=tuple(payload["c_case_ids"]) if payload.get("c_case_ids") is not None else None,
        nccc_case_ids=tuple(payload["nccc_case_ids"]) if payload.get("nccc_case_ids") is not None else None,
        srp_case_ids=tuple(payload["srp_case_ids"]) if payload.get("srp_case_ids") is not None else None,
        write_artifacts=bool(payload.get("write_artifacts", False)),
        data_type=payload.get("data_type", "mole"),
        staged_beds=payload.get("staged_beds", "auto"),
        solver_settings=payload.get("solver_settings"),
        profile_pngs=bool(payload.get("profile_pngs", False)),
        profile_csvs=bool(payload.get("profile_csvs", False)),
        subprocess_timeout_s=payload.get("subprocess_timeout_s"),
        c_case_dataset=payload.get("c_case_dataset", "legacy"),
        nccc_dataset=payload.get("nccc_dataset", "legacy"),
    )


def _terminate_process_tree(process: subprocess.Popen):
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.kill()


def _worker_error_message(returncode, stdout, stderr):
    details = (stderr or stdout or "").strip().splitlines()
    tail = " | ".join(details[-5:])
    return f"Benchmark subprocess failed with return code {returncode}: {tail}"


def _annotate_solver_settings(result, solver_settings):
    result = dict(result)
    result.setdefault(
        "chemical_equilibrium_model",
        _default_chemical_equilibrium_model(result.get("thermo_model")),
    )
    for key in (
        "mesh_points",
        "tol",
        "bc_tol",
        "max_nodes",
        "chemical_equilibrium_model",
        "co2_capture_guess_pct",
        "h2o_capture_guess_pct",
        "eta_psi",
        "mass_transfer_factor",
        "heat_transfer_factor",
        "intercooler_strength",
        "co2_flux_mode",
        "vapor_composition_mode",
        "gas_flow_basis",
        "gas_velocity_area_exponent",
        "gas_velocity_area_reference_m_s",
        "co2_vapor_upper_factor",
        "success_boundary_residual_max",
        "scaling_mode",
        "transform_mode",
        "integrator",
        "root_method",
        "ivp_method",
    ):
        if key in solver_settings:
            result[key] = solver_settings[key]
    return result


def _default_chemical_equilibrium_model(thermo_model):
    normalized = (thermo_model or "").lower()
    if normalized in {
        "epcsaft_reactive_six",
        "epcsaft_reactive_six_concentration",
        "epcsaft_reactive_six_activity",
        "epcsaft_reactive_six_activity_converted",
        "epcsaft_reactive_six_activity_rebased",
        "epcsaft_reactive_nine",
        "epcsaft_reactive_nine_activity",
        "epcsaft_reactive_nine_activity_converted",
        "epcsaft_reactive_nine_activity_rebased",
        "epcsaft_reactive_nine_tabulated",
        "epcsaft_full_species_activity",
        "epcsaft_full_species_activity_converted",
        "epcsaft_full_species_activity_rebased",
    }:
        return normalized
    return "legacy"


def _coerce_row(result):
    return {column: result.get(column) for column in BENCHMARK_COLUMNS}


def _failure_metadata(df, run, method, staged_beds):
    row = df.iloc[run]
    beds_column = "Beds" if "Beds" in df.columns else "beds" if "beds" in df.columns else None
    beds = int(row[beds_column]) if beds_column is not None else None
    intercooler_column = "Intercoolers" if "Intercoolers" in df.columns else "intercoolers" if "intercoolers" in df.columns else None
    intercoolers = int(row[intercooler_column]) if intercooler_column is not None else 0
    if staged_beds == "auto":
        staged = method in {"scipy-bvp", "collocation"} and (
            (beds is not None and beds > 1) or intercoolers > 0
        )
    else:
        staged = bool(staged_beds)
    return {
        "beds": beds,
        "intercoolers": intercoolers,
        "staged_beds": staged,
        "intercooler_model": "liquid_temperature_reset" if staged and intercoolers else "none",
        "intercooler_assumption": "Tl_feed_target" if staged and intercoolers else "none",
        "max_nodes": None,
        "co2_conservation_relative_residual": None,
        "h2o_conservation_relative_residual": None,
        "co2_capture_guess_pct": None,
        "h2o_capture_guess_pct": None,
        "eta_psi": None,
        "mass_transfer_factor": None,
        "heat_transfer_factor": None,
        "intercooler_strength": None,
        "co2_flux_mode": None,
        "vapor_composition_mode": None,
        "gas_flow_basis": None,
        "gas_velocity_area_exponent": None,
        "gas_velocity_area_reference_m_s": None,
        "co2_vapor_upper_factor": None,
        "success_boundary_residual_max": None,
        "integrator": None,
        "root_method": None,
        "ivp_method": None,
        "continuation_stage": "failed_before_solver",
        "continuation_success": False,
        "invalid_state_count": None,
        "domain_guard_counts": None,
        "first_failed_domain": None,
        "jacobian_status": None,
        "solver_rhs_calls": None,
        "solver_rhs_node_evaluations": None,
        "solver_boundary_calls": None,
        "solver_jacobian_calls": None,
        "solver_iterations": None,
        "solver_final_nodes": None,
        "solver_mesh_nodes_added": None,
        "solver_max_rms_residual": None,
        "dense_grid_points": None,
        "dense_ode_residual_max": None,
        "dense_boundary_residual_max": None,
        "scaling_mode": "legacy_flow_enthalpy",
        "transform_mode": "bounded_guarded_raw_state",
        "continuation_path": "none",
        "profile_png": None,
        "profile_csv_dir": None,
        "profile_csv_status": None,
        "profile_csv_files": None,
    }


def _package_versions():
    names = ["mea-absorption-column", "numpy", "pandas", "scipy", "matplotlib"]
    versions = []
    for name in names:
        try:
            versions.append(f"{name}={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{name}=uninstalled")
    return ";".join(versions)


def write_benchmark_artifacts(results: pd.DataFrame, output_dir: Path | str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "benchmark_results.csv", index=False)

    summary_input = results.copy()
    for column in ("capture_error_pct", "temperature_rmse_K", "runtime_s"):
        summary_input[column] = pd.to_numeric(summary_input[column], errors="coerce")
    summary_input["success"] = summary_input["success"].fillna(False).astype(bool)

    summary = (
        summary_input.groupby(["case_source", "method", "thermo_model"], dropna=False)
        .agg(
            runs=("case_id", "count"),
            successes=("success", "sum"),
            capture_mae_pct=("capture_error_pct", lambda s: s.abs().mean()),
            capture_rmse_pct=("capture_error_pct", lambda s: (s.dropna().pow(2).mean()) ** 0.5),
            temperature_rmse_K=("temperature_rmse_K", "mean"),
            runtime_median_s=("runtime_s", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    (output_dir / "benchmark_summary.md").write_text(_to_markdown(summary), encoding="utf-8")
    if {"case_id", "beds", "intercoolers", "capture_error_pct"}.issubset(results.columns):
        write_calibration_artifacts(results, output_dir)


def _to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[_format_markdown_cell(value) for value in record] for record in df.to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows)) if rows else len(str(header))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body]) + "\n"


def _format_markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run MEA absorber benchmark cases.")
    parser.add_argument("--methods", nargs="+", default=list(BenchmarkSettings.methods))
    parser.add_argument("--thermo-models", nargs="+", default=list(BenchmarkSettings.thermo_models))
    parser.add_argument("--output-dir", default=str(BenchmarkSettings.output_dir))
    parser.add_argument("--c-case-limit", type=int, default=None)
    parser.add_argument("--nccc-case-limit", type=int, default=None)
    parser.add_argument("--srp-case-limit", type=int, default=0)
    parser.add_argument("--c-case-ids", nargs="+", default=None)
    parser.add_argument("--nccc-case-ids", nargs="+", default=None)
    parser.add_argument("--srp-case-ids", nargs="+", default=None)
    parser.add_argument("--staged-beds", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--mesh-points", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--bc-tol", type=float, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-runtime-s", type=float, default=None)
    parser.add_argument("--transform-mode", default=None)
    parser.add_argument("--thermal-state-mode", choices=["enthalpy", "temperature"], default=None)
    parser.add_argument("--co2-flux-mode", choices=["bidirectional", "absorption_only"], default=None)
    parser.add_argument("--vapor-composition-mode", choices=["legacy_ratio", "input_o2", "dry_saturated"], default=None)
    parser.add_argument("--gas-flow-basis", choices=["reported_total_wet", "reported_dry_mass"], default=None)
    parser.add_argument("--gas-velocity-area-exponent", type=float, default=None)
    parser.add_argument("--gas-velocity-area-reference-m-s", type=float, default=None)
    parser.add_argument("--co2-vapor-upper-factor", type=float, default=None)
    parser.add_argument("--shooting-integrator", choices=["euler", "solve_ivp", "bdf", "radau", "rk45"], default=None)
    parser.add_argument("--shooting-root-method", default=None)
    parser.add_argument("--co2-capture-guess-pct", type=float, default=None)
    parser.add_argument("--h2o-capture-guess-pct", type=float, default=None)
    parser.add_argument("--eta-psi", type=float, default=None)
    parser.add_argument("--chemical-equilibrium-model", default=None)
    parser.add_argument("--mass-transfer-factor", type=float, default=None)
    parser.add_argument("--heat-transfer-factor", type=float, default=None)
    parser.add_argument("--intercooler-strength", type=float, default=None)
    parser.add_argument(
        "--intercooler-model",
        choices=["liquid_temperature_reset", "pumparound_temperature_approach"],
        default=None,
    )
    parser.add_argument("--success-boundary-residual-max", type=float, default=None)
    parser.add_argument("--success-capture-error-max-pct", type=float, default=None)
    parser.add_argument("--finite-jacobian", action="store_true")
    parser.add_argument("--profile-pngs", action="store_true")
    parser.add_argument("--profile-csvs", action="store_true")
    parser.add_argument("--subprocess-timeout-s", type=float, default=None)
    parser.add_argument("--c-case-dataset", choices=["legacy", "campaign"], default="legacy")
    parser.add_argument("--nccc-dataset", choices=["legacy", "2014", "2017"], default="legacy")
    parser.add_argument("--data-type", choices=["mole", "mass"], default=BenchmarkSettings.data_type)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    staged_beds = args.staged_beds
    if staged_beds == "true":
        staged_beds = True
    elif staged_beds == "false":
        staged_beds = False
    settings = BenchmarkSettings(
        methods=tuple(args.methods),
        thermo_models=tuple(args.thermo_models),
        output_dir=Path(args.output_dir),
        c_case_limit=args.c_case_limit,
        nccc_case_limit=args.nccc_case_limit,
        srp_case_limit=args.srp_case_limit,
        c_case_ids=tuple(args.c_case_ids) if args.c_case_ids is not None else None,
        nccc_case_ids=tuple(args.nccc_case_ids) if args.nccc_case_ids is not None else None,
        srp_case_ids=tuple(args.srp_case_ids) if args.srp_case_ids is not None else None,
        staged_beds=staged_beds,
        write_artifacts=not args.no_write,
        data_type=args.data_type,
        solver_settings=_solver_settings_from_args(args),
        profile_pngs=bool(args.profile_pngs),
        profile_csvs=bool(args.profile_csvs),
        subprocess_timeout_s=args.subprocess_timeout_s,
        c_case_dataset=args.c_case_dataset,
        nccc_dataset=args.nccc_dataset,
    )
    results = run_benchmark(settings)
    print(results.to_string(index=False))


def _solver_settings_from_args(args):
    settings = {}
    if args.mesh_points is not None:
        settings["mesh_points"] = args.mesh_points
    if args.tol is not None:
        settings["tol"] = args.tol
    if args.bc_tol is not None:
        settings["bc_tol"] = args.bc_tol
    if args.max_nodes is not None:
        settings["max_nodes"] = args.max_nodes
    if args.max_runtime_s is not None:
        settings["max_runtime_s"] = args.max_runtime_s
    if args.transform_mode is not None:
        settings["transform_mode"] = args.transform_mode
    if args.thermal_state_mode is not None:
        settings["thermal_state_mode"] = args.thermal_state_mode
    if args.intercooler_model is not None:
        settings["intercooler_model"] = args.intercooler_model
    if args.co2_flux_mode is not None:
        settings["co2_flux_mode"] = args.co2_flux_mode
    if args.vapor_composition_mode is not None:
        settings["vapor_composition_mode"] = args.vapor_composition_mode
    if args.gas_flow_basis is not None:
        settings["gas_flow_basis"] = args.gas_flow_basis
    if args.gas_velocity_area_exponent is not None:
        settings["gas_velocity_area_exponent"] = args.gas_velocity_area_exponent
    if args.gas_velocity_area_reference_m_s is not None:
        settings["gas_velocity_area_reference_m_s"] = args.gas_velocity_area_reference_m_s
    if args.co2_vapor_upper_factor is not None:
        settings["co2_vapor_upper_factor"] = args.co2_vapor_upper_factor
    if args.shooting_integrator is not None:
        settings["integrator"] = args.shooting_integrator
        if args.shooting_integrator in {"bdf", "radau", "rk45"}:
            settings["ivp_method"] = {"bdf": "BDF", "radau": "Radau", "rk45": "RK45"}[args.shooting_integrator]
    if args.shooting_root_method is not None:
        settings["root_method"] = args.shooting_root_method
    if args.co2_capture_guess_pct is not None:
        settings["co2_capture_guess_pct"] = args.co2_capture_guess_pct
    if args.h2o_capture_guess_pct is not None:
        settings["h2o_capture_guess_pct"] = args.h2o_capture_guess_pct
    if args.eta_psi is not None:
        settings["eta_psi"] = args.eta_psi
    if args.chemical_equilibrium_model is not None:
        settings["chemical_equilibrium_model"] = args.chemical_equilibrium_model
    if args.mass_transfer_factor is not None:
        settings["mass_transfer_factor"] = args.mass_transfer_factor
    if args.heat_transfer_factor is not None:
        settings["heat_transfer_factor"] = args.heat_transfer_factor
    if args.intercooler_strength is not None:
        settings["intercooler_strength"] = args.intercooler_strength
    if args.success_boundary_residual_max is not None:
        settings["success_boundary_residual_max"] = args.success_boundary_residual_max
    if args.success_capture_error_max_pct is not None:
        settings["success_capture_error_max_pct"] = args.success_capture_error_max_pct
    if args.finite_jacobian:
        settings["use_finite_jacobian"] = True
    return settings or None


def _write_profile_png(result, output_dir: Path | str, case_source: str):
    profiles = result.get("_profiles") or {}
    if "T" not in profiles:
        return ""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) / "temperature_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result['case_id']}_{case_source}_{result['method']}_{result['thermo_model']}.png".replace(" ", "_")
    path = output_dir / filename
    profile = profiles["T"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    if "Tl" in profile:
        ax.plot(profile.index, profile["Tl"], label="Liquid model", linewidth=2)
    if "Tv" in profile:
        ax.plot(profile.index, profile["Tv"], label="Vapor model", linewidth=2)
    ax.set_xlabel("Normalized column position")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"{result['case_id']} | {result['method']} | {result['thermo_model']}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_profile_csvs_for_result(result, output_dir: Path | str, case_source: str):
    profiles = result.get("_profiles") or {}
    if not profiles:
        return {
            "profile_csv_dir": "",
            "profile_csv_status": "empty",
            "profile_csv_files": "",
        }
    profile_dir = _profile_csv_dir(result, output_dir, case_source)
    metadata = {
        **(result.get("_case_metadata") or {}),
        "case_id": result.get("case_id", ""),
        "case_source": case_source,
        "method": result.get("method", ""),
        "thermo_model": result.get("thermo_model", ""),
        "success": bool(result.get("success", False)),
        "message": result.get("message", ""),
        "runtime_s": result.get("runtime_s"),
        "runtime_label": _format_runtime_label(result.get("runtime_s")),
        "beds": result.get("beds", ""),
        "intercoolers": result.get("intercoolers", ""),
        "total_packed_height_m": (result.get("_case_metadata") or {}).get("total_packed_height_m"),
        "single_bed_height_m": (result.get("_case_metadata") or {}).get("single_bed_height_m"),
        "intercooler_model": result.get("intercooler_model", ""),
        "intercooler_assumption": result.get("intercooler_assumption", ""),
        "profile_status": "clean" if bool(result.get("success", False)) else "diagnostic",
        "position_orientation": "global_normalized_bottom_to_top",
    }
    export = write_profile_csvs(
        _profiles_with_coordinates(profiles, metadata),
        profile_dir,
        metadata,
    )
    _write_profile_rerun_files(profile_dir, metadata, output_dir, result)
    return export


def _profiles_with_coordinates(profiles, metadata):
    converted = {}
    for sheetname, profile in profiles.items():
        if "Position" in profile.columns:
            converted[sheetname] = profile
            continue
        # Sort to enforce a stable bottom-to-top (ascending Position) CSV contract
        # while preserving each row's value coupling from the original profile index.
        profile = profile.sort_index()
        positions = profile.index.to_numpy(dtype=float)
        coordinate_frame = build_profile_coordinate_frame(
            positions,
            total_packed_height_m=metadata.get("total_packed_height_m"),
            beds=metadata.get("beds", 1),
        )
        converted[sheetname] = pd.concat(
            [coordinate_frame.reset_index(drop=True), profile.reset_index(drop=True)],
            axis=1,
        )
    return converted


def _profile_csv_dir(result, output_dir: Path | str, case_source: str):
    parts = [
        Path(output_dir),
        "profiles",
        _safe_path_part(case_source),
        _safe_path_part(result.get("case_id", "unknown_case")),
        _safe_path_part(result.get("method", "unknown_method")),
        _safe_path_part(result.get("thermo_model", "unknown_thermo")),
    ]
    path = parts[0]
    for part in parts[1:]:
        path = path / part
    return path


def _safe_path_part(value):
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _format_runtime_label(runtime_s):
    try:
        runtime_s = float(runtime_s)
    except (TypeError, ValueError):
        return ""
    if pd.isna(runtime_s):
        return ""
    if runtime_s < 60.0:
        return f"{runtime_s:.2f} s"
    minutes, seconds = divmod(runtime_s, 60.0)
    return f"{int(minutes)} min {seconds:.1f} s"


def _write_profile_rerun_files(profile_dir: Path, metadata: dict, output_dir: Path | str, result: dict):
    solver_settings = {}
    for key in (
        "mesh_points",
        "tol",
        "bc_tol",
        "max_nodes",
        "co2_capture_guess_pct",
        "h2o_capture_guess_pct",
        "eta_psi",
        "mass_transfer_factor",
        "heat_transfer_factor",
        "intercooler_strength",
        "co2_flux_mode",
        "vapor_composition_mode",
        "gas_flow_basis",
        "gas_velocity_area_exponent",
        "gas_velocity_area_reference_m_s",
        "co2_vapor_upper_factor",
        "success_boundary_residual_max",
        "scaling_mode",
        "transform_mode",
        "integrator",
        "root_method",
        "ivp_method",
    ):
        value = result.get(key)
        if value is not None and not pd.isna(value):
            solver_settings[key] = value
    spec = {
        "case_source": metadata.get("case_source"),
        "case_id": metadata.get("case_id"),
        "method": metadata.get("method"),
        "thermo_model": metadata.get("thermo_model"),
        "staged_beds": "auto",
        "output_dir": str(output_dir),
        "solver_settings": solver_settings,
    }
    spec_path = profile_dir / "run_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True, default=str), encoding="utf-8")
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
            'repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"',
            'cd -- "$repo_root"',
            'uv run python analyses/nccc_validation/scripts/run_case_profile.py --spec "$script_dir/run_spec.json"',
            "",
        ]
    )
    rerun_path = profile_dir / "rerun_profile.sh"
    rerun_path.write_text(script, encoding="utf-8")
    rerun_path.chmod(rerun_path.stat().st_mode | 0o111)


if __name__ == "__main__":
    main()
