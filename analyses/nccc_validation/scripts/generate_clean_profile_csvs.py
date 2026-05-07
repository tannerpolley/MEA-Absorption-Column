from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mea_absorption_column.benchmark import BENCHMARK_COLUMNS, BenchmarkSettings, run_benchmark


ROOT = Path(__file__).resolve().parents[3]
FINAL_TABLES = ROOT / "analyses" / "nccc_validation" / "results" / "final" / "tables"
DEFAULT_OUTPUT = ROOT / "analyses" / "nccc_validation" / "results" / "runs" / "clean_profile_csvs"


@dataclass(frozen=True)
class ProfileJob:
    case_source: str
    case_id: str
    method: str
    thermo_model: str
    solver_settings: dict


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "benchmark_results.csv"
    log_path = output_dir / "profile_generation_log.csv"
    jobs = _jobs(args.suite, args.case_ids)
    existing = _existing_keys(rows_path) if args.resume else set()
    rows = _read_existing_rows(rows_path) if args.resume else []
    log_rows = _read_existing_log(log_path) if args.resume else []

    for job in jobs:
        key = (job.case_source, job.case_id, job.method, job.thermo_model)
        if key in existing:
            print(f"skip existing {job.case_id} {job.thermo_model}", flush=True)
            continue
        start = time.time()
        try:
            result = _run_job(job, output_dir, args.per_case_timeout_s)
            row = result.iloc[0].to_dict()
            status = "ok" if bool(row.get("success")) and row.get("profile_csv_status") == "written" else "diagnostic"
        except Exception as exc:
            row = _failure_row(job, output_dir, exc, time.time() - start)
            status = "error"
        rows = _replace_row(rows, row, key)
        log_rows.append(
            {
                "case_source": job.case_source,
                "case_id": job.case_id,
                "method": job.method,
                "thermo_model": job.thermo_model,
                "status": status,
                "runtime_s": row.get("runtime_s"),
                "success": row.get("success"),
                "profile_csv_status": row.get("profile_csv_status"),
                "message": row.get("message"),
            }
        )
        _write_rows(rows, rows_path)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)
        _write_runtime_index(rows, output_dir)
        print(
            f"{job.case_id} {job.thermo_model}: {status} "
            f"success={row.get('success')} profile={row.get('profile_csv_status')} "
            f"runtime={row.get('runtime_s')}",
            flush=True,
        )
    _write_runtime_index(rows, output_dir)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate dense profile CSVs for accepted validation rows.")
    parser.add_argument("--suite", choices=["c", "k", "all"], default="all")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--per-case-timeout-s",
        type=float,
        default=60.0,
        help="Wall-clock timeout for each case subprocess. Timed-out cases are logged and skipped.",
    )
    parser.add_argument("--case-ids", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    return parser.parse_args(argv)


def _jobs(suite: str, case_ids: list[str] | None = None) -> list[ProfileJob]:
    selected = {str(case_id) for case_id in case_ids} if case_ids else None
    jobs: list[ProfileJob] = []
    if suite in {"c", "all"}:
        c_cases = pd.read_csv(FINAL_TABLES / "verified_c_case_thermo_benchmark.csv")
        for row in c_cases.itertuples(index=False):
            if not _truthy(row.success):
                continue
            if selected is not None and str(row.case_id) not in selected:
                continue
            jobs.append(
                ProfileJob(
                    case_source="C_cases_data",
                    case_id=str(row.case_id),
                    method="scipy-bvp",
                    thermo_model=str(row.thermo_model),
                    solver_settings={},
                )
            )
    if suite in {"k", "all"}:
        k_cases = pd.read_csv(FINAL_TABLES / "verified_staged_kcase_benchmark.csv")
        for row in k_cases.itertuples(index=False):
            if not _truthy(row.success):
                continue
            if selected is not None and str(row.case_id) not in selected:
                continue
            jobs.append(
                ProfileJob(
                    case_source="NCCC_Data",
                    case_id=str(row.case_id),
                    method="scipy-bvp",
                    thermo_model=str(row.thermo_model),
                    solver_settings=_parse_continuation_path(getattr(row, "continuation_path", "")),
                )
            )
    return jobs


def _run_job(job: ProfileJob, output_dir: Path, timeout_s: float) -> pd.DataFrame:
    settings = BenchmarkSettings(
        methods=(job.method,),
        thermo_models=(job.thermo_model,),
        output_dir=output_dir,
        c_case_limit=0 if job.case_source != "C_cases_data" else None,
        nccc_case_limit=0 if job.case_source != "NCCC_Data" else None,
        c_case_ids=(job.case_id,) if job.case_source == "C_cases_data" else None,
        nccc_case_ids=(job.case_id,) if job.case_source == "NCCC_Data" else None,
        staged_beds="auto",
        solver_settings=job.solver_settings or None,
        profile_csvs=True,
        profile_pngs=True,
        subprocess_timeout_s=timeout_s,
        write_artifacts=False,
    )
    return run_benchmark(settings)


def _parse_continuation_path(value) -> dict:
    settings = {
        "mesh_points": 5,
        "tol": 0.5,
        "bc_tol": 0.001,
        "max_nodes": 80,
        "max_runtime_s": 45.0,
        "success_capture_error_max_pct": 8.0,
    }
    text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    if not text or text == "none":
        return settings
    for part in text.split(";"):
        if "=" not in part:
            continue
        key, raw = [item.strip() for item in part.split("=", 1)]
        if key == "capture_guess":
            settings["co2_capture_guess_pct"] = float(raw)
        elif key in {"mass_transfer_factor", "intercooler_strength", "co2_vapor_upper_factor"}:
            settings[key] = float(raw)
        elif key == "co2_flux_mode":
            settings[key] = raw
    return settings


def _existing_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    data = pd.read_csv(path)
    required = {"case_source", "case_id", "method", "thermo_model"}
    if not required <= set(data.columns):
        return set()
    data = data[data.get("profile_csv_status", "").astype(str).eq("written")]
    return {
        (str(row.case_source), str(row.case_id), str(row.method), str(row.thermo_model))
        for row in data.itertuples(index=False)
    }


def _read_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _read_existing_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=BENCHMARK_COLUMNS).to_csv(path, index=False)


def _replace_row(rows: list[dict], row: dict, key: tuple[str, str, str, str]) -> list[dict]:
    replaced = False
    output = []
    for existing in rows:
        existing_key = (
            str(existing.get("case_source")),
            str(existing.get("case_id")),
            str(existing.get("method")),
            str(existing.get("thermo_model")),
        )
        if existing_key == key:
            if not replaced:
                output.append(row)
                replaced = True
        else:
            output.append(existing)
    if not replaced:
        output.append(row)
    return output


def _failure_row(job: ProfileJob, output_dir: Path, exc: Exception, runtime_s: float) -> dict:
    row = {column: None for column in BENCHMARK_COLUMNS}
    row.update(
        {
            "case_source": job.case_source,
            "case_id": job.case_id,
            "method": job.method,
            "thermo_model": job.thermo_model,
            "success": False,
            "message": f"profile generation failed before benchmark row: {exc}",
            "runtime_s": float(runtime_s),
            "profile_csv_dir": str(output_dir / "profiles" / job.case_source / job.case_id / job.method / job.thermo_model),
            "profile_csv_status": "error",
            "profile_csv_files": "",
        }
    )
    return row


def _write_runtime_index(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    records = []
    for row in rows:
        runtime_s = _float_or_none(row.get("runtime_s"))
        records.append(
            {
                "case_source": row.get("case_source"),
                "case_id": row.get("case_id"),
                "method": row.get("method"),
                "thermo_model": row.get("thermo_model"),
                "success": row.get("success"),
                "profile_csv_status": row.get("profile_csv_status"),
                "runtime_s": runtime_s,
                "runtime_label": _runtime_label(runtime_s),
                "profile_csv_dir": row.get("profile_csv_dir"),
                "message": row.get("message"),
            }
        )
    index = pd.DataFrame(records)
    index.to_csv(output_dir / "profile_runtime_index.csv", index=False)
    _refresh_profile_manifests(index)


def _refresh_profile_manifests(index: pd.DataFrame) -> None:
    for row in index.itertuples(index=False):
        if not isinstance(row.profile_csv_dir, str) or not row.profile_csv_dir:
            continue
        profile_dir = Path(row.profile_csv_dir)
        if not profile_dir.exists():
            continue
        updates = {
            "runtime_s": row.runtime_s,
            "runtime_label": row.runtime_label,
        }
        manifest_json = profile_dir / "profile_manifest.json"
        if manifest_json.exists():
            data = json.loads(manifest_json.read_text(encoding="utf-8"))
            data.update(updates)
            manifest_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        manifest_csv = profile_dir / "profile_manifest.csv"
        if manifest_csv.exists():
            data = pd.read_csv(manifest_csv)
            for key, value in updates.items():
                data[key] = value
            data.to_csv(manifest_csv, index=False)


def _runtime_label(runtime_s: float | None) -> str:
    if runtime_s is None:
        return ""
    if runtime_s < 60.0:
        return f"{runtime_s:.2f} s"
    minutes, seconds = divmod(runtime_s, 60.0)
    return f"{int(minutes)} min {seconds:.1f} s"


def _float_or_none(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _truthy(value) -> bool:
    return str(value).lower() == "true" or value is True


if __name__ == "__main__":
    raise SystemExit(main())
