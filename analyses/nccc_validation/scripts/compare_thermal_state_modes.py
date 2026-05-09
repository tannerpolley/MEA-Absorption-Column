from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = ROOT / "analyses" / "nccc_validation" / "results" / "final" / "tables"
REPORT_DIR = ROOT / "analyses" / "nccc_validation" / "results" / "final" / "reports"
PYTHON = Path(sys.executable)

CASE_SOURCES = {
    "C_cases_data": ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_data.csv",
    "SRP_method_cases": ROOT / "src" / "mea_absorption_column" / "data" / "SRP_method_cases.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case-source")
    parser.add_argument("--case-id")
    parser.add_argument("--thermal-state-mode", choices=["enthalpy", "temperature"])
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(_run_worker(args.case_source, args.case_id, args.thermal_state_mode), sort_keys=True))
        return

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_source, path in CASE_SOURCES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        for case_id in df.index.astype(str):
            for mode in ("enthalpy", "temperature"):
                rows.append(_run_subprocess(case_source, case_id, mode, args.timeout_s))

    results = pd.DataFrame(rows)
    comparisons = _build_comparisons(results)
    results.to_csv(TABLE_DIR / "thermal_state_mode_runs.csv", index=False)
    comparisons.to_csv(TABLE_DIR / "thermal_state_mode_parity.csv", index=False)
    _write_report(results, comparisons)

    print(f"Wrote {TABLE_DIR / 'thermal_state_mode_runs.csv'}")
    print(f"Wrote {TABLE_DIR / 'thermal_state_mode_parity.csv'}")
    print(f"Wrote {REPORT_DIR / 'thermal_state_mode_parity.md'}")


def _run_subprocess(case_source: str, case_id: str, mode: str, timeout_s: float) -> dict:
    command = [
        str(PYTHON),
        str(Path(__file__).resolve()),
        "--worker",
        "--case-source",
        case_source,
        "--case-id",
        case_id,
        "--thermal-state-mode",
        mode,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "case_source": case_source,
            "case_id": case_id,
            "thermal_state_mode": mode,
            "success": False,
            "message": f"subprocess timeout after {timeout_s:g} s",
            "timeout": True,
        }
    if completed.returncode != 0:
        return {
            "case_source": case_source,
            "case_id": case_id,
            "thermal_state_mode": mode,
            "success": False,
            "message": completed.stderr[-1000:] or completed.stdout[-1000:],
            "timeout": False,
        }
    line = completed.stdout.strip().splitlines()[-1]
    row = json.loads(line)
    row["timeout"] = False
    return row


def _run_worker(case_source: str, case_id: str, mode: str) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from mea_absorption_column.Run_Model import run_model

    df = pd.read_csv(CASE_SOURCES[case_source], index_col=0)
    run = list(df.index.astype(str)).index(case_id)
    settings = {
        "mesh_points": 11,
        "tol": 1.0,
        "bc_tol": 0.05,
        "max_nodes": 160,
        "thermal_state_mode": mode,
        "enhancement_factor_model": "implicit",
    }
    if mode == "temperature":
        settings["seed_from_enthalpy"] = True
    result = run_model(
        df,
        method="scipy-bvp",
        run=run,
        thermo_model="ideal_henry",
        return_details=True,
        staged_beds=False,
        solver_settings=settings,
    )
    return {
        "case_source": case_source,
        "case_id": case_id,
        "thermal_state_mode": mode,
        "success": bool(result.get("success")),
        "message": result.get("message", ""),
        "runtime_s": result.get("runtime_s"),
        "capture_pct": result.get("capture_pct"),
        "capture_error_pct": result.get("capture_error_pct"),
        "temperature_rmse_K": result.get("temperature_rmse_K"),
        "boundary_residual_norm": result.get("boundary_residual_norm"),
        "continuation_path": result.get("continuation_path", "none"),
    }


def _build_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case_source, case_id), group in results.groupby(["case_source", "case_id"], dropna=False):
        by_mode = {row.thermal_state_mode: row for row in group.itertuples(index=False)}
        enthalpy = by_mode.get("enthalpy")
        temperature = by_mode.get("temperature")
        if enthalpy is None or temperature is None:
            continue
        capture_delta = _delta(getattr(temperature, "capture_pct", None), getattr(enthalpy, "capture_pct", None))
        rmse_delta = _delta(getattr(temperature, "temperature_rmse_K", None), getattr(enthalpy, "temperature_rmse_K", None))
        parity_ok = (
            bool(getattr(enthalpy, "success", False))
            and bool(getattr(temperature, "success", False))
            and capture_delta is not None
            and abs(capture_delta) <= 0.05
            and (rmse_delta is None or abs(rmse_delta) <= 0.1)
        )
        rows.append(
            {
                "case_source": case_source,
                "case_id": case_id,
                "enthalpy_success": bool(getattr(enthalpy, "success", False)),
                "temperature_success": bool(getattr(temperature, "success", False)),
                "capture_delta_pct_point": capture_delta,
                "temperature_rmse_delta_K": rmse_delta,
                "enthalpy_runtime_s": getattr(enthalpy, "runtime_s", None),
                "temperature_runtime_s": getattr(temperature, "runtime_s", None),
                "parity_ok": parity_ok,
            }
        )
    return pd.DataFrame(rows)


def _delta(a, b):
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a) - float(b)


def _write_report(results: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    accepted = int(comparisons["parity_ok"].sum()) if "parity_ok" in comparisons else 0
    total = len(comparisons)
    lines = [
        "# Thermal State Mode Parity",
        "",
        "This report compares the legacy enthalpy-state BVP with the direct temperature-state BVP.",
        "Temperature-state runs are warm-started from the converged enthalpy profile and then solved",
        "with temperature as the thermal state variable.",
        "",
        f"Accepted parity rows: {accepted} of {total}.",
        "",
        "| Case source | Case | Capture delta, pct-pt | Temperature RMSE delta, K | Parity |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in comparisons.itertuples(index=False):
        rmse = "" if row.temperature_rmse_delta_K is None or pd.isna(row.temperature_rmse_delta_K) else f"{row.temperature_rmse_delta_K:.4f}"
        cap = "" if row.capture_delta_pct_point is None or pd.isna(row.capture_delta_pct_point) else f"{row.capture_delta_pct_point:.4f}"
        lines.append(f"| {row.case_source} | {row.case_id} | {cap} | {rmse} | {row.parity_ok} |")
    lines.extend(
        [
            "",
            "Rows that fail parity should keep the enthalpy formulation as the validation reference",
            "until the direct temperature equations and initialization are improved for that regime.",
        ]
    )
    (REPORT_DIR / "thermal_state_mode_parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
