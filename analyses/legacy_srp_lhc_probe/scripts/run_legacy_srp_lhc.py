from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
DATA_PATH = SRC_ROOT / "mea_absorption_column" / "data" / "LHC_design_w_SRP_cases.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "analyses" / "legacy_srp_lhc_probe" / "results"


def _install_pcsaft_import_shim() -> None:
    """Provide the unused legacy pcsaft symbols needed for import-time setup."""
    if "pcsaft" in sys.modules:
        return

    module = types.ModuleType("pcsaft")

    class InputError(Exception):
        pass

    def pcsaft_den(*args, **kwargs):
        return 1.0

    def pcsaft_fugcoef(*args, **kwargs):
        x = kwargs.get("x")
        if x is None and len(args) >= 3:
            x = args[2]
        n = len(x) if x is not None else 1
        return np.ones(n)

    module.InputError = InputError
    module.pcsaft_den = pcsaft_den
    module.pcsaft_fugcoef = pcsaft_fugcoef
    sys.modules["pcsaft"] = module


def _load_lhc() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.sort_values("Run").reset_index(drop=True)


def _legacy_input_frame(lhc: pd.DataFrame) -> pd.DataFrame:
    """Map SRP-LHC columns to the legacy run_model input shape."""
    mapped = pd.DataFrame(
        {
            "L_G": lhc["L"] / lhc["V"],
            "Fv_T": lhc["V"],
            "alpha": lhc["alpha"],
            "w_MEA_unloaded": lhc["w_MEA"],
            "y_CO2": lhc["y_CO2"],
            "Tl_z": lhc["Tl"],
            "Tv_0": lhc["Tv"],
            "P": 109180.0,
            "beds": 1.0,
            "y_H2O_explicit": lhc["y_H2O"],
        }
    )
    return mapped


def _row_metadata(row_index: int) -> dict[str, object]:
    row = _load_lhc().iloc[row_index]
    return {
        "run": int(row["Run"]),
        "L": float(row["L"]),
        "V": float(row["V"]),
        "L_G": float(row["L"] / row["V"]),
        "alpha": float(row["alpha"]),
        "w_MEA": float(row["w_MEA"]),
        "y_CO2": float(row["y_CO2"]),
        "y_H2O_input": float(row["y_H2O"]),
        "y_H2O_legacy_converter": float(row["y_CO2"] * 0.9626010166),
        "Tl_K": float(row["Tl"]),
        "Tv_K": float(row["Tv"]),
    }


def _probe_convert_data(
    df: pd.DataFrame,
    run: int = 0,
    type: str = "mole",
    geometry: str = "srp",
    water_mode: str = "explicit",
):
    """Legacy convert_data equivalent with selectable SRP vapor-water handling."""
    if type != "mole":
        raise ValueError("The SRP-LHC probe only supports mole inputs.")

    from mea_absorption_column.config.Constants import MWs_l, MWs_v, column_params, packing_params, n

    row = df.iloc[run]
    L_G = float(row["L_G"])
    Fv_T = float(row["Fv_T"])
    alpha = float(row["alpha"])
    w_MEA_unloaded = float(row["w_MEA_unloaded"])
    y_CO2 = float(row["y_CO2"])
    Tl_z = float(row["Tl_z"])
    Tv_0 = float(row["Tv_0"])
    P = float(row["P"])
    beds = float(row["beds"])
    if water_mode == "explicit":
        y_H2O = float(row["y_H2O_explicit"])
    elif water_mode == "legacy-ratio":
        y_H2O = y_CO2 * 0.9626010166
    else:
        raise ValueError(f"Unknown water mode: {water_mode}")

    MW_CO2 = MWs_l[0]
    MW_MEA = MWs_l[1]
    MW_H2O = MWs_l[2]
    MW_N2 = MWs_v[2]
    MW_O2 = MWs_v[3]

    alpha_O2_N2 = 0.08485753604

    Fl_T = L_G * Fv_T
    x_MEA_unloaded = w_MEA_unloaded / (MW_MEA / MW_H2O + w_MEA_unloaded * (1 - MW_MEA / MW_H2O))
    x_H2O_unloaded = 1 - x_MEA_unloaded

    Fl_MEA_b = Fl_T * x_MEA_unloaded
    Fl_H2O_b = Fl_T * x_H2O_unloaded
    Fl_CO2_b = Fl_MEA_b * alpha
    Fl = [Fl_CO2_b, Fl_MEA_b, Fl_H2O_b]

    y_N2 = (1 - y_CO2 - y_H2O) / (1 + alpha_O2_N2)
    y_O2 = y_N2 * alpha_O2_N2
    if min(y_CO2, y_H2O, y_N2, y_O2) <= 0:
        raise ValueError(
            f"Invalid vapor composition for row {run + 1}: "
            f"CO2={y_CO2}, H2O={y_H2O}, N2={y_N2}, O2={y_O2}"
        )

    Fv_CO2_a = y_CO2 * Fv_T
    Fv_H2O_a = y_H2O * Fv_T
    Fv_N2_a = y_N2 * Fv_T
    Fv_O2_a = y_O2 * Fv_T
    Fv = [Fv_CO2_a, Fv_H2O_a, Fv_N2_a, Fv_O2_a]

    geometry_key = "SRP" if geometry == "srp" else "NCCC"
    D = column_params[geometry_key]["D"]
    H = column_params[geometry_key]["H"] * beds

    packing_data = packing_params["MellapakPlus252Y"]
    packing = (
        packing_data["a_p"],
        packing_data["eps"],
        packing_data["Cl"],
        packing_data["Cv"],
        packing_data["Cs"],
        packing_data["Cp_0"],
        packing_data["Ch"],
    )

    A = np.pi * 0.25 * D**2
    z = np.linspace(0, 1, n)
    X = (L_G, Fv_T, alpha, w_MEA_unloaded, y_CO2, Tl_z, Tv_0, P, beds)
    return [Fl, Fv, Tl_z, Tv_0, z, H, A, P, packing], X


def _bounded_single_shoot_solve(Y_a_scaled, Y_b_scaled, z, parameters):
    """Runner-local bounded shooting probe around the legacy Euler/ABS kernel."""
    from scipy.optimize import least_squares
    from mea_absorption_column.BVP.ABS_Column import abs_column
    from mea_absorption_column.BVP.Methods.Integration_Methods import eulers

    Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a, Hlf_a_guess, Hvf_a, P_a = Y_a_scaled
    Fl_CO2_b, Fl_H2O_b, _Fv_CO2_b_guess, _Fv_H2O_b_guess, Hlf_b, _Hvf_b_guess, _P_b = Y_b_scaled
    target = np.array([Fl_CO2_b, Fl_H2O_b, Hlf_b], dtype=float)
    scale = np.maximum(np.abs(target), 1.0)
    x0 = np.array([Fl_CO2_a_guess, Fl_H2O_a_guess, Hlf_a_guess], dtype=float)

    h_low = min(0.2 * Hlf_a_guess, 5.0 * Hlf_a_guess)
    h_high = max(0.2 * Hlf_a_guess, 5.0 * Hlf_a_guess)
    lower = np.array([1.0e-12, 1.0e-12, h_low], dtype=float)
    upper = np.array([max(10.0 * Fl_CO2_a_guess, 1.0), max(10.0 * Fl_H2O_a_guess, 1.0), h_high], dtype=float)
    x0 = np.clip(x0, lower + 1.0e-12, upper - 1.0e-12)

    def residual(x):
        Y_a_trial = [x[0], x[1], Fv_CO2_a, Fv_H2O_a, x[2], Hvf_a, P_a]
        try:
            with np.errstate(all="ignore"):
                Y_scaled, _, _, _ = eulers(abs_column, Y_a_trial, z, args=parameters)
        except Exception:
            return np.array([1.0e6, 1.0e6, 1.0e6])
        if not np.all(np.isfinite(Y_scaled)):
            return np.array([1.0e6, 1.0e6, 1.0e6])
        simulated = np.array([Y_scaled[0, -1], Y_scaled[1, -1], Y_scaled[4, -1]], dtype=float)
        if not np.all(np.isfinite(simulated)):
            return np.array([1.0e6, 1.0e6, 1.0e6])
        return (simulated - target) / scale

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        max_nfev=80,
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )

    Fl_CO2_a, Fl_H2O_a, Hlf_a = result.x
    Y_a_final = [Fl_CO2_a, Fl_H2O_a, Fv_CO2_a, Fv_H2O_a, Hlf_a, Hvf_a, P_a]
    Y_scaled, z, _success, _message = eulers(abs_column, Y_a_final, z, args=parameters)
    residual_norm = float(np.linalg.norm(residual(result.x), ord=np.inf))
    success = bool(result.success and np.all(np.isfinite(Y_scaled)) and residual_norm < 1.0e-2)
    message = f"{result.message}; scaled_residual_inf={residual_norm:.3e}; nfev={result.nfev}"
    return Y_scaled, z, "Bounded Single Shooting Probe", success, message


def _run_worker(row_index: int, method: str, geometry: str, water_mode: str) -> dict[str, object]:
    sys.path.insert(0, str(SRC_ROOT))
    _install_pcsaft_import_shim()

    import mea_absorption_column.Run_Model as run_model_module

    if geometry not in {"srp", "legacy-nccc"}:
        raise ValueError(f"Unknown geometry mode: {geometry}")

    lhc = _load_lhc()
    model_input = _legacy_input_frame(lhc)
    row = lhc.iloc[row_index]
    run_model_module.convert_data = lambda df, run=0, type="mole": _probe_convert_data(
        df, run=run, type=type, geometry=geometry, water_mode=water_mode
    )
    method_for_run = method
    if method == "single-bounded":
        run_model_module.single_shoot_solve = _bounded_single_shoot_solve
        method_for_run = "single"

    start = time.perf_counter()
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stdout):
        capture_pct, success = run_model_module.run_model(
            model_input,
            method=method_for_run,
            data_type="mole",
            run=row_index,
            show_info=False,
            save_run_results=False,
            plot_temperature=False,
        )
    runtime_s = time.perf_counter() - start

    return {
        "run": int(row["Run"]),
        "method": method,
        "geometry": geometry,
        "water_mode": water_mode,
        "status": "success" if bool(success) else "solver_failure",
        "success": bool(success),
        "capture_pct": float(capture_pct),
        "runtime_s": runtime_s,
        "message": captured_stdout.getvalue().strip()[:1000],
        "L": float(row["L"]),
        "V": float(row["V"]),
        "L_G": float(row["L"] / row["V"]),
        "alpha": float(row["alpha"]),
        "w_MEA": float(row["w_MEA"]),
        "y_CO2": float(row["y_CO2"]),
        "y_H2O_input": float(row["y_H2O"]),
        "y_H2O_legacy_converter": float(row["y_CO2"] * 0.9626010166),
        "y_H2O_used": float(row["y_H2O"] if water_mode == "explicit" else row["y_CO2"] * 0.9626010166),
        "Tl_K": float(row["Tl"]),
        "Tv_K": float(row["Tv"]),
    }


def _run_subprocess(row_index: int, method: str, geometry: str, water_mode: str, timeout_s: float) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--row-index",
        str(row_index),
        "--method",
        method,
        "--geometry",
        geometry,
        "--water-mode",
        water_mode,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        runtime_s = time.perf_counter() - start
        result = {
            "run": row_index + 1,
            "method": method,
            "geometry": geometry,
            "water_mode": water_mode,
            "status": "timeout",
            "success": False,
            "capture_pct": np.nan,
            "runtime_s": runtime_s,
            "wall_runtime_s": runtime_s,
            "timeout_s": timeout_s,
            "message": f"Timed out after {timeout_s:.1f} s",
        }
        result.update(_row_metadata(row_index))
        result["y_H2O_used"] = result["y_H2O_input"] if water_mode == "explicit" else result["y_H2O_legacy_converter"]
        return result

    runtime_s = time.perf_counter() - start
    if proc.returncode != 0:
        result = {
            "run": row_index + 1,
            "method": method,
            "geometry": geometry,
            "water_mode": water_mode,
            "status": "error",
            "success": False,
            "capture_pct": np.nan,
            "runtime_s": runtime_s,
            "wall_runtime_s": runtime_s,
            "timeout_s": timeout_s,
            "message": (proc.stderr or proc.stdout).strip()[:1000],
        }
        result.update(_row_metadata(row_index))
        result["y_H2O_used"] = result["y_H2O_input"] if water_mode == "explicit" else result["y_H2O_legacy_converter"]
        return result

    try:
        row = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {
            "run": row_index + 1,
            "method": method,
            "geometry": geometry,
            "water_mode": water_mode,
            "status": "bad_worker_output",
            "success": False,
            "capture_pct": np.nan,
            "runtime_s": runtime_s,
            "wall_runtime_s": runtime_s,
            "timeout_s": timeout_s,
            "message": proc.stdout.strip()[:1000],
        }
        result.update(_row_metadata(row_index))
        result["y_H2O_used"] = result["y_H2O_input"] if water_mode == "explicit" else result["y_H2O_legacy_converter"]
        return result
    row["wall_runtime_s"] = runtime_s
    row["timeout_s"] = timeout_s
    return row


def _write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results_path = output_dir / "legacy_srp_lhc_matrix.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby(["geometry", "water_mode", "method", "status"], dropna=False)
        .agg(
            runs=("run", "count"),
            median_runtime_s=("runtime_s", "median"),
            median_wall_runtime_s=("wall_runtime_s", "median"),
            median_capture_pct=("capture_pct", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "legacy_srp_lhc_summary.csv", index=False)

    lines = [
        "# Legacy SRP-LHC Probe Results",
        "",
        f"Rows tested: {len(results)}",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "Notes:",
        "- `single` is the legacy shooting-method path.",
        "- The `srp` geometry mode uses the SRP dimensions already present in `Constants.py`.",
        "- `explicit` uses the SRP-LHC `y_H2O` column; `legacy-ratio` reproduces the old converter's fixed CO2-water ratio.",
        "- `y_H2O_input`, `y_H2O_legacy_converter`, and `y_H2O_used` are reported so the inlet-water assumption is visible.",
        "- Interpretation for the manuscript: shooting is fast for smoother SRP-like cases, finite difference can be useful as an intermediate method, and collocation remains the more defensible reference method for NCCC-style validation because it handles coupled boundary conditions and sharper thermal behavior more systematically.",
    ]
    (output_dir / "legacy_srp_lhc_report.md").write_text("\n".join(lines), encoding="utf-8")


def _validate_existing_outputs(output_dir: Path) -> None:
    summary_path = output_dir / "legacy_srp_lhc_summary.csv"
    matrix_path = output_dir / "legacy_srp_lhc_matrix.csv"
    bounded_path = output_dir / "bounded_failed_rows" / "legacy_srp_lhc_summary.csv"
    missing = [path for path in [summary_path, matrix_path, bounded_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expected probe artifact(s): " + ", ".join(str(path) for path in missing))

    summary = pd.read_csv(summary_path)
    required = {
        ("srp", "explicit", "single", "success"): 20,
        ("srp", "explicit", "single", "error"): 5,
        ("srp", "legacy-ratio", "single", "success"): 25,
    }
    for key, expected_runs in required.items():
        geometry, water_mode, method, status = key
        rows = summary[
            (summary["geometry"] == geometry)
            & (summary["water_mode"] == water_mode)
            & (summary["method"] == method)
            & (summary["status"] == status)
        ]
        if rows.empty:
            raise AssertionError(f"Missing summary row for {key}")
        actual_runs = int(rows.iloc[0]["runs"])
        if actual_runs != expected_runs:
            raise AssertionError(f"Expected {expected_runs} runs for {key}, found {actual_runs}")

    bounded = pd.read_csv(bounded_path)
    rows = bounded[
        (bounded["geometry"] == "srp")
        & (bounded["water_mode"] == "explicit")
        & (bounded["method"] == "single-bounded")
        & (bounded["status"] == "solver_failure")
    ]
    if rows.empty or int(rows.iloc[0]["runs"]) != 5:
        raise AssertionError("Expected five bounded-probe failures for explicit-water failed rows.")
    print(f"Validated legacy SRP-LHC probe artifacts in {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["single"], choices=["single", "single-bounded", "collocation", "finite"])
    parser.add_argument("--geometry", default="srp", choices=["srp", "legacy-nccc"])
    parser.add_argument("--water-modes", nargs="+", default=["explicit"], choices=["explicit", "legacy-ratio"])
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rows", nargs="+", type=int, default=None, help="1-based SRP-LHC Run numbers to execute.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--validate-results-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--method", default="single", choices=["single", "single-bounded", "collocation", "finite"])
    parser.add_argument("--water-mode", default="explicit", choices=["explicit", "legacy-ratio"])
    args = parser.parse_args()

    if args.validate_results_only:
        _validate_existing_outputs(args.output_dir)
        return 0

    if args.worker:
        result = _run_worker(args.row_index, args.method, args.geometry, args.water_mode)
        print(json.dumps(result))
        return 0

    n_rows = len(_load_lhc())
    if args.rows:
        row_indices = [row_number - 1 for row_number in args.rows]
    else:
        if args.limit is not None:
            n_rows = min(n_rows, args.limit)
        row_indices = list(range(n_rows))

    rows: list[dict[str, object]] = []
    for water_mode in args.water_modes:
        for method in args.methods:
            for row_index in row_indices:
                result = _run_subprocess(row_index, method, args.geometry, water_mode, args.timeout_s)
                rows.append(result)
                print(
                    f"{water_mode:12s} {method:11s} row {row_index + 1:02d}: "
                    f"{result.get('status')} capture={result.get('capture_pct')} "
                    f"runtime={result.get('runtime_s'):.2f}s"
                )

    _write_outputs(rows, args.output_dir)
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
