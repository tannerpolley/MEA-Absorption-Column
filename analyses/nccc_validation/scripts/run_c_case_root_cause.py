from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mea_absorption_column.benchmark import BenchmarkSettings, run_benchmark


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ANALYSIS / "results" / "runs" / "c_case_trend_root_cause"


@dataclass(frozen=True)
class RunConfig:
    label: str
    solver_settings: dict[str, object]


CONFIGS: tuple[RunConfig, ...] = (
    RunConfig(
        "baseline_51_95",
        {
            "mesh_points": 51,
            "tol": 0.5,
            "bc_tol": 0.001,
            "max_nodes": 1000,
            "co2_capture_guess_pct": 95.0,
        },
    ),
    RunConfig(
        "coarse_21_80",
        {
            "mesh_points": 21,
            "tol": 1.0,
            "bc_tol": 0.01,
            "max_nodes": 400,
            "co2_capture_guess_pct": 80.0,
        },
    ),
    RunConfig(
        "fine_101_99",
        {
            "mesh_points": 101,
            "tol": 0.2,
            "bc_tol": 0.0001,
            "max_nodes": 2000,
            "co2_capture_guess_pct": 99.0,
        },
    ),
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for config in CONFIGS:
        config_dir = output_root / config.label
        settings = BenchmarkSettings(
            methods=("scipy-bvp",),
            thermo_models=tuple(args.thermo_models),
            output_dir=config_dir,
            c_case_limit=None,
            nccc_case_limit=0,
            srp_case_limit=0,
            c_case_ids=tuple(args.case_ids),
            staged_beds=False,
            solver_settings=config.solver_settings,
            profile_pngs=True,
            profile_csvs=True,
            subprocess_timeout_s=args.subprocess_timeout_s,
        )
        result = run_benchmark(settings)
        result.insert(0, "run_config", config.label)
        all_rows.append(result)

    rows = pd.concat(all_rows, ignore_index=True)
    rows_path = output_root / "c_case_trend_root_cause_rows.csv"
    rows.to_csv(rows_path, index=False)

    summary = _summarize(rows)
    summary_path = output_root / "c_case_trend_root_cause_summary.csv"
    summary.to_csv(summary_path, index=False)

    profile_deltas = _summarize_profile_deltas(rows, baseline_label="baseline_51_95")
    profile_deltas_path = output_root / "c_case_trend_root_cause_profile_deltas.csv"
    profile_deltas.to_csv(profile_deltas_path, index=False)

    report = _write_report(rows_path, summary_path, profile_deltas_path, rows, summary, profile_deltas)
    report_path = output_root / "c_case_trend_root_cause_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {profile_deltas_path}")
    print(f"Wrote {report_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded one-bed C-case root-cause probes.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case-ids", nargs="+", default=["1C", "3C", "7C"])
    parser.add_argument("--thermo-models", nargs="+", default=["epcsaft_neutral"])
    parser.add_argument("--subprocess-timeout-s", type=float, default=60.0)
    return parser.parse_args(argv)


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    numeric = rows.copy()
    for column in ("runtime_s", "capture_error_pct", "temperature_rmse_K", "boundary_residual_norm"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    summary_rows = []
    for (run_config, thermo_model), group in numeric.groupby(["run_config", "thermo_model"], sort=False):
        capture = pd.to_numeric(group["capture_error_pct"], errors="coerce")
        runtime = pd.to_numeric(group["runtime_s"], errors="coerce")
        summary_rows.append(
            {
                "run_config": run_config,
                "thermo_model": thermo_model,
                "runs": int(len(group)),
                "successes": int(group["success"].astype(str).str.lower().eq("true").sum()),
                "timeouts": int(group["jacobian_status"].astype(str).str.contains("timeout", case=False, na=False).sum()),
                "capture_mae_pct": _safe_round(capture.abs().mean()),
                "capture_bias_pct": _safe_round(capture.mean()),
                "capture_error_min_pct": _safe_round(capture.min()),
                "capture_error_max_pct": _safe_round(capture.max()),
                "runtime_median_s": _safe_round(runtime.median()),
                "temperature_rmse_median_K": _safe_round(pd.to_numeric(group["temperature_rmse_K"], errors="coerce").median()),
                "boundary_residual_median": _safe_round(pd.to_numeric(group["boundary_residual_norm"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(summary_rows)


def _summarize_profile_deltas(rows: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    baseline = rows[rows["run_config"] == baseline_label].copy()
    if baseline.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for _, variant_row in rows[rows["run_config"] != baseline_label].iterrows():
        case_id = variant_row["case_id"]
        thermo_model = variant_row["thermo_model"]
        config = variant_row["run_config"]

        baseline_match = baseline[(baseline["case_id"] == case_id) & (baseline["thermo_model"] == thermo_model)]
        if baseline_match.empty:
            continue
        baseline_row = baseline_match.iloc[0]
        delta = _compare_temperature_profiles(baseline_row, variant_row)
        delta.update(
            {
                "case_id": case_id,
                "thermo_model": thermo_model,
                "run_config": config,
                "baseline_runtime_s": baseline_row.get("runtime_s"),
                "variant_runtime_s": variant_row.get("runtime_s"),
                "baseline_capture_error_pct": baseline_row.get("capture_error_pct"),
                "variant_capture_error_pct": variant_row.get("capture_error_pct"),
            }
        )
        records.append(delta)

    return pd.DataFrame(records)


def _compare_temperature_profiles(baseline_row: pd.Series, variant_row: pd.Series) -> dict[str, object]:
    baseline_dir = baseline_row.get("profile_csv_dir")
    variant_dir = variant_row.get("profile_csv_dir")
    baseline_path = Path(str(baseline_dir)) / "T.csv"
    variant_path = Path(str(variant_dir)) / "T.csv"
    if not baseline_path.exists() or not variant_path.exists():
        return {
            "profile_points": math.nan,
            "tl_profile_rmse_K": math.nan,
            "tv_profile_rmse_K": math.nan,
            "tl_profile_max_abs_K": math.nan,
            "tv_profile_max_abs_K": math.nan,
        }

    baseline = pd.read_csv(baseline_path)
    variant = pd.read_csv(variant_path)
    merged = _align_profile_frames(baseline, variant)
    if merged.empty:
        return {
            "profile_points": math.nan,
            "tl_profile_rmse_K": math.nan,
            "tv_profile_rmse_K": math.nan,
            "tl_profile_max_abs_K": math.nan,
            "tv_profile_max_abs_K": math.nan,
        }

    tl_diff = merged["Tl_variant"] - merged["Tl_baseline"]
    tv_diff = merged["Tv_variant"] - merged["Tv_baseline"]
    return {
        "profile_points": int(len(merged)),
        "tl_profile_rmse_K": _safe_round(float((tl_diff.pow(2).mean()) ** 0.5)),
        "tv_profile_rmse_K": _safe_round(float((tv_diff.pow(2).mean()) ** 0.5)),
        "tl_profile_max_abs_K": _safe_round(float(tl_diff.abs().max())),
        "tv_profile_max_abs_K": _safe_round(float(tv_diff.abs().max())),
    }


def _align_profile_frames(baseline: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline.copy()
    variant = variant.copy()
    if "Position" not in baseline.columns or "Position" not in variant.columns:
        return pd.DataFrame()

    merged_positions = pd.Index(sorted(set(baseline["Position"].astype(float)) | set(variant["Position"].astype(float))))
    baseline_interp = baseline.set_index("Position").reindex(merged_positions).interpolate(method="index").ffill().bfill()
    variant_interp = variant.set_index("Position").reindex(merged_positions).interpolate(method="index").ffill().bfill()
    aligned = pd.DataFrame(
        {
            "Position": merged_positions.astype(float),
            "Tl_baseline": baseline_interp["Tl"].astype(float),
            "Tv_baseline": baseline_interp["Tv"].astype(float),
            "Tl_variant": variant_interp["Tl"].astype(float),
            "Tv_variant": variant_interp["Tv"].astype(float),
        }
    )
    return aligned


def _write_report(
    rows_path: Path,
    summary_path: Path,
    profile_deltas_path: Path,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    profile_deltas: pd.DataFrame,
) -> str:
    lines = [
        "# C-Case Trend Root Cause Probe",
        "",
        "This probe checks whether the 1C-to-7C capture trend is being forced by solver mesh or initial-capture guesses.",
        "",
        f"- Rows: `{rows_path.as_posix()}`",
        f"- Summary: `{summary_path.as_posix()}`",
        f"- Profile deltas: `{profile_deltas_path.as_posix()}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No summary rows_",
        "",
        "## Profile Deltas vs Baseline",
        "",
        profile_deltas.to_markdown(index=False) if not profile_deltas.empty else "_No profile delta rows_",
        "",
        "## Per-Run Rows",
        "",
        rows[
            [
                "run_config",
                "case_id",
                "thermo_model",
                "success",
                "runtime_s",
                "capture_pct",
                "capture_error_pct",
                "temperature_rmse_K",
                "mesh_points",
                "tol",
                "bc_tol",
                "max_nodes",
                "co2_capture_guess_pct",
                "profile_csv_dir",
                "profile_png",
            ]
        ].to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def _safe_round(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    return round(float(value), 4)


if __name__ == "__main__":
    raise SystemExit(main())
