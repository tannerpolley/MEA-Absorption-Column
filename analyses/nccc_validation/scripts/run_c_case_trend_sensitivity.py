from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mea_absorption_column.benchmark import BenchmarkSettings, run_benchmark


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
RUNS = ANALYSIS / "results" / "runs"
FINAL_REPORTS = ANALYSIS / "results" / "final" / "reports"
CASE_DATA = ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_data.csv"


@dataclass(frozen=True)
class SensitivityConfig:
    label: str
    solver_settings: dict[str, object]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = tuple(args.case_ids)
    configs = _configs(args.config_set)
    all_rows: list[pd.DataFrame] = []

    for config in configs:
        config_dir = output_root / config.label
        settings = BenchmarkSettings(
            methods=("scipy-bvp",),
            thermo_models=tuple(args.thermo_models),
            output_dir=config_dir,
            c_case_ids=cases,
            nccc_case_limit=0,
            srp_case_limit=0,
            staged_beds=False,
            solver_settings=config.solver_settings,
            subprocess_timeout_s=args.timeout_s,
        )
        result = run_benchmark(settings)
        result.insert(0, "sensitivity_label", config.label)
        result.insert(1, "sensitivity_settings", _format_settings(config.solver_settings))
        all_rows.append(result)

    data = pd.concat(all_rows, ignore_index=True)
    data = _augment_with_case_inputs(data)
    rows_path = output_root / "c_case_trend_sensitivity_rows.csv"
    summary_path = output_root / "c_case_trend_sensitivity_summary.csv"
    report_path = output_root / "c_case_trend_sensitivity_report.md"
    data.to_csv(rows_path, index=False)
    summary = _summarize(data)
    summary.to_csv(summary_path, index=False)
    report = _write_report(data, summary, rows_path, summary_path)
    report_path.write_text(report, encoding="utf-8")

    if args.promote_report:
        FINAL_REPORTS.mkdir(parents=True, exist_ok=True)
        (FINAL_REPORTS / "c_case_trend_sensitivity_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded C-case sensitivity tests for capture-error trend diagnosis."
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=["1C", "3C", "7C"],
        help="C cases to run. Use all seven only after the smoke set is informative.",
    )
    parser.add_argument(
        "--thermo-models",
        nargs="+",
        default=["ideal_henry"],
        help="Thermo lanes to run. Start with ideal_henry because the trend is shared by both lanes.",
    )
    parser.add_argument(
        "--config-set",
        choices=["smoke", "mass-transfer", "thermal", "full"],
        default="smoke",
    )
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--output-root",
        default=str(RUNS / "c_case_trend_sensitivity"),
        help="Run-owned output directory.",
    )
    parser.add_argument("--promote-report", action="store_true")
    return parser.parse_args(argv)


def _configs(config_set: str) -> list[SensitivityConfig]:
    baseline = SensitivityConfig("baseline", {})
    mass = [
        SensitivityConfig(f"mass_transfer_{value:g}", {"mass_transfer_factor": value})
        for value in (0.35, 0.5, 0.75, 1.25)
    ]
    thermal = [
        SensitivityConfig(f"heat_transfer_{value:g}", {"heat_transfer_factor": value})
        for value in (0.5, 0.75, 1.25, 1.5)
    ]
    flux = [SensitivityConfig("absorption_only_flux", {"co2_flux_mode": "absorption_only"})]
    if config_set == "smoke":
        return [
            baseline,
            SensitivityConfig("mass_transfer_0.5", {"mass_transfer_factor": 0.5}),
            SensitivityConfig("mass_transfer_0.75", {"mass_transfer_factor": 0.75}),
            SensitivityConfig("heat_transfer_0.75", {"heat_transfer_factor": 0.75}),
            SensitivityConfig("input_o2_composition", {"vapor_composition_mode": "input_o2"}),
            flux[0],
        ]
    if config_set == "mass-transfer":
        return [baseline, *mass]
    if config_set == "thermal":
        return [baseline, *thermal]
    return [baseline, *mass, *thermal, *flux]


def _augment_with_case_inputs(data: pd.DataFrame) -> pd.DataFrame:
    inputs = pd.read_csv(CASE_DATA).rename(columns={"Case": "case_id", "CO2 %": "target_capture_pct"})
    merged = data.merge(inputs, on="case_id", how="left")
    merged["L_over_G"] = pd.to_numeric(merged["L/G"], errors="coerce")
    merged["G"] = pd.to_numeric(merged["G"], errors="coerce")
    merged["alpha"] = pd.to_numeric(merged["alpha"], errors="coerce")
    merged["y_CO2"] = pd.to_numeric(merged["y_CO2"], errors="coerce")
    merged["target_capture_pct"] = pd.to_numeric(merged["target_capture_pct"], errors="coerce")
    merged["success_bool"] = merged["success"].astype(str).str.lower().eq("true")
    return merged


def _summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, thermo_model), group in data.groupby(["sensitivity_label", "thermo_model"], sort=False):
        error = pd.to_numeric(group["capture_error_pct"], errors="coerce")
        runtime = pd.to_numeric(group["runtime_s"], errors="coerce")
        rows.append(
            {
                "sensitivity_label": label,
                "thermo_model": thermo_model,
                "runs": int(len(group)),
                "successes": int(group["success_bool"].sum()),
                "timeouts": int(group["jacobian_status"].astype(str).str.contains("timeout", case=False, na=False).sum()),
                "capture_mae_pct": _safe_round(error.abs().mean()),
                "capture_bias_pct": _safe_round(error.mean()),
                "max_abs_capture_error_pct": _safe_round(error.abs().max()),
                "runtime_median_s": _safe_round(runtime.median()),
                "pearson_error_L_over_G": _safe_round(_corr(error, group["L_over_G"])),
                "pearson_error_G": _safe_round(_corr(error, group["G"])),
                "settings": group["sensitivity_settings"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def _write_report(data: pd.DataFrame, summary: pd.DataFrame, rows_path: Path, summary_path: Path) -> str:
    lines = [
        "# C-Case Capture-Trend Sensitivity",
        "",
        "This diagnostic tests global model settings against the one-bed NCCC C-case capture-error trend. "
        "It is a model-closure screen, not a case-specific tuning table.",
        "",
        f"- Detailed rows: `{rows_path.as_posix()}`",
        f"- Summary rows: `{summary_path.as_posix()}`",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Per-Case Rows",
        "",
        _markdown_table(data[
            [
                "sensitivity_label",
                "case_id",
                "thermo_model",
                "success",
                "runtime_s",
                "capture_pct",
                "capture_error_pct",
                "temperature_rmse_K",
                "L_over_G",
                "G",
                "jacobian_status",
            ]
        ]),
        "",
    ]
    return "\n".join(lines)


def _format_settings(settings: dict[str, object]) -> str:
    if not settings:
        return "baseline"
    return ";".join(f"{key}={value}" for key, value in sorted(settings.items()))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    text = frame.copy()
    for column in text.columns:
        text[column] = text[column].map(lambda value: "" if pd.isna(value) else str(value))

    header = "| " + " | ".join(text.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = [
        "| " + " | ".join(row[column] for column in text.columns) + " |"
        for _, row in text.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _corr(a: pd.Series, b: pd.Series) -> float:
    frame = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(frame) < 3:
        return math.nan
    if frame.iloc[:, 0].std() == 0 or frame.iloc[:, 1].std() == 0:
        return math.nan
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def _safe_round(value: float) -> float:
    if pd.isna(value):
        return math.nan
    return round(float(value), 4)


if __name__ == "__main__":
    raise SystemExit(main())
