from __future__ import annotations

from pathlib import Path

import argparse
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mea_absorption_column.benchmark import BenchmarkSettings, run_benchmark


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_data.csv"
RUNS = ANALYSIS / "results" / "runs" / "c_case_sensitivity_matrix"


@dataclass(frozen=True)
class SensitivityCase:
    label: str
    mass_transfer_factor: float | None = None
    heat_transfer_factor: float | None = None
    co2_flux_mode: str | None = None


SENSITIVITY_CASES: tuple[SensitivityCase, ...] = (
    SensitivityCase("baseline"),
    SensitivityCase("mass_transfer_0.80", mass_transfer_factor=0.8),
    SensitivityCase("mass_transfer_1.20", mass_transfer_factor=1.2),
    SensitivityCase("heat_transfer_0.80", heat_transfer_factor=0.8),
    SensitivityCase("heat_transfer_1.20", heat_transfer_factor=1.2),
    SensitivityCase("mass_1.20_heat_0.90", mass_transfer_factor=1.2, heat_transfer_factor=0.9),
    SensitivityCase("co2_flux_absorption_only", co2_flux_mode="absorption_only"),
)


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small bounded c-case sensitivity matrix for trend diagnosis."
    )
    parser.add_argument("--output-dir", default=str(RUNS))
    parser.add_argument("--c-case-ids", nargs="+", default=None)
    parser.add_argument("--c-case-limit", type=int, default=None)
    parser.add_argument("--subprocess-timeout-s", type=float, default=60.0)
    parser.add_argument("--max-runtime-s", type=float, default=20.0)
    parser.add_argument("--mesh-points", type=int, default=51)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--co2-capture-guess-pct", type=float, default=95.0)
    parser.add_argument("--run-epcsaft", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = _case_ids()
    if args.c_case_ids is not None:
        case_ids = [case_id for case_id in case_ids if case_id in set(args.c_case_ids)]
    if args.c_case_limit is not None:
        case_ids = case_ids[: args.c_case_limit]
    if not case_ids:
        raise ValueError("No c-case ids selected for sensitivity matrix.")

    baseline_rows = _run_suite(
        case_ids=case_ids,
        label="ideal_henry",
        output_dir=output_dir,
        args=args,
        include_epcsaft=False,
    )

    epcsaft_rows: list[dict] = []
    if args.run_epcsaft:
        epcsaft_rows = _run_suite(
            case_ids=case_ids,
            label="epcsaft_ionic",
            output_dir=output_dir,
            args=args,
            include_epcsaft=True,
        )

    all_rows = baseline_rows + epcsaft_rows
    if not all_rows:
        raise RuntimeError("No benchmark rows were produced.")

    results = pd.DataFrame(all_rows)
    results = _coerce_row_types(results)
    results_path = output_dir / "sensitivity_matrix_results.csv"
    results.to_csv(results_path, index=False)

    case_inputs = _load_case_inputs(case_ids)
    enriched = _add_case_proxies(results, case_inputs)
    enriched_path = output_dir / "sensitivity_matrix_results_enriched.csv"
    enriched.to_csv(enriched_path, index=False)

    report = _build_report(enriched)
    report_path = output_dir / "sensitivity_matrix_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote {len(enriched)} rows to {enriched_path}")
    print(f"Report: {report_path}")
    print(f"Failed/timeout rows: {int((~enriched['success']).sum())}")


def _run_suite(
    case_ids: list[str],
    label: str,
    output_dir: Path,
    args: argparse.Namespace,
    include_epcsaft: bool,
) -> list[dict]:
    solver_settings = {
        "mesh_points": args.mesh_points,
        "tol": 0.5,
        "bc_tol": 0.001,
        "max_nodes": args.max_nodes,
        "max_runtime_s": args.max_runtime_s,
        "co2_capture_guess_pct": args.co2_capture_guess_pct,
        "transform_mode": "bounded_guarded_raw_state",
    }
    rows: list[dict] = []
    for config in SENSITIVITY_CASES:
        config_solver = dict(solver_settings)
        run_solver = _to_settings_updates(config)
        config_solver.update(run_solver)
        run_settings = BenchmarkSettings(
            methods=("scipy-bvp",),
            thermo_models=(label,),
            output_dir=output_dir / _safe_path_part(label) / _safe_path_part(config.label),
            c_case_ids=tuple(case_ids),
            write_artifacts=False,
            c_case_limit=None,
            solver_settings=config_solver,
            profile_csvs=False,
            profile_pngs=False,
            subprocess_timeout_s=args.subprocess_timeout_s,
        )

        try:
            batch = run_benchmark(run_settings)
        except Exception as exc:
            for case_id in case_ids:
                rows.append(
                    _failed_row(
                        case_source="C_cases_data",
                        case_id=case_id,
                        method="scipy-bvp",
                        thermo_model=label,
                        success=False,
                        message=f"Suite run failed before per-case loops: {exc}",
                        config=config,
                        include_epcsaft=include_epcsaft,
                    )
                )
            continue

        for row in batch.to_dict("records"):
            row["sensitivity_label"] = config.label
            if config.mass_transfer_factor is not None:
                row["mass_transfer_factor"] = config.mass_transfer_factor
            if config.heat_transfer_factor is not None:
                row["heat_transfer_factor"] = config.heat_transfer_factor
            if config.co2_flux_mode is not None:
                row["co2_flux_mode"] = config.co2_flux_mode
            row["include_epcsaft"] = include_epcsaft
            row["run_label"] = f"{label}:{config.label}"
            rows.append(row)
    return rows


def _to_settings_updates(config: SensitivityCase) -> dict[str, object]:
    updates: dict[str, object] = {}
    if config.mass_transfer_factor is not None:
        updates["mass_transfer_factor"] = config.mass_transfer_factor
    if config.heat_transfer_factor is not None:
        updates["heat_transfer_factor"] = config.heat_transfer_factor
    if config.co2_flux_mode is not None:
        updates["co2_flux_mode"] = config.co2_flux_mode
    return updates


def _failed_row(
    case_source: str,
    case_id: str,
    method: str,
    thermo_model: str,
    success: bool,
    message: str,
    config: SensitivityCase,
    include_epcsaft: bool,
) -> dict:
    return {
        "case_source": case_source,
        "case_id": case_id,
        "method": method,
        "thermo_model": thermo_model,
        "success": success,
        "message": message,
        "runtime_s": float("nan"),
        "capture_error_pct": float("nan"),
        "capture_pct": float("nan"),
        "raw_capture_pct": float("nan"),
        "temperature_rmse_K": float("nan"),
        "boundary_residual_norm": float("nan"),
        "mass_transfer_factor": config.mass_transfer_factor,
        "heat_transfer_factor": config.heat_transfer_factor,
        "co2_flux_mode": config.co2_flux_mode or "bidirectional",
        "sensitivity_label": config.label,
        "include_epcsaft": include_epcsaft,
        "run_label": f"{thermo_model}:{config.label}",
    }



def _case_ids() -> list[str]:
    data = pd.read_csv(CASES_PATH)
    return [str(case_id) for case_id in data["Case"].tolist()]


def _load_case_inputs(case_ids: list[str]) -> pd.DataFrame:
    case_inputs = pd.read_csv(CASES_PATH)
    case_inputs = case_inputs.set_index("Case")
    return case_inputs.loc[case_ids].copy().reset_index().rename(columns={"Case": "case_id"})


def _add_case_proxies(results: pd.DataFrame, case_inputs: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(case_inputs, on="case_id", how="left")
    if merged.empty:
        return merged
    merged["case_seq"] = merged["case_id"].apply(_case_sequence)
    merged["gas_rate_G"] = merged["G"]
    merged["L_over_G"] = merged["L/G"]
    merged["L_proxy"] = merged["G"] * merged["L/G"]
    merged["lean_loading_proxy"] = merged["y_CO2"] / merged["w_MEA"].replace(0, np.nan)
    merged["inlet_co2_mol_frac"] = merged["y_CO2"]
    merged["capacity_ratio_proxy"] = merged["CO2 %"] / merged["lean_loading_proxy"]
    merged["capture_success"] = merged["success"].fillna(False).astype(bool)
    return merged


def _coerce_row_types(results: pd.DataFrame) -> pd.DataFrame:
    for col in ("capture_error_pct", "runtime_s", "temperature_rmse_K", "boundary_residual_norm"):
        results[col] = pd.to_numeric(results[col], errors="coerce")
    results["success"] = results["success"].astype(str).str.lower().isin({"1", "true", "yes"})
    return results


def _build_report(enriched: pd.DataFrame) -> str:
    report_lines = ["# C-Case Sensitivity Matrix Report", ""]
    report_lines.append("## Scope")
    report_lines.append("- Method: `scipy-bvp`")
    report_lines.append("- Cases: 1C–7C")
    report_lines.append("- Thermo: `ideal_henry` (baseline matrix); ePC-SAFT rows are excluded unless `--run-epcsaft` is set.")

    ideal = enriched[enriched["thermo_model"] == "ideal_henry"].copy()
    if ideal.empty:
        return "\n".join(report_lines + ["", "No ideal_henry rows were produced."])

    report_lines.append("")
    report_lines.append("## Run Statistics")
    report_lines.append(f"- Total rows: {len(enriched)}")
    report_lines.append(f"- Successful rows: {int(ideal['success'].sum()) + int((enriched[enriched['thermo_model'] != 'ideal_henry']['success']).sum())}")
    report_lines.append(f"- Failed/timeout rows: {int((~ideal['success']).sum())} (ideal_henry)")

    failure_rows = ideal[~ideal["success"]][
        ["case_id", "sensitivity_label", "message"]
    ].fillna({"message": "no message"})
    report_lines.append("")
    report_lines.append("## Failure Rows (ideal_henry)")
    if failure_rows.empty:
        report_lines.append("- None")
    else:
        report_lines.append(failure_rows.to_markdown(index=False))

    report_lines.extend(_section_trends(ideal))
    report_lines.extend(_section_correlations(ideal))
    report_lines.extend(_section_holdout(ideal))
    return "\n".join(report_lines) + "\n"


def _section_trends(ideal: pd.DataFrame) -> list[str]:
    lines: list[str] = ["", "## Trend and Case Ordering"]
    valid = ideal.dropna(subset=["capture_error_pct", "case_seq"]).copy()
    if valid.empty:
        lines.append("- No valid rows for trend fit.")
        return lines
    trend_rows = []
    for label, subset in valid.groupby("sensitivity_label", dropna=False):
        if len(subset) < 3:
            continue
        x = subset["case_seq"].to_numpy(dtype=float)
        y = subset["capture_error_pct"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(np.unique(x)) > 1 else math.nan
        mean = float(np.nanmean(np.abs(y)))
        trend_rows.append((label, slope, mean))
    lines.append("| config | slope(capture_error vs case index) | mean |abs| capture_error |")
    lines.append("| --- | ---: | ---: |")
    for label, slope, mean in sorted(trend_rows, key=lambda item: item[0]):
        lines.append(f"| {label} | {slope:+.3f} | {mean:.3f} |")
    return lines


def _section_correlations(ideal: pd.DataFrame) -> list[str]:
    lines: list[str] = ["", "## Correlations (ideal_henry)"]
    proxy_columns = [
        "L_over_G",
        "gas_rate_G",
        "w_MEA",
        "lean_loading_proxy",
        "inlet_co2_mol_frac",
        "CO2 %",
        "L_proxy",
        "capacity_ratio_proxy",
    ]
    lines.append("| config | " + " | ".join(proxy_columns) + " |")
    lines.append("| --- |" + " --- |" * len(proxy_columns))
    for label, subset in ideal.groupby("sensitivity_label", dropna=False):
        subset = subset.dropna(subset=proxy_columns + ["capture_error_pct"]).copy()
        if subset.empty:
            rows = [""] * len(proxy_columns)
        else:
            rows = [f"{subset['capture_error_pct'].corr(subset[col]):+.3f}" for col in proxy_columns]
        lines.append(f"| {label} | " + " | ".join(rows) + " |")
    return lines


def _section_holdout(ideal: pd.DataFrame) -> list[str]:
    lines: list[str] = ["", "## Holdout Check (1C-6C train, 7C holdout)"]
    train_mask = ideal["case_seq"] <= 6
    holdout_mask = ideal["case_seq"] == 7
    rows = []
    for label, subset in ideal.groupby("sensitivity_label", dropna=False):
        train = subset[train_mask & (subset["case_id"].notna())]
        holdout = subset[holdout_mask & (subset["case_id"].notna())]
        if train.empty:
            train_mae = math.nan
        else:
            train_vals = train.loc[train["success"], "capture_error_pct"].to_numpy(dtype=float)
            train_mae = float(np.nanmean(np.abs(train_vals)))
        if holdout.empty:
            holdout_mae = math.nan
        else:
            holdout_vals = holdout.loc[holdout["success"], "capture_error_pct"].to_numpy(dtype=float)
            holdout_mae = float(np.nanmean(np.abs(holdout_vals)))
        rows.append((label, train_mae, holdout_mae))

    lines.append("| config | train MAE (1C-6C) | holdout MAE (7C) |")
    lines.append("| --- | ---: | ---: |")
    for label, train_mae, holdout_mae in sorted(rows, key=lambda item: item[0]):
        lines.append(f"| {label} | {train_mae:.3f} | {holdout_mae:.3f} |")
    baseline = next((row for row in rows if row[0] == "baseline"), None)
    if baseline is not None:
        b_train, b_holdout = baseline[1], baseline[2]
        lines.append("")
        lines.append("### Δ vs baseline")
        lines.append("| config | delta train MAE | delta holdout MAE |")
        lines.append("| --- | ---: | ---: |")
        for label, train_mae, holdout_mae in rows:
            if label == "baseline":
                continue
            dt = train_mae - b_train if not (math.isnan(train_mae) or math.isnan(b_train)) else math.nan
            dh = holdout_mae - b_holdout if not (math.isnan(holdout_mae) or math.isnan(b_holdout)) else math.nan
            lines.append(f"| {label} | {dt:+.3f} | {dh:+.3f} |")
    return lines


def _case_sequence(case_id: str) -> int:
    match = re.match(r"^([0-9]+)C$", str(case_id).strip())
    if match:
        return int(match.group(1))
    return 999


if __name__ == "__main__":
    main()
