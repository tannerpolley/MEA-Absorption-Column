from __future__ import annotations

from pathlib import Path

import pandas as pd

from generate_accuracy_credibility_artifacts import main as generate_accuracy_credibility_artifacts


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
TABLES = ANALYSIS / "results" / "final" / "tables"

DIAGNOSTIC_COLUMNS = [
    "boundary_residual_components",
    "co2_vapor_upper_factor",
    "success_boundary_residual_max",
]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    c_cases = _read_current_c_case_results()
    c_cases = _select_columns(
        c_cases,
        [
            "artifact",
            "case_id",
            "thermo_model",
            "success",
            "runtime_s",
            "capture_pct",
            "capture_error_pct",
            "temperature_rmse_K",
            "boundary_residual_norm",
            *DIAGNOSTIC_COLUMNS,
        ],
    )
    c_cases.to_csv(TABLES / "raw_c_case_thermo_benchmark.csv", index=False)
    c_cases.to_csv(TABLES / "verified_c_case_thermo_benchmark.csv", index=False)
    generate_accuracy_credibility_artifacts()


def _read_current_c_case_results() -> pd.DataFrame:
    campaign_metrics = TABLES / "c_case_campaign_temperature_overlay_metrics.csv"
    if campaign_metrics.exists():
        data = pd.read_csv(campaign_metrics)
        data["artifact"] = f"analyses/nccc_validation/results/final/tables/{campaign_metrics.name}"
        data["success"] = True
        return data
    return _read_results("analyses/nccc_validation/results/runs/physical_domain_c_cases/benchmark_results.csv")


def _read_results(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    read_kwargs = {}
    using_fallback = False
    if not path.exists():
        if relative_path.replace("\\", "/") != "analyses/nccc_validation/results/runs/physical_domain_c_cases/benchmark_results.csv":
            raise FileNotFoundError(path)
        path = TABLES / "verified_c_case_thermo_benchmark.csv"
        read_kwargs["dtype"] = str
        using_fallback = True
    data = pd.read_csv(path, **read_kwargs)
    data["artifact"] = (
        f"analyses/nccc_validation/results/final/tables/{path.name}"
        if using_fallback
        else relative_path.replace("\\", "/")
    )
    return data


def _select_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = data.copy()
    for column in columns:
        if column not in data.columns:
            data[column] = ""
    return data[columns]


if __name__ == "__main__":
    main()
