from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
RUNS = ANALYSIS / "results" / "runs"
TABLES = ANALYSIS / "results" / "final" / "tables"

DIAGNOSTIC_COLUMNS = [
    "boundary_residual_components",
    "co2_vapor_upper_factor",
    "success_boundary_residual_max",
]


def _select_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = data.copy()
    for column in columns:
        if column not in data.columns:
            data[column] = ""
    return data[columns]


def _read_results(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    read_kwargs = {}
    using_fallback = False
    if not path.exists():
        fallback_name = {
            "analyses/nccc_validation/results/runs/physical_domain_c_cases/benchmark_results.csv": "verified_c_case_thermo_benchmark.csv",
            "analyses/nccc_validation/results/runs/goal_k1_k3_flux_strength_factor_multistart_v2/benchmark_results.csv": "verified_staged_kcase_benchmark.csv",
        }.get(relative_path.replace("\\", "/"))
        if fallback_name is None:
            raise FileNotFoundError(path)
        path = TABLES / fallback_name
        read_kwargs["dtype"] = str
        using_fallback = True
    data = pd.read_csv(path, **read_kwargs)
    data["artifact"] = (
        f"analyses/nccc_validation/results/final/tables/{path.name}"
        if using_fallback
        else relative_path.replace("\\", "/")
    )
    return data


def _normalize_artifact(data: pd.DataFrame, filename: str) -> pd.DataFrame:
    data = data.copy()
    data["artifact"] = f"analyses/nccc_validation/results/final/tables/{filename}"
    return data


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    c_cases = _read_results("analyses/nccc_validation/results/runs/physical_domain_c_cases/benchmark_results.csv")
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

    staged = _read_results("analyses/nccc_validation/results/runs/goal_k1_k3_flux_strength_factor_multistart_v2/benchmark_results.csv")
    staged = _add_nccc_metadata(staged)
    staged = _select_columns(
        staged,
        [
            "artifact",
            "case_id",
            "thermo_model",
            "success",
            "runtime_s",
            "capture_pct",
            "capture_error_pct",
            "boundary_residual_norm",
            "beds",
            "intercoolers",
            "staged_beds",
            "intercooler_assumption",
            *DIAGNOSTIC_COLUMNS,
            "continuation_path",
            "message",
        ],
    )
    staged.to_csv(TABLES / "raw_staged_kcase_benchmark.csv", index=False)
    staged.to_csv(TABLES / "verified_staged_kcase_benchmark.csv", index=False)

    epcsaft_smoke = _normalize_artifact(
        pd.read_csv(TABLES / "staged_epcsaft_smoke.csv", dtype=str),
        "staged_epcsaft_smoke.csv",
    )
    for column in DIAGNOSTIC_COLUMNS:
        if column not in epcsaft_smoke.columns:
            epcsaft_smoke[column] = ""
    epcsaft_smoke.to_csv(TABLES / "staged_epcsaft_smoke.csv", index=False)
    epcsaft_smoke.to_csv(TABLES / "raw_staged_epcsaft_smoke.csv", index=False)

    unresolved = _normalize_artifact(
        pd.read_csv(TABLES / "kcase_unresolved_diagnostics.csv", dtype=str),
        "kcase_unresolved_diagnostics.csv",
    )
    for column in DIAGNOSTIC_COLUMNS:
        if column not in unresolved.columns:
            unresolved[column] = ""
    unresolved.to_csv(TABLES / "kcase_unresolved_diagnostics.csv", index=False)
    unresolved.to_csv(TABLES / "raw_kcase_unresolved_diagnostics.csv", index=False)

    recoveries = _normalize_artifact(
        pd.read_csv(TABLES / "kcase_sensitivity_recoveries.csv", dtype=str),
        "kcase_sensitivity_recoveries.csv",
    )
    for column in DIAGNOSTIC_COLUMNS:
        if column not in recoveries.columns:
            recoveries[column] = ""
    recoveries.to_csv(TABLES / "kcase_sensitivity_recoveries.csv", index=False)
    recoveries.to_csv(TABLES / "raw_kcase_sensitivity_recoveries.csv", index=False)

    epcsaft_recovery = _normalize_artifact(
        pd.read_csv(TABLES / "staged_epcsaft_recovery_probe.csv", dtype=str),
        "staged_epcsaft_recovery_probe.csv",
    )
    for column in DIAGNOSTIC_COLUMNS:
        if column not in epcsaft_recovery.columns:
            epcsaft_recovery[column] = ""
    epcsaft_recovery.to_csv(TABLES / "staged_epcsaft_recovery_probe.csv", index=False)
    epcsaft_recovery.to_csv(TABLES / "raw_staged_epcsaft_recovery_probe.csv", index=False)

    k2_blend = _normalize_artifact(
        pd.read_csv(TABLES / "staged_epcsaft_k2_blend_probe.csv", dtype=str),
        "staged_epcsaft_k2_blend_probe.csv",
    )
    for column in DIAGNOSTIC_COLUMNS:
        if column not in k2_blend.columns:
            k2_blend[column] = ""
    k2_blend.to_csv(TABLES / "staged_epcsaft_k2_blend_probe.csv", index=False)
    k2_blend.to_csv(TABLES / "raw_staged_epcsaft_k2_blend_probe.csv", index=False)


def _add_nccc_metadata(data: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.read_csv(ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_Data.csv")
    metadata = metadata[["Runs", "Beds", "Intercoolers"]].rename(
        columns={"Runs": "case_id", "Beds": "beds", "Intercoolers": "intercoolers"}
    )
    data = data.drop(columns=[c for c in ("beds", "intercoolers", "staged_beds", "intercooler_assumption") if c in data.columns])
    data = data.merge(metadata, on="case_id", how="left")
    data["staged_beds"] = data["beds"].astype(float).gt(1) | data["intercoolers"].astype(float).gt(0)
    data["intercooler_assumption"] = data["intercoolers"].astype(float).gt(0).map(
        {True: "Tl_feed_target", False: "none"}
    )
    return data


if __name__ == "__main__":
    main()
