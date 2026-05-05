from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"


def _read_results(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    data = pd.read_csv(path)
    data["artifact"] = relative_path.replace("\\", "/")
    return data


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    c_cases = _read_results("benchmark_artifacts/physical_domain_c_cases/benchmark_results.csv")
    c_cases = c_cases[
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
        ]
    ]
    c_cases.to_csv(OUT / "raw_c_case_thermo_benchmark.csv", index=False)

    staged = _read_results("benchmark_artifacts/goal_k1_k3_flux_strength_factor_multistart_v2/benchmark_results.csv")
    staged = staged[
        [
            "artifact",
            "case_id",
            "thermo_model",
            "success",
            "runtime_s",
            "capture_pct",
            "capture_error_pct",
            "boundary_residual_norm",
            "continuation_path",
            "message",
        ]
    ]
    staged.to_csv(OUT / "raw_staged_kcase_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
