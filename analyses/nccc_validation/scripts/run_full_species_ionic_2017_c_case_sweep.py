from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_IDS = ("1C", "2C", "3C", "4C", "5C", "6C", "7C")
CONFIG_NAME = "2025_Figiel_empirical_fitted_Born_SSM_DS"
THERMO_MODEL = "epcsaft_reactive_nine_activity_rebased"
DATASET_NAME = "MEA_CO2_H2O_ionic_fit"
INPUT_SOURCE = REPO_ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_model_inputs_mass.csv"
CONFIG_TABLE = (
    REPO_ROOT
    / "analyses"
    / "nccc_validation"
    / "results"
    / "final"
    / "tables"
    / "epcsaft_electrolyte_config_user_options.csv"
)
RAW_ROOT = (
    REPO_ROOT
    / "analyses"
    / "nccc_validation"
    / "results"
    / "runs"
    / "full_species_ionic_2017_c_cases"
)
FINAL_CSV = (
    REPO_ROOT
    / "analyses"
    / "nccc_validation"
    / "results"
    / "final"
    / "tables"
    / "full_species_ionic_2017_c_case_sweep.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and/or aggregate the corrected NCCC 2017 C-case sweep for the "
            "nine-species ePC-SAFT activity-rebased speciation model."
        )
    )
    parser.add_argument(
        "--run-cases",
        action="store_true",
        help="Run the seven benchmark cases before aggregating the final CSV.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=list(CASE_IDS),
        help="Case IDs to run or aggregate. Defaults to 1C through 7C.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
        help="Run-output root containing one benchmark_results.csv per case.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=FINAL_CSV,
        help="Curated final CSV to write.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for benchmark subprocesses.",
    )
    parser.add_argument(
        "--subprocess-timeout-s",
        type=float,
        default=900.0,
        help="Timeout passed through to mea_absorption_column.benchmark.",
    )
    return parser.parse_args()


def _git_head() -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=True,
        )
        full = result.stdout.strip()
        return full[:7], full
    except Exception:
        git_file = REPO_ROOT / ".git"
        if not git_file.exists():
            return "", ""
        text = git_file.read_text(encoding="utf-8").strip()
        git_dir = Path(text.replace("gitdir:", "").strip()) if text.startswith("gitdir:") else git_file
        head_text = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head_text.startswith("ref: "):
            ref = head_text.replace("ref: ", "", 1)
            full = (git_dir / ref).read_text(encoding="utf-8").strip()
        else:
            full = head_text
        return full[:7], full


def _config_row() -> pd.Series:
    configs = pd.read_csv(CONFIG_TABLE)
    match = configs.loc[configs["config"] == CONFIG_NAME]
    if len(match) != 1:
        raise RuntimeError(f"Expected exactly one config row for {CONFIG_NAME}, found {len(match)}")
    return match.iloc[0]


def _portable_source_note(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(
        r"[A-Za-z]:[\\/][^;`|]*?[\\/]data[\\/]reference[\\/]epcsaft_parameters[\\/]"
        r"([^\\/;`|]+)[\\/]user_options\.json",
        r"epcsaft reference data: epcsaft_parameters/\1/user_options.json",
        text,
    )


def _benchmark_env(user_options_json: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MEA_EPCSAFT_DATASET_NAME": DATASET_NAME,
            "MEA_EPCSAFT_USER_OPTIONS_JSON": user_options_json,
            "MEA_EPCSAFT_REACTIVE_MAX_ITERATIONS": "160",
        }
    )
    return env


def _run_case(case_id: str, raw_root: Path, python_executable: str, timeout_s: float, env: dict[str, str]) -> None:
    output_dir = raw_root / case_id
    command = [
        python_executable,
        "-m",
        "mea_absorption_column.benchmark",
        "--methods",
        "scipy-bvp",
        "--thermo-models",
        THERMO_MODEL,
        "--output-dir",
        str(output_dir),
        "--nccc-dataset",
        "2017",
        "--data-type",
        "mass",
        "--nccc-case-ids",
        case_id,
        "--c-case-limit",
        "0",
        "--srp-case-limit",
        "0",
        "--staged-beds",
        "false",
        "--mesh-points",
        "7",
        "--tol",
        "10",
        "--bc-tol",
        "0.5",
        "--max-nodes",
        "80",
        "--subprocess-timeout-s",
        str(timeout_s),
    ]
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _aggregate(case_ids: list[str], raw_root: Path, output_csv: Path, config_row: pd.Series) -> pd.DataFrame:
    inputs = pd.read_csv(INPUT_SOURCE, index_col=0)
    head_short, head_full = _git_head()
    rows = []
    for case_id in case_ids:
        raw_csv = raw_root / case_id / "benchmark_results.csv"
        if not raw_csv.exists():
            raise FileNotFoundError(f"Missing raw result CSV for {case_id}: {raw_csv}")
        result = pd.read_csv(raw_csv)
        if len(result) != 1:
            raise RuntimeError(f"Expected one result row in {raw_csv}, found {len(result)}")
        row = result.iloc[0].to_dict()
        if str(row.get("case_id")) != case_id:
            raise RuntimeError(f"Result case_id mismatch in {raw_csv}: {row.get('case_id')} != {case_id}")
        if row.get("case_source") != "NCCC_2017_cases":
            raise RuntimeError(f"Unexpected case_source for {case_id}: {row.get('case_source')}")
        if case_id not in inputs.index:
            raise RuntimeError(f"Missing corrected input row for {case_id} in {INPUT_SOURCE}")
        inp = inputs.loc[case_id].to_dict()

        rows.append(
            {
                "case_id": case_id,
                "case_source": row.get("case_source"),
                "input_source_file": "src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv",
                "nccc_dataset": "2017",
                "data_type": "mass",
                "git_head_short": head_short,
                "git_head_full": head_full,
                "thermo_model": row.get("thermo_model"),
                "chemical_equilibrium_model": row.get("chemical_equilibrium_model"),
                "epcsaft_dataset_name": DATASET_NAME,
                "epcsaft_config": CONFIG_NAME,
                "epcsaft_user_options_json": config_row.get("user_options_json"),
                "epcsaft_config_source_note": _portable_source_note(config_row.get("source_note")),
                "method": row.get("method"),
                "success": row.get("success"),
                "message": row.get("message"),
                "runtime_s": row.get("runtime_s"),
                "co2_capture_pct": row.get("capture_pct"),
                "target_co2_capture_pct": inp.get("CO2  %"),
                "capture_error_pct_pt": row.get("capture_error_pct"),
                "temperature_rmse_K": row.get("temperature_rmse_K"),
                "boundary_residual_norm": row.get("boundary_residual_norm"),
                "invalid_state_count": row.get("invalid_state_count"),
                "guard_penalty_count": row.get("guard_penalty_count"),
                "epcsaft_chemistry_solve_s": row.get("epcsaft_chemistry_solve_s"),
                "epcsaft_chemistry_max_mass_residual": row.get("epcsaft_chemistry_max_mass_residual"),
                "epcsaft_chemistry_max_reaction_residual": row.get("epcsaft_chemistry_max_reaction_residual"),
                "epcsaft_chemistry_max_charge_residual": row.get("epcsaft_chemistry_max_charge_residual"),
                "epcsaft_chemistry_failed_count": row.get("epcsaft_chemistry_failed_count"),
                "epcsaft_chemistry_accepted_best_effort_count": row.get(
                    "epcsaft_chemistry_accepted_best_effort_count"
                ),
                "epcsaft_chemistry_last_iterations": row.get("epcsaft_chemistry_last_iterations"),
                "epcsaft_chemistry_last_native_success": row.get("epcsaft_chemistry_last_native_success"),
                "epcsaft_chemistry_last_message": row.get("epcsaft_chemistry_last_message"),
                "mesh_points": row.get("mesh_points"),
                "tol": row.get("tol"),
                "bc_tol": row.get("bc_tol"),
                "max_nodes": row.get("max_nodes"),
                "staged_beds": row.get("staged_beds"),
                "beds": row.get("beds"),
                "intercoolers": row.get("intercoolers"),
                "continuation_success": row.get("continuation_success"),
                "scaling_mode": row.get("scaling_mode"),
                "transform_mode": row.get("transform_mode"),
                "python_version": row.get("python_version"),
                "platform": row.get("platform"),
                "package_versions": row.get("package_versions"),
                "raw_result_csv": str(raw_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
                "benchmark_command": (
                    "python -m mea_absorption_column.benchmark --methods scipy-bvp "
                    f"--thermo-models {THERMO_MODEL} --nccc-dataset 2017 --data-type mass "
                    "--nccc-case-ids <case_id> --c-case-limit 0 --srp-case-limit 0 "
                    "--staged-beds false --mesh-points 7 --tol 10 --bc-tol 0.5 "
                    "--max-nodes 80 --subprocess-timeout-s 900"
                ),
                "env_OPENBLAS_NUM_THREADS": "1",
                "env_OMP_NUM_THREADS": "1",
                "env_MKL_NUM_THREADS": "1",
                "env_MEA_EPCSAFT_REACTIVE_MAX_ITERATIONS": "160",
                "input_L": inp.get("L"),
                "input_G": inp.get("G"),
                "input_alpha": inp.get("alpha"),
                "input_w_MEA": inp.get("w_MEA"),
                "input_y_CO2": inp.get("y_CO2"),
                "input_y_O2": inp.get("y_O2"),
                "input_Tl_K": inp.get("Tl"),
                "input_Tv_K": inp.get("Tv"),
                "input_P_Pa": inp.get("P"),
                "input_Beds": inp.get("Beds"),
                "input_Intercoolers": inp.get("Intercoolers"),
                "input_lean_solvent_temp_imputed": inp.get("lean_solvent_temp_imputed"),
                "input_lean_solvent_temp_imputed_C": inp.get("lean_solvent_temp_imputed_C"),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    return out


def _print_summary(results: pd.DataFrame, output_csv: Path) -> None:
    summary = {
        "rows": int(len(results)),
        "successes": int(results["success"].astype(str).str.lower().eq("true").sum()),
        "total_runtime_s": float(results["runtime_s"].sum()),
        "max_abs_capture_error_pct_pt": float(results["capture_error_pct_pt"].abs().max()),
        "invalid_state_count_total": int(results["invalid_state_count"].fillna(0).sum()),
        "guard_penalty_count_total": int(results["guard_penalty_count"].fillna(0).sum()),
        "chem_failed_total": int(results["epcsaft_chemistry_failed_count"].fillna(0).sum()),
        "max_mass_residual": float(results["epcsaft_chemistry_max_mass_residual"].max()),
        "max_reaction_residual": float(results["epcsaft_chemistry_max_reaction_residual"].max()),
        "max_charge_residual": float(results["epcsaft_chemistry_max_charge_residual"].max()),
    }
    print(output_csv)
    print(summary)
    print(
        results[
            [
                "case_id",
                "success",
                "runtime_s",
                "co2_capture_pct",
                "target_co2_capture_pct",
                "capture_error_pct_pt",
                "boundary_residual_norm",
                "invalid_state_count",
                "guard_penalty_count",
                "epcsaft_chemistry_solve_s",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    config_row = _config_row()
    case_ids = [str(case_id) for case_id in args.case_ids]
    raw_root = args.raw_root if args.raw_root.is_absolute() else REPO_ROOT / args.raw_root
    output_csv = args.output_csv if args.output_csv.is_absolute() else REPO_ROOT / args.output_csv
    if args.run_cases:
        env = _benchmark_env(str(config_row["user_options_json"]))
        for case_id in case_ids:
            _run_case(case_id, raw_root, args.python_executable, args.subprocess_timeout_s, env)
    results = _aggregate(case_ids, raw_root, output_csv, config_row)
    _print_summary(results, output_csv)


if __name__ == "__main__":
    main()
