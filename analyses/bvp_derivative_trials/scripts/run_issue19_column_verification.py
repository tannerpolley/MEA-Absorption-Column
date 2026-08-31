from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_epcsaft_integration import load_contract, resolve_epcsaft
from mea_absorption_column.Thermodynamics.epcsaft_v02 import (
    dataset_content_sha256,
    parameter_document_content_sha256,
)
from mea_absorption_column.Thermodynamics.thermo_models import MEA_THERMODYNAMICS_EPCSAFT_DATASET
from mea_absorption_column.benchmark import BenchmarkSettings, run_benchmark


ANALYSIS = ROOT / "analyses/bvp_derivative_trials"
RUNS = ANALYSIS / "results/runs/issue19_scipy_bvp"
TABLES = ANALYSIS / "results/final/tables"
ROWS_PATH = TABLES / "issue19_scipy_bvp_candidate_rows.csv"
SUMMARY_PATH = TABLES / "issue19_scipy_bvp_summary.json"
COMMAND_BASE = "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python analyses/bvp_derivative_trials/scripts/run_issue19_column_verification.py"
CONSERVATION_TOLERANCE = 1.0e-8
CAPTURE_CLUSTER_TOLERANCE_PCT = 0.5


@dataclass(frozen=True)
class Attempt:
    study_axis: str
    setting_id: str
    case_id: str
    case_source: str
    data_type: str
    thermo_model: str
    initialization_id: str
    capture_guess_pct: float
    mesh_points: int = 21
    tol: float = 0.5
    bc_tol: float = 1.0e-3
    max_nodes: int = 1000


BASELINE = Attempt("case_comparison", "baseline", "3C", "NCCC_2017_cases", "mass", "epcsaft_ionic", "capture_95", 95.0)
ATTEMPTS = (
    BASELINE,
    Attempt("initialization", "init_25", "3C", "NCCC_2017_cases", "mass", "epcsaft_ionic", "capture_25", 25.0),
    Attempt("initialization", "init_60", "3C", "NCCC_2017_cases", "mass", "epcsaft_ionic", "capture_60", 60.0),
    Attempt("mesh", "mesh_11", "3C", "NCCC_2017_cases", "mass", "epcsaft_ionic", "capture_95", 95.0, mesh_points=11),
    Attempt("mesh", "mesh_41", "3C", "NCCC_2017_cases", "mass", "epcsaft_ionic", "capture_95", 95.0, mesh_points=41),
    Attempt("tolerance", "tol_1", "3C", "NCCC_2017_cases", "mass", "epcsaft_ionic", "capture_95", 95.0, tol=1.0),
    Attempt("tolerance", "tol_0p25", "3C", "NCCC_2017_cases", "mass", "epcsaft_ionic", "capture_95", 95.0, tol=0.25),
    Attempt("boundary_tolerance", "bc_1e-2", "3C", "NCCC_2017_cases", "mass", "epcsaft_ionic", "capture_95", 95.0, bc_tol=1.0e-2),
    Attempt("boundary_tolerance", "bc_1e-4", "3C", "NCCC_2017_cases", "mass", "epcsaft_ionic", "capture_95", 95.0, bc_tol=1.0e-4),
    Attempt("case_comparison", "nccc_henry", "3C", "NCCC_2017_cases", "mass", "ideal_henry", "capture_95", 95.0),
    Attempt("case_comparison", "srp_henry", "SRP-LG7", "SRP_method_cases", "mole", "ideal_henry", "capture_95", 95.0),
    Attempt("case_comparison", "srp_epcsaft", "SRP-LG7", "SRP_method_cases", "mole", "epcsaft_ionic", "capture_95", 95.0),
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(case_timeout_s: float) -> str:
    return f"{COMMAND_BASE} --case-timeout-s {case_timeout_s:g}"


def _identity(case_timeout_s: float) -> dict[str, object]:
    contract = load_contract()
    expected = contract["final_identity"]
    resolved = resolve_epcsaft(contract)
    if resolved.get("wheel_sha256") != expected["wheel_sha256"] or resolved.get("core_sha256") != expected["core_sha256"]:
        raise RuntimeError("Installed ePC-SAFT wheel does not match integration/epcsaft_contract.json")
    dataset = Path(MEA_THERMODYNAMICS_EPCSAFT_DATASET)
    return {
        "repository_code_commit": _git("rev-parse", "HEAD"),
        "repository_base_commit": _git("merge-base", "HEAD", "origin/main"),
        "generator_sha256": _sha256(Path(__file__)),
        "engine_commit": expected["engine_commit"],
        "engine_version": resolved["version"],
        "engine_wheel_filename": expected["wheel_filename"],
        "engine_wheel_sha256": expected["wheel_sha256"],
        "engine_core_sha256": expected["core_sha256"],
        "dataset_id": dataset.name,
        "dataset_content_sha256": dataset_content_sha256(str(dataset)),
        "parameter_document_content_sha256": parameter_document_content_sha256(str(dataset)),
        "machine_node": platform.node(),
        "machine_platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "case_timeout_s": case_timeout_s,
        "reproduction_command": _command(case_timeout_s),
    }


def _settings(attempt: Attempt, timeout_s: float, run_dir: Path) -> BenchmarkSettings:
    case_kwargs = (
        {"nccc_case_limit": None, "nccc_case_ids": (attempt.case_id,), "nccc_dataset": "2017", "srp_case_limit": 0}
        if attempt.case_source == "NCCC_2017_cases"
        else {"nccc_case_limit": 0, "srp_case_limit": None, "srp_case_ids": (attempt.case_id,)}
    )
    return BenchmarkSettings(
        methods=("scipy-bvp",),
        thermo_models=(attempt.thermo_model,),
        output_dir=run_dir,
        c_case_limit=0,
        data_type=attempt.data_type,
        staged_beds=False,
        write_artifacts=False,
        subprocess_timeout_s=timeout_s,
        solver_settings={
            "chemical_equilibrium_model": "legacy",
            "mesh_points": attempt.mesh_points,
            "tol": attempt.tol,
            "bc_tol": attempt.bc_tol,
            "max_nodes": attempt.max_nodes,
            "co2_capture_guess_pct": attempt.capture_guess_pct,
            "success_boundary_residual_max": 1.0,
            "accept_low_residual_final_iterate": False,
        },
        **case_kwargs,
    )


def _number(row: dict[str, object], key: str) -> float | None:
    value = pd.to_numeric(row.get(key), errors="coerce")
    return None if pd.isna(value) else float(value)


def _classify(row: dict[str, object]) -> tuple[str, str, str]:
    message = str(row.get("message") or "")
    if row.get("jacobian_status") == "subprocess_timeout" or "exceeded subprocess_timeout_s" in message:
        return "campaign_watchdog", "campaign_timeout", "not_established"
    if message.startswith("Benchmark subprocess failed with return code") or message == "Benchmark subprocess completed without writing output row.":
        return "subprocess", "subprocess_failure", "not_established"
    if not bool(row.get("success")):
        return "solver", "numerical_convergence_failure", "boundary_at_state"
    if not row["certificate_pass"]:
        return "certificate_check", "certificate_failure", "boundary_at_state"
    if not row["basic_state_check_pass"]:
        return "basic_state_check", "physical_invalidity", "boundary_at_state"
    return "none", "evaluated", "result"


def _annotate(raw: dict[str, object], attempt: Attempt, identity: dict[str, object], sequence: int) -> dict[str, object]:
    machine_local_columns = {"epcsaft_dataset", "profile_png", "profile_csv_dir", "profile_csv_files"}
    row = {key: value for key, value in raw.items() if key not in machine_local_columns}
    dense_ode = _number(row, "dense_ode_residual_max")
    dense_boundary = _number(row, "dense_boundary_residual_max")
    boundary = _number(row, "boundary_residual_norm")
    co2_balance = _number(row, "co2_conservation_relative_residual")
    h2o_balance = _number(row, "h2o_conservation_relative_residual")
    capture = _number(row, "capture_pct")
    row.update(
        {
            "attempt_sequence": sequence,
            "attempted": True,
            "candidate_id": "direct_scipy_bvp_fixed_chemistry",
            **asdict(attempt),
            "fixed_chemistry": True,
            "dense_ode_check_pass": dense_ode is not None and dense_ode <= attempt.tol,
            "dense_boundary_check_pass": dense_boundary is not None and dense_boundary <= attempt.bc_tol,
            "reported_boundary_check_pass": boundary is not None and boundary <= 1.0,
            "conservation_tolerance": CONSERVATION_TOLERANCE,
            "conservation_check_pass": co2_balance is not None and h2o_balance is not None and max(co2_balance, h2o_balance) <= CONSERVATION_TOLERANCE,
            "basic_state_check_pass": capture is not None and 0.0 <= capture <= 100.0 and int(_number(row, "invalid_state_count") or 0) == 0 and int(_number(row, "guard_penalty_count") or 0) == 0,
            **identity,
        }
    )
    row["certificate_pass"] = bool(row["dense_ode_check_pass"] and row["dense_boundary_check_pass"] and row["reported_boundary_check_pass"] and row["conservation_check_pass"])
    row["stopped_by"], row["outcome"], row["claim_strength"] = _classify(row)
    row["validation_pass"] = bool(row.get("success") and row["certificate_pass"] and row["basic_state_check_pass"])
    run_payload = json.dumps({"attempt": asdict(attempt), "identity": identity}, sort_keys=True, separators=(",", ":"))
    row["run_id"] = hashlib.sha256(run_payload.encode()).hexdigest()
    row["timing_kind"] = "cold_isolated_subprocess_wall"
    return row


def _assign_capture_clusters(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["capture_cluster_id"] = ""
        row["capture_cluster_tolerance_pct"] = CAPTURE_CLUSTER_TOLERANCE_PCT
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["validation_pass"]:
            groups.setdefault((str(row["case_id"]), str(row["thermo_model"])), []).append(row)
    for group in groups.values():
        cluster_references: list[float] = []
        for row in sorted(group, key=lambda item: float(item["capture_pct"])):
            capture = float(row["capture_pct"])
            cluster = next((index for index, reference in enumerate(cluster_references, 1) if abs(capture - reference) <= CAPTURE_CLUSTER_TOLERANCE_PCT), None)
            if cluster is None:
                cluster_references.append(capture)
                cluster = len(cluster_references)
            row["capture_cluster_id"] = f"capture_cluster_{cluster}"


def _reference_deltas(rows: list[dict[str, object]]) -> None:
    reference = next(row for row in rows if row["setting_id"] == "baseline")
    reference_capture = _number(reference, "capture_pct")
    for row in rows:
        capture = _number(row, "capture_pct")
        row["capture_delta_from_reference_pct"] = "" if capture is None or reference_capture is None else capture - reference_capture


def _validate(rows: pd.DataFrame, summary: dict[str, object]) -> None:
    required = {"run_id", "stopped_by", "outcome", "claim_strength", "dense_ode_residual_max", "dense_boundary_residual_max", "capture_cluster_id", "capture_cluster_tolerance_pct", "case_timeout_s", "engine_wheel_sha256", "repository_code_commit"}
    missing = required - set(rows.columns)
    if missing:
        raise AssertionError(f"Retained Issue 19 rows are missing columns: {sorted(missing)}")
    if len(rows) != len(ATTEMPTS) or rows["run_id"].nunique() != len(rows):
        raise AssertionError("Retained Issue 19 table must contain one unique row per declared attempt.")
    init = rows.loc[(rows["case_id"].astype(str) == "3C") & (rows["thermo_model"] == "epcsaft_ionic") & (rows["initialization_id"].isin(["capture_25", "capture_60", "capture_95"])) & (rows["mesh_points"] == 21) & (rows["tol"] == 0.5) & (rows["bc_tol"] == 0.001)]
    if set(init["initialization_id"]) != {"capture_25", "capture_60", "capture_95"}:
        raise AssertionError("Three distinct NCCC Case 3C initialization outcomes are required.")
    if set(rows["study_axis"]) < {"case_comparison", "initialization", "mesh", "tolerance", "boundary_tolerance"}:
        raise AssertionError("Mesh, tolerance, boundary-tolerance, initialization, and case evidence are required.")
    if not rows.loc[rows["outcome"] != "evaluated", "message"].fillna("").astype(str).str.len().gt(0).all():
        raise AssertionError("Every incomplete or failed row must retain its diagnostic message.")
    if summary["row_count"] != len(rows):
        raise AssertionError("Issue 19 summary row count is stale.")
    timeout_values = pd.to_numeric(rows["case_timeout_s"], errors="coerce").dropna().unique()
    if len(timeout_values) != 1 or float(timeout_values[0]) != float(summary["case_timeout_s"]):
        raise AssertionError("Retained rows and summary must record one matching per-case timeout.")
    if rows["reproduction_command"].nunique() != 1 or rows["reproduction_command"].iloc[0] != summary["reproduction_command"]:
        raise AssertionError("Retained rows and summary must record the same exact reproduction command.")
    retained_text = rows.to_csv(index=False) + json.dumps(summary, sort_keys=True)
    if "/home/" in retained_text or ".codex/worktrees" in retained_text:
        raise AssertionError("Retained Issue 19 evidence contains a machine-local path.")


def _write(rows: list[dict[str, object]], identity: dict[str, object]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROWS_PATH, index=False, lineterminator="\n")
    initialization = frame.loc[(frame["case_id"].astype(str) == "3C") & (frame["thermo_model"] == "epcsaft_ionic") & frame["initialization_id"].isin(["capture_25", "capture_60", "capture_95"]) & (frame["mesh_points"] == 21) & (frame["tol"] == 0.5) & (frame["bc_tol"] == 0.001)]
    summary = {
        "claim_boundary": "Fixed-chemistry numerical solution evidence only; the 21-state physical reactive-film campaign was not attempted.",
        "row_count": len(frame),
        "validation_pass_count": int(frame["validation_pass"].sum()),
        "campaign_timeout_count": int(frame["outcome"].eq("campaign_timeout").sum()),
        "failed_or_incomplete_count": int(frame["outcome"].ne("evaluated").sum()),
        "three_initialization_outcomes_retained": len(initialization) == 3,
        "three_initializations_same_capture_cluster": bool(len(initialization) == 3 and initialization["validation_pass"].all() and initialization["capture_cluster_id"].nunique() == 1),
        "capture_cluster_rule": f"Within each case and thermodynamic model, validated captures are assigned in ascending order to the first cluster whose initial capture reference differs by no more than {CAPTURE_CLUSTER_TOLERANCE_PCT:g} percentage point.",
        "capture_cluster_tolerance_pct": CAPTURE_CLUSTER_TOLERANCE_PCT,
        "true_profile_branch_identity_established": False,
        "profile_branch_claim_boundary": "Capture clustering compares scalar outlet capture only; retained profiles are absent, so numerical solution-profile or branch identity is not established.",
        "attempt_matrix": [asdict(attempt) for attempt in ATTEMPTS],
        "identity": identity,
        "candidate_rows_sha256": _sha256(ROWS_PATH),
        "case_timeout_s": identity["case_timeout_s"],
        "reproduction_command": identity["reproduction_command"],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate(frame, summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-timeout-s", type=float, default=100.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        rows = pd.read_csv(ROWS_PATH)
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        if summary["candidate_rows_sha256"] != _sha256(ROWS_PATH):
            raise AssertionError("Issue 19 candidate table hash does not match the retained summary.")
        _validate(rows, summary)
        print("Issue 19 retained SciPy BVP rows are internally consistent.")
        return 0

    identity = _identity(args.case_timeout_s)
    rows = []
    for sequence, attempt in enumerate(ATTEMPTS, 1):
        run_dir = RUNS / f"{sequence:02d}_{attempt.setting_id}"
        result = run_benchmark(_settings(attempt, args.case_timeout_s, run_dir)).iloc[0].to_dict()
        rows.append(_annotate(result, attempt, identity, sequence))
    _assign_capture_clusters(rows)
    _reference_deltas(rows)
    _write(rows, identity)
    print(f"Wrote {ROWS_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
