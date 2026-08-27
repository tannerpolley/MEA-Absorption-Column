from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_epcsaft_integration import load_contract, resolve_epcsaft
from mea_absorption_column.Thermodynamics.epcsaft_v02 import (
    dataset_content_sha256,
    parameter_document_content_sha256,
)
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    epcsaft_state_contribution_diagnostics,
)


TABLE_DIR = REPO_ROOT / "analyses/nccc_validation/results/final/tables"
RUN_ROOT = REPO_ROOT / "analyses/nccc_validation/results/runs/epcsaft_v02_validation"
CONTRIBUTION_TABLE = TABLE_DIR / "epcsaft_v02_contribution_table.csv"
COLUMN_TABLE = TABLE_DIR / "epcsaft_v02_column_row.csv"
NEUTRAL_PARAMETERS = REPO_ROOT / "src/mea_absorption_column/data/epcsaft_neutral/parameters.json"
DATASET = Path(MEA_THERMODYNAMICS_EPCSAFT_DATASET)
REPRODUCTION_COMMAND = (
    "uv run python analyses/nccc_validation/scripts/generate_epcsaft_v02_validation.py"
)
NEUTRAL_X = (0.02, 0.24, 0.74)
IONIC_X = (1.0e-8, 0.055, 0.888, 0.028, 0.027, 0.001)
CONTRIBUTION_ZERO_ATOL = 1.0e-12
CONTRIBUTION_NONZERO_ATOL = 1.0e-8
CONSERVATION_RTOL = 1.0e-8


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _claim_strength(outcome: str) -> str:
    if outcome == "evaluated":
        return "result"
    if outcome in {"not_attempted", "campaign_timeout", "subprocess_failure"}:
        return "not_established"
    return "boundary_at_state"


def _float(row: dict[str, object], key: str) -> float:
    return float(row[key])


def _identity() -> dict[str, str]:
    contract = load_contract()
    resolved = resolve_epcsaft(contract)
    expected = contract["final_identity"]
    if resolved.get("wheel_sha256") != expected["wheel_sha256"]:
        raise RuntimeError("Installed ePC-SAFT wheel does not match integration/epcsaft_contract.json")
    if Path(resolved.get("wheel_path", "")).name != expected["wheel_filename"]:
        raise RuntimeError("Installed ePC-SAFT wheel filename does not match the final identity")
    return {
        "engine_commit": expected["engine_commit"],
        "engine_version": str(resolved["version"]),
        "engine_wheel_filename": expected["wheel_filename"],
        "engine_wheel_sha256": expected["wheel_sha256"],
        "repository_base_commit": _git("merge-base", "HEAD", "origin/main"),
        "repository_head_at_generation": _git("rev-parse", "HEAD"),
        "generator_sha256": _sha256(Path(__file__)),
        "reproduction_command": REPRODUCTION_COMMAND,
    }


def _contribution_rows(identity: dict[str, str]) -> list[dict[str, object]]:
    parameter_identities = {
        "neutral": {
            "dataset_id": "epcsaft_neutral/parameters.json",
            "dataset_content_sha256": dataset_content_sha256(str(NEUTRAL_PARAMETERS)),
            "parameter_document_content_sha256": _sha256(NEUTRAL_PARAMETERS),
        },
        "ionic": {
            "dataset_id": DATASET.name,
            "dataset_content_sha256": dataset_content_sha256(str(DATASET)),
            "parameter_document_content_sha256": parameter_document_content_sha256(str(DATASET)),
        },
    }
    rows = []
    for mixture_kind, composition in (("neutral", NEUTRAL_X), ("ionic", IONIC_X)):
        diagnostics = epcsaft_state_contribution_diagnostics(
            323.15,
            109500.0,
            composition,
            phase="liquid",
            mixture_kind=mixture_kind,
        )
        terms = diagnostics["ares_terms"]
        electrolyte_present = (
            abs(float(terms["ion"])) > CONTRIBUTION_NONZERO_ATOL
            and abs(float(terms["born"])) > CONTRIBUTION_NONZERO_ATOL
        )
        expected = mixture_kind == "ionic"
        check_pass = (
            electrolyte_present
            if expected
            else abs(float(terms["ion"])) <= CONTRIBUTION_ZERO_ATOL
            and abs(float(terms["born"])) <= CONTRIBUTION_ZERO_ATOL
        )
        co2_fugacity_pa = (
            float(diagnostics["composition"][0])
            * float(diagnostics["phi_co2"])
            * float(diagnostics["pressure_Pa"])
        )
        rows.append(
            {
                "state_id": f"fixed_{mixture_kind}_323.15K_109500Pa",
                "mixture_kind": mixture_kind,
                "species_json": json.dumps(diagnostics["species"], separators=(",", ":")),
                "composition_json": json.dumps(diagnostics["composition"], separators=(",", ":")),
                "temperature_K": diagnostics["temperature_K"],
                "pressure_Pa": diagnostics["pressure_Pa"],
                "density_mol_m3": diagnostics["density_mol_m3"],
                "phi_co2": diagnostics["phi_co2"],
                "co2_fugacity_Pa": co2_fugacity_pa,
                "a_hard_chain": terms["hc"],
                "a_dispersion": terms["disp"],
                "a_association": terms["assoc"],
                "a_ion": terms["ion"],
                "a_born": terms["born"],
                "electrolyte_contributions_expected": expected,
                "electrolyte_contributions_present": electrolyte_present,
                "contribution_check_pass": check_pass and co2_fugacity_pa > 0.0,
                "zero_atol": CONTRIBUTION_ZERO_ATOL,
                "nonzero_atol": CONTRIBUTION_NONZERO_ATOL,
                **parameter_identities[mixture_kind],
                "provider_parameter_fingerprint": diagnostics["parameter_fingerprint"],
                "provider_parameter_fingerprint_scope": "checkout-path-local; not portable provenance",
                **identity,
            }
        )
    return rows


def _profile_checks(profile_dir: Path) -> dict[str, object]:
    fl = sorted(_read_rows(profile_dir / "Fl.csv"), key=lambda row: float(row["Position"]))
    fv = sorted(_read_rows(profile_dir / "Fv.csv"), key=lambda row: float(row["Position"]))
    temperatures = _read_rows(profile_dir / "T.csv")
    transport = _read_rows(profile_dir / "transport.csv")

    def balance(liquid: str, vapor: str) -> float:
        inlet = float(fl[-1][liquid]) + float(fv[0][vapor])
        outlet = float(fl[0][liquid]) + float(fv[-1][vapor])
        return abs(inlet - outlet) / max(abs(inlet), 1.0e-30)

    co2_residual = balance("Fl_CO2", "Fv_CO2")
    h2o_residual = balance("Fl_H2O", "Fv_H2O")
    temperatures_in_bounds = all(
        250.0 <= float(row[key]) <= 500.0
        for row in temperatures
        for key in ("Tl", "Tv")
    )
    pressure_positive = all(float(row["P"]) > 0.0 for row in transport)
    return {
        "co2_conservation_relative_residual": co2_residual,
        "h2o_conservation_relative_residual": h2o_residual,
        "conservation_relative_tolerance": CONSERVATION_RTOL,
        "conservation_check_pass": max(co2_residual, h2o_residual) <= CONSERVATION_RTOL,
        "temperature_bounds_K": "[250,500]",
        "temperature_check_pass": temperatures_in_bounds,
        "pressure_check_pass": pressure_positive,
    }


def _column_row(identity: dict[str, str], provider_fingerprint: str) -> dict[str, object]:
    run_dir = RUN_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    relative_run_dir = run_dir.relative_to(REPO_ROOT).as_posix()
    command = [
        "uv", "run", "python", "-m", "mea_absorption_column.benchmark",
        "--methods", "scipy-bvp",
        "--thermo-models", "epcsaft_ionic",
        "--chemical-equilibrium-model", "legacy",
        "--c-case-ids", "3C",
        "--nccc-case-limit", "0",
        "--srp-case-limit", "0",
        "--staged-beds", "false",
        "--mesh-points", "21",
        "--tol", "1",
        "--bc-tol", "0.05",
        "--max-nodes", "200",
        "--max-runtime-s", "60",
        "--subprocess-timeout-s", "75",
        "--profile-csvs",
        "--output-dir", relative_run_dir,
    ]
    env = os.environ.copy()
    env["MEA_EPCSAFT_DATASET_NAME"] = DATASET.name
    env["PYTHONPATH"] = "src"
    env.pop("MEA_EPCSAFT_USER_OPTIONS_JSON", None)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=100,
        )
        returncode = completed.returncode
        diagnostic = completed.stderr.strip()
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        diagnostic = f"subprocess timeout after 100 s\nstdout={exc.stdout or ''}\nstderr={exc.stderr or ''}"
        timed_out = True
    wall_time = time.perf_counter() - started

    result_path = run_dir / "benchmark_results.csv"
    results = _read_rows(result_path) if result_path.exists() else []
    result = results[0] if len(results) == 1 else {}
    solver_success = len(results) == 1 and _as_bool(result.get("success"))
    profile_dir_text = str(result.get("profile_csv_dir", ""))
    profile_dir = Path(profile_dir_text)
    if profile_dir_text and not profile_dir.is_absolute():
        profile_dir = REPO_ROOT / profile_dir
    try:
        checks = _profile_checks(profile_dir)
        check_diagnostic = ""
    except Exception as exc:
        checks = {
            "co2_conservation_relative_residual": "",
            "h2o_conservation_relative_residual": "",
            "conservation_relative_tolerance": CONSERVATION_RTOL,
            "conservation_check_pass": False,
            "temperature_bounds_K": "[250,500]",
            "temperature_check_pass": False,
            "pressure_check_pass": False,
        }
        check_diagnostic = f"{type(exc).__name__}: {exc}"

    capture_in_bounds = False
    boundary_check = False
    zero_invalid_states = False
    if result:
        try:
            capture_in_bounds = 0.0 <= _float(result, "capture_pct") <= 100.0
            boundary_check = _float(result, "boundary_residual_norm") <= 1.0
            zero_invalid_states = (
                int(float(result.get("invalid_state_count", 0))) == 0
                and int(float(result.get("guard_penalty_count", 0))) == 0
            )
        except (KeyError, TypeError, ValueError):
            pass
    physical_pass = (
        capture_in_bounds
        and boundary_check
        and zero_invalid_states
        and bool(checks["temperature_check_pass"])
        and bool(checks["pressure_check_pass"])
    )
    validation_pass = solver_success and physical_pass and bool(checks["conservation_check_pass"])
    if timed_out:
        stopped_by, outcome = "subprocess", "subprocess_failure"
    elif returncode != 0 or len(results) != 1:
        stopped_by, outcome = "subprocess", "subprocess_failure"
    elif not solver_success:
        stopped_by, outcome = "solver", "numerical_convergence_failure"
    elif not checks["conservation_check_pass"]:
        stopped_by, outcome = "certificate_check", "certificate_failure"
    elif not physical_pass:
        stopped_by, outcome = "physical_check", "physical_invalidity"
    else:
        stopped_by, outcome = "none", "evaluated"

    return {
        "attempted": True,
        "case_id": result.get("case_id", "3C"),
        "case_source": result.get("case_source", "C_cases_data"),
        "thermo_model": result.get("thermo_model", "epcsaft_ionic"),
        "chemical_equilibrium_model": result.get("chemical_equilibrium_model", "legacy"),
        "fixed_chemistry": True,
        "solver_method": result.get("method", "scipy-bvp"),
        "solver_success": solver_success,
        "validation_pass": validation_pass,
        "stopped_by": stopped_by,
        "outcome": outcome,
        "claim_strength": _claim_strength(outcome),
        "message": result.get("message", ""),
        "diagnostic": diagnostic,
        "check_diagnostic": check_diagnostic,
        "command_returncode": returncode,
        "run_wall_time_s": wall_time,
        "solver_runtime_s": result.get("runtime_s", ""),
        "capture_pct": result.get("capture_pct", ""),
        "capture_error_pct": result.get("capture_error_pct", ""),
        "capture_in_physical_bounds": capture_in_bounds,
        "temperature_rmse_K": result.get("temperature_rmse_K", ""),
        "boundary_residual_norm": result.get("boundary_residual_norm", ""),
        "boundary_residual_check_pass": boundary_check,
        "invalid_state_count": result.get("invalid_state_count", ""),
        "guard_penalty_count": result.get("guard_penalty_count", ""),
        "zero_invalid_or_guarded_states": zero_invalid_states,
        **checks,
        "dataset_id": DATASET.name,
        "dataset_content_sha256": dataset_content_sha256(str(DATASET)),
        "parameter_document_content_sha256": parameter_document_content_sha256(str(DATASET)),
        "provider_parameter_fingerprint": provider_fingerprint,
        "provider_parameter_fingerprint_scope": "checkout-path-local; not portable provenance",
        "benchmark_command": shlex.join(command),
        "benchmark_environment_json": json.dumps(
            {"MEA_EPCSAFT_DATASET_NAME": DATASET.name, "PYTHONPATH": "src"},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "run_directory_at_generation": relative_run_dir,
        **identity,
    }


def main() -> int:
    identity = _identity()
    contributions = _contribution_rows(identity)
    _write_csv(CONTRIBUTION_TABLE, contributions)
    ionic = next(row for row in contributions if row["mixture_kind"] == "ionic")
    column = _column_row(identity, str(ionic["provider_parameter_fingerprint"]))
    _write_csv(COLUMN_TABLE, [column])
    print(f"Wrote {CONTRIBUTION_TABLE.relative_to(REPO_ROOT)}")
    print(f"Wrote {COLUMN_TABLE.relative_to(REPO_ROOT)}")
    return 0 if all(row["contribution_check_pass"] for row in contributions) and column["validation_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
