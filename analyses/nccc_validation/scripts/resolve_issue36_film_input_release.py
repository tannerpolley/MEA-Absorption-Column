"""Retain the blocked packet-bound Work Package B film-input release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from resolve_issue40_apparent_true_species import read_bundle  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses/nccc_validation"
INPUT = ANALYSIS / "inputs/issue36_film_input_release.json"
TABLE = ANALYSIS / "results/final/tables/issue36_film_input_release.csv"
SUMMARY = ANALYSIS / "results/final/tables/issue36_film_input_release_summary.json"
REPORT = ANALYSIS / "results/final/reports/issue36_film_input_release.md"

OUTPUT_PATHS = (TABLE, SUMMARY, REPORT)
OUTPUT_RELATIVE_PATHS = tuple(path.relative_to(ROOT).as_posix() for path in OUTPUT_PATHS)

RESULT_FIELDS = [
    "state_id", "case_id", "position", "temperature_K", "pressure_Pa",
    "issue40_source_basis_status", "issue40_packet_bound_status", "issue40_scientific_admission", "issue40_reason",
    "issue41_packet_bound_status", "issue41_scientific_admission", "issue41_reason",
    "issue42_state_domain_status", "issue42_candidate_A_status", "issue42_candidate_B_status", "issue42_decision", "issue42_reason",
    "basis_gate", "kinetics_gate", "transport_gate", "bulk_equilibrium_status", "detailed_balance_status",
    "rate_comparison_status", "transport_comparison_status", "uncertainty_propagation_status", "film_input_release_status",
    "release_status", "downstream_issue30_status", "admission_decision", "failure_summary",
    "issue40_result_identity", "issue41_result_identity", "issue42_result_identity",
    "source_revision", "input_sha256", "generator_sha256", "exact_command", "machine", "workers", "run_id",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def git_output(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def git_revision() -> tuple[str, bool]:
    revision = git_output(ROOT, "rev-parse", "HEAD")
    dirty = bool(git_output(ROOT, "status", "--porcelain"))
    return revision, dirty


def git_blob_sha256(revision: str, relative_path: str) -> str:
    result = subprocess.run(["git", "show", f"{revision}:{relative_path}"], cwd=ROOT, check=True, capture_output=True)
    return sha256_bytes(result.stdout)


def source_revision_has_issue36_outputs(revision: str) -> bool:
    return any(
        subprocess.run(["git", "cat-file", "-e", f"{revision}:{path}"], cwd=ROOT, capture_output=True).returncode == 0
        for path in OUTPUT_RELATIVE_PATHS
    )


def dependency_map(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    dependencies = {item["id"]: item for item in config["dependencies"]}
    require(len(dependencies) == len(config["dependencies"]), "Issue 36 dependency identities are duplicated")
    return dependencies


def result_identity(dependencies: dict[str, dict[str, str]], name: str) -> str:
    item = dependencies[name]
    return f"{item['path']}@sha256:{item['sha256']}"


def validate_work_package_a(config: dict[str, Any]) -> dict[str, Any]:
    owner = config["work_package_a"]
    root = Path(owner["local_checkout"])
    require((root / ".git").exists(), f"Work Package A checkout is missing: {root}")
    require(git_output(root, "rev-parse", owner["owner_revision"]) == owner["owner_revision"], "Work Package A owner revision is unavailable")
    observed_files = []
    for item in owner["files"]:
        result = subprocess.run(
            ["git", "show", f"{owner['owner_revision']}:{item['path']}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        observed = sha256_bytes(result.stdout)
        require(observed == item["sha256"], f"Work Package A owner file changed: {item['path']}")
        observed_files.append({"path": item["path"], "sha256": observed})
    for path in config["release_guard"]["external_revision_paths"]:
        require(
            subprocess.run(["git", "cat-file", "-e", f"{owner['owner_revision']}:{path}"], cwd=root, capture_output=True).returncode != 0,
            f"Work Package A owner revision already contains a version-2 release file: {path}",
        )
    return {**owner, "files": observed_files, "owner_revision_verified": True, "version_2_release_absent_at_owner_revision": True}


def validate_release_guard(config: dict[str, Any]) -> dict[str, Any]:
    paths = [ROOT / path for path in config["release_guard"]["paths"]]
    paths.extend(Path(path) for path in config["release_guard"]["external_paths"])
    observed = [{"path": str(path), "exists": path.is_file()} for path in paths]
    require(not any(item["exists"] for item in observed), "a canonical version-2 release file exists; refusing to overwrite it")
    return {
        "paths": config["release_guard"]["paths"],
        "external_paths": config["release_guard"]["external_paths"],
        "observed": observed,
        "partial_release_absent": True,
    }


def validate_bundle(config: dict[str, Any]) -> dict[str, Any]:
    bundle_config = config["bundle"]
    path = Path(bundle_config["path"])
    archive, _, manifest = read_bundle(path, bundle_config)
    try:
        require(manifest["bundle_id"] == bundle_config["bundle_id"], "bundle identity changed")
        member_paths = {item["path"] for item in manifest["files"]}
        require(set(bundle_config["required_members"]).issubset(member_paths), "required bundle member is missing")
        require(
            next(item["path"] for item in manifest["files"] if item["path"].startswith("engine/"))
            == f"engine/{bundle_config['engine_wheel_filename']}",
            "bundle wheel filename changed",
        )
        return {
            **bundle_config,
            "manifest_members": manifest["files"],
            "outer_and_member_hashes_verified": True,
            "extracted_wheel_member_verified": True,
        }
    finally:
        archive.close()


def validate_dependency_files(config: dict[str, Any]) -> dict[str, Any]:
    dependencies = dependency_map(config)
    for item in dependencies.values():
        path = ROOT / item["path"]
        require(path.is_file(), f"dependency is missing: {item['path']}")
        require(sha256_path(path) == item["sha256"], f"dependency hash changed: {item['path']}")

    issue40 = load_json(ROOT / dependencies["issue40_summary"]["path"])
    issue41 = load_json(ROOT / dependencies["issue41_summary"]["path"])
    issue42 = load_json(ROOT / dependencies["issue42_summary"]["path"])
    require(issue40["row_counts"]["scientifically_admitted_rows"] == 0, "Issue 40 admitted a true-species row")
    require(issue40["gates"]["position_1_source_basis_unresolved"] is True, "Issue 40 basis gate changed")
    require(issue40["gates"]["no_capture_inference"] is True, "Issue 40 capture inference gate changed")
    require(issue41["row_counts"]["scientifically_admitted_packet_rows"] == 0, "Issue 41 admitted a kinetic row")
    require(issue41["gates"]["packet_activity_closure_attempted"] is False, "Issue 41 activity closure was attempted")
    require(issue41["gates"]["physical_reactive_film_adoption"] is False, "Issue 41 physical film was adopted")
    require(issue41["gates"]["source_f3_coefficient_recovered"] is False, "Issue 41 F3 status changed")
    require(issue42["row_counts"]["scientifically_admitted_states"] == 0, "Issue 42 admitted a transport state")
    require(issue42["gates"]["physical_transport_adoption"] is False, "Issue 42 physical transport was adopted")
    require(issue42["gates"]["no_package_or_parameter_identity_used"] is True, "Issue 42 package identity gate changed")
    require(issue42["gates"]["no_capture_inference"] is True, "Issue 42 capture inference gate changed")

    issue40_rows = csv_rows(ROOT / dependencies["issue40_table"]["path"])
    issue41_rows = csv_rows(ROOT / dependencies["issue41_packet"]["path"])
    issue42_rows = csv_rows(ROOT / dependencies["issue42_comparison"]["path"])
    expected_states = set(config["required_state_ids"])
    require({row["source_row_id"] for row in issue40_rows} == expected_states, "Issue 40 state set changed")
    require({row["source_row_id"] for row in issue41_rows} == expected_states, "Issue 41 state set changed")
    require({row["state_id"] for row in issue42_rows} == expected_states, "Issue 42 state set changed")
    require(all(row["scientific_admission"] == "basis_unresolved" for row in issue40_rows), "Issue 40 admission boundary changed")
    require(all(row["scientific_admission"] == "basis_unresolved" for row in issue41_rows), "Issue 41 admission boundary changed")
    require(all(row["decision"] == "blocked_not_evaluable" for row in issue42_rows), "Issue 42 decision boundary changed")
    require(all(row["candidate_A_status"] == "blocked_missing_source_complete_species_set_or_source_defined_lump" for row in issue42_rows), "Issue 42 Candidate A boundary changed")
    require(all(row["candidate_B_status"] == "blocked_missing_complete_primary_mobility_law" for row in issue42_rows), "Issue 42 Candidate B boundary changed")

    issue40_summary_path = ROOT / dependencies["issue40_summary"]["path"]
    require(issue40["result_table_sha256"] == sha256_path(ROOT / dependencies["issue40_table"]["path"]), "Issue 40 result hash is stale")
    for key, path_key in {
        "packet_bound_comparison": "issue41_packet",
        "report": "issue41_report",
    }.items():
        require(issue41["output_sha256"][key] == dependencies[path_key]["sha256"], f"Issue 41 output hash is stale: {key}")
    for key, path_key in {
        "transport_comparison": "issue42_comparison",
        "report": "issue42_report",
    }.items():
        require(issue42["output_sha256"][key] == dependencies[path_key]["sha256"], f"Issue 42 output hash is stale: {key}")
    return {
        "issue40": issue40,
        "issue41": issue41,
        "issue42": issue42,
        "issue40_rows": issue40_rows,
        "issue41_rows": issue41_rows,
        "issue42_rows": issue42_rows,
        "issue40_summary_path": issue40_summary_path,
    }


def run_metadata(config: dict[str, Any], revision: str, input_hash: str, generator_hash: str) -> dict[str, Any]:
    reproduction = config["reproduction"]
    require(platform.platform() == reproduction["machine"], "current machine differs from pinned Issue 36 generation machine")
    return {
        "source_revision": revision,
        "input_sha256": input_hash,
        "generator_sha256": generator_hash,
        "exact_command": reproduction["command"],
        "machine": reproduction["machine"],
        "workers": reproduction["workers"],
        "run_id": f"{reproduction['run_id_prefix']}{revision[:12]}",
    }


def build_rows(
    config: dict[str, Any], dependency_data: dict[str, Any], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    dependencies = dependency_map(config)
    issue40 = {row["source_row_id"]: row for row in dependency_data["issue40_rows"]}
    issue41 = {row["source_row_id"]: row for row in dependency_data["issue41_rows"]}
    issue42 = {row["state_id"]: row for row in dependency_data["issue42_rows"]}
    rows = []
    for state_id in config["required_state_ids"]:
        i40, i41, i42 = issue40[state_id], issue41[state_id], issue42[state_id]
        rows.append({
            "state_id": state_id,
            "case_id": i40["case_id"],
            "position": i40["position"],
            "temperature_K": i40["temperature_K"],
            "pressure_Pa": i40["pressure_Pa"],
            "issue40_source_basis_status": i40["source_basis_status"],
            "issue40_packet_bound_status": i40["packet_bound_status"],
            "issue40_scientific_admission": i40["scientific_admission"],
            "issue40_reason": i40["admission_reason"],
            "issue41_packet_bound_status": i41["packet_bound_status"],
            "issue41_scientific_admission": i41["scientific_admission"],
            "issue41_reason": i41["admission_reason"],
            "issue42_state_domain_status": i42["state_domain_status"],
            "issue42_candidate_A_status": i42["candidate_A_status"],
            "issue42_candidate_B_status": i42["candidate_B_status"],
            "issue42_decision": i42["decision"],
            "issue42_reason": i42["reason"],
            "basis_gate": "blocked_basis_unresolved",
            "kinetics_gate": "blocked_no_admitted_kinetic_state",
            "transport_gate": "blocked_no_admitted_transport_state",
            "bulk_equilibrium_status": "not_attempted_no_admitted_state",
            "detailed_balance_status": "not_attempted_no_admitted_state",
            "rate_comparison_status": "not_attempted_no_admitted_kinetic_state",
            "transport_comparison_status": "not_attempted_no_admitted_transport_state",
            "uncertainty_propagation_status": "not_attempted_no_admitted_state",
            "film_input_release_status": "blocked_no_complete_dependency_intersection",
            "release_status": "blocked",
            "downstream_issue30_status": "blocked",
            "admission_decision": "not_admitted",
            "failure_summary": "basis_unresolved; no_admitted_kinetic_state; no_admitted_transport_state",
            "issue40_result_identity": result_identity(dependencies, "issue40_summary"),
            "issue41_result_identity": result_identity(dependencies, "issue41_summary"),
            "issue42_result_identity": result_identity(dependencies, "issue42_summary"),
            **metadata,
        })
    require(len(rows) == 5, "Issue 36 must retain exactly five declared state failures")
    return rows


def dependency_summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue40": {
            "source_repository_commit": data["issue40"].get("source_repository_commit"),
            "generator_sha256": data["issue40"].get("generator_sha256"),
            "result_table_sha256": data["issue40"].get("result_table_sha256"),
            "scientifically_admitted_rows": data["issue40"]["row_counts"]["scientifically_admitted_rows"],
        },
        "issue41": {
            "source_revision": data["issue41"].get("source_revision"),
            "generator_sha256": data["issue41"].get("generator_sha256"),
            "output_sha256": data["issue41"]["output_sha256"],
            "scientifically_admitted_packet_rows": data["issue41"]["row_counts"]["scientifically_admitted_packet_rows"],
        },
        "issue42": {
            "source_revision": data["issue42"].get("source_revision"),
            "generator_sha256": data["issue42"].get("generator_sha256"),
            "output_sha256": data["issue42"]["output_sha256"],
            "scientifically_admitted_states": data["issue42"]["row_counts"]["scientifically_admitted_states"],
        },
    }


def canonical_summary(
    config: dict[str, Any],
    metadata: dict[str, Any],
    bundle: dict[str, Any],
    owner: dict[str, Any],
    release_guard: dict[str, Any],
    dependency_data: dict[str, Any],
    rows: list[dict[str, Any]],
    output_sha256: dict[str, str],
) -> dict[str, Any]:
    dependencies = dependency_map(config)
    return {
        "schema_version": "issue36_film_input_release_result_v1",
        "issue": 36,
        "claim_label": config["claim_label"],
        "source_revision": metadata["source_revision"],
        "source_worktree_clean_at_generation": True,
        "source_revision_protocol": config["reproduction"]["source_revision_protocol"],
        **metadata,
        "bundle": bundle,
        "work_package_a": owner,
        "release_guard": release_guard,
        "dependencies": list(dependencies.values()),
        "dependency_summaries": dependency_summary(dependency_data),
        "species_order": config["species_order"],
        "reaction_order": config["reaction_order"],
        "row_counts": {"declared_states": len(rows), "blocked_states": len(rows), "admitted_states": 0, "release_files_created": 0},
        "state_gate_failures": rows,
        "gates": {
            "source_revision_clean": True,
            "source_revision_excludes_issue36_outputs": True,
            "bundle_outer_and_member_hashes_verified": True,
            "bundle_extracted_wheel_verified": True,
            "work_package_a_owner_verified": True,
            "issue40_scientifically_admitted_rows": 0,
            "issue41_scientifically_admitted_packet_rows": 0,
            "issue42_scientifically_admitted_states": 0,
            "bulk_equilibrium_attempted": False,
            "detailed_balance_evaluated": False,
            "rate_comparison_attempted": False,
            "transport_comparison_attempted": False,
            "uncertainty_propagation_attempted": False,
            "physical_film_input_release": False,
            "partial_version_2_release_absent": True,
            "no_fallback_values": True,
            "no_capture_inference": True,
            "no_thermodynamic_kinetic_transport_fit": True,
            "downstream_issue30_blocked": True,
            "supported_negative": True,
        },
        "release": {
            "status": "blocked",
            "version": 2,
            "input_path": "data/reference/MEA/film_chemistry_inputs/2/input.json",
            "schema_path": "data/reference/MEA/film_chemistry_inputs/2/schema.json",
            "receipt_path": "data/reference/MEA/film_chemistry_inputs/2.receipt.json",
            "files_created": [],
            "reason": "No state has the complete concentration mapping, kinetics, and transport intersection required for physical film-input release.",
        },
        "downstream": {
            "issue30": "blocked",
            "issue32": "closure_declared_for_reviewed_supported_negative; remains_open_until_PR_acceptance",
            "reason": "The retained Work Package B result blocks downstream physical adoption and creates no v2 input set.",
        },
        "claim_boundary": config["claim_boundary"],
        "limitations": config["limitations"],
        "output_paths": {
            "table": TABLE.relative_to(ROOT).as_posix(),
            "summary": SUMMARY.relative_to(ROOT).as_posix(),
            "report": REPORT.relative_to(ROOT).as_posix(),
        },
        "output_sha256": output_sha256,
    }


def report_text(
    config: dict[str, Any],
    metadata: dict[str, Any],
    bundle: dict[str, Any],
    owner: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    return f"""# Issue 36: packet-bound MEA film-input release

Status: **supported-negative release blocked**. The immutable packet is available and verified, but no declared state has the complete concentration mapping, kinetic, and transport intersection required for physical film-input release. No v2 input, schema, receipt, rate, diffusivity, residual, uncertainty, or fallback value is created.

## State result

The retained table has {len(rows)} declared states and {sum(row['admission_decision'] == 'admitted' for row in rows)} admitted states. Every row is blocked with the same three dependency gates: `basis_unresolved`, `no_admitted_kinetic_state`, and `no_admitted_transport_state`. Issue 40 contributes zero scientifically admitted concentration rows, Issue 41 contributes zero packet-bound kinetic rows, and Issue 42 contributes zero physical transport rows.

The Position 1 packet mapping in Issue 40 remains diagnostic only. Its source prepared/loaded basis is unresolved. The two source-label rows and two out-of-common-domain rows remain visible in the five-row state set. No bulk equilibrium, detailed-balance, concentration/activity rate comparison, transport comparison, uncertainty propagation, film flux, or packed-column calculation is attempted.

## Immutable identities

The bundle outer SHA-256 is `{bundle['outer_sha256']}`; the parameter document is `{bundle['parameter_document_sha256']}`; the extracted wheel member is `{bundle['engine_wheel_sha256']}`; the state packet is `{bundle['state_packet_sha256']}`; the chemistry member is `{bundle['chemistry_sha256']}`; and the loaded parameter fingerprint recorded by the bundle is `{bundle['parameter_fingerprint']}`. The Work Package A owner is `{owner['repository']}` at revision `{owner['owner_revision']}`, with its three source files verified by hash.

The source/result protocol records source revision `{metadata['source_revision']}`, input SHA-256 `{metadata['input_sha256']}`, generator SHA-256 `{metadata['generator_sha256']}`, machine `{metadata['machine']}`, worker count `1`, and run identity `{metadata['run_id']}`. The source revision contains no Issue 36 generated outputs. The bundle was independently checked in a clean temporary Python 3.13 environment with its extracted wheel and `verify_bundle.py`.

## Release and downstream boundary

The release status is `blocked`; the canonical version-2 paths remain absent. Downstream issue #30 is `blocked` because it has no physically admitted film-input set. No thermodynamic, kinetic, transport, interfacial-area, transfer, or capture quantity is fitted or retuned. The result supports only the typed negative readiness/release decision; it does not support film, column, or manuscript claims.

The parent issue #32 explicitly permits closure by a reviewed supported-negative Work Package B result that blocks downstream scientific adoption. This draft PR declares that closure for review; #32 remains open until the PR is accepted.

Regenerate with:

```text
{config['reproduction']['command']}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    revision, dirty = git_revision()
    require(not dirty, "Issue 36 generation requires a clean source worktree")
    require(not source_revision_has_issue36_outputs(revision), "Issue 36 source revision must contain no generated result files")
    require(not any(path.exists() for path in OUTPUT_PATHS), "Issue 36 generated outputs already exist; use a clean source worktree")

    config = load_json(INPUT)
    require(config["schema_version"] == "issue36_packet_bound_film_input_release_v1", "unexpected Issue 36 input record")
    require(config["species_order"] == ["CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-"], "Issue 36 species order changed")
    require(config["reaction_order"] == ["R1", "R2", "R3", "R4", "R5"], "Issue 36 reaction order changed")
    require(args.bundle.resolve() == Path(config["bundle"]["path"]).resolve(), "bundle path differs from the pinned Issue 36 authority")

    input_hash = sha256_path(INPUT)
    generator_hash = sha256_path(Path(__file__))
    input_relative = INPUT.relative_to(ROOT).as_posix()
    generator_relative = Path(__file__).relative_to(ROOT).as_posix()
    require(git_blob_sha256(revision, input_relative) == input_hash, "Issue 36 input is not the committed source input")
    require(git_blob_sha256(revision, generator_relative) == generator_hash, "Issue 36 resolver is not the committed source resolver")

    bundle = validate_bundle(config)
    owner = validate_work_package_a(config)
    release_guard = validate_release_guard(config)
    dependency_data = validate_dependency_files(config)
    metadata = run_metadata(config, revision, input_hash, generator_hash)
    rows = build_rows(config, dependency_data, metadata)

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(config, metadata, bundle, owner, rows), encoding="utf-8")
    summary = canonical_summary(
        config,
        metadata,
        bundle,
        owner,
        release_guard,
        dependency_data,
        rows,
        {"table": sha256_path(TABLE), "report": sha256_path(REPORT)},
    )
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": SUMMARY.relative_to(ROOT).as_posix(), "row_counts": summary["row_counts"], "release_status": summary["release"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
