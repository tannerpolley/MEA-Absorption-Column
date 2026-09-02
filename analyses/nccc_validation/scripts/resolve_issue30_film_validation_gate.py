"""Retain the Issue 30 supported-negative film-validation no-run decision."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from resolve_issue36_film_input_release import (
    REPORT as ISSUE36_REPORT,
    SUMMARY as ISSUE36_SUMMARY,
    TABLE as ISSUE36_TABLE,
    git_blob_sha256,
    git_revision,
    load_json,
    sha256_path,
    validate_bundle,
    validate_dependency_files,
    validate_release_guard,
    validate_work_package_a,
)


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses/nccc_validation"
INPUT = ANALYSIS / "inputs/issue30_film_validation_gate.json"
TABLE = ANALYSIS / "results/final/tables/issue30_film_validation_gate.csv"
SUMMARY = ANALYSIS / "results/final/tables/issue30_film_validation_gate_summary.json"
REPORT = ANALYSIS / "results/final/reports/issue30_film_validation_gate.md"
OUTPUT_PATHS = (TABLE, SUMMARY, REPORT)
OUTPUT_RELATIVE_PATHS = tuple(path.relative_to(ROOT).as_posix() for path in OUTPUT_PATHS)

RESULT_FIELDS = [
    "case_class", "role", "required_evidence", "attempt_status",
    "non_execution_reason_type", "non_execution_reason",
    "physical_values_retained", "model_disagreement_status", "decision_status",
    "issue36_release_status", "issue36_admitted_states", "issue19_status",
    "issue19_campaign_budget_status", "issue31_status", "source_revision",
    "input_sha256", "generator_sha256", "exact_command", "machine", "workers", "run_id",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        capture_output=True,
    ).returncode == 0


def git_commit_exists(revision: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
    ).returncode == 0


def source_revision_has_issue30_outputs(revision: str) -> bool:
    return any(
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}:{path}"],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
        for path in OUTPUT_RELATIVE_PATHS
    )


def check_no_unauthorized_issue30_outputs() -> None:
    expected = set(OUTPUT_PATHS)
    roots = (ANALYSIS / "results/final", ANALYSIS / "results/runs")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            require(
                not (path.is_file() and "issue30" in path.name.lower() and path not in expected),
                f"unauthorized Issue 30 result exists: {path.relative_to(ROOT)}",
            )


def validate_issue36_binding(config: dict[str, Any]) -> dict[str, Any]:
    binding = config["issue36"]
    issue36_input = ROOT / binding["input_path"]
    require(sha256_path(issue36_input) == binding["input_sha256"], "Issue 36 input hash changed")
    issue36_config = load_json(issue36_input)
    require(issue36_config["schema_version"] == "issue36_packet_bound_film_input_release_v1", "Issue 36 input type changed")

    issue36_summary = load_json(ISSUE36_SUMMARY)
    require(issue36_summary["issue"] == 36, "Issue 36 result issue changed")
    require(issue36_summary["source_revision"] == binding["source_revision"], "Issue 36 source revision changed")
    require(git_commit_exists(binding["accepted_result_commit"]), "accepted Issue 36 result commit is unavailable")
    current_revision = git_revision()[0]
    require(git_ancestor(binding["accepted_result_commit"], current_revision), "Issue 36 accepted result is not an ancestor")

    for item in binding["outputs"]:
        path = ROOT / item["path"]
        require(path.is_file() and sha256_path(path) == item["sha256"], f"Issue 36 output changed: {item['path']}")
        require(sha256_path(path) == git_blob_sha256(binding["accepted_result_commit"], item["path"]), f"Issue 36 accepted result differs: {item['path']}")
    require(issue36_summary["output_sha256"]["table"] == binding["outputs"][0]["sha256"], "Issue 36 table identity changed")
    require(issue36_summary["output_sha256"]["report"] == binding["outputs"][2]["sha256"], "Issue 36 report identity changed")
    require(issue36_summary["row_counts"] == {"declared_states": 5, "blocked_states": 5, "admitted_states": 0, "release_files_created": 0}, "Issue 36 state counts changed")
    require(issue36_summary["gates"]["supported_negative"] is True, "Issue 36 supported-negative gate changed")
    require(issue36_summary["gates"]["physical_film_input_release"] is False, "Issue 36 released a physical film input")
    require(issue36_summary["release"]["status"] == "blocked" and issue36_summary["release"]["files_created"] == [], "Issue 36 release boundary changed")
    require(issue36_summary["downstream"]["issue30"] == "blocked", "Issue 36 no longer blocks Issue 30")
    require(issue36_summary["bundle"]["outer_sha256"] == issue36_config["bundle"]["outer_sha256"], "Issue 36 bundle identity changed")

    bundle = validate_bundle(issue36_config)
    owner = validate_work_package_a(issue36_config)
    release_guard = validate_release_guard(issue36_config)
    validate_dependency_files(issue36_config)
    require(issue36_summary["bundle"] == bundle, "Issue 36 bundle record is stale")
    require(issue36_summary["work_package_a"] == owner, "Issue 36 owner record is stale")
    require(issue36_summary["release_guard"] == release_guard, "Issue 36 release guard is stale")
    return {
        "input": {
            "path": binding["input_path"],
            "sha256": binding["input_sha256"],
        },
        "source_revision": binding["source_revision"],
        "accepted_result_commit": binding["accepted_result_commit"],
        "outputs": binding["outputs"],
        "summary_sha256": sha256_path(ISSUE36_SUMMARY),
        "table_sha256": sha256_path(ISSUE36_TABLE),
        "report_sha256": sha256_path(ISSUE36_REPORT),
        "result_summary": issue36_summary,
        "bundle": bundle,
        "work_package_a": owner,
        "release_guard": release_guard,
        "dependencies": issue36_summary["dependencies"],
        "dependency_summaries": issue36_summary["dependency_summaries"],
    }


def run_metadata(config: dict[str, Any], revision: str, input_hash: str, generator_hash: str) -> dict[str, Any]:
    reproduction = config["reproduction"]
    require(platform.platform() == reproduction["machine"], "current machine differs from pinned Issue 30 gate machine")
    return {
        "source_revision": revision,
        "input_sha256": input_hash,
        "generator_sha256": generator_hash,
        "exact_command": reproduction["command"],
        "machine": reproduction["machine"],
        "workers": reproduction["workers"],
        "run_id": f"{reproduction['run_id_prefix']}{revision[:12]}",
    }


def build_rows(config: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in config["case_classes"]:
        rows.append({
            "case_class": case["id"],
            "role": case["role"],
            "required_evidence": case["required_evidence"],
            "attempt_status": "not_attempted",
            "non_execution_reason_type": case["non_execution_reason_type"],
            "non_execution_reason": case["non_execution_reason"],
            "physical_values_retained": "none",
            "model_disagreement_status": "not_evaluated",
            "decision_status": "blocked_no_run",
            "issue36_release_status": "blocked",
            "issue36_admitted_states": "0",
            "issue19_status": "open",
            "issue19_campaign_budget_status": config["reproduction"]["campaign_budget_status"],
            "issue31_status": "blocked_not_executable",
            **metadata,
        })
    require(len(rows) == 7, "Issue 30 predeclared case-class count changed")
    require(all(row["attempt_status"] == "not_attempted" for row in rows), "Issue 30 no-run rows were evaluated")
    require(all(row["model_disagreement_status"] == "not_evaluated" for row in rows), "Issue 30 non-execution was labeled model disagreement")
    return rows


def canonical_summary(
    config: dict[str, Any],
    metadata: dict[str, Any],
    issue36: dict[str, Any],
    rows: list[dict[str, Any]],
    output_sha256: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "issue30_film_validation_gate_result_v1",
        "issue": 30,
        "claim_label": config["claim_label"],
        "owner": config["owner"],
        "repository": {
            **config["repository"],
            "source_revision": metadata["source_revision"],
            "source_worktree_clean_at_generation": True,
            "source_revision_excludes_issue30_outputs": True,
            "source_revision_protocol": config["reproduction"]["source_revision_protocol"],
        },
        **metadata,
        "execution": {
            "mode": config["reproduction"]["execution_mode"],
            "physical_campaign_launched": False,
            "declared_case_classes": len(rows),
            "attempted_case_classes": 0,
            "physical_values_retained": False,
            "model_disagreement_evaluated": False,
            "timeout_meaning": config["reproduction"]["timeout_meaning"],
            "campaign_budget_status": config["reproduction"]["campaign_budget_status"],
        },
        "issue36": issue36,
        "case_results": rows,
        "row_counts": {
            "declared_case_classes": len(rows),
            "not_attempted_case_classes": len(rows),
            "evaluated_case_classes": 0,
            "admitted_physical_states": 0,
        },
        "gates": {
            "source_revision_clean": True,
            "source_revision_excludes_issue30_outputs": True,
            "issue36_accepted_result_bound": True,
            "issue36_zero_admitted_states": True,
            "issue36_v2_release_absent": True,
            "issue19_open": True,
            "issue19_campaign_budget_recorded": False,
            "physical_bvp_campaign_launched": False,
            "limiting_cases_attempted": False,
            "desorption_attempted": False,
            "initialization_or_order_attempted": False,
            "rate_observation_comparison_attempted": False,
            "application_stress_campaign_attempted": False,
            "model_disagreement_evaluated": False,
            "model_revision_required": False,
            "physical_film_admitted": False,
            "column_replacement_blocked": True,
            "downstream_issue31_blocked": True,
            "no_physical_values_in_result": True,
            "no_timeout_or_failure_as_disagreement": True,
            "supported_negative_no_run": True,
        },
        "decision": {
            "supported_negative_no_run": True,
            "physical_film_admitted": "not_reached",
            "model_revision_required": "not_reached",
            "reason": "Zero admitted physical states is incomplete evidence; Issue 36 blocks the film-input dependency and Issue 19 supplies neither a reviewed completion nor an investigator-approved campaign budget.",
        },
        "downstream": {
            "column_replacement": "blocked",
            "issue31_execution": "blocked",
            "reason": "Issue 30 has no admitted film validation result and therefore cannot authorize the coupled column adoption decision.",
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


def report_text(config: dict[str, Any], metadata: dict[str, Any], issue36: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    issue36_summary = issue36["result_summary"]
    case_lines = "\n".join(
        f"| `{row['case_class']}` | `{row['role']}` | `{row['attempt_status']}` | `{row['non_execution_reason_type']}` | {row['non_execution_reason']} |"
        for row in rows
    )
    return f"""# Issue 30: nonlinear reactive-film validation gate

Status: **supported-negative no-run decision**. The accepted Issue 36 result retains zero admitted physical film-input states. Issue 30 therefore records the predeclared validation cases as not attempted; it launches no physical BVP campaign and retains no physical film values.

## Case-class result

All seven predeclared case classes remain visible. `attempt_status=not_attempted` means dependency-blocked or incomplete evidence, not model disagreement. `model_disagreement_status=not_evaluated` for every row. A timeout or failed physical state cannot be reported because no physical state was launched.

| Case class | Role | Attempt | Reason type | Non-execution reason |
| --- | --- | --- | --- | --- |
{case_lines}

The limiting, desorption, initialization/order, source-rate, independent-observation, and 21-state Case 3C checks remain unrun. No flux, residual, branch, mesh, timing, uncertainty, or observation-comparison value is invented.

## Decision boundary

The accepted Issue 36 release result has {issue36_summary['row_counts']['declared_states']} declared states, {issue36_summary['row_counts']['admitted_states']} admitted states, and no version-2 input, schema, or receipt files. Issue 19 remains open and has no recorded investigator-approved timing campaign budget. The Issue 30 outcomes **Physical film admitted** and **Model revision required** are both `not_reached`; neither no-run evidence nor a dependency blocker is a physical disagreement.

The selected issue-level result is **supported negative, no run**. It blocks column replacement and downstream Issue 31 execution. No thermodynamic, kinetic, transport, film, area, transfer, or capture quantity is fitted or changed.

## Source/result lineage

Repository `tannerpolley/MEA-Absorption-Column`, base revision `{config['repository']['base_revision']}`, source revision `{metadata['source_revision']}`; input SHA-256 `{metadata['input_sha256']}`; generator SHA-256 `{metadata['generator_sha256']}`; exact command `{metadata['exact_command']}`; machine `{metadata['machine']}`; workers `{metadata['workers']}`; run identity `{metadata['run_id']}`. The source revision contains none of the three generated Issue 30 files.

Issue 36 source revision `{issue36['source_revision']}` and accepted result commit `{issue36['accepted_result_commit']}` are bound by the exact Issue 36 input, table, summary, report, bundle, Work Package A owner, release guard, and dependency hashes retained in the generated summary. The Issue 36 bundle outer SHA-256 is `{issue36['bundle']['outer_sha256']}`, the extracted wheel is `{issue36['bundle']['engine_wheel_sha256']}`, the parameter document is `{issue36['bundle']['parameter_document_sha256']}`, the state packet is `{issue36['bundle']['state_packet_sha256']}`, and the chemistry member is `{issue36['bundle']['chemistry_sha256']}`.

## Claim boundary

This record supports only the Issue 30 no-run decision. It does not validate limiting behavior, desorption, numerical convergence, initialization, rate observations, film flux, packed-column capture, or a model disagreement.

Regenerate with:

```text
{metadata['exact_command']}
```
"""


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=RESULT_FIELDS, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def validate_outputs() -> None:
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    require(not dirty, "Issue 30 validator requires a clean current worktree")
    require(all(path.is_file() for path in (INPUT, *OUTPUT_PATHS)), "Issue 30 input or output is missing")
    config = load_json(INPUT)
    require(config["schema_version"] == "issue30_film_validation_gate_v1", "Issue 30 input type changed")
    current_revision = git_revision()[0]
    summary = load_json(SUMMARY)
    source_revision = summary["source_revision"]
    require(source_revision != current_revision, "Issue 30 result must record a distinct source revision")
    require(git_commit_exists(source_revision), "Issue 30 source revision is unavailable")
    require(git_ancestor(source_revision, current_revision), "Issue 30 source revision is not an ancestor")
    require(git_ancestor(config["repository"]["base_revision"], source_revision), "Issue 30 source is stale against the recorded base")
    require(not source_revision_has_issue30_outputs(source_revision), "Issue 30 source revision contains generated outputs")
    input_hash = sha256_path(INPUT)
    generator = Path(__file__)
    generator_hash = sha256_path(generator)
    require(git_blob_sha256(source_revision, INPUT.relative_to(ROOT).as_posix()) == input_hash, "Issue 30 input is not the committed source input")
    require(git_blob_sha256(source_revision, generator.relative_to(ROOT).as_posix()) == generator_hash, "Issue 30 resolver is not the committed source resolver")
    check_no_unauthorized_issue30_outputs()
    issue36 = validate_issue36_binding(config)
    metadata = run_metadata(config, source_revision, input_hash, generator_hash)
    rows = build_rows(config, metadata)
    expected_report = report_text(config, metadata, issue36, rows)
    expected_summary = canonical_summary(
        config,
        metadata,
        issue36,
        rows,
        {"table": sha256_bytes(csv_bytes(rows)), "report": sha256_bytes(expected_report.encode("utf-8"))},
    )
    require(TABLE.read_bytes() == csv_bytes(rows), "Issue 30 table differs from deterministic replay")
    require(REPORT.read_text(encoding="utf-8") == expected_report, "Issue 30 report differs from deterministic replay")
    require(summary == expected_summary, "Issue 30 summary differs from deterministic replay")
    require(summary["output_sha256"]["table"] == sha256_path(TABLE), "Issue 30 table hash is stale")
    require(summary["output_sha256"]["report"] == sha256_path(REPORT), "Issue 30 report hash is stale")


def main() -> int:
    revision, dirty = git_revision()
    require(not dirty, "Issue 30 generation requires a clean source worktree")
    require(not source_revision_has_issue30_outputs(revision), "Issue 30 source revision must contain no generated output files")
    require(not any(path.exists() for path in OUTPUT_PATHS), "Issue 30 outputs already exist; use a clean source worktree")
    config = load_json(INPUT)
    require(config["schema_version"] == "issue30_film_validation_gate_v1", "unexpected Issue 30 input type")
    require(config["repository"]["base_revision"] == "fc6fd8369ec4567694eca389c5937ce1159577b9", "Issue 30 base revision changed")
    require(git_ancestor(config["repository"]["base_revision"], revision), "Issue 30 source is stale against the recorded base")
    check_no_unauthorized_issue30_outputs()
    input_hash = sha256_path(INPUT)
    generator = Path(__file__)
    generator_hash = sha256_path(generator)
    require(git_blob_sha256(revision, INPUT.relative_to(ROOT).as_posix()) == input_hash, "Issue 30 input is not the committed source input")
    require(git_blob_sha256(revision, generator.relative_to(ROOT).as_posix()) == generator_hash, "Issue 30 resolver is not the committed source resolver")
    issue36 = validate_issue36_binding(config)
    metadata = run_metadata(config, revision, input_hash, generator_hash)
    rows = build_rows(config, metadata)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_bytes(csv_bytes(rows))
    report = report_text(config, metadata, issue36, rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    summary = canonical_summary(config, metadata, issue36, rows, {"table": sha256_path(TABLE), "report": sha256_path(REPORT)})
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": SUMMARY.relative_to(ROOT).as_posix(), "row_counts": summary["row_counts"], "decision": summary["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
