from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from resolve_issue35_transport import (  # noqa: E402
    CORRELATION_FIELDS,
    build_rows as build_issue35_rows,
    load_config as load_issue35_config,
    validate_config as validate_issue35_config,
    validate_dependencies as validate_issue35_dependencies,
    validate_sources as validate_issue35_sources,
)


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses/nccc_validation"
INPUT = ANALYSIS / "inputs/issue42_film_transport.json"
TABLES = ANALYSIS / "results/final/tables"
REPORTS = ANALYSIS / "results/final/reports"
SOURCE_TABLE = TABLES / "issue42_transport_inputs.csv"
COMPARISON_TABLE = TABLES / "issue42_transport_comparison.csv"
SUMMARY = TABLES / "issue42_film_transport_summary.json"
REPORT = REPORTS / "issue42_film_transport.md"

ISSUE35_INPUT = ANALYSIS / "inputs/issue35_transport.json"
ISSUE35_CORRELATIONS = TABLES / "issue35_transport_correlations.csv"
ISSUE35_SUMMARY = TABLES / "issue35_transport_summary.json"
ISSUE35_REPORT = REPORTS / "issue35_transport_inputs.md"
ISSUE40_INPUT = ANALYSIS / "inputs/issue40_apparent_true_species.json"
ISSUE40_SUMMARY = TABLES / "issue40_apparent_true_species_summary.json"
ISSUE40_TABLE = TABLES / "issue40_apparent_true_species.csv"
ISSUE41_INPUT = ANALYSIS / "inputs/issue41_reversible_kinetics.json"
ISSUE41_SUMMARY = TABLES / "issue41_reversible_kinetics_summary.json"
ISSUE41_PACKET = TABLES / "issue41_packet_bound_comparison.csv"
ISSUE41_REPORT = REPORTS / "issue41_reversible_kinetics.md"

ISSUE42_OUTPUT_PATHS = (
    "analyses/nccc_validation/results/final/tables/issue42_transport_inputs.csv",
    "analyses/nccc_validation/results/final/tables/issue42_transport_comparison.csv",
    "analyses/nccc_validation/results/final/tables/issue42_film_transport_summary.json",
    "analyses/nccc_validation/results/final/reports/issue42_film_transport.md",
)

SOURCE_FIELDS = [
    "record_id", "record_kind", "species", "quantity_type", "quantity_type_note", "units",
    "source_status", "source", "source_locator", "source_chain", "conversion", "standard_state_basis",
    "domain", "uncertainty_status", "value_status", "admission_decision", "source_revision",
    "input_sha256", "generator_sha256", "exact_command", "machine", "workers", "run_id",
]
COMPARISON_FIELDS = [
    "comparison_id", "state_id", "case_id", "position", "temperature_K", "pressure_Pa",
    "source_basis_status", "issue40_packet_bound_status", "issue41_kinetic_status", "state_domain_status",
    "candidate_A_status", "candidate_B_status", "candidate_A_co2_flux_mol_m2_s",
    "candidate_B_co2_flux_mol_m2_s", "paired_delta_J_interval_mol_m2_s",
    "candidate_A_uncertainty_halfwidth_mol_m2_s", "candidate_B_uncertainty_halfwidth_mol_m2_s",
    "candidate_A_numerical_error_mol_m2_s", "candidate_B_numerical_error_mol_m2_s",
    "charge_balance_residual", "zero_current_residual", "conservation_residual", "transfer_direction",
    "positivity_status", "attempt_status", "decision", "reason", "source_revision", "input_sha256",
    "generator_sha256", "exact_command", "machine", "workers", "run_id",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git_revision() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return revision, dirty


def git_blob_sha256(revision: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"], cwd=ROOT, check=True, capture_output=True
    )
    return sha256_bytes(result.stdout)


def source_revision_contains_issue42_outputs(revision: str) -> bool:
    return any(
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}:{path}"], cwd=ROOT, capture_output=True
        ).returncode == 0
        for path in ISSUE42_OUTPUT_PATHS
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def validate_source_documents(config: dict) -> list[dict]:
    verified = []
    for source in config["source_documents"]:
        path = Path(source["local_pdf_path"])
        require(path.is_file(), f"source PDF is missing: {source['id']}")
        actual = sha256(path)
        require(actual == source["source_pdf_sha256"], f"source PDF hash changed: {source['id']}")
        verified.append({"id": source["id"], "sha256": actual, "role": source["role"], "status": "verified"})
    return verified


def canonical_source_documents(config: dict, verified: list[dict]) -> list[dict]:
    verified_by_id = {item["id"]: item for item in verified}
    return [
        {
            **source,
            "verified_sha256": verified_by_id[source["id"]]["sha256"],
            "verification_status": "verified",
        }
        for source in config["source_documents"]
    ]


def canonical_run_metadata(config: dict, revision: str, input_hash: str, generator_hash: str) -> dict:
    canonical = config["reproduction"]["canonical_run_metadata"]
    require(canonical["exact_command"] == config["reproduction"]["command"], "canonical command differs from reproduction command")
    require(canonical["workers"] == config["reproduction"]["workers"] == 1, "Issue 42 worker count changed")
    return {
        "source_revision": revision,
        "input_sha256": input_hash,
        "generator_sha256": generator_hash,
        "exact_command": canonical["exact_command"],
        "machine": canonical["machine"],
        "workers": canonical["workers"],
        "run_id": f"{canonical['run_id_prefix']}{revision[:12]}",
    }


def validate_dependencies(config: dict) -> tuple[dict, dict, dict, dict]:
    for item in config["dependencies"].values():
        path = ROOT / item["path"]
        require(path.is_file(), f"dependency is missing: {item['path']}")
        require(sha256(path) == item["sha256"], f"dependency hash changed: {item['path']}")

    issue35 = load_json(ISSUE35_SUMMARY)
    require(issue35["input_sha256"] == config["dependencies"]["issue35_input"]["sha256"], "Issue 35 input identity is stale")
    require(issue35["gates"]["supported_negative"] is True, "Issue 35 supported-negative result is missing")
    require(issue35["gates"]["physical_transport_adoption"] is False, "Issue 35 physical transport was adopted")
    require(issue35["diagnostics"]["source_observation_counts"] == {
        "amundsen_weiland_density": 23,
        "amundsen_weiland_unloaded_density": 5,
        "hartono_density": 48,
        "hartono_viscosity": 48,
    }, "Issue 35 source reconstruction counts changed")
    require(issue35["diagnostics"]["snijder_row_count"] == 16, "Issue 35 Snijder source-state count changed")
    issue35_rows = csv_rows(ISSUE35_CORRELATIONS)
    require(len(issue35_rows) == 322, "Issue 35 correlation table row count changed")

    issue40 = load_json(ISSUE40_SUMMARY)
    issue40_rows = csv_rows(ISSUE40_TABLE)
    require(issue40["row_counts"]["scientifically_admitted_rows"] == 0, "Issue 40 admitted a scientific row")
    require(len(issue40_rows) == 5 and all(row["source_basis_status"] == "basis_unresolved" for row in issue40_rows), "Issue 40 basis boundary changed")

    issue41 = load_json(ISSUE41_SUMMARY)
    issue41_packet = csv_rows(ISSUE41_PACKET)
    require(issue41["row_counts"]["scientifically_admitted_packet_rows"] == 0, "Issue 41 admitted a packet row")
    require(issue41["gates"]["source_f3_coefficient_recovered"] is False, "Issue 41 F3 status changed")
    require(issue41["gates"]["physical_reactive_film_adoption"] is False, "Issue 41 physical film was adopted")
    require(len(issue41_packet) == 5 and all(row["scientific_admission"] == "basis_unresolved" for row in issue41_packet), "Issue 41 packet boundary changed")
    return issue35, issue40, issue41, {"issue35_rows": issue35_rows, "issue40_rows": issue40_rows, "issue41_packet": issue41_packet}


def replay_issue35(config: dict) -> dict:
    issue35_config = load_issue35_config()
    validate_issue35_config(issue35_config)
    validate_issue35_dependencies(issue35_config)
    validate_issue35_sources(issue35_config)
    rows, diagnostics = build_issue35_rows(issue35_config)
    replay_hash = sha256_bytes(csv_bytes(rows, CORRELATION_FIELDS))
    require(replay_hash == sha256(ISSUE35_CORRELATIONS), "Issue 35 correlation replay differs from its retained table")
    require(diagnostics["positive_evaluated_values"], "Issue 35 known source values are not positive")
    return {"row_count": len(rows), "replay_sha256": replay_hash, "diagnostics": diagnostics}


def source_rows(config: dict, run_metadata: dict) -> list[dict]:
    rows = []
    for record in config["transport_records"]:
        row = {field: record.get(field, "") for field in SOURCE_FIELDS}
        row["record_id"] = record["id"]
        row.update(run_metadata)
        rows.append(row)
    require(len(rows) == 13, "Issue 42 transport record count changed")
    require({row["species"] for row in rows if row["species"]} <= set(config["species_order"]), "unknown species in transport records")
    require({row["species"] for row in rows if row["record_kind"] == "ionic_diffusivity"} == set(config["charged_species"]), "charged species coverage changed")
    return rows


def canonical_decision_summaries(
    config: dict,
    source_documents: list[dict],
    source_rows: list[dict],
    comparison_rows: list[dict],
    dependency_rows: dict,
) -> dict[str, dict]:
    evidence = {
        "source_documents_verified": all(item["verification_status"] == "verified" for item in source_documents),
        "source_pdf_sha256": {item["id"]: item["verified_sha256"] for item in source_documents},
        "transport_records": len(source_rows),
        "ionic_diffusivity_records": sum(row["record_kind"] == "ionic_diffusivity" for row in source_rows),
        "declared_comparison_states": len(comparison_rows),
        "evaluated_comparison_states": sum(row["attempt_status"] != "not_attempted" for row in comparison_rows),
        "scientifically_admitted_states": sum(row["decision"] == "admitted" for row in comparison_rows),
        "issue40_scientifically_admitted_rows": sum(
            row.get("scientific_admission") == "admitted" for row in dependency_rows["issue40_rows"]
        ),
        "issue41_scientifically_admitted_packet_rows": sum(
            row.get("scientific_admission") == "admitted" for row in dependency_rows["issue41_packet"]
        ),
    }
    return {
        "candidate_A": {**config["candidate_A"], "canonical_evidence": evidence},
        "candidate_B": {**config["candidate_B"], "canonical_evidence": evidence},
    }


def canonical_summary(
    config: dict,
    revision: str,
    source_documents: list[dict],
    replay: dict,
    source_output_rows: list[dict],
    comparison: list[dict],
    dependency_summaries: tuple[dict, dict, dict],
    dependency_rows: dict,
    run_metadata: dict,
    issue35_correlation_sha256: str,
    output_sha256: dict,
) -> dict:
    issue35, issue40, issue41 = dependency_summaries
    unrecovered_ids = {item["id"] for item in config["unrecovered_source_records"]}
    record_by_id = {item["id"]: item for item in config["transport_records"]}
    source_pdf_hashes_verified = all(item["verification_status"] == "verified" for item in source_documents)
    issue40_admitted = issue40["row_counts"]["scientifically_admitted_rows"]
    issue41_admitted = issue41["row_counts"]["scientifically_admitted_packet_rows"]
    n2o_chain_complete = not {"Ko_primary_N2O", "Jamal_primary_N2O", "Ying_Eimer_primary_N2O"} & unrecovered_ids
    water_diffusivity_complete = record_by_id["water_self_diffusivity_holz2000"]["source_status"] != "missing_source_complete_local_record"
    ion_diffusivity_input_complete = all(
        record_by_id[f"ion_diffusivity_{species}"]["source_status"]
        != "not_recovered_source_complete_primary_ion_record"
        for species in config["charged_species"]
    )
    candidate_A_complete = not config["candidate_A"]["status"].startswith("blocked")
    candidate_B_complete = not config["candidate_B"]["status"].startswith("blocked")
    physical_outputs_evaluated = any(
        row["candidate_A_co2_flux_mol_m2_s"] or row["candidate_B_co2_flux_mol_m2_s"] for row in comparison
    )
    row_counts = {
        "transport_records": len(source_output_rows),
        "ionic_diffusivity_records": sum(row["record_kind"] == "ionic_diffusivity" for row in source_output_rows),
        "declared_comparison_states": len(comparison),
        "evaluated_comparison_states": sum(row["attempt_status"] != "not_attempted" for row in comparison),
        "scientifically_admitted_states": sum(row["decision"] == "admitted" for row in comparison),
        "candidate_A_evaluated_states": sum(row["candidate_A_status"] != config["candidate_A"]["status"] for row in comparison),
        "candidate_B_evaluated_states": sum(row["candidate_B_status"] != config["candidate_B"]["status"] for row in comparison),
    }
    gates = {
        "source_pdf_hashes_verified": source_pdf_hashes_verified,
        "source_input_and_generator_committed": True,
        "issue35_reconstruction_replayed_exactly": replay["replay_sha256"] == issue35_correlation_sha256,
        "issue35_supported_negative_preserved": issue35["gates"]["supported_negative"],
        "issue40_scientifically_admitted_rows": issue40_admitted,
        "issue41_scientifically_admitted_packet_rows": issue41_admitted,
        "n2o_chain_complete": n2o_chain_complete,
        "water_diffusivity_complete": water_diffusivity_complete,
        "ion_diffusivity_input_complete": ion_diffusivity_input_complete,
        "candidate_A_all_species_effective_diffusivities_complete": candidate_A_complete,
        "candidate_A_source_defined_equal_ion_lump": False,
        "candidate_B_complete_mobility_law": candidate_B_complete,
        "candidate_B_gamma_and_admitted_true_state_available": False,
        "identical_state_flux_comparison_evaluated": physical_outputs_evaluated,
        "paired_delta_J_evaluated": any(row["paired_delta_J_interval_mol_m2_s"] for row in comparison),
        "physical_checks_evaluated": any(row["positivity_status"] != "not_attempted" for row in comparison),
        "physical_transport_adoption": False,
        "no_package_or_parameter_identity_used": config["package_parameter_identity"]["used"] is False,
        "no_capture_inference": True,
        "supported_negative": True,
    }
    decision_summaries = canonical_decision_summaries(
        config, source_documents, source_output_rows, comparison, dependency_rows
    )
    return {
        "schema_version": "issue42_film_transport_result_v1",
        "issue": config["issue"],
        "claim_label": config["claim_label"],
        "source_revision": revision,
        "source_worktree_clean_at_generation": True,
        "source_revision_protocol": config["reproduction"]["source_revision_protocol"],
        "input_sha256": run_metadata["input_sha256"],
        "generator_sha256": run_metadata["generator_sha256"],
        "exact_command": run_metadata["exact_command"],
        "machine": run_metadata["machine"],
        "workers": run_metadata["workers"],
        "run_id": run_metadata["run_id"],
        "source_documents": source_documents,
        "unrecovered_source_records": config["unrecovered_source_records"],
        "source_reconstruction": {
            "issue35_correlation_table_sha256": issue35_correlation_sha256,
            "issue35_replay_sha256": replay["replay_sha256"],
            "issue35_replay_exact_match": replay["replay_sha256"] == issue35_correlation_sha256,
            "issue35_source_observation_counts": replay["diagnostics"]["source_observation_counts"],
            "issue35_source_model_evaluation_counts": replay["diagnostics"]["source_model_evaluation_counts"],
            "issue35_snijder_row_count": replay["diagnostics"]["snijder_row_count"],
            "issue35_snijder_max_relative_residual": replay["diagnostics"]["snijder_max_relative_residual"],
            "issue35_snijder_max_absolute_residual_m2_s": replay["diagnostics"]["snijder_max_absolute_residual_m2_s"],
        },
        "dependencies": config["dependencies"],
        "row_counts": row_counts,
        "gates": gates,
        "candidate_A": decision_summaries["candidate_A"],
        "candidate_B": decision_summaries["candidate_B"],
        "package_parameter_identity": config["package_parameter_identity"],
        "dependencies_have_zero_admitted_physical_rows": issue40_admitted == 0 and issue41_admitted == 0,
        "output_paths": {
            "transport_inputs": SOURCE_TABLE.relative_to(ROOT).as_posix(),
            "transport_comparison": COMPARISON_TABLE.relative_to(ROOT).as_posix(),
            "summary": SUMMARY.relative_to(ROOT).as_posix(),
            "report": REPORT.relative_to(ROOT).as_posix(),
        },
        "claim_boundary": config["claim_boundary"],
        "limitations": config["limitations"],
        "regeneration_command": config["reproduction"]["command"],
        "output_sha256": output_sha256,
    }


def state_domain_status(row: dict[str, str], common_domain: dict) -> str:
    if row["source_row_id"].startswith("Putta"):
        return "not_evaluable_source_label_only"
    temperature = float(row["temperature_K"])
    lower, upper = common_domain["temperature_K"]
    if not lower <= temperature <= upper:
        return "outside_common_temperature_domain"
    return "basis_unresolved"


def comparison_rows(config: dict, issue40_rows: list[dict[str, str]], run_metadata: dict) -> list[dict]:
    expected = {item["state_id"]: item for item in config["required_state_rows"]}
    require(set(expected) == {row["source_row_id"] for row in issue40_rows}, "Issue 40 state rows do not match Issue 42 state records")
    rows = []
    for source in issue40_rows:
        state_id = source["source_row_id"]
        requirement = expected[state_id]
        row = {field: "" for field in COMPARISON_FIELDS}
        row.update({
            "comparison_id": "identical_state_transport",
            "state_id": state_id,
            "case_id": source["case_id"],
            "position": source["position"],
            "temperature_K": source["temperature_K"],
            "pressure_Pa": source["pressure_Pa"],
            "source_basis_status": source["source_basis_status"],
            "issue40_packet_bound_status": source["packet_bound_status"],
            "issue41_kinetic_status": "no_admitted_kinetic_rows",
            "state_domain_status": state_domain_status(source, config["common_domain"]),
            "candidate_A_status": config["candidate_A"]["status"],
            "candidate_B_status": config["candidate_B"]["status"],
            "positivity_status": "not_attempted",
            "attempt_status": "not_attempted",
            "decision": "blocked_not_evaluable",
            "reason": requirement["reason"],
        })
        row.update(run_metadata)
        rows.append(row)
    require(all(row["attempt_status"] == "not_attempted" for row in rows), "blocked transport rows were evaluated")
    require(len(rows) == 5, "Issue 42 state row count changed")
    return rows


def report_text(config: dict, source_files: list[dict], replay: dict, comparison: list[dict], run_metadata: dict) -> str:
    diagnostics = replay["diagnostics"]
    return f"""# Issue 42: source-only species-resolved film transport

Status: **supported-negative source-only transport evidence complete**. The retained record extends the Issue 35 source reconstruction with the nine-species transport inventory, explicit quantity types, source chains, conversions, uncertainty states, and a structurally blocked Candidate A/B comparison. No physical film flux is calculated or adopted.

## Source recovery and reconstruction

The seven retained local Zotero PDFs were rehashed before generation. The Issue 35 correlation table was replayed from its committed input and resolver with an exact CSV hash match. Its retained counts remain 23 Weiland loaded-density states, 48 Hartono density states, 48 Hartono viscosity states, and 16 Snijder diffusivity source states. The maximum Snijder reconstruction differences remain {diagnostics['snijder_max_relative_residual']:.6f} relative and {diagnostics['snijder_max_absolute_residual_m2_s']:.9e} m2/s absolute against the displayed rounded values.

The molecular records preserve the Luo CO2-water and modified Stokes-Einstein relationships and the Snijder free-MEA relationship with their source quantity labels, units, domains, conversions, and uncertainty statements. Snijder's dispersion-derived coefficient remains `not_reported` for tracer/Fick/Maxwell--Stefan classification at the retained locator. The retained source chain does not supply a source-complete H2O self-diffusivity record or the primary Ko, Jamal, and Ying--Eimer N2O inputs. It also supplies no species-resolved ionic diffusivity record for {', '.join(config['charged_species'])}, no source-defined equal-ion lump, and no complete primary unequal-ion mobility/friction law. The legacy scalar ion expression remains rejected and is not used.

## Candidate decision

Candidate A, `{config['candidate_A']['id']}`, is **blocked** because all nine source-complete effective diffusivities or a cited source-defined lump are unavailable, and the concentration basis is unresolved. Candidate B, `{config['candidate_B']['id']}`, is **blocked** because its complete generic unequal-ion electrochemical-potential mobility/friction law, unequal-ion inputs, ePC-SAFT Gamma, and admitted true-species state are unavailable. Its quantity type remains `not_reported`; scalar diffusivities are not converted into a mobility matrix.

The comparison table retains {len(comparison)} declared states: two source-label rows, two out-of-common-temperature rows, and the packet-evaluated Position 1 row. All {len(comparison)} rows are `not_attempted`; evaluated states, CO2 fluxes, species fluxes, paired Delta J intervals, uncertainty widths, numerical-error bounds, charge/current residuals, transfer directions, and positivity results remain blank. The packet and kinetic dependencies admit zero physical rows.

## Provenance and claim boundary

Source revision: `{run_metadata['source_revision']}`; generator SHA-256: `{run_metadata['generator_sha256']}`; input SHA-256: `{run_metadata['input_sha256']}`; machine: `{run_metadata['machine']}`; workers: `1`; run identity: `{run_metadata['run_id']}`.

No ePC-SAFT package, parameter document, parameter bundle, or mutable sibling checkout was used. The result does not establish Candidate A or Candidate B adequacy, universal unequal-ion mobility/friction adequacy, thermodynamic or kinetic validation, packed-column capture, or a manuscript result. Physical transport selection remains unresolved until source-complete inputs, an admitted common true-species/kinetic state, and the stated physical checks exist.

Regenerate with:

```text
{run_metadata['exact_command']}
```
"""


def main() -> int:
    revision, dirty = git_revision()
    require(not dirty, "source worktree must be clean before writing Issue 42 outputs")
    require(
        not source_revision_contains_issue42_outputs(revision),
        "Issue 42 source revision must exclude all generated output paths",
    )
    config = load_json(INPUT)
    require(config["schema_version"] == "issue42_species_resolved_transport_v1", "unexpected Issue 42 input schema")
    require(config["species_order"] == ["CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-"], "Issue 42 species order changed")
    generator = Path(__file__).relative_to(ROOT).as_posix()
    input_hash = sha256(INPUT)
    generator_hash = sha256(Path(__file__))
    require(git_blob_sha256(revision, INPUT.relative_to(ROOT).as_posix()) == input_hash, "Issue 42 input is not the committed source input")
    require(git_blob_sha256(revision, generator) == generator_hash, "Issue 42 resolver is not the committed source resolver")
    source_files = validate_source_documents(config)
    issue35, issue40, issue41, dependency_rows = validate_dependencies(config)
    replay = replay_issue35(config)
    canonical = config["reproduction"]["canonical_run_metadata"]
    require(platform.platform() == canonical["machine"], "current machine differs from pinned Issue 42 generation machine")
    run_metadata = canonical_run_metadata(config, revision, input_hash, generator_hash)
    source_output_rows = source_rows(config, run_metadata)
    comparison = comparison_rows(config, dependency_rows["issue40_rows"], run_metadata)
    source_documents = canonical_source_documents(config, source_files)
    write_csv(SOURCE_TABLE, source_output_rows, SOURCE_FIELDS)
    write_csv(COMPARISON_TABLE, comparison, COMPARISON_FIELDS)
    REPORT.write_text(report_text(config, source_documents, replay, comparison, run_metadata), encoding="utf-8")
    summary = canonical_summary(
        config,
        revision,
        source_documents,
        replay,
        source_output_rows,
        comparison,
        (issue35, issue40, issue41),
        dependency_rows,
        run_metadata,
        sha256(ISSUE35_CORRELATIONS),
        {
            "transport_inputs": sha256(SOURCE_TABLE),
            "transport_comparison": sha256(COMPARISON_TABLE),
            "report": sha256(REPORT),
        },
    )
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": SUMMARY.relative_to(ROOT).as_posix(), "gates": summary["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
