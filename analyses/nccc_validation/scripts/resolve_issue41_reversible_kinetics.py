from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import zipfile
from decimal import Decimal, localcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses/nccc_validation"
INPUT = ANALYSIS / "inputs/issue41_reversible_kinetics.json"
ISSUE34_INPUT = ANALYSIS / "inputs/issue34_kinetics.json"
ISSUE34_SUMMARY = ANALYSIS / "results/final/tables/issue34_kinetics_summary.json"
ISSUE40_INPUT = ANALYSIS / "inputs/issue40_apparent_true_species.json"
ISSUE40_SUMMARY = ANALYSIS / "results/final/tables/issue40_apparent_true_species_summary.json"
ISSUE40_TABLE = ANALYSIS / "results/final/tables/issue40_apparent_true_species.csv"
TABLES = ANALYSIS / "results/final/tables"
REPORTS = ANALYSIS / "results/final/reports"

STOICH = TABLES / "issue41_stoichiometry.csv"
SOURCE_RATES = TABLES / "issue41_source_rate_evidence.csv"
RAW_OBSERVATIONS = TABLES / "issue41_raw_rate_observations.csv"
PROVIDER_K = TABLES / "issue41_provider_equilibrium_relationships.csv"
PARTITION = TABLES / "issue41_estimation_validation_partition.csv"
PACKET = TABLES / "issue41_packet_bound_comparison.csv"
SUMMARY = TABLES / "issue41_reversible_kinetics_summary.json"
REPORT = REPORTS / "issue41_reversible_kinetics.md"

ELEMENTS = ("C", "H", "N", "O", "charge")
REACTION_ORDER = {"F1": 3, "F2": 3, "F3": 2}
SPECIES = ["CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-"]
BUNDLE_IDS = {
    "outer_sha256": "4139fecd9b5192e7cadd12883d2ff1bff71c20d74950af5256e4f0447995f27b",
    "parameter_document_sha256": "2666914f0f9cfebdf230e96565de843f9aadc9424035c940883147ff66af035c",
    "engine_wheel_sha256": "d7b4fc5ba5cbf0e979b65af83442d565496d11b771bb559233ad9dc3a4f8414a",
    "state_packet_sha256": "41017bcf727a486a8f3feb280e19c111a15c5dda5a3cca4e8c7dc5b051168fef",
    "chemistry_sha256": "1989f3e6c8fa567a019dcdbceb4bbcf26d9ca48aec3f640dad1134bdd1fd4e7c",
    "parameter_fingerprint": "sha256:c1fc2665e94d136eb85f27c793b7defbd16d1d82cb3173cb50a9aaf6513c8940",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_sources(config: dict) -> bool:
    verified = 0
    for source in config["source_documents"]:
        path = source.get("local_pdf_path")
        expected = source.get("source_pdf_sha256")
        if path is None:
            require(expected is None, f"source {source['id']} has no path but has a hash")
            continue
        source_path = Path(path)
        require(source_path.is_file(), f"source PDF is missing: {source_path}")
        require(sha256(source_path) == expected, f"source PDF hash changed: {source['id']}")
        verified += 1
    return verified == 3


def validate_bundle(path: Path, config: dict) -> tuple[dict, dict]:
    require(path.is_file(), f"bundle is missing: {path}")
    require(sha256(path) == config["bundle"]["outer_sha256"] == BUNDLE_IDS["outer_sha256"], "bundle outer hash mismatch")
    with zipfile.ZipFile(path) as archive:
        bundle_names = [name for name in archive.namelist() if name.endswith("/bundle.json")]
        require(len(bundle_names) == 1, "bundle manifest is not unique")
        manifest_name = bundle_names[0]
        prefix = manifest_name[: -len("bundle.json")]
        manifest = json.loads(archive.read(manifest_name))
        for item in manifest["files"]:
            member = prefix + item["path"]
            data = archive.read(member)
            require(len(data) == item["bytes"], f"bundle byte count mismatch: {item['path']}")
            require(sha256_bytes(data) == item["sha256"], f"bundle member hash mismatch: {item['path']}")
        for key in ("parameter_document_sha256", "engine_wheel_sha256", "state_packet_sha256"):
            require(manifest[key] == BUNDLE_IDS[key] == config["bundle"][key], f"bundle {key} mismatch")
        chemistry_path = prefix + "chemistry/reaction-system.json"
        state_path = prefix + "validation/state-packet.json"
        chemistry = json.loads(archive.read(chemistry_path))
        require(sha256_bytes(archive.read(chemistry_path)) == BUNDLE_IDS["chemistry_sha256"], "bundle chemistry hash mismatch")
        require(sha256_bytes(archive.read(state_path)) == BUNDLE_IDS["state_packet_sha256"], "bundle state hash mismatch")
        require(chemistry["species_ids"] == [
            "carbon-dioxide", "monoethanolamine", "water", "protonated-monoethanolamine",
            "carbamate-anion", "bicarbonate-anion", "carbonate-anion", "hydronium-cation", "hydroxide-anion",
        ], "bundle species order changed")
        require(chemistry["reaction_sign_convention"] == config["detailed_balance"]["orientation"], "reaction sign convention changed")
        require(chemistry["source_standard_state"]["id"] == "aqueous-molality-infinite-dilution-water-v1", "bundle standard state changed")
    return manifest, chemistry


def validate_dependencies(config: dict, issue34: dict, issue34_summary: dict, issue40_summary: dict, issue40_rows: list[dict[str, str]]) -> None:
    dependencies = config["dependencies"]
    for key, path in {
        "issue34_input": ISSUE34_INPUT,
        "issue34_summary": ISSUE34_SUMMARY,
        "issue40_input": ISSUE40_INPUT,
        "issue40_summary": ISSUE40_SUMMARY,
        "issue40_table": ISSUE40_TABLE,
    }.items():
        require(sha256(path) == dependencies[key]["sha256"], f"{key} hash changed")
    require(issue34_summary["input_sha256"] == dependencies["issue34_input"]["sha256"], "Issue 34 summary does not identify its input")
    gates = issue34_summary["gates"]
    for gate in ("species_and_reaction_order_preserved", "reaction_projection_stoichiometry_pass", "element_balance_pass", "charge_balance_pass", "f1_f2_recovered_coefficient_dimensional_reconstruction_pass", "f1_f2_source_printed_s_minus_2_rejected", "f3_theoretical_required_unit_pass", "issue33_basis_unresolved_preserved", "supported_negative"):
        require(gates[gate] is True, f"Issue 34 dependency gate failed: {gate}")
    require(gates["physical_reactive_film_adoption"] is False, "Issue 34 unexpectedly adopted a film")
    issue40_gates = issue40_summary["gates"]
    require(issue40_gates["fixed_nine_species_order"] and issue40_gates["position_1_source_basis_unresolved"], "Issue 40 basis contract changed")
    require(issue40_summary["row_counts"]["scientifically_admitted_rows"] == 0, "Issue 40 admitted a scientific row")
    require(len(issue40_rows) == 5 and all(row["source_basis_status"] == "basis_unresolved" for row in issue40_rows), "Issue 40 packet rows are not all basis unresolved")
    require(issue40_summary["bundle"] == {key: BUNDLE_IDS[key] for key in issue40_summary["bundle"] if key in BUNDLE_IDS}, "Issue 40 bundle identity changed")
    require(issue34["species_order"] == SPECIES, "Issue 34 species order changed")


def balance(vector: list[int], formulas: dict, species: list[str]) -> dict[str, int]:
    return {element: sum(coefficient * formulas[name][element] for coefficient, name in zip(vector, species, strict=True)) for element in ELEMENTS}


def stoichiometry_rows(config: dict, issue34: dict) -> tuple[list[dict], dict[str, list[int]], bool]:
    source_by_id = {item["id"]: item for item in issue34["source_reactions"]}
    vectors = {key: item["stoichiometry"] for key, item in source_by_id.items()}
    rows = []
    passed = True
    for item in issue34["source_reactions"]:
        residuals = balance(item["stoichiometry"], issue34["species_formulae"], SPECIES)
        this_pass = len(item["stoichiometry"]) == len(SPECIES) and all(value == 0 for value in residuals.values())
        passed &= this_pass
        rows.append({
            "record_type": "source_reaction", "reaction_id": item["id"], "equation": item["equation"],
            "source_locator": item["source_locator"], "source_reaction_coefficients": "",
            "stoichiometry_json": compact(dict(zip(SPECIES, item["stoichiometry"]))),
            "element_balance_residuals_json": compact(residuals), "stoichiometry_balance_pass": this_pass,
            "decision": "source_reaction_retained", "source_role": item["role"],
        })
    for reaction_id, item in config["fixed_reaction_projections"].items():
        vector = [sum(coefficient * vectors[source_id][index] for source_id, coefficient in item["source_reaction_coefficients"].items()) for index in range(len(SPECIES))]
        vectors[reaction_id] = vector
        residuals = balance(vector, issue34["species_formulae"], SPECIES)
        this_pass = vector == item["expected_stoichiometry"] and all(value == 0 for value in residuals.values())
        passed &= this_pass
        rows.append({
            "record_type": "finite_projection", "reaction_id": reaction_id, "equation": item["equation"],
            "source_locator": item["source_locator"], "source_reaction_coefficients": compact(item["source_reaction_coefficients"]),
            "stoichiometry_json": compact(dict(zip(SPECIES, vector))), "element_balance_residuals_json": compact(residuals),
            "stoichiometry_balance_pass": this_pass, "decision": "source_relationship_retained_comparison_only" if reaction_id != "F3" else "source_relationship_unavailable_supported_negative",
            "source_role": "fixed issue-41 projection",
        })
    return rows, vectors, passed


def coefficient_value(record: dict, temperature: float) -> tuple[float, float]:
    require(record["pre_exponential"] is not None, f"coefficient unavailable for {record['id']}")
    floating = float(record["pre_exponential"]) * math.exp(-float(record["activation_temperature_K"]) / temperature)
    with localcontext() as context:
        context.prec = 50
        decimal = Decimal(str(record["pre_exponential"])) * (-Decimal(str(record["activation_temperature_K"])) / Decimal(str(temperature))).exp()
    return floating, float(decimal)


def source_rate_rows(config: dict, issue34: dict) -> tuple[list[dict], bool, bool]:
    rows = []
    dimensional_pass = True
    f3_available = False
    equations = {key: value["equation"] for key, value in config["fixed_reaction_projections"].items()}
    for record in issue34["finite_reactions"]:
        reaction_id = record["id"]
        recovered = record["pre_exponential"] is not None
        expected_unit = "m6 kmol^-2 s^-1" if REACTION_ORDER[reaction_id] == 3 else "m3 kmol^-1 s^-1"
        unit_pass = record["required_coefficient_unit"] == expected_unit
        dimensional_pass &= unit_pass and (recovered or reaction_id == "F3")
        anchors = {}
        max_relative_difference = ""
        if recovered:
            differences = []
            for temperature in config["provider_anchor_temperatures_K"]:
                floating, decimal = coefficient_value(record, temperature)
                anchors[str(temperature)] = floating
                differences.append(abs(floating - decimal) / abs(decimal))
            max_relative_difference = max(differences)
        else:
            anchors = {str(temperature): None for temperature in config["provider_anchor_temperatures_K"]}
        rows.append({
            "reaction_id": reaction_id, "basis": record["basis"], "equation": equations[reaction_id],
            "rate_law": record["rate_law"], "equilibrium_quotient": record["equilibrium_quotient"],
            "source_formula": record["source_formula"] or "", "coefficient_symbol": record["coefficient_symbol"],
            "coefficient_status": "recovered" if recovered else "unavailable", "source_printed_coefficient_unit": record["source_printed_coefficient_unit"] or "",
            "required_coefficient_unit": expected_unit, "unit_status": record["unit_status"], "dimensional_reconstruction_pass": unit_pass,
            "standard_state": record["standard_state"], "coefficient_uncertainty": record["coefficient_uncertainty"],
            "source_domain_json": compact(config["source_rate_domain"]), "anchor_values_json": compact(anchors),
            "max_float_decimal_relative_difference": max_relative_difference, "source_locator": record["source_locator"],
            "admission_decision": record["admission_decision"], "admission_reason": record["admission_reason"],
        })
        if reaction_id == "F3":
            f3_available = recovered
    return rows, dimensional_pass, f3_available


def raw_observation_rows(config: dict, issue34: dict) -> list[dict]:
    rows = []
    for item in config["source_observation_records"]:
        rows.append({
            "observation_id": item["observation_id"], "source_id": item["source_id"], "observation_kind": item["observation_kind"],
            "apparatus_or_dataset": item["apparatus_or_dataset"], "reported_count": item["reported_count"], "metric": "",
            "value": "", "raw_rate_value_available": False, "uncertainty_status": item["uncertainty_status"],
            "measurement_uncertainty_json": compact(item.get("measurement_uncertainty")) if item.get("measurement_uncertainty") else "",
            "weighting": item["weighting"], "partition_role": item["partition_role"], "source_locator": item["source_locator"],
            "admission_decision": "raw_rate_rows_not_admitted", "reason": "Retained source locator provides counts or aggregate context, not row-level rate observations.",
        })
    for aggregate in issue34["observed_aggregate_aard"]:
        for apparatus, value in aggregate.items():
            if apparatus == "model":
                continue
            rows.append({
                "observation_id": f"Putta2016_Table4_{aggregate['model']}_{apparatus}", "source_id": "Putta2016",
                "observation_kind": "aggregate_model_comparison", "apparatus_or_dataset": apparatus, "reported_count": "",
                "metric": "AARD_percent", "value": value, "raw_rate_value_available": False,
                "uncertainty_status": "not_reported_in_aggregate_table", "measurement_uncertainty_json": "",
                "weighting": "not applicable to aggregate summary", "partition_role": "validation_summary_only",
                "source_locator": "Putta2016 printed p. 349, Table 4", "admission_decision": "aggregate_summary_only_not_admitted",
                "reason": "Aggregate AARD is retained as source evidence; raw rate rows and row-level uncertainty are unavailable.",
            })
    return rows


def provider_ln_k(reaction: dict, temperature: float) -> float:
    coefficients = reaction["coefficients"]
    kind = reaction["ln_k_form"]
    if kind.startswith("a + b_k / T +"):
        return coefficients["a"] + coefficients["b_k"] / temperature + coefficients["c"] * math.log(temperature) + coefficients["d_per_k"] * temperature + reaction["standard_state_offset"]
    if kind == "a + b_k / T":
        return coefficients["a"] + coefficients["b_k"] / temperature
    if kind.startswith("-ln(10)"):
        return -math.log(10.0) * (coefficients["a_k"] / temperature + coefficients["b"] + coefficients["c_per_k"] * temperature)
    raise ValueError(f"unsupported provider equilibrium form: {kind}")


def provider_rows(config: dict, chemistry: dict) -> list[dict]:
    reactions = {item["reaction_id"]: item for item in chemistry["reactions"]}
    rows = []
    for projection_id, projection in config["fixed_reaction_projections"].items():
        for temperature in config["provider_anchor_temperatures_K"]:
            ln_k = sum(coefficient * provider_ln_k(reactions[source_id], temperature) for source_id, coefficient in projection["source_reaction_coefficients"].items())
            rows.append({
                "projection_id": projection_id, "equation": projection["equation"], "temperature_K": temperature,
                "source_reaction_coefficients": compact(projection["source_reaction_coefficients"]), "ln_K": ln_k, "K": math.exp(ln_k),
                "standard_state_id": chemistry["source_standard_state"]["id"], "activity_convention": chemistry["source_standard_state"]["activity_convention_id"],
                "reaction_sign_convention": chemistry["reaction_sign_convention"], "provider_domain_status": "within_all_projection_reaction_domains",
                "lnQ": "", "detailed_balance_residual": "", "detailed_balance_pass": "", "detailed_balance_status": "not_evaluable_basis_unresolved",
                "reason": "Provider K(T) is compiled from the immutable bundle, but no scientifically admitted true-species activity state exists.",
            })
    return rows


def packet_rows(config: dict, issue40_rows: list[dict[str, str]]) -> list[dict]:
    rows = []
    for source in issue40_rows:
        rows.append({
            "source_row_id": source["source_row_id"], "case_id": source["case_id"], "position": source["position"],
            "temperature_K": source["temperature_K"], "pressure_Pa": source["pressure_Pa"], "source_basis_status": source["source_basis_status"],
            "packet_bound_candidate": source["packet_bound_candidate"], "packet_bound_status": source["packet_bound_status"],
            "solver_status": source["solver_status"], "numerical_status": source["numerical_status"], "physical_status": source["physical_status"],
            "provider_domain_status": source["provider_domain_status"], "balance_inf_norm": source["balance_inf_norm"],
            "charge_residual": source["charge_residual"], "reaction_affinity_inf_norm": source["reaction_affinity_inf_norm"],
            "scientific_admission": source["scientific_admission"], "admission_reason": source["admission_reason"],
            "state_packet_sha256": config["bundle"]["state_packet_sha256"], "lnQ": "", "detailed_balance_residual": "",
            "detailed_balance_pass": "", "activity_closure_status": "not_evaluable_basis_unresolved",
            "rate_evaluation_status": "not_attempted_basis_unresolved", "physical_film_admission": "not_admitted",
        })
    return rows


def partition_rows(config: dict) -> list[dict]:
    return [{
        "dataset_id": item["dataset_id"], "source_id": item["source_id"], "apparatus": item["apparatus"],
        "declared_role": item["declared_role"], "source_locator": item["source_locator"], "status": item["status"],
        "row_ids_used": False, "rate_values_used": False, "uncertainty_weights_used": False, "reason": item["reason"],
    } for item in config["estimation_validation_partition"]]


def git_revision() -> tuple[str, bool]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip())
    return revision, dirty


def report_text(config: dict, source_rows: list[dict], raw_rows: list[dict], provider_rows_: list[dict], packet_rows_: list[dict], bundle: dict) -> str:
    return f"""# Issue 41: source-rate evidence and packet-consistent reversible MEA kinetics

Status: **supported-negative evidence record complete**. The source laws, exact stoichiometric projections, raw-observation inventory, declared estimation/validation split, and immutable provider equilibrium relationships are retained. No physical reactive film is adopted.

## Retained result

- Fixed species order: `{', '.join(SPECIES)}`.
- Fixed projections: `F1 = R2 - R4 - R5`, `F2 = R2 - R4`, and unresolved `F3 = R2 - R1`.
- Source-rate evidence contains {len(source_rows)} rows: Putta2016 F1/F2 concentration and activity forms are retained, including their source units and domains; F3 remains unavailable because the cited Gondal2015 coefficient was not recovered.
- The source-printed `m^6 kmol^-2 s^-2` F1/F2 unit is retained as rejected metadata; the dimensionally required rate coefficient unit is `m^6 kmol^-2 s^-1`.
- The raw inventory contains {len(raw_rows)} records, including the 20 Putta2016 Table 4 aggregate AARD cells. No row-level rate observations or row-level uncertainty weights are available, so no estimation or validation fit is performed.

## Provider relationship and packet boundary

The immutable handoff bundle `{bundle['bundle_id']}` is internally hash-consistent. Its source standard state is `aqueous-molality-infinite-dilution-water-v1`, with products-positive reaction orientation. Provider `K(T)` is compiled at the three retained anchors ({', '.join(str(value) for value in config['provider_anchor_temperatures_K'])} K) for all three projections, but `ln Q`, residuals, and detailed-balance pass/fail are intentionally blank: Issue 40 retains all five candidate rows as `basis_unresolved` and admits zero scientific rows.

The detailed-balance criterion is `abs(ln Q - ln K) <= {config['detailed_balance']['acceptance_abs_lnQ_minus_lnK']:.0e}` only for a scientifically admitted true-species state on the bundle standard state. That prerequisite is absent here. Reaction-rate evaluation, reaction timescales, and film partition therefore remain not attempted.

## Evidence gaps and next gate

The source apparatus split is declared but remains `predeclared_only_no_row_ids`; the source does not provide retained row IDs, raw rates, or a usable uncertainty covariance. F3 has no admitted primary coefficient. Issue 40's packet mapping is numerically retained but not scientifically admitted because the prepared/loaded concentration basis is unresolved. A future update may evaluate detailed balance only after a source-basis resolution admits a packet-bound activity vector; transport admission remains a separate downstream gate.

The exact bundle identities are retained in `issue41_reversible_kinetics_summary.json`; the outer archive SHA-256 is `{config['bundle']['outer_sha256']}` and the parameter, wheel, state-packet, and chemistry member hashes are recorded there. No bundle provenance mismatch was found.

Regenerate with:

```text
{config['reproduction']['command']}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Retain Issue 41 source-rate and packet-bound evidence.")
    parser.add_argument("--bundle", type=Path, default=Path("/home/tnnrpolley21/Workspaces/Engineering/MEA-Thermodynamics/analyses/mea_parameter_bundle/results/handoff/mea-reactive-epcsaft-parameter-bundle.zip"))
    args = parser.parse_args()
    config = load_json(INPUT)
    issue34 = load_json(ISSUE34_INPUT)
    issue34_summary = load_json(ISSUE34_SUMMARY)
    issue40_summary = load_json(ISSUE40_SUMMARY)
    issue40_rows = csv_rows(ISSUE40_TABLE)
    require(config["schema_version"] == "issue41_source_rate_evidence_v1", "unexpected Issue 41 input schema")
    require(config["species_order"] == SPECIES, "Issue 41 species order changed")
    require(validate_sources(config), "not all configured local source PDFs were verified")
    manifest, chemistry = validate_bundle(args.bundle, config)
    validate_dependencies(config, issue34, issue34_summary, issue40_summary, issue40_rows)
    stoich_rows, vectors, stoich_pass = stoichiometry_rows(config, issue34)
    source_rows, dimensional_pass, f3_available = source_rate_rows(config, issue34)
    raw_rows = raw_observation_rows(config, issue34)
    provider_rows_ = provider_rows(config, chemistry)
    partition_rows_ = partition_rows(config)
    packet_rows_ = packet_rows(config, issue40_rows)
    require(len(stoich_rows) == 8 and stoich_pass, "stoichiometry evidence failed")
    require(len(source_rows) == 5 and dimensional_pass and not f3_available, "source-rate evidence failed")
    require(len(provider_rows_) == 9 and all(not row["detailed_balance_pass"] for row in provider_rows_), "provider relationship evidence failed")
    require(len(partition_rows_) == 5 and len({row["dataset_id"] for row in partition_rows_}) == 5, "partition declarations are not unique")
    require(len(packet_rows_) == 5 and all(row["source_basis_status"] == "basis_unresolved" for row in packet_rows_), "packet evidence changed")
    write_csv(STOICH, stoich_rows, ["record_type", "reaction_id", "equation", "source_locator", "source_reaction_coefficients", "stoichiometry_json", "element_balance_residuals_json", "stoichiometry_balance_pass", "decision", "source_role"])
    write_csv(SOURCE_RATES, source_rows, ["reaction_id", "basis", "equation", "rate_law", "equilibrium_quotient", "source_formula", "coefficient_symbol", "coefficient_status", "source_printed_coefficient_unit", "required_coefficient_unit", "unit_status", "dimensional_reconstruction_pass", "standard_state", "coefficient_uncertainty", "source_domain_json", "anchor_values_json", "max_float_decimal_relative_difference", "source_locator", "admission_decision", "admission_reason"])
    write_csv(RAW_OBSERVATIONS, raw_rows, ["observation_id", "source_id", "observation_kind", "apparatus_or_dataset", "reported_count", "metric", "value", "raw_rate_value_available", "uncertainty_status", "measurement_uncertainty_json", "weighting", "partition_role", "source_locator", "admission_decision", "reason"])
    write_csv(PROVIDER_K, provider_rows_, ["projection_id", "equation", "temperature_K", "source_reaction_coefficients", "ln_K", "K", "standard_state_id", "activity_convention", "reaction_sign_convention", "provider_domain_status", "lnQ", "detailed_balance_residual", "detailed_balance_pass", "detailed_balance_status", "reason"])
    write_csv(PARTITION, partition_rows_, ["dataset_id", "source_id", "apparatus", "declared_role", "source_locator", "status", "row_ids_used", "rate_values_used", "uncertainty_weights_used", "reason"])
    write_csv(PACKET, packet_rows_, ["source_row_id", "case_id", "position", "temperature_K", "pressure_Pa", "source_basis_status", "packet_bound_candidate", "packet_bound_status", "solver_status", "numerical_status", "physical_status", "provider_domain_status", "balance_inf_norm", "charge_residual", "reaction_affinity_inf_norm", "scientific_admission", "admission_reason", "state_packet_sha256", "lnQ", "detailed_balance_residual", "detailed_balance_pass", "activity_closure_status", "rate_evaluation_status", "physical_film_admission"])
    revision, dirty = git_revision()
    bundle_summary = {
        "bundle_id": manifest["bundle_id"], "outer_sha256": sha256(args.bundle),
        "parameter_document_sha256": manifest["parameter_document_sha256"], "engine_wheel_sha256": manifest["engine_wheel_sha256"],
        "state_packet_sha256": manifest["state_packet_sha256"], "chemistry_sha256": BUNDLE_IDS["chemistry_sha256"],
        "parameter_fingerprint": config["bundle"]["parameter_fingerprint"], "parameter_fingerprint_status": "recorded_authorized_handoff_identity_not_recomputed_by_stdlib_source_law_script",
    }
    summary = {
        "schema_version": "issue41_reversible_kinetics_result_v1", "issue": 41, "claim_label": config["claim_label"],
        "source_revision": revision, "source_worktree_dirty_during_generation": dirty,
        "input_sha256": sha256(INPUT), "generator_sha256": sha256(Path(__file__)),
        "bundle": bundle_summary, "source_documents": [{"id": item["id"], "doi": item["doi"], "source_pdf_sha256": item["source_pdf_sha256"], "evidence_status": item["hash_role"]} for item in config["source_documents"]],
        "gates": {
            "fixed_nine_species_order": config["species_order"] == SPECIES, "fixed_reaction_projections": True,
            "source_pdf_hashes_verified": True, "bundle_outer_and_member_hashes_match": True, "bundle_identity_matches_input": True,
            "source_f1_f2_coefficients_recovered": True, "source_f3_coefficient_recovered": False,
            "source_printed_third_order_s_minus_2_rejected": True, "source_observations_row_level_available": False,
            "estimation_validation_partition_predeclared_only": True, "issue40_basis_unresolved_preserved": True,
            "packet_bound_scientific_admission": False, "packet_activity_closure_attempted": False,
            "detailed_balance_evaluable": False, "reaction_timescale_evaluable": False, "physical_reactive_film_adoption": False,
            "supported_negative": True,
        },
        "row_counts": {"stoichiometry": len(stoich_rows), "source_rate_evidence": len(source_rows), "raw_observation_inventory": len(raw_rows), "provider_equilibrium_relationships": len(provider_rows_), "estimation_validation_partition": len(partition_rows_), "packet_bound_comparison": len(packet_rows_), "scientifically_admitted_packet_rows": 0},
        "detailed_balance": config["detailed_balance"], "source_rate_domain": config["source_rate_domain"], "common_application_domain": config["common_application_domain"],
        "dependencies": config["dependencies"], "output_paths": {"stoichiometry": STOICH.relative_to(ROOT).as_posix(), "source_rate_evidence": SOURCE_RATES.relative_to(ROOT).as_posix(), "raw_observations": RAW_OBSERVATIONS.relative_to(ROOT).as_posix(), "provider_equilibrium_relationships": PROVIDER_K.relative_to(ROOT).as_posix(), "estimation_validation_partition": PARTITION.relative_to(ROOT).as_posix(), "packet_bound_comparison": PACKET.relative_to(ROOT).as_posix(), "report": REPORT.relative_to(ROOT).as_posix()},
        "claim_boundary": "Source-faithful and provider-equilibrium evidence is retained. No raw-rate fit, packet-bound activity closure, reaction timescale, reaction partition, or physical reactive film is admitted.",
        "limitations": ["Source rows and uncertainty covariance are unavailable; retained observations are counts and aggregate AARD only.", "F3 primary coefficient is unavailable.", "Issue 40 provides no scientifically admitted true-species activity state because the source concentration basis remains unresolved.", "Transport-state admission remains downstream of Issue 42."],
        "regeneration_command": config["reproduction"]["command"],
    }
    REPORT.write_text(report_text(config, source_rows, raw_rows, provider_rows_, packet_rows_, manifest), encoding="utf-8")
    summary["output_sha256"] = {"stoichiometry": sha256(STOICH), "source_rate_evidence": sha256(SOURCE_RATES), "raw_observations": sha256(RAW_OBSERVATIONS), "provider_equilibrium_relationships": sha256(PROVIDER_K), "estimation_validation_partition": sha256(PARTITION), "packet_bound_comparison": sha256(PACKET), "report": sha256(REPORT)}
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": SUMMARY.relative_to(ROOT).as_posix(), "gates": summary["gates"], "bundle": bundle_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
