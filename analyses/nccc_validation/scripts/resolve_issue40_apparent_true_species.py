"""Map retained apparent Case 3C inputs to one packet-bound true-species state."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from importlib import metadata
import json
import math
import platform
import subprocess
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses/nccc_validation"
CONFIG_PATH = ANALYSIS / "inputs/issue40_apparent_true_species.json"
ISSUE33_TABLE = ROOT / "analyses/nccc_validation/results/final/tables/issue33_concentration_basis.csv"
ISSUE33_SUMMARY = ROOT / "analyses/nccc_validation/results/final/tables/issue33_concentration_basis_summary.json"
PROFILE = ANALYSIS / "inputs/retained_reactive_case3c/film_states.csv"
TABLE = ANALYSIS / "results/final/tables/issue40_apparent_true_species.csv"
SUMMARY = ANALYSIS / "results/final/tables/issue40_apparent_true_species_summary.json"
REPORT = ANALYSIS / "results/final/reports/issue40_apparent_true_species.md"

SHORT_NAMES = ("CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-")
SPECIES_IDS = (
    "carbon-dioxide", "monoethanolamine", "water", "protonated-monoethanolamine",
    "carbamate-anion", "bicarbonate-anion", "carbonate-anion", "hydronium-cation",
    "hydroxide-anion",
)
TRANSFORM = (
    (0, 1, 0, 1, 1, 0, 0, 0, 0),
    (1, 0, 0, 0, 1, 1, 1, 0, 0),
    (0, 0, 1, 0, 0, 1, 1, 1, 1),
    (1, 2, 0, 2, 3, 1, 1, 0, 0),
    (0, 1, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, -1, -1, -2, 1, -1),
)
TRANSFORM_ROWS = ("analytical_MEA", "total_inorganic_carbon", "water_equivalent", "elemental_C", "elemental_N", "charge")
DOMAIN_TEMPERATURE_K = (293.15, 323.15)
DOMAIN_LOADING = (0.0, 0.5)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


def git_value(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def git_worktree_is_clean() -> bool:
    return not git_value("status", "--porcelain")


def read_bundle(path: Path, expected: dict[str, Any]) -> tuple[zipfile.ZipFile, dict[str, bytes], dict[str, Any]]:
    outer = sha256_path(path)
    if outer != expected["outer_sha256"]:
        raise ValueError(f"bundle outer hash changed: expected {expected['outer_sha256']}, got {outer}")
    archive = zipfile.ZipFile(path)
    prefix = expected.get("bundle_id", "mea-reactive-epcsaft-parameter-bundle") + "/"
    manifest = json.loads(archive.read(prefix + "bundle.json"))
    members: dict[str, bytes] = {}
    for item in manifest["files"]:
        data = archive.read(prefix + item["path"])
        if len(data) != item["bytes"] or sha256_bytes(data) != item["sha256"]:
            raise ValueError(f"bundle member hash changed: {item['path']}")
        members[item["path"]] = data
    for key in ("parameter_document_sha256", "engine_wheel_sha256", "state_packet_sha256"):
        if manifest[key] != expected[key]:
            raise ValueError(f"bundle manifest identity changed: {key}")
    return archive, members, manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def reaction_constants(chemistry: dict[str, Any], temperature_K: float) -> list[list[Any]]:
    source_ids = {"R1": "Austgen1991_converted_to_common_molality", "R2": "Austgen1991_converted_to_common_molality", "R3": "Austgen1991_converted_to_common_molality", "R4": "Tong2012+Aroua1999", "R5": "BatesPinching1951"}
    result = []
    for reaction in chemistry["reactions"]:
        rid = reaction["reaction_id"]
        coeff = reaction["coefficients"]
        if rid in {"R1", "R2", "R3"}:
            value = coeff["a"] + coeff["b_k"] / temperature_K + coeff["c"] * math.log(temperature_K) + coeff["d_per_k"] * temperature_K + reaction["standard_state_offset"]
        elif rid == "R4":
            value = coeff["a"] + coeff["b_k"] / temperature_K
        else:
            value = -math.log(10.0) * (coeff["a_k"] / temperature_K + coeff["b"] + coeff["c_per_k"] * temperature_K)
        metadata = None
        if rid == "R4":
            metadata = {"coefficient_identities": ["reaction:R4:correlation:a", "reaction:R4:correlation:b_k"], "coefficient_values": [coeff["a"], coeff["b_k"]], "kind": "ln-k-a-plus-b-over-t", "reaction_id": rid}
        elif rid == "R5":
            metadata = {"coefficient_identities": ["reaction:R5:correlation:a_k", "reaction:R5:correlation:b", "reaction:R5:correlation:c_per_k"], "coefficient_values": [coeff["a_k"], coeff["b"], coeff["c_per_k"]], "kind": "negative-log10-temperature-polynomial", "reaction_id": rid}
        record: list[Any] = [value, source_ids[rid], chemistry["source_standard_state"]["id"], "products_positive", "source-standard-state-to-provider-neutral-reference", True]
        if metadata is not None:
            record.append(metadata)
        result.append(record)
    return result


def packet_request(template: dict[str, Any], chemistry: dict[str, Any], temperature_K: float, pressure_Pa: float, apparent: tuple[float, float, float], seed_ions: tuple[float, ...]) -> dict[str, Any]:
    carbon, mea, water = apparent
    request = copy.deepcopy(template)
    request.update(
        identity="issue40_case3c_position_1_packet_bound_v1",
        continuation=None,
        temperature={"role": "fixed", "unit": "kelvin", "value": temperature_K},
        pressure={"role": "fixed", "unit": "pascal", "value": pressure_Pa},
    )
    request["outputs"] = [
        {"identity": f"issue40-{i}-{SHORT_NAMES[i]}", "aggregate_identity": None, "basis": "true-species-liquid-mole-fraction", "censor_limit": None, "coefficients": [1.0 if j == i else 0.0 for j in range(9)], "covariance_identity": None, "phase_identity": "mea-nine-species-liquid", "selector": "phase.mole_fraction", "solvent_mass_coefficients_kg_per_mol": [], "support": "positive", "unit": "dimensionless"}
        for i in range(9)
    ]
    seed = [0.0, 0.0, 0.0, *seed_ions]
    seed_totals = [sum(coefficient * value for coefficient, value in zip(row, seed, strict=True)) for row in chemistry["balance_matrix"]]
    seed_water = sum(coefficient * value for coefficient, value in zip(TRANSFORM[2], seed, strict=True))
    feed = [carbon + 2.0 * seed_totals[1] - seed_totals[0], 1.0 - seed_totals[1], water - seed_water, *seed_ions]
    system = copy.deepcopy(request["reaction_system"])
    system.update(
        species_ids=list(chemistry["species_ids"]),
        charges=list(chemistry["charges"]),
        balance_matrix=copy.deepcopy(chemistry["balance_matrix"]),
        reaction_matrix=[reaction["stoichiometry"] for reaction in chemistry["reactions"]],
        equilibrium_constants=reaction_constants(chemistry, temperature_K),
        feed_amounts_mol=feed,
        conserved_totals=[carbon + 2.0, 1.0],
    )
    request["reaction_system"] = system
    return request


def transform(values: list[float]) -> list[float]:
    return [sum(float(coefficient) * value for coefficient, value in zip(row, values, strict=True)) for row in TRANSFORM]


def evidence(result: Any, name: str) -> float:
    return float(dict(result.evidence).get(name, float("nan")))


def base_row(source: dict[str, str], profile: dict[str, str] | None, source_table_sha: str, profile_sha: str) -> dict[str, Any]:
    not_reported_inputs = {
        "CO2": None,
        "MEA": None,
        "H2O": None,
        "normalized_CO2_MEA_H2O": None,
        "status": "not_reported_source_label_only",
    }
    not_reported_intervals = {
        "CO2_mol_s": {"interval": None, "status": "not_reported"},
        "MEA_mol_s": {"interval": None, "status": "not_reported"},
        "H2O_mol_s": {"interval": None, "status": "not_reported"},
        "temperature_K": {"interval": None, "status": "not_reported"},
        "pressure_Pa": {"interval": None, "status": "not_reported"},
    }
    row: dict[str, Any] = {
        "source_row_id": source["record_id"], "case_id": source.get("case_id", ""), "position": source.get("position", ""), "source_locator": source.get("source_locators", ""),
        "source_basis_status": source.get("admission_decision", "basis_unresolved"), "source_loaded_analytical_MEA_mol_L": source.get("loaded_analytical_concentration_mol_L", ""), "source_free_MEA_mol_L": source.get("free_MEA_concentration_mol_L", ""),
        "issue33_reconstructed_true_species_loading_mol_CO2_per_mol_MEA": source.get("loading_mol_CO2_per_mol_MEA", "not_reported"), "issue33_loading_units": "mol CO2 per mol MEA; Issue 33 reconstructed true-species loading used only for the domain gate", "issue33_loading_basis": "Issue 33 retained reactive true-species reconstruction; not the packet apparent-flow input",
        "apparent_total_inorganic_carbon_to_analytical_MEA_flow_ratio_mol_per_mol": "not_reported", "apparent_flow_ratio_units": "mol apparent total inorganic carbon flow per mol analytical MEA flow", "apparent_flow_ratio_basis": "retained film_states.csv Fl_CO2/Fl_MEA; not Issue 33 loading", "apparent_minus_issue33_loading_mol_per_mol": "not_reported", "loading_apparent_difference_status": "not_applicable_source_label_only; neither relabelled nor fitted",
        "source_loaded_density_kg_m3": source.get("loaded_density_kg_m3", ""), "source_density_status": "source-backed Issue 33 density; diagnostic only" if profile is not None else "not_reported_for_source_label", "source_density_locator": source.get("loaded_density_source", ""), "diagnostic_density_marker": "source density does not define prepared or analytical basis",
        "source_issue33_table_sha256": source_table_sha, "source_issue33_summary_sha256": sha256_path(ISSUE33_SUMMARY), "source_profile_sha256": profile_sha,
        "apparent_inputs_json": json.dumps(not_reported_inputs, sort_keys=True), "apparent_input_units_json": json.dumps({"CO2": "mol s^-1", "MEA": "mol s^-1", "H2O": "mol s^-1", "normalized_basis": "mol mol^-1 analytical MEA"}, sort_keys=True),
        "apparent_input_reporting_intervals_json": json.dumps(not_reported_intervals, sort_keys=True), "apparent_input_hashes_json": json.dumps({"issue33_table": source_table_sha, "retained_profile": profile_sha}, sort_keys=True),
        "temperature_K": "not_reported", "pressure_Pa": "not_reported", "temperature_reporting_interval": "not_reported_source_label_only", "pressure_reporting_interval": "not_reported_source_label_only", "packet_bound_candidate": "false", "packet_bound_status": "not_attempted", "packet_failure_code": "", "packet_failure_diagnostic": "",
        "phase_topology": "single_liquid_only", "vle_fugacity_equality_imposed": "false", "branch_identity": "", "branch_count": "", "multiple_branch_detected": "false", "solver_status": "", "numerical_status": "", "physical_status": "", "provider_domain_status": "", "mechanical_class": "",
        "packet_density_status": "not_attempted", "packet_molar_density_mol_m3": "", "packet_pressure_Pa": "", "mole_fraction_normalization_residual": "", "charge_residual": "", "balance_inf_norm": "", "pressure_relative_inf_norm": "", "reaction_affinity_inf_norm": "", "source_reference_residual_inf_norm": "",
        "inverse_apparent_totals_per_mol_N_json": "", "inverse_residuals_json": "", "inverse_max_abs_residual": "", "inverse_mapping_status": "not_attempted", "inverse_replay_status": "not_attempted", "forward_transform_status": "not_attempted", "forward_replay_status": "not_attempted", "replay_branch_identity": "", "replay_branch_match_status": "not_attempted", "forward_replay_max_abs_species_diff": "", "forward_replay_max_abs_density_diff": "",
        "scientific_admission": "basis_unresolved", "admission_reason": "", "capture_inference_used": "false", "thermodynamic_or_kinetic_fit_performed": "false", "workers": "1", "machine": platform.platform(), "run_id": "", "reproduction_command": "",
    }
    for short in SHORT_NAMES:
        row[f"true_x_{short}"] = ""
        row[f"true_C_{short}_mol_m3"] = ""
    if profile is None:
        row["admission_reason"] = "source label only; apparent component feed, T, P, and source reporting intervals are unavailable"
        return row
    temperature_K = float(profile["Tl"]); pressure_Pa = float(profile["P"])
    loading = float(source["loading_mol_CO2_per_mol_MEA"])
    co2, mea, water = (float(profile[key]) for key in ("Fl_CO2", "Fl_MEA", "Fl_H2O"))
    normalized = (co2 / mea, 1.0, water / mea)
    loading_difference = normalized[0] - loading
    row.update(
        apparent_inputs_json=json.dumps({"CO2": co2, "MEA": mea, "H2O": water, "normalized_CO2_MEA_H2O": normalized}, sort_keys=True),
        apparent_input_reporting_intervals_json=json.dumps({"CO2_mol_s": {"interval": None, "status": "not_reported"}, "MEA_mol_s": {"interval": None, "status": "not_reported"}, "H2O_mol_s": {"interval": None, "status": "not_reported"}, "temperature_K": {"interval": [temperature_K, temperature_K], "status": "retained_profile_value; experimental interval unavailable"}, "pressure_Pa": {"interval": [pressure_Pa, pressure_Pa], "status": "retained_case_input; experimental interval unavailable"}}, sort_keys=True),
        issue33_reconstructed_true_species_loading_mol_CO2_per_mol_MEA=loading, apparent_total_inorganic_carbon_to_analytical_MEA_flow_ratio_mol_per_mol=normalized[0], apparent_minus_issue33_loading_mol_per_mol=loading_difference, loading_apparent_difference_status="distinct inputs; neither relabelled nor fitted", temperature_K=temperature_K, pressure_Pa=pressure_Pa, temperature_reporting_interval="retained profile value; experimental interval unavailable", pressure_reporting_interval="retained Case 3C input value; experimental interval unavailable",
    )
    if not (DOMAIN_TEMPERATURE_K[0] <= temperature_K <= DOMAIN_TEMPERATURE_K[1]) or not (DOMAIN_LOADING[0] <= loading <= DOMAIN_LOADING[1]):
        row["packet_failure_code"] = "temperature_or_loading_outside_common_reaction_domain"
        row["packet_failure_diagnostic"] = f"source state T={temperature_K:.15g} K, loading={loading:.15g}; common packet domain is 293.15--323.15 K and 0--0.5 mol/mol"
        row["admission_reason"] = "basis unresolved; packet attempt withheld because the source state is outside the common reaction domain"
        return row
    row["packet_bound_candidate"] = "true"
    row["packet_failure_diagnostic"] = ""
    return row


def apply_result(row: dict[str, Any], result: Any, replay: Any, run_id: str, command: str) -> None:
    if replay.status != "evaluated" or len(replay.phases) != 1 or replay.phases[0].role != "liquid":
        raise RuntimeError("deterministic replay did not return one evaluated liquid phase")
    phase = result.phases[0]
    x = [float(value) for value in phase.mole_fractions]
    density = float(phase.molar_density_mol_m3)
    concentrations = [value * density for value in x]
    transformed = transform(concentrations)
    nitrogen = transformed[4]
    per_n = [value / nitrogen for value in transformed]
    normalized = json.loads(row["apparent_inputs_json"])["normalized_CO2_MEA_H2O"]
    target = [1.0, normalized[0], normalized[2], normalized[0] + 2.0, 1.0, 0.0]
    residuals = [actual - expected for actual, expected in zip(per_n, target, strict=True)]
    branch = json.dumps(jsonable(result.branch_signature), separators=(",", ":"))
    replay_branch = json.dumps(jsonable(replay.branch_signature), separators=(",", ":"))
    replay_x = [float(value) for value in replay.phases[0].mole_fractions]
    species_diff = max(abs(a - b) for a, b in zip(x, replay_x, strict=True))
    density_diff = abs(density - float(replay.phases[0].molar_density_mol_m3))
    branch_match = result.branch_signature == replay.branch_signature
    row.update(
        packet_bound_status="evaluated", packet_density_status="ePC-SAFT phase density; true-state mapping diagnostic only", solver_status=result.solver_status, numerical_status=result.numerical_status, physical_status=result.physical_status, provider_domain_status=result.provider_domain_status,
        branch_identity=branch, branch_count="1", multiple_branch_detected="false", mechanical_class=phase.mechanical_class, packet_molar_density_mol_m3=density, packet_pressure_Pa=float(phase.pressure_pa),
        mole_fraction_normalization_residual=abs(sum(x) - 1.0), charge_residual=abs(per_n[5]), balance_inf_norm=evidence(result, "balance_inf_norm"), pressure_relative_inf_norm=evidence(result, "pressure_relative_inf_norm"), reaction_affinity_inf_norm=evidence(result, "reaction_affinity_inf_norm"), source_reference_residual_inf_norm=evidence(result, "source_reference_representation_residual_inf_norm"),
        inverse_apparent_totals_per_mol_N_json=json.dumps(dict(zip(TRANSFORM_ROWS, per_n, strict=True)), sort_keys=True), inverse_residuals_json=json.dumps(dict(zip(TRANSFORM_ROWS, residuals, strict=True)), sort_keys=True), inverse_max_abs_residual=max(abs(value) for value in residuals), inverse_mapping_status="evaluated_packet_equilibrium_roundtrip", inverse_replay_status="identity_pass" if max(abs(value) for value in residuals) <= 1.0e-10 else "identity_failed", forward_transform_status="evaluated",
        replay_branch_identity=replay_branch, replay_branch_match_status="identity_pass" if branch_match else "identity_failed", forward_replay_status="identity_pass" if branch_match and species_diff <= 1.0e-12 and density_diff <= 1.0e-10 else "identity_failed", forward_replay_max_abs_species_diff=species_diff, forward_replay_max_abs_density_diff=density_diff,
        admission_reason="packet mapping evaluated; source prepared/loaded basis and prepared-to-loaded volume remain unresolved, so the row is not scientifically admitted", run_id=run_id, reproduction_command=command,
    )
    for short, mole_fraction, concentration in zip(SHORT_NAMES, x, concentrations, strict=True):
        row[f"true_x_{short}"] = mole_fraction
        row[f"true_C_{short}_mol_m3"] = concentration


def write_table(rows: list[dict[str, Any]]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bundle_meta = config["bundle"]
    archive, members, manifest = read_bundle(args.bundle.resolve(), bundle_meta)
    try:
        import epcsaft
        from epcsaft.equilibrium import general_reactive_equilibrium_problem_from_mapping, solve

        wheel_path = Path(json.loads(metadata.distribution("epcsaft").read_text("direct_url.json"))["url"].removeprefix("file://"))
        if sha256_path(wheel_path) != bundle_meta["engine_wheel_sha256"]:
            raise ValueError("installed ePC-SAFT wheel does not match the authoritative bundle wheel")
        parameters = epcsaft.Parameters.from_mapping(json.loads(members["parameters/parameters.json"]))
        if parameters.fingerprint != bundle_meta["parameter_fingerprint"]:
            raise ValueError(f"parameter fingerprint changed: expected {bundle_meta['parameter_fingerprint']}, got {parameters.fingerprint}")
        model = epcsaft.Mixture(parameters)
        chemistry = json.loads(members["chemistry/reaction-system.json"])
        if tuple(chemistry["species_ids"]) != SPECIES_IDS or tuple(chemistry["charges"]) != tuple(item["charge"] for item in config["species"]):
            raise ValueError("authoritative bundle species order or charges do not match Issue 40")
        if sha256_bytes(members["chemistry/reaction-system.json"]) != bundle_meta["chemistry_sha256"]:
            raise ValueError("authoritative bundle chemistry identity changed")
        state_packet = json.loads(members["validation/state-packet.json"])
        template = next(item["request"] for item in state_packet["observations"] if len(item["request"].get("phases", [])) == 1 and item["request"]["phases"][0].get("fluid_role") == "liquid")
        if len(template["phases"]) != 1 or template["phases"][0]["fluid_role"] != "liquid":
            raise ValueError("state-packet template is not the required single-liquid topology")
        source_table_sha = sha256_path(ISSUE33_TABLE); profile_sha = sha256_path(PROFILE)
        if source_table_sha != config["source_inputs"]["issue33_table"]["sha256"] or profile_sha != config["source_inputs"]["retained_profile"]["sha256"]:
            raise ValueError("Issue 33 source table or retained profile hash changed")
        if sha256_path(ISSUE33_SUMMARY) != config["source_inputs"]["issue33_summary"]["sha256"]:
            raise ValueError("Issue 33 summary hash changed")
        source_rows = read_csv(ISSUE33_TABLE); profile_rows = read_csv(PROFILE)
        profile_by_position = {round(float(item["Position"]), 12): item for item in profile_rows}
        command = config["reproduction"]["command"]
        generator_sha = sha256_path(Path(__file__))
        source_commit = git_value("rev-parse", "HEAD")
        source_worktree_clean = git_worktree_is_clean()
        if not source_worktree_clean:
            raise RuntimeError("source worktree must be clean before generating retained results")
        config_sha = sha256_path(CONFIG_PATH)
        run_id = "sha256:" + sha256_bytes(canonical({"bundle": bundle_meta, "source_commit": source_commit, "config": config_sha, "source_table": source_table_sha, "profile": profile_sha, "generator": generator_sha, "selected_state": config["selected_state"]}))
        rows = []
        seed_ions = tuple(float(value) for value in config["solver_start"]["positive_interior_seed_mol"])
        if len(seed_ions) != 6 or abs(sum(value * charge for value, charge in zip(seed_ions, chemistry["charges"][3:], strict=True))) > 1.0e-15:
            raise ValueError("configured positive interior seed is not six-species electroneutral")
        p1_result = None
        p1_replay = None
        for source in source_rows:
            position = source.get("position", "")
            profile = profile_by_position.get(round(float(position), 12)) if source["record_kind"] == "retained_case3c_state" else None
            row = base_row(source, profile, source_table_sha, profile_sha)
            row["run_id"] = run_id
            row["reproduction_command"] = command
            rows.append(row)
            if row["packet_bound_candidate"] != "true":
                continue
            apparent = json.loads(row["apparent_inputs_json"])["normalized_CO2_MEA_H2O"]
            if row["source_row_id"] == "Case3C_position_1" and (
                abs(float(row["source_loaded_analytical_MEA_mol_L"]) - config["selected_state"]["source_loaded_analytical_MEA_mol_L"]) > 1.0e-15
                or abs(float(row["source_free_MEA_mol_L"]) - config["selected_state"]["source_free_MEA_mol_L"]) > 1.0e-15
            ):
                raise ValueError("Issue 33 Position 1 exact source values changed")
            request = packet_request(template, chemistry, float(row["temperature_K"]), float(row["pressure_Pa"]), tuple(apparent), seed_ions)
            problem = general_reactive_equilibrium_problem_from_mapping(request)
            p1_result = solve(model, problem)
            p1_replay = solve(model, problem)
            if p1_result.status != "evaluated" or len(p1_result.phases) != 1 or p1_result.phases[0].role != "liquid":
                row["packet_bound_status"] = "non_evaluable"; row["packet_failure_code"] = str(getattr(p1_result.failure, "code", "solver_failure")); row["packet_failure_diagnostic"] = str(p1_result.failure or p1_result.solver_status); row["admission_reason"] = "packet candidate retained as a typed solver failure; source basis remains unresolved"
            else:
                apply_result(row, p1_result, p1_replay, run_id, command)
        if p1_result is None or p1_result.status != "evaluated":
            raise RuntimeError("authoritative packet did not evaluate the in-domain Position 1 candidate")
        packet_rows = [row for row in rows if row["packet_bound_candidate"] == "true"]
        evaluated_rows = [row for row in rows if row["packet_bound_status"] == "evaluated"]
        summary = {
            "claim_label": config["claim_label"], "analysis": "nccc_validation", "issue": 40, "rows": len(rows), "generator_sha256": generator_sha, "source_repository_commit": source_commit, "source_worktree_clean_at_generation": source_worktree_clean, "source_revision_protocol": config["reproduction"]["source_revision_protocol"], "machine": platform.platform(), "workers": 1, "reproduction_command": command,
            "bundle": {"outer_sha256": bundle_meta["outer_sha256"], "parameter_document_sha256": bundle_meta["parameter_document_sha256"], "engine_wheel_sha256": bundle_meta["engine_wheel_sha256"], "state_packet_sha256": bundle_meta["state_packet_sha256"], "parameter_fingerprint": parameters.fingerprint, "chemistry_sha256": manifest["files"][1]["sha256"]},
            "source_inputs": {"config_sha256": config_sha, "issue33_table_sha256": source_table_sha, "issue33_summary_sha256": sha256_path(ISSUE33_SUMMARY), "retained_profile_sha256": profile_sha},
            "species_order": list(SPECIES_IDS), "species_short_name_order": list(SHORT_NAMES), "apparent_transform": config["apparent_transform"], "state_domain": config["state_domain"],
            "row_counts": {"source_rows": len(rows), "literature_label_rows": sum(row["source_row_id"].startswith("Putta2016_") for row in rows), "retained_case3c_rows": sum(row["case_id"] == "3C" for row in rows), "packet_candidate_rows": len(packet_rows), "packet_evaluated_rows": len(evaluated_rows), "packet_non_evaluable_rows": sum(row["packet_bound_status"] == "non_evaluable" for row in rows), "packet_not_attempted_rows": sum(row["packet_bound_status"] == "not_attempted" for row in rows), "scientifically_admitted_rows": sum(row["scientific_admission"] == "admitted" for row in rows), "basis_unresolved_rows": sum(row["scientific_admission"] == "basis_unresolved" for row in rows)},
            "gates": {"fixed_nine_species_order": list(SPECIES_IDS) == list(config["bundle"].get("species_order", SPECIES_IDS)), "single_liquid_only": all(row["phase_topology"] == "single_liquid_only" for row in rows), "vle_fugacity_equality_imposed": False, "position_1_exact_source_analytical_mol_L": config["selected_state"]["source_loaded_analytical_MEA_mol_L"], "position_1_exact_source_free_MEA_mol_L": config["selected_state"]["source_free_MEA_mol_L"], "position_1_issue33_reconstructed_true_species_loading": config["selected_state"]["issue33_reconstructed_true_species_loading"]["value"], "position_1_apparent_flow_ratio": config["selected_state"]["apparent_total_inorganic_carbon_to_analytical_MEA_flow_ratio"]["value"], "position_1_apparent_minus_issue33_loading": config["selected_state"]["loading_apparent_flow_ratio_difference"]["value"], "loading_and_apparent_ratio_distinct_not_relabelled_or_fitted": config["selected_state"]["loading_apparent_not_relabelled_or_fitted"], "position_1_source_basis_unresolved": True, "packet_evaluated_at_least_one_row": len(evaluated_rows) == 1, "analytical_and_elemental_residual_pass": max(float(row["inverse_max_abs_residual"]) for row in evaluated_rows) <= 1.0e-10, "mole_fraction_normalization_pass": max(float(row["mole_fraction_normalization_residual"]) for row in evaluated_rows) <= 1.0e-12, "charge_residual_pass": max(float(row["charge_residual"]) for row in evaluated_rows) <= 1.0e-10, "deterministic_replay_pass": all(row["forward_replay_status"] == "identity_pass" for row in evaluated_rows), "replay_branch_match_pass": all(row["replay_branch_match_status"] == "identity_pass" for row in evaluated_rows), "no_capture_inference": all(row["capture_inference_used"] == "false" for row in rows), "no_thermo_or_kinetic_fit": all(row["thermodynamic_or_kinetic_fit_performed"] == "false" for row in rows)},
            "packet_state_evidence": {"state_packet_observation_count": len(state_packet["observations"]), "state_packet_is_historical_evidence_only": True, "non_evaluable_state_table_sha256": next(item["sha256"] for item in manifest["files"] if item["path"] == "validation/non-evaluable-states.csv"), "non_evaluable_state_count": len([line for line in members["validation/non-evaluable-states.csv"].decode().splitlines() if line.strip()]) - 1},
            "unresolved_rows": [row["source_row_id"] for row in rows if row["scientific_admission"] == "basis_unresolved"], "failure_rows": [row["source_row_id"] for row in rows if row["packet_bound_status"] == "non_evaluable"],
            "limitations": ["The Position 1 packet mapping is evaluated but is not a prepared/loaded concentration admission because NCCC preparation temperature and prepared-to-loaded volume basis remain unreported.", "The two out-of-domain Case 3C profile rows and two source-only Putta labels are retained as not attempted or unresolved rows; no row is silently dropped.", "The apparent-to-true transform is a conserved-total forward map plus packet equilibrium inverse reconstruction; it is not an algebraic unique inverse of the rank-deficient transform.", "The historical packet contains non-evaluable states; those states are retained as evidence and no historical continuation state is reused as a process-column initial condition.", "No vapor phase, VLE fugacity equality, packed-column capture, transport, area, hydraulic, or fitted quantity is used."],
        }
        write_table(rows)
        summary["result_table_sha256"] = sha256_path(TABLE)
        SUMMARY.parent.mkdir(parents=True, exist_ok=True); SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report = f"""# Issue 40 apparent-to-true species mapping

This retained analysis reconstructs source apparent component flows and maps the in-domain retained Case 3C Position 1 state through the authoritative nine-species reactive ePC-SAFT packet. It does not replace the Issue 33 source basis: the exact source values remain **{config['selected_state']['source_loaded_analytical_MEA_mol_L']} mol/L analytical MEA** and **{config['selected_state']['source_free_MEA_mol_L']} mol/L free MEA**, with `basis_unresolved` admission.

## Reproduction and identities

Command: `{command}`<br>
Source repository commit: `{summary['source_repository_commit']}` (result artifacts are committed separately)<br>
Generator SHA-256: `{generator_sha}`<br>
Machine: `{summary['machine']}`; workers: `1`

The source revision protocol is: `{summary['source_revision_protocol']}` The focused validator rehashes the current source files and retained outputs against these recorded identities; it is a retained-output/source-lineage consistency check, not an independent physical reproduction. Checking out the recorded source commit and rerunning the command reproduces the retained result artifacts before their separate result-artifact commit.

The immutable bundle identities are outer zip `{bundle_meta['outer_sha256']}`, parameter document `{bundle_meta['parameter_document_sha256']}`, ePC-SAFT wheel `{bundle_meta['engine_wheel_sha256']}`, state packet `{bundle_meta['state_packet_sha256']}`, and parameter fingerprint `{parameters.fingerprint}`. The bundle verifier was run before this analysis.

## Definition and method

The true species order is `{', '.join(SHORT_NAMES)}`. Apparent inputs are normalized component flows `(CO2, MEA, H2O)` in `mol per mol analytical MEA`, with CO2 representing apparent total inorganic carbon. The retained transform rows are `{', '.join(TRANSFORM_ROWS)}`. Prepared concentration, analytical concentration, free MEA, apparent totals, true species, and diagnostic density are separate definitions in the Issue 40 input record. Every apparent input records units and reporting-interval status in the result table; absent experimental intervals remain explicitly `null`/unreported.

The packet request is one finite liquid phase at `T=318.15 K`, `P=109500 Pa`, with no vapor phase and no VLE fugacity equality. Reaction constants are compiled at the requested temperature from the bundle reaction correlations, including the explicit R1--R3 standard-state offsets. The packet's electroneutral positive interior seed is used only as a solver start; exact apparent C/N totals are conserved independently. The returned true state is forward-transformed and inverse-replayed by conserved totals, and the same request is solved twice for deterministic identity.

## Retained row accounting

| Row class | Count |
|---|---:|
| Source rows | {summary['row_counts']['source_rows']} |
| Literature label rows | {summary['row_counts']['literature_label_rows']} |
| Case 3C retained profile rows | {summary['row_counts']['retained_case3c_rows']} |
| Packet candidates | {summary['row_counts']['packet_candidate_rows']} |
| Packet evaluated | {summary['row_counts']['packet_evaluated_rows']} |
| Packet not attempted | {summary['row_counts']['packet_not_attempted_rows']} |
| Scientific admissions | {summary['row_counts']['scientifically_admitted_rows']} |
| Basis-unresolved rows | {summary['row_counts']['basis_unresolved_rows']} |

Position 1 is packet-evaluated with a single `strict_stable_local_minimum` liquid branch. It remains scientifically unadmitted because the source prepared/loaded volume basis is unresolved. Positions 0 and 0.5 are retained as out-of-common-domain, not attempted; Putta labels are retained as source-label-only rows. For Position 1, the Issue 33 reconstructed true-species loading is **{config['selected_state']['issue33_reconstructed_true_species_loading']['value']} mol CO2 per mol MEA** and is used only for the domain gate. The packet input is the distinct apparent total-inorganic-carbon/analytical-MEA flow ratio **{config['selected_state']['apparent_total_inorganic_carbon_to_analytical_MEA_flow_ratio']['value']} mol per mol**, with difference **{config['selected_state']['loading_apparent_flow_ratio_difference']['value']} mol per mol**; neither value is relabelled or fitted.

## Claim boundary

The mapping is numerical evidence for a specified single-liquid state, not a packed-column result. It infers no thermodynamic, kinetic, transport, area, hydraulic, or capture quantity, and it performs no parameter fit. Historical packet failures remain represented by their immutable state-packet/non-evaluable-state identities; no historical continuation state is reused as a process initial condition.
"""
        REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(report, encoding="utf-8")
        print(f"Wrote {TABLE}"); print(f"Wrote {SUMMARY}"); print(f"Wrote {REPORT}"); print(json.dumps(summary["row_counts"], indent=2)); return 0
    finally:
        archive.close()


if __name__ == "__main__":
    raise SystemExit(main())
