from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "integration" / "reactive_mea_application_contract.json"
ELEMENTS = ("C", "H", "N", "O")
ATOMIC_MASS_KG_PER_MOL = {
    "C": 0.012011,
    "H": 0.001008,
    "N": 0.014007,
    "O": 0.015999,
}
MOLAR_MASS_ABS_TOLERANCE_KG_PER_MOL = 1.0e-5


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _rank(matrix: list[list[int]]) -> int:
    work = [[Fraction(value) for value in row] for row in matrix]
    if not work:
        return 0
    rows = len(work)
    columns = len(work[0])
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(rows):
            if row == rank or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(work[row], work[rank], strict=True)
            ]
        rank += 1
        if rank == rows:
            break
    return rank


def _float_close(observed: str | float, expected: float, *, atol: float = 1.0e-10) -> bool:
    return abs(float(observed) - expected) <= atol


def _unique_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
    matches = [row for row in rows if row.get(key) == value]
    return matches[0] if len(matches) == 1 else None


def _check_structure(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != 1:
        errors.append("contract schema_version must be 1")
    if contract.get("identity") != "mea-reactive-vle-application-contract-v1":
        errors.append("unexpected contract identity")

    endpoint = contract.get("scientific_endpoint", {})
    if endpoint.get("kind") != "predictive_reactive_vle":
        errors.append("scientific endpoint is not predictive_reactive_vle")
    forbidden_inputs = set(endpoint.get("forbidden_prediction_inputs", []))
    if forbidden_inputs != {"observed_total_pressure", "observed_CO2_partial_pressure"}:
        errors.append("predictive endpoint must forbid both observed pressure inputs")

    chemistry = contract.get("chemistry", {})
    species = chemistry.get("species", [])
    reactions = chemistry.get("reactions", [])
    if len(species) != 9:
        errors.append(f"expected 9 species, found {len(species)}")
        return errors
    if len(reactions) != 5:
        errors.append(f"expected 5 reactions, found {len(reactions)}")
        return errors

    source_ids = [item["source_id"] for item in species]
    provider_ids = [item["provider_id"] for item in species]
    if len(set(source_ids)) != len(source_ids):
        errors.append("source species identities are not unique")
    if len(set(provider_ids)) != len(provider_ids):
        errors.append("provider species identities are not unique")

    for item in species:
        formula = item["formula"]
        calculated = sum(formula[element] * ATOMIC_MASS_KG_PER_MOL[element] for element in ELEMENTS)
        declared = float(item["molar_mass_kg_per_mol"])
        if abs(calculated - declared) > MOLAR_MASS_ABS_TOLERANCE_KG_PER_MOL:
            errors.append(
                f"{item['source_id']} molar mass {declared} is inconsistent with formula value {calculated}"
            )
    carbamate = next(item for item in species if item["source_id"] == "MEACOO-")
    if not _float_close(carbamate["molar_mass_kg_per_mol"], 0.10408, atol=1.0e-12):
        errors.append("MEACOO- molar mass must be 0.10408 kg/mol")

    stoichiometry = [reaction["stoichiometry"] for reaction in reactions]
    if any(len(row) != len(species) for row in stoichiometry):
        errors.append("reaction stoichiometry width does not match species order")
        return errors
    if _rank(stoichiometry) != chemistry.get("declared_reaction_rank"):
        errors.append("declared reaction rank does not match exact stoichiometric rank")

    for reaction, coefficients in zip(reactions, stoichiometry, strict=True):
        for element in ELEMENTS:
            balance = sum(
                coefficient * item["formula"][element]
                for coefficient, item in zip(coefficients, species, strict=True)
            )
            if balance != 0:
                errors.append(f"{reaction['id']} does not conserve {element}: {balance}")
        charge = sum(
            coefficient * item["charge"]
            for coefficient, item in zip(coefficients, species, strict=True)
        )
        if charge != 0:
            errors.append(f"{reaction['id']} does not conserve charge: {charge}")

    split = contract.get("observations", {}).get("pressure_speciation_split", {})
    if split.get("training_states") + split.get("reserved_states") != split.get("total_states"):
        errors.append("pressure/speciation split totals are inconsistent")
    if sum(split.get("training_by_family", {}).values()) != split.get("training_states"):
        errors.append("training family counts do not sum to the training total")
    if sum(split.get("reserved_by_family", {}).values()) != split.get("reserved_states"):
        errors.append("reserved family counts do not sum to the reserved total")

    hierarchy = contract.get("nested_model_hierarchy", {})
    if not hierarchy.get("discrete_selection_outside_optimizer"):
        errors.append("discrete model selection must remain outside the optimizer")
    if contract.get("observations", {}).get("reserved_policy") != (
        "reserved_observations_may_reject_but_may_not_redesign_or_refit_the_model"
    ):
        errors.append("reserved-set policy is not fail closed")
    return errors


def _check_source(contract: dict[str, Any], source_root: Path) -> list[str]:
    errors: list[str] = []
    if not (source_root / ".git").exists():
        return [f"source root is not a Git checkout: {source_root}"]

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )
    observed_commit = head.stdout.strip()
    expected_commit = contract["source_binding"]["commit"]
    if head.returncode != 0 or observed_commit != expected_commit:
        errors.append(f"source commit mismatch: expected {expected_commit}, observed {observed_commit}")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        errors.append("source checkout is not clean")

    artifacts = {item["id"]: item for item in contract["source_binding"]["artifacts"]}
    for artifact in artifacts.values():
        path = source_root / artifact["path"]
        if not path.is_file():
            errors.append(f"missing source artifact: {artifact['path']}")
            continue
        observed_hash = _sha256(path)
        if observed_hash != artifact["sha256"]:
            errors.append(
                f"source artifact hash mismatch for {artifact['path']}: "
                f"expected {artifact['sha256']}, observed {observed_hash}"
            )

    reaction_contract = _load_json(source_root / artifacts["reaction_source_contract"]["path"])
    chemistry = contract["chemistry"]
    if reaction_contract.get("identity") != chemistry["source_contract_identity"]:
        errors.append("reaction source-contract identity mismatch")
    if reaction_contract.get("status") != chemistry["source_contract_status"]:
        errors.append("reaction source-contract status mismatch")
    if reaction_contract.get("species_order") != [item["source_id"] for item in chemistry["species"]]:
        errors.append("reaction source species order mismatch")
    if reaction_contract.get("provider_species_order") != [
        item["provider_id"] for item in chemistry["species"]
    ]:
        errors.append("reaction provider species order mismatch")
    if [item["reaction_id"] for item in reaction_contract.get("reactions", [])] != [
        item["id"] for item in chemistry["reactions"]
    ]:
        errors.append("reaction order mismatch")
    if [item["stoichiometry"] for item in reaction_contract.get("reactions", [])] != [
        item["stoichiometry"] for item in chemistry["reactions"]
    ]:
        errors.append("reaction stoichiometry mismatch")
    if reaction_contract.get("common_source_standard_state", {}).get("identity") != chemistry[
        "common_source_standard_state"
    ]:
        errors.append("common source standard-state identity mismatch")
    provider_transform = reaction_contract.get("provider_transform", {})
    if provider_transform.get("identity") != chemistry["provider_transform"]:
        errors.append("provider transform identity mismatch")
    if provider_transform.get("ready") is not chemistry["provider_transform_ready"]:
        errors.append("provider transform readiness mismatch")

    parameter_rows = _rows(source_root / artifacts["phase2_pure_parameter_seed"]["path"])
    for species in chemistry["species"]:
        row = _unique_row(parameter_rows, "component", species["parameter_component"])
        if row is None:
            errors.append(f"parameter seed must contain exactly one row for {species['parameter_component']}")
            continue
        if not _float_close(row["MW"], species["molar_mass_kg_per_mol"], atol=1.0e-12):
            errors.append(f"parameter-seed molar mass mismatch for {species['parameter_component']}")
        if int(row["z"]) != species["charge"]:
            errors.append(f"parameter-seed charge mismatch for {species['parameter_component']}")

    observations = contract["observations"]
    grouped_rows = _rows(source_root / artifacts["grouped_pressure_speciation_split"]["path"])
    role_counts = Counter(row["role"] for row in grouped_rows)
    family_role_counts = Counter((row["target_family"], row["role"]) for row in grouped_rows)
    split = observations["pressure_speciation_split"]
    if role_counts != Counter(
        {"active_training": split["training_states"], "reserved_validation": split["reserved_states"]}
    ):
        errors.append(f"pressure/speciation role counts drifted: {dict(role_counts)}")
    for family, expected in split["training_by_family"].items():
        if family_role_counts[(family, "active_training")] != expected:
            errors.append(f"training count drifted for {family}")
    for family, expected in split["reserved_by_family"].items():
        if family_role_counts[(family, "reserved_validation")] != expected:
            errors.append(f"reserved count drifted for {family}")

    membership_rows = _rows(source_root / artifacts["speciation_target_membership"]["path"])
    metrology_rows = _rows(source_root / artifacts["pressure_metrology"]["path"])
    volumetric_rows = _rows(source_root / artifacts["volumetric_observation_contract"]["path"])
    eligible = observations["eligible_targets"]
    if sum(row["target_eligible"] == "yes" for row in membership_rows) != eligible["speciation"]:
        errors.append("eligible speciation target count drifted")
    if sum(row["target_eligible"] == "yes" for row in metrology_rows) != eligible["pressure"]:
        errors.append("eligible pressure target count drifted")
    if sum(row["target_eligible"] == "yes" for row in volumetric_rows) != eligible["volumetric"]:
        errors.append("eligible volumetric target count drifted")

    volumetric_split_rows = _rows(source_root / artifacts["volumetric_split"]["path"])
    volumetric_roles = Counter(row["role"] for row in volumetric_split_rows)
    if volumetric_roles != Counter(
        {
            "future_training": observations["volumetric_split"]["training_states"],
            "reserved_validation": observations["volumetric_split"]["reserved_states"],
        }
    ):
        errors.append(f"volumetric split counts drifted: {dict(volumetric_roles)}")

    tracer = observations["reduced_tracer"]
    pressure = tracer["pressure_row"]
    vle_rows = _rows(source_root / artifacts["vle_observations"]["path"])
    vle = _unique_row(vle_rows, "observation_id", pressure["id"])
    metrology = _unique_row(metrology_rows, "observation_id", pressure["id"])
    if vle is None or metrology is None:
        errors.append("reduced-tracer pressure row is missing or duplicated")
    else:
        pressure_checks = (
            _float_close(float(vle["temperature_canonical_C"]) + 273.15, pressure["temperature_k"]),
            _float_close(vle["MEA_weight_fraction"], pressure["mea_mass_fraction"]),
            _float_close(vle["CO2_loading"], pressure["co2_loading_mol_per_mol_mea"]),
            _float_close(float(vle["CO2_pressure"]) * 1000.0, pressure["observed_co2_partial_pressure_pa"]),
            _float_close(metrology["state_pressure_pa"], pressure["observed_total_pressure_pa"]),
            metrology["target_eligible"] == "yes",
        )
        if not all(pressure_checks):
            errors.append("reduced-tracer pressure payload drifted")

    speciation = tracer["speciation_row"]
    speciation_rows = _rows(source_root / artifacts["speciation_observations"]["path"])
    spec = _unique_row(speciation_rows, "record_id", speciation["id"])
    if spec is None:
        errors.append("reduced-tracer speciation row is missing or duplicated")
    else:
        speciation_checks = (
            _float_close(spec["temperature_K"], speciation["temperature_k"]),
            _float_close(spec["mea_mass_fraction"], speciation["mea_mass_fraction"]),
            _float_close(spec["co2_loading_mol_per_mol_mea"], speciation["co2_loading_mol_per_mol_mea"]),
            spec["species"] == speciation["species"],
            _float_close(spec["reported_value"], speciation["observed_mole_fraction"]),
            spec["reported_unit"] == "mole_fraction",
        )
        if not all(speciation_checks):
            errors.append("reduced-tracer speciation payload drifted")
    return errors


def _check_final(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("overall_status") != "accepted":
        errors.append(f"overall contract status is {contract.get('overall_status')!r}, not 'accepted'")
    upstream = contract.get("upstream_acceptance", {})
    for capability, status in upstream.get("capabilities", {}).items():
        if status != "accepted":
            errors.append(f"upstream capability {capability} is {status!r}, not 'accepted'")
    if upstream.get("installed_artifact_receipt") is None:
        errors.append("installed upstream artifact receipt is missing")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen MEA reactive-VLE application contract.")
    parser.add_argument("--mode", choices=("structure", "source", "final"), default="structure")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)

    contract = _load_json(CONTRACT_PATH)
    errors = _check_structure(contract)
    if args.mode in {"source", "final"}:
        if args.source_root is None:
            errors.append("--source-root is required for source and final modes")
        else:
            errors.extend(_check_source(contract, args.source_root.resolve()))
    if args.mode == "final":
        errors.extend(_check_final(contract))

    print(f"contract: {contract['identity']}")
    print(f"mode: {args.mode}")
    print(f"status: {contract['overall_status']}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("acceptance: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
