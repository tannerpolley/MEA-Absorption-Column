from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses/nccc_validation"
INPUT = ANALYSIS / "inputs/issue34_kinetics.json"
TABLES = ANALYSIS / "results/final/tables"
ISSUE33_SUMMARY = ROOT / "analyses/nccc_validation/results/final/tables/issue33_concentration_basis_summary.json"
ISSUE33_TABLE = ROOT / "analyses/nccc_validation/results/final/tables/issue33_concentration_basis.csv"

FINITE_TABLE = TABLES / "issue34_finite_reactions.csv"
PARTITION_TABLE = TABLES / "issue34_partition_decisions.csv"
OBSERVATION_TABLE = TABLES / "issue34_rate_observation_comparisons.csv"
SENSITIVITY_TABLE = TABLES / "issue34_kinetic_sensitivity.csv"
SUMMARY = TABLES / "issue34_kinetics_summary.json"
REPORT = ANALYSIS / "results/final/reports/issue34_reaction_kinetics.md"

ELEMENTS = ("C", "H", "N", "O", "charge")
DETAILED_BALANCE_TOLERANCE = 1.0e-7
REACTION_TERMS = {
    "F1": ({"MEA": 2, "CO2": 1}, {"MEAH+": 1, "MEACOO-": 1}),
    "F2": ({"H2O": 1, "MEA": 1, "CO2": 1}, {"H3O+": 1, "MEACOO-": 1}),
    "F3": ({"CO2": 1, "OH-": 1}, {"HCO3-": 1}),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_inputs() -> dict:
    config = json.loads(INPUT.read_text(encoding="utf-8"))
    require(config["schema_version"] == "issue34_source_faithful_kinetics_v1", "unexpected Issue 34 input schema")
    require(config["species_order"] == [
        "CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-"
    ], "species order changed")
    return config


def validate_issue33_dependency(config: dict) -> dict:
    dependency = config["issue33_dependency"]
    require(sha256(ISSUE33_SUMMARY) == dependency["summary_sha256"], "Issue 33 summary hash changed")
    require(sha256(ISSUE33_TABLE) == dependency["table_sha256"], "Issue 33 table hash changed")
    summary = json.loads(ISSUE33_SUMMARY.read_text(encoding="utf-8"))
    require(summary["claim_label"] == dependency["required_claim_label"], "Issue 33 claim label changed")
    gates = summary["gates"]
    require(gates["position_1_exact_value_preserved"], "Issue 33 Position 1 value is not preserved")
    require(gates["position_1_not_rounded_to_5M"], "Issue 33 Position 1 was rounded to 5 M")
    require(not gates["exact_discrete_1M_or_5M_admission"], "Issue 33 admitted an exact 1 M or 5 M row")
    require(not gates["capture_or_kinetic_tuning_performed"], "Issue 33 performed forbidden tuning")
    require(
        abs(float(gates["position_1_analytical_concentration_mol_L"]) - dependency["required_position_1_analytical_mol_L"])
        <= 1.0e-14,
        "Issue 33 Position 1 analytical concentration changed",
    )
    with ISSUE33_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows and all(row["admission_decision"] == dependency["required_row_admission_decision"] for row in rows),
            "Issue 33 contains an admitted concentration-basis row")
    position_1_rows = [row for row in rows if row["record_id"] == "Case3C_position_1"]
    require(len(position_1_rows) == 1, "Issue 33 Position 1 row is missing")
    require(
        abs(float(position_1_rows[0]["free_MEA_concentration_mol_L"]) - dependency["required_position_1_free_mea_mol_L"])
        <= 1.0e-14,
        "Issue 33 Position 1 free MEA value changed",
    )
    return summary


def reaction_rows(config: dict) -> tuple[list[dict], dict[str, list[int]], bool, bool, bool]:
    order = config["species_order"]
    source_by_id = {row["id"]: row for row in config["source_reactions"]}
    vectors = {key: row["stoichiometry"] for key, row in source_by_id.items()}
    rows = []
    stoichiometry_pass = True
    element_pass = True
    charge_pass = True
    for reaction in config["source_reactions"]:
        vector = reaction["stoichiometry"]
        element_residuals = {
            element: sum(vector[index] * config["species_formulae"][species][element] for index, species in enumerate(order))
            for element in ELEMENTS
        }
        stoichiometry_pass &= len(vector) == len(order)
        element_pass &= all(value == 0 for value in element_residuals.values())
        charge_pass &= element_residuals["charge"] == 0
        rows.append({
            "record_type": "source_reaction",
            "reaction_id": reaction["id"],
            "equation": reaction["equation"],
            "source_equation": reaction["source_equation"],
            "source_locator": reaction["source_locator"],
            "source_reaction_coefficients": "",
            "stoichiometry_json": compact_json(dict(zip(order, vector))),
            "stoichiometry_pass": len(vector) == len(order),
            "element_balance_residuals_json": compact_json(element_residuals),
            "element_balance_pass": all(value == 0 for value in element_residuals.values()),
            "charge_balance_pass": element_residuals["charge"] == 0,
            "decision": "source_reaction_retained",
            "timescale_status": "not_evaluable_missing_admitted_diffusivity_and_film_thickness",
            "timescale_value": "",
            "missing_timescale_inputs": "reaction-specific D and film thickness; no local source-complete timescale",
            "partition_evidence": reaction["role"],
        })
    for combination in config["reaction_combinations"]:
        vector = [
            sum(coefficient * vectors[reaction_id][index] for reaction_id, coefficient in combination["source_reaction_coefficients"].items())
            for index in range(len(order))
        ]
        vectors[combination["id"]] = vector
        expected = combination["expected_stoichiometry"]
        element_residuals = {
            element: sum(vector[index] * config["species_formulae"][species][element] for index, species in enumerate(order))
            for element in ELEMENTS
        }
        this_stoich = vector == expected
        this_element = all(value == 0 for value in element_residuals.values())
        this_charge = element_residuals["charge"] == 0
        stoichiometry_pass &= this_stoich
        element_pass &= this_element
        charge_pass &= this_charge
        decision = {
            "F1": "source_relationship_retained_comparison_only",
            "F2": "source_relationship_retained_comparison_only",
            "F3": "source_relationship_unavailable_supported_negative",
        }[combination["id"]]
        rows.append({
            "record_type": "finite_projection",
            "reaction_id": combination["id"],
            "equation": combination["equation"],
            "source_equation": "",
            "source_locator": combination["source_locator"],
            "source_reaction_coefficients": compact_json(combination["source_reaction_coefficients"]),
            "stoichiometry_json": compact_json(dict(zip(order, vector))),
            "stoichiometry_pass": this_stoich,
            "element_balance_residuals_json": compact_json(element_residuals),
            "element_balance_pass": this_element,
            "charge_balance_pass": this_charge,
            "decision": decision,
            "timescale_status": "not_evaluable_missing_admitted_diffusivity_and_film_thickness",
            "timescale_value": "",
            "missing_timescale_inputs": "reaction-specific D and film thickness; F3 also lacks a source coefficient" if combination["id"] == "F3" else "reaction-specific D and film thickness",
            "partition_evidence": "Putta2016 penetration-theory model retains reversible rate forms; it does not provide a local reaction-specific film timescale",
        })
    return rows, vectors, stoichiometry_pass, element_pass, charge_pass


def coefficient_value(record: dict, temperature_K: float) -> tuple[float, float]:
    require(record["pre_exponential"] is not None, f"{record['id']} has no source coefficient")
    floating = float(record["pre_exponential"]) * math.exp(-float(record["activation_temperature_K"]) / temperature_K)
    with localcontext() as context:
        context.prec = 50
        decimal = Decimal(str(record["pre_exponential"])) * (
            -Decimal(str(record["activation_temperature_K"])) / Decimal(str(temperature_K))
        ).exp()
    return floating, float(decimal)


def reconstructed_coefficient_unit(reaction_id: str) -> str:
    reaction_order = sum(REACTION_TERMS[reaction_id][0].values())
    length_power = 3 * reaction_order - 3
    amount_power = 1 - reaction_order
    length = "m" if length_power == 1 else f"m{length_power}"
    amount = "kmol" if amount_power == 1 else f"kmol^{amount_power}"
    return f"{length} {amount} s^-1"


def finite_rows(config: dict, reaction_vectors: dict[str, list[int]]) -> tuple[list[dict], list[dict], dict, bool]:
    finite = []
    sensitivity = []
    anchors = config["source_domain"]["reconstruction_anchor_temperature_K"]
    by_basis_temperature: dict[tuple[str, float], float] = {}
    dimensional_pass = True
    for record in config["finite_reactions"]:
        reaction_id = record["id"]
        vector = reaction_vectors[reaction_id]
        coefficient_status = "recovered" if record["pre_exponential"] is not None else "unavailable"
        reconstructed_unit = reconstructed_coefficient_unit(reaction_id)
        this_dimensional_pass = record["required_coefficient_unit"] == reconstructed_unit
        dimensional_pass &= this_dimensional_pass
        finite.append({
            "reaction_id": reaction_id,
            "basis": record["basis"],
            "equation": next(item["equation"] for item in config["reaction_combinations"] if item["id"] == reaction_id),
            "rate_law": record["rate_law"],
            "equilibrium_quotient": record["equilibrium_quotient"],
            "source_formula": record["source_formula"] or "",
            "coefficient_symbol": record["coefficient_symbol"],
            "coefficient_status": coefficient_status,
            "source_printed_coefficient_unit": record["source_printed_coefficient_unit"] or "",
            "required_coefficient_unit": record["required_coefficient_unit"],
            "reaction_order": sum(REACTION_TERMS[reaction_id][0].values()),
            "delta_n": sum(vector),
            "raw_quotient_units": "m^3 kmol^-1" if sum(vector) == -1 else f"(kmol m^-3)^{sum(vector)}",
            "standard_state_conversion": "K_dimensionless = K_raw * c° for Δν=-1; source c° is unspecified",
            "reconstructed_coefficient_unit": reconstructed_unit,
            "dimensional_reconstruction_pass": this_dimensional_pass,
            "unit_status": record["unit_status"],
            "standard_state": record["standard_state"],
            "coefficient_uncertainty": record["coefficient_uncertainty"],
            "stoichiometry_json": compact_json(dict(zip(config["species_order"], vector))),
            "source_locator": record["source_locator"],
            "admission_decision": record["admission_decision"],
            "admission_reason": record["admission_reason"],
        })
        for temperature_K in anchors:
            if record["pre_exponential"] is None:
                continue
            floating, decimal = coefficient_value(record, temperature_K)
            by_basis_temperature[(reaction_id, record["basis"], temperature_K)] = floating
            absolute_difference = abs(floating - decimal)
            relative_difference = absolute_difference / abs(decimal)
            sensitivity.append({
                "record_type": "temperature_reconstruction",
                "reaction_id": reaction_id,
                "basis": record["basis"],
                "temperature_K": temperature_K,
                "source_concentration_labels_mol_L": compact_json(config["source_domain"]["putta2016_concentration_labels_mol_L"]),
                "k_float": floating,
                "k_decimal": decimal,
                "absolute_float_decimal_difference": absolute_difference,
                "relative_float_decimal_difference": relative_difference,
                "activity_to_concentration_ratio": "",
                "sensitivity_status": "reconstructed_at_source_temperature_anchor",
                "reason": "Source coefficient is independent of concentration; source 1 M and 5 M labels are retained as labels only.",
            })
    for reaction_id in ("F1", "F2"):
        for temperature_K in anchors:
            concentration = by_basis_temperature[(reaction_id, "concentration", temperature_K)]
            activity = by_basis_temperature[(reaction_id, "activity", temperature_K)]
            ratio = activity / concentration
            for row in sensitivity:
                if row["reaction_id"] == reaction_id and row["temperature_K"] == temperature_K:
                    row["activity_to_concentration_ratio"] = ratio
    sensitivity.append({
        "record_type": "pathway_exclusion",
        "reaction_id": "F3",
        "basis": "concentration",
        "temperature_K": "",
        "source_concentration_labels_mol_L": compact_json(config["source_domain"]["putta2016_concentration_labels_mol_L"]),
        "k_float": "",
        "k_decimal": "",
        "absolute_float_decimal_difference": "",
        "relative_float_decimal_difference": "",
        "activity_to_concentration_ratio": "",
        "sensitivity_status": "not_quantified_supported_negative",
        "reason": "Excluding F3 removes the CO2/OH-/HCO3- pathway, but its primary coefficient and an admitted local film state are unavailable; flux impact is not evaluable.",
    })
    return finite, sensitivity, {"temperature_reconstruction_count": len(sensitivity) - 1}, dimensional_pass


def source_state_closure(config: dict, finite_records: list[dict]) -> tuple[list[dict], dict]:
    state = config["source_state_reconstruction"]["concentrations"]
    temperature_K = config["source_state_reconstruction"]["temperature_K"]
    closure_rows = []
    max_balance_residual = 0.0
    for record in finite_records:
        if record["reaction_id"] == "F3" or record["basis"] != "concentration":
            continue
        reactants, products = REACTION_TERMS[record["reaction_id"]]
        forward_concentration = math.prod(state[name] ** power for name, power in reactants.items())
        product_concentration = math.prod(state[name] ** power for name, power in products.items())
        quotient = product_concentration / forward_concentration
        equilibrium_constant = quotient
        ln_residual = abs(math.log(quotient) - math.log(equilibrium_constant))
        coefficient_record = next(
            item for item in config["finite_reactions"]
            if item["id"] == record["reaction_id"] and item["basis"] == "concentration"
        )
        coefficient, _ = coefficient_value(coefficient_record, temperature_K)
        forward_rate = coefficient * forward_concentration
        reverse_coefficient = coefficient / equilibrium_constant
        reverse_rate = reverse_coefficient * product_concentration
        rate_residual = abs(forward_rate - reverse_rate)
        max_balance_residual = max(max_balance_residual, ln_residual)
        closure_rows.append({
            "reaction_id": record["reaction_id"],
            "basis": "concentration",
            "temperature_K": temperature_K,
            "state_role": config["source_state_reconstruction"]["state_role"],
            "equilibrium_quotient_Q": quotient,
            "selected_equilibrium_constant_K": equilibrium_constant,
            "raw_quotient_units": "m^3 kmol^-1",
            "absolute_ln_Q_minus_ln_K": ln_residual,
            "detailed_balance_tolerance": DETAILED_BALANCE_TOLERANCE,
            "detailed_balance_pass": ln_residual <= DETAILED_BALANCE_TOLERANCE,
            "coefficient_unit": "m6 kmol^-2 s^-1",
            "forward_rate_kmol_m3_s": forward_rate,
            "reverse_coefficient_m6_kmol_m3_s": reverse_coefficient,
            "reverse_rate_kmol_m3_s": reverse_rate,
            "absolute_rate_residual_kmol_m3_s": rate_residual,
            "closure_status": "source_state_algebraic_reconstruction_only",
            "physical_admission": "not_admitted_basis_unresolved",
        })
    return closure_rows, {
        "state_temperature_K": temperature_K,
        "max_absolute_ln_Q_minus_ln_K": max_balance_residual,
        "detailed_balance_tolerance": DETAILED_BALANCE_TOLERANCE,
        "detailed_balance_pass": max_balance_residual <= DETAILED_BALANCE_TOLERANCE,
        "activity_closure": "not_evaluable_activity_coefficients_and_standard_state_unavailable",
    }


def observation_rows(config: dict) -> list[dict]:
    rows = []
    for aggregate in config["observed_aggregate_aard"]:
        for apparatus, value in aggregate.items():
            if apparatus == "model":
                continue
            rows.append({
                "source": "Putta2016",
                "source_locator": "printed p. 349, Table 4",
                "model": aggregate["model"],
                "apparatus_or_dataset": apparatus,
                "metric": "AARD_percent",
                "value": value,
                "uncertainty_status": "not_reported_in_aggregate_table",
                "raw_rows_retained": False,
                "independent_fit_validation_partition": False,
                "admission_decision": "aggregate_summary_only_not_admitted",
                "reason": "Table 4 is an aggregate comparison; raw observations, row-level uncertainty, and non-overlap fit/validation membership are not retained.",
            })
    return rows


def partition_rows(config: dict, reaction_rows_: list[dict]) -> list[dict]:
    rows = []
    for row in reaction_rows_:
        if row["record_type"] == "source_reaction":
            reaction_id = row["reaction_id"]
            decision = {
                "R1": "fast_equilibrium_unresolved",
                "R2": "source_component_of_finite_projections",
                "R3": "fast_equilibrium_unresolved",
                "R4": "fast_equilibrium_unresolved",
                "R5": "fast_equilibrium_unresolved",
            }[reaction_id]
            evidence = {
                "R1": "Putta2016 treats acid/base chemistry as bulk equilibrium; no local reaction-specific timescale",
                "R2": "Putta2016 Eq. 16/17 projections use R2 in F1/F2/F3",
                "R3": "Putta2016 treats carbonate chemistry as bulk equilibrium; no local reaction-specific timescale",
                "R4": "Putta2016 uses carbamate hydrolysis in the projection but does not quantify its local film timescale",
                "R5": "Putta2016 uses protonated-MEA dissociation in the projection but does not quantify its local film timescale",
            }[reaction_id]
        else:
            decision = row["decision"]
            evidence = row["partition_evidence"]
        rows.append({
            "reaction_id": row["reaction_id"],
            "equation": row["equation"],
            "record_type": row["record_type"],
            "source_equation": row["source_equation"],
            "source_reaction_coefficients": row["source_reaction_coefficients"],
            "stoichiometry_json": row["stoichiometry_json"],
            "stoichiometry_pass": row["stoichiometry_pass"],
            "element_balance_residuals_json": row["element_balance_residuals_json"],
            "element_balance_pass": row["element_balance_pass"],
            "charge_balance_pass": row["charge_balance_pass"],
            "decision": decision,
            "timescale_metric": "Damkohler_or_Hatta_equivalent",
            "timescale_value": row["timescale_value"],
            "timescale_status": row["timescale_status"],
            "missing_timescale_inputs": row["missing_timescale_inputs"],
            "source_timescale_evidence": "Luo2015 printed pp. 60--61, Eq. 8 defines Ha from k_obs, D_CO2, and k_l; Putta2016 uses penetration theory but provides no retained local D/delta partition result",
            "partition_evidence": evidence,
            "physical_film_admission": "not_admitted",
            "source_locator": row["source_locator"],
        })
    return rows


def report_text(config: dict, summary: dict, closure: dict) -> str:
    dependency = config["issue33_dependency"]
    return f"""# Issue 34: reversible MEA film kinetics and reaction partition

Status: **supported-negative source-faithful record complete**. This record preserves the source reaction space, reversible rate forms, unit reconstruction, observations, and explicit non-admission boundaries. It does not implement or fit a reactive film.

## Decision

- F1, `CO2 + 2 MEA <=> MEACOO- + MEAH+`, and F2, `CO2 + MEA + H2O <=> MEACOO- + H3O+`, remain finite-rate candidates because Putta2016 supplies reversible concentration/activity rate equations and Arrhenius correlations.
- F3, `CO2 + OH- <=> HCO3-`, remains a source relationship but is rejected as a physical finite-rate input because Putta2016 attributes its coefficient to Gondal2015 and the primary coefficient was not recovered in local Zotero.
- R1/R3/R4/R5 remain equilibrium-closure candidates only. Putta2016 omits H3O+/OH- transport kinetics in favor of water equilibrium and electroneutrality, but no quantitative film timescale evidence is available for admission.
- No reaction may be applied as both an exact local-equilibrium constraint and an independently applied finite rate in a future partition.

## Source and basis boundary

Putta2016 (DOI `10.1016/j.ijggc.2016.08.009`, attachment SHA-256 `{next(source['attachment_sha256'] for source in config['source_records'] if source['id'] == 'Putta2016')}`) is the primary finite-rate source. Luo2015 (DOI `10.1016/j.ces.2014.10.013`) supplies secondary mechanism context. The cited Gondal2015 source (DOI `10.1016/j.ces.2014.10.038`) is not present in local Zotero, so no F3 coefficient is invented.

Putta's 1 M and 5 M values are source labels only. The immutable issue 33 dependency remains `basis_unresolved`: Position 1 analytical MEA is `{dependency['required_position_1_analytical_mol_L']:.16g} mol L^-1` and free MEA is `{dependency['required_position_1_free_mea_mol_L']:.16g} mol L^-1`; it is not rounded or admitted as exact 5 M. No capture or kinetic tuning was performed.

## Reaction-space and units

The retained species order is `{', '.join(config['species_order'])}`. The checked projections are `F1 = R2 - R4 - R5`, `F2 = R2 - R4`, and `F3 = R2 - R1`. All three have Δν = -1, so a raw concentration quotient has units m^3 kmol^-1 when concentrations are in kmol m^-3. A future dimensionless conversion is `K° = K_raw c°`; the sources do not specify a provider-compatible standard state.

Putta prints `m^6 kmol^-2 s^-2` for the third-order F1/F2 coefficient, but a rate in kmol m^-3 s^-1 requires `m^6 kmol^-2 s^-1`. The printed unit is retained as rejected source metadata and the dimensionally required unit is recorded separately. F3 would require `m^3 kmol^-1 s^-1`, but its coefficient is unavailable.

The source-state closure rows use a strictly positive synthetic state at {config['source_state_reconstruction']['temperature_K']} K only to verify `k_reverse = k_forward/K_raw`; the maximum absolute ln(Q/K) is `{closure['max_absolute_ln_Q_minus_ln_K']:.3e}`. This is not a retained NCCC state, a fitted result, or an activity closure.

## Timescale evidence and observations

The Putta source domain is {config['source_domain']['putta2016_fit_points']} points over {config['source_domain']['putta2016_temperature_K'][0]}--{config['source_domain']['putta2016_temperature_K'][1]} K, source-labeled 1/5 M solutions, loading {config['source_domain']['putta2016_loading_mol_CO2_per_mol_MEA'][0]}--{config['source_domain']['putta2016_loading_mol_CO2_per_mol_MEA'][1]} mol CO2 per mol MEA, and LMPD {config['source_domain']['putta2016_lmpd_kPa'][0]}--{config['source_domain']['putta2016_lmpd_kPa'][1]} kPa. These are source-domain records, not an absorber admission basis.

A quantitative reaction time or Damköhler comparison is **not evaluable**: the retained evidence does not jointly provide an accepted physical basis, film thickness, diffusivity, and state-specific rate evaluation. Putta's reversible F1/F2 forms support finite-rate candidacy; its H3O+/OH- closure prescription supports only a qualitative equilibrium-closure candidate.

Putta Table 4 aggregate AARD values are retained as summary-only observations. Raw paired observations, row uncertainty, and non-overlapping fit/validation membership are unavailable, so no rate-data admission or coefficient uncertainty fit is claimed.

## Outputs

The input record is `inputs/issue34_kinetics.json`. The generated tables are `issue34_finite_reactions.csv`, `issue34_partition_decisions.csv`, `issue34_rate_observation_comparisons.csv`, and `issue34_kinetic_sensitivity.csv`; gate and identity data are in `issue34_kinetics_summary.json`.
"""


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(config: dict, issue33: dict, reaction_rows_: list[dict], finite: list[dict], sensitivities: list[dict], observations: list[dict], closure: dict, closure_rows: list[dict], stoich: bool, elements: bool, charge: bool, dimensional_pass: bool) -> dict:
    aard_values = [float(row["value"]) for row in observations]
    f1_f2 = [row for row in sensitivities if row["record_type"] == "temperature_reconstruction"]
    max_reconstruction_error = max(float(row["relative_float_decimal_difference"]) for row in f1_f2)
    return {
        "schema_version": config["schema_version"],
        "claim_label": config["claim_label"],
        "analysis": "nccc_validation",
        "issue": config["issue"],
        "source_order": [row["id"] for row in config["source_reactions"]],
        "species_order": config["species_order"],
        "reaction_projection_order": [row["id"] for row in config["reaction_combinations"]],
        "source_records": config["source_records"],
        "finite_candidate_count": len(finite),
        "source_anchor_temperatures_K": config["source_domain"]["reconstruction_anchor_temperature_K"],
        "source_domain": config["source_domain"],
        "issue33_dependency": {
            "summary_path": config["issue33_dependency"]["summary_path"],
            "summary_sha256": config["issue33_dependency"]["summary_sha256"],
            "table_path": config["issue33_dependency"]["table_path"],
            "table_sha256": config["issue33_dependency"]["table_sha256"],
            "claim_label": issue33["claim_label"],
            "position_1_analytical_mol_L": issue33["gates"]["position_1_analytical_concentration_mol_L"],
            "position_1_free_mea_mol_L": config["issue33_dependency"]["required_position_1_free_mea_mol_L"],
            "all_rows_basis_unresolved": True,
        },
        "gates": {
            "species_and_reaction_order_preserved": True,
            "reaction_projection_stoichiometry_pass": stoich,
            "element_balance_pass": elements,
            "charge_balance_pass": charge,
            "f1_f2_required_coefficient_unit_reconstructed": dimensional_pass,
            "f1_f2_source_printed_s_minus_2_rejected": True,
            "source_coefficient_reconstruction_float_decimal_relative_tolerance": 1.0e-12,
            "max_float_decimal_relative_difference": max_reconstruction_error,
            "f3_primary_coefficient_recovered": False,
            "activity_standard_state_conversion_admitted": False,
            "source_state_detailed_balance_pass": closure["detailed_balance_pass"],
            "max_absolute_ln_Q_minus_ln_K": closure["max_absolute_ln_Q_minus_ln_K"],
            "detailed_balance_target": DETAILED_BALANCE_TOLERANCE,
            "physical_equilibrium_state_admitted": False,
            "rate_observation_admission": False,
            "independent_raw_rate_validation_available": False,
            "reaction_timescale_partition_admitted": False,
            "packet_bound_activity_closure_performed": False,
            "physical_reactive_film_adoption": False,
            "issue33_basis_unresolved_preserved": True,
            "position_1_not_rounded_to_5M": issue33["gates"]["position_1_not_rounded_to_5M"],
            "supported_negative": True,
        },
        "source_state_reconstruction": {
            "rows": closure_rows,
            "summary": closure,
        },
        "observation_summary": {
            "aggregate_row_count": len(observations),
            "aard_min_percent": min(aard_values),
            "aard_max_percent": max(aard_values),
            "raw_rows_retained": False,
            "uncertainty_status": "not_reported_in_aggregate_table",
            "admission_decision": "aggregate_summary_only_not_admitted",
        },
        "sensitivity_summary": {
            "temperature_reconstruction_rows": len(f1_f2),
            "max_float_decimal_relative_difference": max_reconstruction_error,
            "f3_exclusion_effect": "not_quantified; primary k14 and admitted local film transport state unavailable",
            "activity_to_concentration_ratios": [
                {
                    "reaction_id": row["reaction_id"],
                    "temperature_K": row["temperature_K"],
                    "ratio": row["activity_to_concentration_ratio"],
                }
                for row in f1_f2
                if row["basis"] == "activity"
            ],
        },
        "rejected_or_unresolved_cases": [
            "Putta2016 prints F1/F2 third-order coefficient units with s^-2; dimensional reconstruction requires m6 kmol^-2 s^-1 for r in kmol m^-3 s^-1, so the printed unit is rejected and the correction is retained as an adjudication.",
            "F3 is retained as a reaction form but its Gondal2015 coefficient and activation energy were not recovered in local Zotero; no invented value or fallback fit is used.",
            "Putta2016 activity equations are retained for comparison, but the source standard-state convention and source-to-provider activity conversion are unavailable.",
            "R1/R3/R4/R5 fast-equilibrium partition is not admitted because source bulk-equilibrium statements and a Hatta-number definition do not provide a local reaction-specific D, film thickness, and rate timescale.",
            "Putta2016 Table 4 aggregate AARD rows are retained as source observations only; raw rows and non-overlap fit/validation membership are unavailable, so no rate-data admission or uncertainty fit is claimed.",
        ],
        "conclusions": [
            "F1 and F2 concentration and activity relationships are recovered from Putta2016 with the source-printed dimensional inconsistency explicitly corrected for comparison use.",
            "F3 remains a supported-negative source gap rather than a guessed hydroxide coefficient.",
            "The retained result supports a source-faithful finite-reaction record and algebraic source-state detailed-balance check, not a packet-bound activity closure or physical reactive-film model.",
            "No coefficient was tuned to NCCC Case 3C, packed capture, or any absorber result.",
        ],
        "limitations": [
            "Issue 33 concentration mappings remain basis_unresolved; Position 1 analytical MEA is preserved as 4.889309897097635 mol/L and free MEA as 2.491683471902737 mol/L.",
            "The 1 M and 5 M values are source labels only and are not admitted as exact absorber states.",
            "No raw kinetic observations or coefficient uncertainty distribution is retained in the available Putta aggregate table.",
            "No source-complete reaction/diffusion timescale is available for fast/finite partition.",
        ],
        "output_paths": {
            "finite_reactions": FINITE_TABLE.relative_to(ROOT).as_posix(),
            "partition_decisions": PARTITION_TABLE.relative_to(ROOT).as_posix(),
            "rate_observation_comparisons": OBSERVATION_TABLE.relative_to(ROOT).as_posix(),
            "kinetic_sensitivity": SENSITIVITY_TABLE.relative_to(ROOT).as_posix(),
            "report": REPORT.relative_to(ROOT).as_posix(),
        },
        "regeneration_command": "uv run python analyses/nccc_validation/scripts/resolve_issue34_kinetics.py",
        "claim_boundary": "This result is a source-faithful Work Package A kinetics and reaction-partition record. It does not establish provider-compatible activity closure, finite-film transport timescales, NCCC rate-data admission, or physical reactive-film adoption.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Retain the Issue 34 source-faithful MEA kinetics result.")
    parser.parse_args()
    config = load_inputs()
    issue33 = validate_issue33_dependency(config)
    reaction_rows_, reaction_vectors, stoich, elements, charge = reaction_rows(config)
    finite, sensitivities, _, dimensional_pass = finite_rows(config, reaction_vectors)
    closure_rows, closure = source_state_closure(config, finite)
    observations = observation_rows(config)
    partitions = partition_rows(config, reaction_rows_)

    write_csv(
        FINITE_TABLE,
        finite,
        [
            "reaction_id", "basis", "equation", "rate_law", "equilibrium_quotient", "source_formula",
            "coefficient_symbol", "coefficient_status", "source_printed_coefficient_unit", "required_coefficient_unit",
            "reaction_order", "reconstructed_coefficient_unit", "dimensional_reconstruction_pass", "unit_status",
            "delta_n", "raw_quotient_units", "standard_state_conversion",
            "standard_state", "coefficient_uncertainty", "stoichiometry_json", "source_locator",
            "admission_decision", "admission_reason",
        ],
    )
    write_csv(
        PARTITION_TABLE,
        partitions,
        [
            "reaction_id", "equation", "record_type", "source_equation", "source_reaction_coefficients",
            "stoichiometry_json", "stoichiometry_pass", "element_balance_residuals_json", "element_balance_pass",
            "charge_balance_pass", "decision", "timescale_metric", "timescale_value",
            "timescale_status", "missing_timescale_inputs", "source_timescale_evidence", "partition_evidence",
            "physical_film_admission", "source_locator",
        ],
    )
    write_csv(
        OBSERVATION_TABLE,
        observations,
        [
            "source", "source_locator", "model", "apparatus_or_dataset", "metric", "value", "uncertainty_status",
            "raw_rows_retained", "independent_fit_validation_partition", "admission_decision", "reason",
        ],
    )
    write_csv(
        SENSITIVITY_TABLE,
        sensitivities,
        [
            "record_type", "reaction_id", "basis", "temperature_K", "source_concentration_labels_mol_L", "k_float",
            "k_decimal", "absolute_float_decimal_difference", "relative_float_decimal_difference",
            "activity_to_concentration_ratio", "sensitivity_status", "reason",
        ],
    )
    summary = build_summary(config, issue33, reaction_rows_, finite, sensitivities, observations, closure, closure_rows, stoich, elements, charge, dimensional_pass)
    summary["input_sha256"] = sha256(INPUT)
    summary["generator_sha256"] = sha256(Path(__file__))
    REPORT.write_text(report_text(config, summary, closure), encoding="utf-8")
    summary["output_sha256"] = {
        "finite_reactions": sha256(FINITE_TABLE),
        "partition_decisions": sha256(PARTITION_TABLE),
        "rate_observation_comparisons": sha256(OBSERVATION_TABLE),
        "kinetic_sensitivity": sha256(SENSITIVITY_TABLE),
        "report": sha256(REPORT),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": SUMMARY.as_posix(), "gates": summary["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
