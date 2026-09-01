from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses/nccc_validation"
INPUT = ANALYSIS / "inputs/issue35_transport.json"
TABLES = ANALYSIS / "results/final/tables"
CORRELATION_TABLE = TABLES / "issue35_transport_correlations.csv"
SENSITIVITY_TABLE = TABLES / "issue35_transport_sensitivity.csv"
SUMMARY = TABLES / "issue35_transport_summary.json"
REPORT = ANALYSIS / "results/final/reports/issue35_transport_inputs.md"
ISSUE33_SUMMARY = ROOT / "analyses/nccc_validation/results/final/tables/issue33_concentration_basis_summary.json"
ISSUE33_TABLE = ROOT / "analyses/nccc_validation/results/final/tables/issue33_concentration_basis.csv"

CORRELATION_FIELDS = [
    "record_type", "record_id", "property", "source", "source_locator", "equation",
    "temperature_K", "concentration_mol_m3", "mass_fraction_percent", "loading_mol_CO2_per_mol_MEA",
    "value", "value_unit", "reference_value", "reference_unit", "relative_residual",
    "source_domain_status", "common_issue_domain_status", "uncertainty_status",
    "admission_decision", "reason",
]
SENSITIVITY_FIELDS = [
    "comparison_id", "candidate_A_status", "candidate_B_status", "candidate_A_co2_flux_mol_m2_s",
    "candidate_B_co2_flux_mol_m2_s", "charge_residual", "current_residual", "transfer_direction",
    "decision", "reason",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def in_domain(value: float | None, bounds: list[float] | None) -> str:
    if value is None or bounds is None:
        return "not_evaluable"
    return "pass" if bounds[0] <= value <= bounds[1] else "outside_source_domain"


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_config() -> dict:
    config = json.loads(INPUT.read_text(encoding="utf-8"))
    require(config["schema_version"] == "issue35_source_faithful_transport_v1", "unexpected Issue 35 input schema")
    require(config["candidate_A"]["id"] == "A_equal_ion_effective_fick", "Candidate A identity changed")
    require(config["candidate_B"]["id"] == "B_unequal_ion_zero_current", "Candidate B identity changed")
    require(len(config["snijder_table_iii"]) == 16, "Snijder Table III row count changed")
    return config


def validate_dependencies(config: dict) -> dict:
    issue33 = config["issue33_dependency"]
    require(sha256(ISSUE33_SUMMARY) == issue33["summary_sha256"], "Issue 33 summary hash changed")
    require(sha256(ISSUE33_TABLE) == issue33["table_sha256"], "Issue 33 table hash changed")
    issue33_summary = json.loads(ISSUE33_SUMMARY.read_text(encoding="utf-8"))
    require(issue33_summary["claim_label"] == issue33["required_claim_label"], "Issue 33 claim label changed")
    gates = issue33_summary["gates"]
    require(gates["position_1_exact_value_preserved"], "Issue 33 Position 1 value is not preserved")
    require(gates["position_1_not_rounded_to_5M"], "Issue 33 Position 1 was rounded")
    require(not gates["exact_discrete_1M_or_5M_admission"], "Issue 33 admitted an exact 1 M or 5 M row")
    require(not gates["capture_or_kinetic_tuning_performed"], "Issue 33 performed forbidden tuning")
    require(abs(float(gates["position_1_analytical_concentration_mol_L"]) - issue33["required_position_1_analytical_mol_L"]) <= 1e-14, "Issue 33 analytical concentration changed")
    with ISSUE33_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows and all(row["admission_decision"] == issue33["required_row_admission_decision"] for row in rows), "Issue 33 contains an admitted row")
    position_1 = next(row for row in rows if row["record_id"] == "Case3C_position_1")
    require(abs(float(position_1["free_MEA_concentration_mol_L"]) - issue33["required_position_1_free_mea_mol_L"]) <= 1e-14, "Issue 33 free MEA value changed")

    issue34 = config["issue34_dependency"]
    issue34_report = ROOT / issue34["report_path"]
    require(sha256(issue34_report) == issue34["report_sha256"], "Issue 34 report hash changed")
    require(issue34["required_claim_label"] in issue34_report.read_text(encoding="utf-8") or "supported-negative" in issue34_report.read_text(encoding="utf-8"), "Issue 34 supported-negative result is missing")
    return issue33_summary


def validate_sources(config: dict) -> None:
    require(len(config["source_records"]) == 10, "source record count changed")
    for source in config["source_records"]:
        require(source["id"] and source["evidence_status"] and source["locators"], f"incomplete source record: {source.get('id')}")
        if source["evidence_status"] == "local_zotero_pdf_inspected":
            require(source["doi"] and source["zotero_parent_key"] and source["zotero_attachment_key"] and source["attachment_sha256_inspection_receipt"], f"incomplete local Zotero receipt: {source['id']}")


def water_viscosity(T_K: float, params: dict) -> float:
    t_C = T_K - 273.15
    exponent = (1.3272 * (20.0 - t_C) - 0.001053 * (t_C - 20.0) ** 2) / (t_C + 105.0)
    return params["eta_water_at_20C_mPa_s"] * 1e-3 * 10.0**exponent


def weiland_viscosity(T_K: float, mass_percent: float, loading: float, correlation: dict) -> float:
    p = correlation["parameters"]
    eta_water = water_viscosity(T_K, p)
    exponent = (((p["a"] * mass_percent + p["b"]) * T_K + (p["c"] * mass_percent + p["d"])) * (loading * (p["e"] * mass_percent + p["f"] * T_K + p["g"]) + 1.0) * mass_percent) / T_K**2
    return eta_water * math.exp(exponent)


def row(**values: object) -> dict:
    return {field: values.get(field, "") for field in CORRELATION_FIELDS}


def build_rows(config: dict) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    domain = config["common_domain"]
    co2_water = config["correlations"]["co2_water"]
    modified_stokes = config["correlations"]["co2_amine_modified_stokes"]
    mea_free = config["correlations"]["mea_free"]
    weiland = config["correlations"]["weiland_viscosity"]
    water_params = weiland["parameters"]

    for T_K in (298.15, 313.15, 323.15):
        value = 2.35e-6 * math.exp(-2119.0 / T_K)
        rows.append(row(record_type="correlation_evaluation", record_id=co2_water["id"], property="CO2 diffusivity in water", source=co2_water["source"], source_locator=co2_water["source_locator"], equation=co2_water["equation"], temperature_K=T_K, value=value, value_unit=co2_water["output_unit"], source_domain_status=in_domain(T_K, co2_water["temperature_domain_K"]), common_issue_domain_status=in_domain(T_K, domain["temperature_K"]), uncertainty_status=co2_water["uncertainty"], admission_decision="source_reconstruction_only_not_physical_admission", reason="Positive source correlation evaluated at source-reported temperature anchors; no absorber state or flux is inferred."))

        for loading in (0.0, 0.25, 0.5):
            mu_water = water_viscosity(T_K, water_params)
            mu_amine = weiland_viscosity(T_K, 30.0, loading, weiland)
            D_CO2 = value * (mu_water / mu_amine) ** 0.8
            source_status = "pass" if in_domain(T_K, modified_stokes["temperature_domain_K"]) == "pass" and 20.0 <= 30.0 <= 40.0 and 0.0 <= loading <= 0.5 else "outside_source_domain"
            rows.append(row(record_type="correlation_evaluation", record_id=weiland["id"], property="MEA-water viscosity", source="Amundsen2009", source_locator=weiland["source_locator"], equation=weiland["equation"], temperature_K=T_K, mass_fraction_percent=30.0, loading_mol_CO2_per_mol_MEA=loading, value=mu_amine, value_unit=weiland["output_unit"], source_domain_status=source_status, common_issue_domain_status=in_domain(T_K, domain["temperature_K"]), uncertainty_status=weiland["uncertainty"], admission_decision="source_reconstruction_only_not_physical_admission", reason="Weiland viscosity is retained as the source input to Eq. 22; it is not a calibrated absorber viscosity."))
            rows.append(row(record_type="correlation_evaluation", record_id=modified_stokes["id"], property="CO2 diffusivity in amine solution", source=modified_stokes["source"], source_locator=modified_stokes["source_locator"], equation=modified_stokes["equation"], temperature_K=T_K, mass_fraction_percent=30.0, loading_mol_CO2_per_mol_MEA=loading, value=D_CO2, value_unit=modified_stokes["output_unit"], source_domain_status=source_status, common_issue_domain_status=in_domain(T_K, domain["temperature_K"]), uncertainty_status=modified_stokes["uncertainty"], admission_decision="source_reconstruction_only_not_physical_admission", reason="Modified Stokes-Einstein value is source-labeled and requires a resolved composition basis before any film use."))

    snijder_residuals = []
    for index, observation in enumerate(config["snijder_table_iii"], start=1):
        T_K = observation["temperature_K"]
        concentration = observation["concentration_mol_m3"]
        calculated = math.exp(-13.275 - 2198.3 / T_K - 7.8142e-5 * concentration)
        reported = observation["diffusivity_1e9_m2_s"] * 1e-9
        residual = abs(calculated - reported) / reported
        snijder_residuals.append(residual)
        source_status = "pass" if in_domain(T_K, mea_free["temperature_domain_K"]) == "pass" and in_domain(concentration, mea_free["concentration_domain_mol_m3"]) == "pass" else "outside_source_domain"
        common_status = in_domain(T_K, domain["temperature_K"])
        rows.append(row(record_type="source_table_reproduction", record_id=f"snijder_table_iii_{index:02d}", property="free MEA diffusivity", source="Snijder1993", source_locator="Snijder1993 PDF p. 3 (printed p. 477), Table III and Eq. 8", equation=mea_free["equation"], temperature_K=T_K, concentration_mol_m3=concentration, value=calculated, value_unit="m2/s", reference_value=reported, reference_unit="m2/s", relative_residual=residual, source_domain_status=source_status, common_issue_domain_status=common_status, uncertainty_status=mea_free["uncertainty"], admission_decision="source_reconstruction_only_not_physical_admission", reason="Table III reports D x 1e9 in m2/s; the displayed values are rounded and are not an independent absorber-state validation."))
        rows.append(row(record_type="source_observation", record_id=f"snijder_table_iii_{index:02d}", property="solution density", source="Snijder1993", source_locator="Snijder1993 PDF p. 3 (printed p. 477), Table III", equation="", temperature_K=T_K, concentration_mol_m3=concentration, value=observation["density_kg_m3"], value_unit="kg/m3", source_domain_status=source_status, common_issue_domain_status=common_status, uncertainty_status="not reported at retained Table III locator", admission_decision="required_input_retained_not_admitted", reason="Mass density is retained for provenance; conversion to total molar density is blocked by the unresolved species/composition basis."))
        rows.append(row(record_type="source_observation", record_id=f"snijder_table_iii_{index:02d}", property="solution viscosity", source="Snijder1993", source_locator="Snijder1993 PDF p. 3 (printed p. 477), Table III", equation="", temperature_K=T_K, concentration_mol_m3=concentration, value=observation["viscosity_mPa_s"] * 1e-3, value_unit="Pa s", source_domain_status=source_status, common_issue_domain_status=common_status, uncertainty_status="not reported at retained Table III locator", admission_decision="required_input_retained_not_admitted", reason="Viscosity is retained as a source observation for transport reconstruction only."))

    n2o = config["correlations"]["n2o_analogy"]
    rows.append(row(record_type="missing_source_input", record_id=n2o["id"], property="CO2 diffusivity by N2O analogy", source=n2o["source"], source_locator=n2o["source_locator"], equation=n2o["equation"], value_unit="m2/s", source_domain_status="not_evaluable_missing_inputs", common_issue_domain_status="not_evaluable", uncertainty_status=n2o["uncertainty"], admission_decision="supported_negative_missing_source_complete_primary_N2O_chain", reason="The cited Ko, Jamal, and Ying/Eimer primary N2O inputs were not recovered; no N2O coefficient or uncertainty is invented."))
    ion = config["correlations"]["legacy_ion_scalar"]
    rows.append(row(record_type="rejected_legacy_input", record_id=ion["id"], property="ion diffusivity", source="", source_locator=ion["source_locator"], equation=ion["equation"], value_unit="m2/s", source_domain_status="rejected_unattributed", common_issue_domain_status="rejected_unattributed", uncertainty_status=ion["uncertainty"], admission_decision=ion["status"], reason="The repository scalar has no source, species mapping, or uncertainty and is not retained as a transport default."))
    rows.append(row(record_type="closure_requirement", record_id=config["candidate_B"]["id"], property="unequal-ion mobility and electromigration closure", source="", source_locator="Issue 35 candidate B definition", equation=config["candidate_B"]["flux_equation"], value_unit="", source_domain_status="not_evaluable_missing_complete_mobility_law", common_issue_domain_status="not_evaluable", uncertainty_status="not specified", admission_decision=config["candidate_B"]["status"], reason="Gamma, a complete cited mobility law, source-complete ion diffusivities, and an accepted true-species state are missing."))
    return rows, {"snijder_max_relative_residual": max(snijder_residuals), "snijder_row_count": len(snijder_residuals), "positive_evaluated_values": all(float(item["value"]) > 0.0 for item in rows if item["value"] != "")}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_sensitivity(config: dict) -> list[dict]:
    return [{
        "comparison_id": "identical_state_transport",
        "candidate_A_status": config["candidate_A"]["status"],
        "candidate_B_status": config["candidate_B"]["status"],
        "candidate_A_co2_flux_mol_m2_s": "",
        "candidate_B_co2_flux_mol_m2_s": "",
        "charge_residual": "",
        "current_residual": "",
        "transfer_direction": "",
        "decision": "not_evaluable_blocked",
        "reason": "No identical accepted state, admitted kinetics, complete ion inputs, or complete Candidate B mobility law is available; no flux comparison is performed.",
    }]


def report_text(config: dict, issue33: dict, diagnostics: dict) -> str:
    return f"""# Issue 35: source-faithful transport inputs and unequal-ion closure

Status: **supported-negative source-faithful transport record complete**. The retained result reconstructs positive source correlations and preserves the missing-input decision. It does not adopt a physical transport closure or compare film fluxes.

## Decision

Candidate A, the equal-ion effective-Fick reduction, is retained as a reduced-form requirement only. A physical admission would require a source-complete effective diffusivity for every retained species or an explicit source-defined ion lump, plus an accepted local concentration basis.

Candidate B, the unequal-ion zero-current form, is not executable. Its minimum requirements are the true-species state, ePC-SAFT Gamma, a complete cited mobility law, source-complete unequal ion diffusivities, molar density, electroneutrality, zero current, and the potential gauge. The retained sensitivity table therefore has no flux values, charge residual, current residual, or transfer direction.

## Source reconstruction

Luo2015 Eq. 21 retains `D_CO2_water = 2.35e-6 exp(-2119/T)` in `m2/s`. Luo2015 Eq. 22 retains the modified Stokes-Einstein form `D_CO2_amine = D_CO2_water (mu_water/mu_amine)^0.8`; Amundsen2009 Weiland Eqs. 9--10 and Table 12 supply the source-labeled viscosity relationship for the evaluated 30 mass% and 0--0.5 loading rows. These values are source reconstructions, not absorber inputs.

Snijder1993 Eq. 8 retains `ln(D_MEA) = -13.275 - 2198.3/T - 7.8142e-5 C`, with `C` in `mol/m3` and `D` in `m2/s`. The 16 Table III observations retain density and viscosity in `kg/m3` and `mPa s`; the density is not converted to molar density while Issue 33 remains basis_unresolved. The maximum relative residual against the displayed, rounded Table III diffusivities is `{diagnostics['snijder_max_relative_residual']:.6f}`. Snijder's source statement that the fit is within 5% is preserved as source metadata; the rounded table alone does not reproduce that statement at every displayed row.

Putta2017 Eq. 12 is retained as a blocked N2O analogy because the cited Ko, Jamal, and Ying/Eimer primary N2O-water and N2O-amine inputs were not recovered with source-complete coefficients and uncertainty. The legacy scalar ion expression in `src/mea_absorption_column/Properties/Transport_Properties.py` is rejected as unattributed and is not retained as a default.

## Dependency and claim boundary

Issue 33 remains `basis_unresolved`: Position 1 analytical MEA is `{issue33['gates']['position_1_analytical_concentration_mol_L']:.16g} mol/L` and free MEA is `{config['issue33_dependency']['required_position_1_free_mea_mol_L']:.16g} mol/L`; neither is rounded to exact 5 M. Issue 34 remains the merged supported-negative kinetics record. No rate comparison, Case 3C tuning, physical film result, or production transport-adoption change is made here.

The input record is `inputs/issue35_transport.json`. Generated tables are `issue35_transport_correlations.csv`, `issue35_transport_sensitivity.csv`, and `issue35_transport_summary.json`.
"""


def main() -> int:
    config = load_config()
    issue33 = validate_dependencies(config)
    validate_sources(config)
    rows, diagnostics = build_rows(config)
    sensitivity = build_sensitivity(config)
    require(diagnostics["positive_evaluated_values"], "known evaluated transport values are not all positive")
    require(diagnostics["snijder_row_count"] == 16, "Snijder reproduction row count changed")
    require(not any(item["candidate_A_co2_flux_mol_m2_s"] or item["candidate_B_co2_flux_mol_m2_s"] for item in sensitivity), "forbidden physical flux comparison was added")

    write_csv(CORRELATION_TABLE, rows, CORRELATION_FIELDS)
    write_csv(SENSITIVITY_TABLE, sensitivity, SENSITIVITY_FIELDS)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(config, issue33, diagnostics), encoding="utf-8")
    summary = {
        "schema_version": config["schema_version"],
        "claim_label": config["claim_label"],
        "analysis": "nccc_validation",
        "issue": config["issue"],
        "source_records": config["source_records"],
        "issue33_dependency": {"summary_path": config["issue33_dependency"]["summary_path"], "summary_sha256": config["issue33_dependency"]["summary_sha256"], "table_path": config["issue33_dependency"]["table_path"], "table_sha256": config["issue33_dependency"]["table_sha256"], "claim_label": issue33["claim_label"], "position_1_analytical_mol_L": issue33["gates"]["position_1_analytical_concentration_mol_L"], "position_1_free_mea_mol_L": config["issue33_dependency"]["required_position_1_free_mea_mol_L"], "all_rows_basis_unresolved": True},
        "issue34_dependency": config["issue34_dependency"],
        "diagnostics": diagnostics,
        "gates": {
            "source_pdf_inspection_receipts_present_for_local_sources": True,
            "known_correlation_positivity_pass": diagnostics["positive_evaluated_values"],
            "snijder_arithmetic_reproduction_status": "rounded_table_residual_exceeds_source_stated_5_percent_at_one_or_more_rows",
            "snijder_source_fit_statement_preserved": True,
            "n2o_chain_complete": False,
            "ion_diffusivity_input_complete": False,
            "candidate_A_all_species_effective_diffusivities_complete": False,
            "candidate_B_complete_mobility_law_and_gamma": False,
            "issue33_basis_unresolved_preserved": True,
            "issue34_supported_negative_preserved": True,
            "identical_state_flux_comparison_evaluated": False,
            "rate_comparison_performed": False,
            "physical_transport_adoption": False,
            "supported_negative": True,
        },
        "candidate_A": config["candidate_A"],
        "candidate_B": config["candidate_B"],
        "rejected_or_unresolved_cases": [
            "Putta2017 Eq. 12 cannot be evaluated without source-complete primary N2O-water and N2O-amine correlations.",
            "The legacy scalar ion expression has no source, species mapping, or uncertainty and is rejected.",
            "Snijder Table III density observations cannot be converted to total molar density while Issue 33 basis remains unresolved.",
            "Candidate B cannot be evaluated without Gamma, a complete mobility law, unequal ion diffusivities, and an accepted true-species state.",
            "No identical-state candidate flux comparison or physical transport sensitivity is performed.",
        ],
        "output_paths": {"transport_correlations": CORRELATION_TABLE.relative_to(ROOT).as_posix(), "transport_sensitivity": SENSITIVITY_TABLE.relative_to(ROOT).as_posix(), "summary": SUMMARY.relative_to(ROOT).as_posix(), "report": REPORT.relative_to(ROOT).as_posix()},
        "regeneration_command": "uv run python analyses/nccc_validation/scripts/resolve_issue35_transport.py",
        "claim_boundary": "This result is a source-faithful Work Package A transport-input record and supported-negative unequal-ion closure decision. It does not establish a physical reactive film, source-complete ionic transport, flux agreement, rate comparison, or packed-column capture.",
    }
    summary["input_sha256"] = sha256(INPUT)
    summary["generator_sha256"] = sha256(Path(__file__))
    summary["output_sha256"] = {"transport_correlations": sha256(CORRELATION_TABLE), "transport_sensitivity": sha256(SENSITIVITY_TABLE), "report": sha256(REPORT)}
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": SUMMARY.as_posix(), "gates": summary["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
