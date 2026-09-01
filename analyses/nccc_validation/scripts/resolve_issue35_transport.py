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
    "record_type", "record_id", "property", "source", "source_locator", "equation", "source_parameters_json",
    "temperature_K", "concentration_mol_m3", "mass_fraction_percent", "loading_mol_CO2_per_mol_MEA",
    "source_value", "source_value_unit", "value", "value_unit", "reference_value", "reference_unit",
    "absolute_residual", "absolute_residual_unit", "relative_residual", "source_domain_status",
    "common_issue_domain_status", "uncertainty_status", "admission_decision", "reason",
]
SENSITIVITY_FIELDS = [
    "comparison_id", "candidate_A_status", "candidate_B_status", "candidate_A_co2_flux_mol_m2_s",
    "candidate_B_co2_flux_mol_m2_s", "charge_residual", "current_residual", "transfer_direction",
    "decision", "reason",
]
EXPECTED_UNITS = {
    "co2_water": "m2/s",
    "co2_amine_modified_stokes": "m2/s",
    "mea_free": "m2/s",
    "n2o_analogy": "m2/s",
    "weiland_density": "kg/m3",
    "weiland_viscosity": "Pa s",
    "hartono_density": "kg/m3",
    "hartono_viscosity": "Pa s",
    "legacy_ion_scalar": "m2/s",
}
DOMAIN_KEYS = (
    "temperature_domain_K", "source_stated_temperature_domain_K", "concentration_domain_mol_m3", "mass_fraction_domain_percent",
    "loading_domain_mol_CO2_per_mol_MEA",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def in_domain(value: float | None, bounds: list[float] | None) -> str:
    if value is None or bounds is None:
        return "not_evaluable"
    return "pass" if bounds[0] <= value <= bounds[1] else "outside_source_domain"


def parameters_json(parameters: dict | None) -> str:
    return json.dumps(parameters or {}, sort_keys=True, separators=(",", ":"))


def load_config() -> dict:
    config = json.loads(INPUT.read_text(encoding="utf-8"))
    require(config["schema_version"] == "issue35_source_faithful_transport_v1", "unexpected Issue 35 input schema")
    require(config["candidate_A"]["id"] == "A_equal_ion_effective_fick", "Candidate A identity changed")
    require(config["candidate_B"]["id"] == "B_unequal_ion_zero_current", "Candidate B identity changed")
    require(len(config["snijder_table_iii"]) == 16, "Snijder Table III row count changed")
    return config


def validate_domain(name: str, bounds: object) -> None:
    require(isinstance(bounds, list) and len(bounds) == 2, f"{name} must be a two-value domain")
    require(all(finite_number(value) for value in bounds), f"{name} must contain finite numeric bounds")
    require(float(bounds[0]) < float(bounds[1]), f"{name} must be strictly ordered")


def validate_parameters(name: str, record: dict) -> None:
    parameters = record.get("parameters", {})
    require(isinstance(parameters, dict), f"{name} parameters must be an object")
    for key, value in parameters.items():
        require(finite_number(value), f"{name} parameter {key} is not finite numeric")
    positive_keys = {
        "pre_exponential_m2_s", "activation_temperature_K", "viscosity_exponent",
        "concentration_coefficient_m3_per_mol", "M1", "V3", "eta_water_at_20C_mPa_s",
        "water_temperature_reference_C", "water_linear_coefficient", "water_quadratic_coefficient",
        "water_denominator_offset_C",
    }
    for key in positive_keys.intersection(parameters):
        require(float(parameters[key]) > 0.0, f"{name} parameter {key} must be positive")


def validate_observation_block(name: str, block: dict, expected_unit: str) -> None:
    require(block["value_unit"] == expected_unit, f"{name} value unit changed")
    validate_domain(f"{name}.temperature_domain_C", block["temperature_domain_C"])
    validate_domain(f"{name}.loading_domain_mol_CO2_per_mol_MEA", block["loading_domain_mol_CO2_per_mol_MEA"])
    require(finite_number(block["mass_fraction_percent"]) and float(block["mass_fraction_percent"]) > 0.0, f"{name} mass fraction is invalid")
    for index, source_row in enumerate(block["rows"], start=1):
        require(finite_number(source_row["temperature_C"]), f"{name} row {index} temperature is invalid")
        require(in_domain(float(source_row["temperature_C"]), block["temperature_domain_C"]) == "pass", f"{name} row {index} temperature is outside its declared domain")
        loadings = source_row["loading"]
        values = source_row["values"]
        require(isinstance(loadings, list) and isinstance(values, list) and len(loadings) == len(values) and loadings, f"{name} row {index} shape is invalid")
        for loading, value in zip(loadings, values):
            require(finite_number(loading) and in_domain(float(loading), block["loading_domain_mol_CO2_per_mol_MEA"]) == "pass", f"{name} row {index} loading is invalid")
            require(finite_number(value) and float(value) > 0.0, f"{name} row {index} value must be positive")


def validate_config(config: dict) -> None:
    common = config["common_domain"]
    validate_domain("common_domain.temperature_K", common["temperature_K"])
    validate_domain("common_domain.loading_mol_CO2_per_mol_MEA", common["loading_mol_CO2_per_mol_MEA"])
    require(common["concentration_basis_status"] == "basis_unresolved_from_issue33", "Issue 33 basis status changed")
    require(common["exact_1M_or_5M_admission"] is False, "exact 1 M/5 M admission changed")

    for name, record in config["correlations"].items():
        require(record.get("output_unit") == EXPECTED_UNITS[name], f"{name} output unit is not exact")
        require(record.get("equation") and record.get("source_locator"), f"{name} equation or source locator is missing")
        validate_parameters(name, record)
        for domain_key in DOMAIN_KEYS:
            if domain_key in record:
                validate_domain(f"{name}.{domain_key}", record[domain_key])

    points = config["evaluation_points"]
    for key in (
        "co2_water_temperature_K", "modified_stokes_temperature_K", "modified_stokes_mass_fraction_percent",
        "modified_stokes_loading_mol_CO2_per_mol_MEA",
    ):
        require(isinstance(points[key], list) and points[key] and all(finite_number(value) for value in points[key]), f"{key} is invalid")
    co2_water = config["correlations"]["co2_water"]
    stokes = config["correlations"]["co2_amine_modified_stokes"]
    for value in points["co2_water_temperature_K"]:
        require(in_domain(float(value), co2_water["temperature_domain_K"]) == "pass", "CO2-water evaluation point is outside its declared domain")
    for value in points["modified_stokes_temperature_K"]:
        require(in_domain(float(value), stokes["temperature_domain_K"]) == "pass", "Stokes temperature evaluation point is outside its declared domain")
    for value in points["modified_stokes_mass_fraction_percent"]:
        require(in_domain(float(value), stokes["mass_fraction_domain_percent"]) == "pass", "Stokes mass-fraction evaluation point is outside its declared domain")
    for value in points["modified_stokes_loading_mol_CO2_per_mol_MEA"]:
        require(in_domain(float(value), stokes["loading_domain_mol_CO2_per_mol_MEA"]) == "pass", "Stokes loading evaluation point is outside its declared domain")

    for index, observation in enumerate(config["snijder_table_iii"], start=1):
        for key in ("temperature_K", "concentration_mol_m3", "diffusivity_1e9_m2_s", "density_kg_m3", "viscosity_mPa_s"):
            require(finite_number(observation[key]), f"Snijder row {index} field {key} is not finite numeric")
        require(observation["temperature_K"] > 0.0 and observation["concentration_mol_m3"] > 0.0, f"Snijder row {index} state is not positive")
        require(observation["diffusivity_1e9_m2_s"] > 0.0 and observation["density_kg_m3"] > 0.0 and observation["viscosity_mPa_s"] > 0.0, f"Snijder row {index} reference is not positive")
        require(in_domain(float(observation["temperature_K"]), config["correlations"]["mea_free"]["temperature_domain_K"]) == "pass", f"Snijder row {index} temperature is outside its declared domain")
        require(in_domain(float(observation["concentration_mol_m3"]), config["correlations"]["mea_free"]["concentration_domain_mol_m3"]) == "pass", f"Snijder row {index} concentration is outside its declared domain")

    validate_observation_block("amundsen_weiland_density", config["source_observations"]["amundsen_weiland_density"], "g/cm3")
    validate_observation_block("hartono_density", config["source_observations"]["hartono_density"], "kg/m3")
    validate_observation_block("hartono_viscosity", config["source_observations"]["hartono_viscosity"], "mPa s")


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
    report_text_34 = issue34_report.read_text(encoding="utf-8")
    require(sha256(issue34_report) == issue34["report_sha256"], "Issue 34 report hash changed")
    require("supported-negative" in report_text_34, "Issue 34 supported-negative result is missing")
    return issue33_summary


def validate_sources(config: dict) -> None:
    require(len(config["source_records"]) == 10, "source record count changed")
    for source in config["source_records"]:
        require(source["id"] and source["evidence_status"] and source["locators"], f"incomplete source record: {source.get('id')}")
        if source["evidence_status"] == "local_zotero_pdf_inspected":
            require(source["doi"] and source["zotero_parent_key"] and source["zotero_attachment_key"] and source["attachment_sha256_inspection_receipt"], f"incomplete local Zotero receipt: {source['id']}")


def water_viscosity(T_K: float, params: dict) -> float:
    t_C = T_K - 273.15
    reference_C = params["water_temperature_reference_C"]
    exponent = (params["water_linear_coefficient"] * (reference_C - t_C) - params["water_quadratic_coefficient"] * (t_C - reference_C) ** 2) / (t_C + params["water_denominator_offset_C"])
    return params["eta_water_at_20C_mPa_s"] * 1e-3 * 10.0**exponent


def weiland_viscosity(T_K: float, mass_percent: float, loading: float, correlation: dict) -> float:
    p = correlation["parameters"]
    eta_water = water_viscosity(T_K, p)
    exponent = (((p["a"] * mass_percent + p["b"]) * T_K + (p["c"] * mass_percent + p["d"])) * (loading * (p["e"] * mass_percent + p["f"] * T_K + p["g"]) + 1.0) * mass_percent) / T_K**2
    return eta_water * math.exp(exponent)


def row(**values: object) -> dict:
    return {field: values.get(field, "") for field in CORRELATION_FIELDS}


def source_status(T_K: float, mass_percent: float | None, loading: float | None, correlation: dict) -> str:
    checks = [in_domain(T_K, correlation.get("temperature_domain_K"))]
    if mass_percent is not None:
        checks.append(in_domain(mass_percent, correlation.get("mass_fraction_domain_percent")))
    if loading is not None:
        checks.append(in_domain(loading, correlation.get("loading_domain_mol_CO2_per_mol_MEA")))
    return "pass" if all(check == "pass" for check in checks) else "outside_source_domain"


def definition_rows(config: dict) -> list[dict]:
    rows = []
    for name in ("co2_water", "co2_amine_modified_stokes", "mea_free", "weiland_density", "weiland_viscosity", "hartono_density", "hartono_viscosity"):
        correlation = config["correlations"][name]
        source = correlation["source"]
        reason = "Source correlation definition retained; no absorber state is admitted."
        if name in ("weiland_density", "weiland_viscosity"):
            source = "Amundsen2009 (parameters reproduced from Weiland et al.)"
            reason = "Amundsen2009 reproduces the Weiland parameter set; the original Weiland publication is not the retained primary source here."
        rows.append(row(record_type="correlation_definition", record_id=correlation["id"], property=name, source=source, source_locator=correlation["source_locator"], equation=correlation["equation"], source_parameters_json=parameters_json(correlation.get("parameters")), value_unit=correlation["output_unit"], source_domain_status="declared_source_domain", common_issue_domain_status="not_evaluated", uncertainty_status=correlation["uncertainty"], admission_decision="source_definition_retained_not_admitted", reason=reason))
    return rows


def add_observation_rows(rows: list[dict], config: dict, block_name: str, correlation_name: str, property_name: str, output_unit: str, factor: float, reason: str) -> int:
    block = config["source_observations"][block_name]
    correlation = config["correlations"][correlation_name]
    count = 0
    for source_row in block["rows"]:
        T_K = float(source_row["temperature_C"]) + 273.15
        for loading, source_value in zip(source_row["loading"], source_row["values"]):
            loading = float(loading)
            source_value = float(source_value)
            rows.append(row(record_type="source_observation", record_id=f"{block_name}_T{source_row['temperature_C']:.2f}_a{loading:.2f}", property=property_name, source=block["source"], source_locator=block["source_locator"], equation=correlation["equation"], temperature_K=T_K, mass_fraction_percent=block["mass_fraction_percent"], loading_mol_CO2_per_mol_MEA=loading, source_value=source_value, source_value_unit=block["value_unit"], value=source_value * factor, value_unit=output_unit, source_domain_status="pass", common_issue_domain_status=in_domain(T_K, config["common_domain"]["temperature_K"]), uncertainty_status=block["uncertainty_status"], admission_decision="required_input_retained_not_admitted", reason=reason))
            count += 1
    return count


def build_rows(config: dict) -> tuple[list[dict], dict]:
    rows = definition_rows(config)
    domain = config["common_domain"]
    points = config["evaluation_points"]
    co2_water = config["correlations"]["co2_water"]
    modified_stokes = config["correlations"]["co2_amine_modified_stokes"]
    mea_free = config["correlations"]["mea_free"]
    weiland = config["correlations"]["weiland_viscosity"]
    water_parameters = co2_water["parameters"]
    stokes_parameters = modified_stokes["parameters"]
    mea_parameters = mea_free["parameters"]

    for T_K in points["co2_water_temperature_K"]:
        T_K = float(T_K)
        D_water = water_parameters["pre_exponential_m2_s"] * math.exp(-water_parameters["activation_temperature_K"] / T_K)
        rows.append(row(record_type="correlation_evaluation", record_id=co2_water["id"], property="CO2 diffusivity in water", source=co2_water["source"], source_locator=co2_water["source_locator"], equation=co2_water["equation"], source_parameters_json=parameters_json(water_parameters), temperature_K=T_K, value=D_water, value_unit=co2_water["output_unit"], source_domain_status=source_status(T_K, None, None, co2_water), common_issue_domain_status=in_domain(T_K, domain["temperature_K"]), uncertainty_status=co2_water["uncertainty"], admission_decision="source_reconstruction_only_not_physical_admission", reason="Positive source correlation evaluated at a configured source-domain anchor; no absorber state or flux is inferred."))

    for T_K in points["modified_stokes_temperature_K"]:
        T_K = float(T_K)
        D_water = water_parameters["pre_exponential_m2_s"] * math.exp(-water_parameters["activation_temperature_K"] / T_K)
        for mass_percent in points["modified_stokes_mass_fraction_percent"]:
            mass_percent = float(mass_percent)
            for loading in points["modified_stokes_loading_mol_CO2_per_mol_MEA"]:
                loading = float(loading)
                mu_water = water_viscosity(T_K, weiland["parameters"])
                mu_amine = weiland_viscosity(T_K, mass_percent, loading, weiland)
                D_CO2 = D_water * (mu_water / mu_amine) ** stokes_parameters["viscosity_exponent"]
                status = source_status(T_K, mass_percent, loading, modified_stokes)
                combined_parameters = {"co2_water": water_parameters, "modified_stokes": stokes_parameters, "weiland_viscosity": weiland["parameters"]}
                rows.append(row(record_type="correlation_evaluation", record_id=weiland["id"], property="MEA-water viscosity", source="Amundsen2009 (parameters reproduced from Weiland et al.)", source_locator=weiland["source_locator"], equation=weiland["equation"], source_parameters_json=parameters_json(weiland["parameters"]), temperature_K=T_K, mass_fraction_percent=mass_percent, loading_mol_CO2_per_mol_MEA=loading, value=mu_amine, value_unit=weiland["output_unit"], source_domain_status=status, common_issue_domain_status=in_domain(T_K, domain["temperature_K"]), uncertainty_status=weiland["uncertainty"], admission_decision="source_reconstruction_only_not_physical_admission", reason="Amundsen2009 reproduces the Weiland viscosity parameters; the value is source-labeled and is not an absorber calibration."))
                rows.append(row(record_type="correlation_evaluation", record_id=modified_stokes["id"], property="CO2 diffusivity in amine solution", source=modified_stokes["source"], source_locator=modified_stokes["source_locator"], equation=modified_stokes["equation"], source_parameters_json=parameters_json(combined_parameters), temperature_K=T_K, mass_fraction_percent=mass_percent, loading_mol_CO2_per_mol_MEA=loading, value=D_CO2, value_unit=modified_stokes["output_unit"], source_domain_status=status, common_issue_domain_status=in_domain(T_K, domain["temperature_K"]), uncertainty_status=modified_stokes["uncertainty"], admission_decision="source_reconstruction_only_not_physical_admission", reason="Modified Stokes-Einstein value is source-labeled and requires a resolved composition basis before any film use."))

    snijder_residuals = []
    snijder_absolute_residuals = []
    for index, observation in enumerate(config["snijder_table_iii"], start=1):
        T_K = float(observation["temperature_K"])
        concentration = float(observation["concentration_mol_m3"])
        calculated = math.exp(mea_parameters["log_intercept"] - mea_parameters["activation_temperature_K"] / T_K - mea_parameters["concentration_coefficient_m3_per_mol"] * concentration)
        reported = float(observation["diffusivity_1e9_m2_s"]) * 1e-9
        absolute_residual = abs(calculated - reported)
        relative_residual = absolute_residual / reported
        snijder_residuals.append(relative_residual)
        snijder_absolute_residuals.append(absolute_residual)
        source_status_value = source_status(T_K, None, None, mea_free)
        source_status_concentration = in_domain(concentration, mea_free["concentration_domain_mol_m3"])
        combined_status = "pass" if source_status_value == "pass" and source_status_concentration == "pass" else "outside_source_domain"
        rows.append(row(record_type="source_table_reproduction", record_id=f"snijder_table_iii_{index:02d}", property="free MEA diffusivity", source="Snijder1993", source_locator="Snijder1993 PDF p. 3 (printed p. 477), Table III and Eq. 8", equation=mea_free["equation"], source_parameters_json=parameters_json(mea_parameters), temperature_K=T_K, concentration_mol_m3=concentration, value=calculated, value_unit=mea_free["output_unit"], reference_value=reported, reference_unit=mea_free["output_unit"], absolute_residual=absolute_residual, absolute_residual_unit=mea_free["output_unit"], relative_residual=relative_residual, source_domain_status=combined_status, common_issue_domain_status=in_domain(T_K, domain["temperature_K"]), uncertainty_status=mea_free["uncertainty"], admission_decision="source_reconstruction_only_not_physical_admission", reason="Table III reports D x 1e9 in m2/s; the displayed values are rounded and are not an independent absorber-state validation."))
        rows.append(row(record_type="source_observation", record_id=f"snijder_table_iii_{index:02d}_density", property="solution density", source="Snijder1993", source_locator="Snijder1993 PDF p. 3 (printed p. 477), Table III", temperature_K=T_K, concentration_mol_m3=concentration, source_value=observation["density_kg_m3"], source_value_unit="kg/m3", value=observation["density_kg_m3"], value_unit="kg/m3", source_domain_status=combined_status, common_issue_domain_status=in_domain(T_K, domain["temperature_K"]), uncertainty_status="not reported at retained Table III locator", admission_decision="required_input_retained_not_admitted", reason="Mass density is retained for provenance; conversion to total molar density is blocked by the unresolved species/composition basis."))
        rows.append(row(record_type="source_observation", record_id=f"snijder_table_iii_{index:02d}_viscosity", property="solution viscosity", source="Snijder1993", source_locator="Snijder1993 PDF p. 3 (printed p. 477), Table III", temperature_K=T_K, concentration_mol_m3=concentration, source_value=observation["viscosity_mPa_s"], source_value_unit="mPa s", value=float(observation["viscosity_mPa_s"]) * 1e-3, value_unit="Pa s", source_domain_status=combined_status, common_issue_domain_status=in_domain(T_K, domain["temperature_K"]), uncertainty_status="not reported at retained Table III locator", admission_decision="required_input_retained_not_admitted", reason="Source viscosity is reported in mPa s and is emitted in this CSV after conversion to Pa s."))

    observation_counts = {
        "amundsen_weiland_density": add_observation_rows(rows, config, "amundsen_weiland_density", "weiland_density", "loaded 30 mass% MEA density", "g/cm3", 1.0, "Amundsen2009 density observation retained in its source units; no conversion to absorber molar density is made."),
        "hartono_density": add_observation_rows(rows, config, "hartono_density", "hartono_density", "loaded 30 mass% MEA density", "kg/m3", 1.0, "Hartono2014 density observation retained in its source units; no conversion to absorber molar density is made."),
        "hartono_viscosity": add_observation_rows(rows, config, "hartono_viscosity", "hartono_viscosity", "loaded 30 mass% MEA viscosity", "Pa s", 1e-3, "Hartono2014 source viscosity is reported in mPa s and is emitted in this CSV after conversion to Pa s."),
    }

    n2o = config["correlations"]["n2o_analogy"]
    rows.append(row(record_type="missing_source_input", record_id=n2o["id"], property="CO2 diffusivity by N2O analogy", source=n2o["source"], source_locator=n2o["source_locator"], equation=n2o["equation"], value_unit=n2o["output_unit"], source_domain_status="not_evaluable_missing_inputs", common_issue_domain_status="not_evaluable", uncertainty_status=n2o["uncertainty"], admission_decision="supported_negative_missing_source_complete_primary_N2O_chain", reason="The cited Ko, Jamal, and Ying/Eimer primary N2O inputs were not recovered; no N2O coefficient or uncertainty is invented."))
    ion = config["correlations"]["legacy_ion_scalar"]
    rows.append(row(record_type="rejected_legacy_input", record_id=ion["id"], property="ion diffusivity", source="", source_locator=ion["source_locator"], equation=ion["equation"], value_unit=ion["output_unit"], source_domain_status="rejected_unattributed", common_issue_domain_status="rejected_unattributed", uncertainty_status=ion["uncertainty"], admission_decision=ion["status"], reason="The repository scalar has no source, species mapping, or uncertainty and is not retained as a transport default."))
    rows.append(row(record_type="closure_requirement", record_id=config["candidate_B"]["id"], property="unequal-ion mobility and electromigration closure", source="", source_locator="Issue 35 candidate B definition", equation=config["candidate_B"]["flux_equation"], value_unit="", source_domain_status="not_evaluable_missing_complete_mobility_law", common_issue_domain_status="not_evaluable", uncertainty_status="not specified", admission_decision=config["candidate_B"]["status"], reason="Gamma, a complete cited mobility law, source-complete ion diffusivities, and an accepted true-species state are missing."))
    diagnostics = {
        "snijder_max_relative_residual": max(snijder_residuals),
        "snijder_max_absolute_residual_m2_s": max(snijder_absolute_residuals),
        "snijder_row_count": len(snijder_residuals),
        "positive_evaluated_values": all(float(item["value"]) > 0.0 for item in rows if item["value"] != ""),
        "source_observation_counts": observation_counts,
    }
    return rows, diagnostics


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

## Source authority and reconstruction

`inputs/issue35_transport.json` is the authoritative coefficient record. Every evaluated correlation row carries its numeric source parameters in `source_parameters_json`; no coefficient is duplicated independently in the resolver. Luo2015 Eq. 21 retains `D_CO2_water = 2.35e-6 exp(-2119/T)` in `m2/s`. Luo2015 Eq. 22 retains `D_CO2_amine = D_CO2_water (mu_water/mu_amine)^0.8`; the evaluated viscosity input uses the Weiland parameters reproduced by Amundsen2009, not a claim that Amundsen originated that correlation.

Snijder1993 Eq. 8 retains `ln(D_MEA) = -13.275 - 2198.3/T - 7.8142e-5 C`, with `C` in `mol/m3` and `D` in `m2/s`. The 16 Table III observations retain density and viscosity. Snijder source viscosity is reported in `mPa s`; its emitted CSV value is converted to `Pa s`. The maximum absolute residual is `{diagnostics['snijder_max_absolute_residual_m2_s']:.9e} m2/s` and the maximum relative residual is `{diagnostics['snijder_max_relative_residual']:.6f}` against the displayed, rounded diffusivities. Snijder's source statement that the fit is within 5% is preserved as source metadata; the rounded table alone does not reproduce that statement at every displayed row.

Amundsen2009 Weiland density parameters and loaded 30 mass% density observations are retained with the source `g/cm3` units. Hartono2014 density observations are retained in `kg/m3`; Hartono2014 viscosity observations are reported in `mPa s` and emitted in `Pa s`. All are source-labeled and non-admitted. Density is not converted to a total molar-density film state while Issue 33 remains basis_unresolved.

Putta2017 Eq. 12 is retained as a blocked N2O analogy because the cited Ko, Jamal, and Ying/Eimer primary N2O-water and N2O-amine inputs were not recovered with source-complete coefficients and uncertainty. The legacy scalar ion expression in `src/mea_absorption_column/Properties/Transport_Properties.py` is rejected as unattributed and is not retained as a default.

## Dependency and claim boundary

Issue 33 remains `basis_unresolved`: Position 1 analytical MEA is `{issue33['gates']['position_1_analytical_concentration_mol_L']:.16g} mol/L` and free MEA is `{config['issue33_dependency']['required_position_1_free_mea_mol_L']:.16g} mol/L`; neither is rounded to exact 5 M. Issue 34 remains the merged supported-negative kinetics record. No rate comparison, Case 3C tuning, physical film result, production transport-adoption change, or packed-column capture claim is made here.

The generated tables are `issue35_transport_correlations.csv`, `issue35_transport_sensitivity.csv`, and `issue35_transport_summary.json`.
"""


def main() -> int:
    config = load_config()
    validate_config(config)
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
        "source_parameters_authority": "analyses/nccc_validation/inputs/issue35_transport.json",
        "source_records": config["source_records"],
        "issue33_dependency": {"summary_path": config["issue33_dependency"]["summary_path"], "summary_sha256": config["issue33_dependency"]["summary_sha256"], "table_path": config["issue33_dependency"]["table_path"], "table_sha256": config["issue33_dependency"]["table_sha256"], "claim_label": issue33["claim_label"], "position_1_analytical_mol_L": issue33["gates"]["position_1_analytical_concentration_mol_L"], "position_1_free_mea_mol_L": config["issue33_dependency"]["required_position_1_free_mea_mol_L"], "all_rows_basis_unresolved": True},
        "issue34_dependency": config["issue34_dependency"],
        "diagnostics": diagnostics,
        "gates": {
            "input_units_domains_signs_and_evaluation_points_validated": True,
            "source_pdf_inspection_receipts_present_for_local_sources": True,
            "known_correlation_positivity_pass": diagnostics["positive_evaluated_values"],
            "snijder_absolute_residual_unit": "m2/s",
            "snijder_relative_residual_retained": True,
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
            "Source density observations cannot be converted to total molar density while Issue 33 basis remains unresolved.",
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
