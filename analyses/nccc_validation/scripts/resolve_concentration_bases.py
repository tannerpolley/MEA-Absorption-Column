from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import (  # noqa: E402
    SPECIES_9,
    tabulated_epcsaft_reactive_chemical_equilibrium,
)


ANALYSIS = ROOT / "analyses/nccc_validation"
INPUT = ANALYSIS / "inputs/issue33_concentration_basis.json"
PROFILE = ANALYSIS / "inputs/retained_reactive_case3c/film_states.csv"
CASE_INPUT = ROOT / "src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv"
SOURCE_CASE_INPUT = ROOT / "src/mea_absorption_column/data/NCCC_2017_cases.csv"
IDENTITY = ANALYSIS / "inputs/issue16_reactive_film_identity.json"
TABLE = ANALYSIS / "results/final/tables/issue33_concentration_basis.csv"
SUMMARY = ANALYSIS / "results/final/tables/issue33_concentration_basis_summary.json"

MEAN_MOLAR_MASS_KG_PER_MOL = 0.06108
CO2_SPECIES = ("CO2", "MEACOO-", "HCO3-", "CO3^2-")
PROFILE_SPECIES_COLUMNS = {
    "CO2": "Cl_CO2_true",
    "MEA": "Cl_MEA_true",
    "H2O": "Cl_H2O_true",
    "MEAH+": "Cl_MEAH_true",
    "MEACOO-": "Cl_MEACOO_true",
}
POSITIONS = (0.0, 0.5, 1.0)
DOMAIN_TEMPERATURE_K = (293.15, 323.15)
DOMAIN_LOADING = (0.0, 0.5)
DISCRETE_LABELS_MOL_L = (1.0, 5.0)
BALANCE_TOLERANCE_MOL_L = 1.0e-10


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _linear(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _bracket(value: float, values: list[float], label: str) -> tuple[float, float]:
    ordered = sorted(set(values))
    if value < ordered[0] - 1.0e-12 or value > ordered[-1] + 1.0e-12:
        raise ValueError(f"{label}={value:.12g} is outside source interpolation range")
    for candidate in ordered:
        if abs(value - candidate) <= 1.0e-12:
            return candidate, candidate
    for lower, upper in zip(ordered, ordered[1:]):
        if lower < value < upper:
            return lower, upper
    raise ValueError(f"could not bracket {label}={value:.12g}")


def _density(
    observations: list[dict[str, float]], temperature_K: float, loading: float
) -> float:
    by_temperature: dict[float, list[dict[str, float]]] = {}
    for observation in observations:
        by_temperature.setdefault(float(observation["temperature_K"]), []).append(observation)
    t0, t1 = _bracket(temperature_K, list(by_temperature), "temperature_K")

    def at_temperature(temperature: float) -> float:
        rows = by_temperature[temperature]
        l0, l1 = _bracket(loading, [float(row["loading"]) for row in rows], "loading")
        values = {float(row["loading"]): float(row["density_kg_m3"]) for row in rows}
        return _linear(loading, l0, l1, values[l0], values[l1])

    return _linear(temperature_K, t0, t1, at_temperature(t0), at_temperature(t1))


def _relative_error_percent(value: float, target: float) -> float:
    return abs(value - target) / abs(target) * 100.0


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash changed: expected {expected}, got {actual}")


def _load_inputs() -> tuple[dict, pd.DataFrame, pd.Series, dict]:
    config = json.loads(INPUT.read_text(encoding="utf-8"))
    _require_hash(PROFILE, config["retained_profile"]["sha256"], "retained profile")
    _require_hash(CASE_INPUT, config["nccc_case"]["input_sha256"], "NCCC case input")
    _require_hash(
        SOURCE_CASE_INPUT,
        config["nccc_case"]["source_preserving_input_sha256"],
        "source-preserving NCCC case input",
    )
    _require_hash(IDENTITY, config["immutable_inputs"]["issue16_identity_sha256"], "Issue 16 identity")

    profile = pd.read_csv(PROFILE)
    case_rows = pd.read_csv(CASE_INPUT)
    case = case_rows.loc[case_rows["case_no"].eq(config["nccc_case"]["case_id"])]
    if len(case) != 1:
        raise ValueError("NCCC case input must contain exactly one Case 3C row")
    case = case.iloc[0]
    for column, expected in (
        ("w_MEA", config["nccc_case"]["nominal_mea_mass_fraction"]),
        ("alpha", config["nccc_case"]["loading_mol_co2_per_mol_mea"]),
        ("Tl", config["nccc_case"]["lean_temperature_K"]),
        ("P", config["nccc_case"]["pressure_Pa"]),
    ):
        if abs(float(case[column]) - float(expected)) > 1.0e-12:
            raise ValueError(f"run-ready Case 3C input changed in {column}")
    source_case_rows = pd.read_csv(SOURCE_CASE_INPUT)
    source_case = source_case_rows.loc[
        source_case_rows["case_no"].eq(config["nccc_case"]["case_id"])
    ]
    if len(source_case) != 1 or not pd.isna(source_case.iloc[0]["absorber_lean_solvent_temp_C"]):
        raise ValueError("source-preserving Case 3C input must retain its missing lean temperature")
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    return config, profile, case, identity


def _source_label_row(config: dict, nominal_mol_L: float) -> dict[str, object]:
    reasons = (
        "Putta literature row supplies a nominal 1/5 mol/L label only; preparation temperature, "
        "local loaded analytical concentration, free-MEA concentration, loaded volume basis, and "
        "row-specific concentration uncertainty are not reported at the retained source locators"
    )
    return {
        "record_kind": "literature_label",
        "record_id": f"Putta2016_{nominal_mol_L:g}M",
        "case_id": "",
        "position": "",
        "temperature_K": "",
        "temperature_source": "not_reported_by_Putta2016",
        "preparation_temperature_K": "",
        "density_conversion_temperature_K": "",
        "loading_mol_CO2_per_mol_MEA": "",
        "loading_source": "not_reported_for_local_literature_row",
        "nominal_mea_mass_fraction": "",
        "nominal_mea_mass_fraction_uncertainty_relative_expanded_percent": "",
        "prepared_density_kg_m3": "",
        "prepared_density_uncertainty_kg_m3": "",
        "prepared_density_source": "not_reported_for_local_literature_row",
        "prepared_concentration_mol_L": nominal_mol_L,
        "prepared_concentration_uncertainty_composition_only_mol_L": "",
        "loaded_density_kg_m3": "",
        "loaded_density_uncertainty_kg_m3": "",
        "loaded_density_source": "not_reported_for_local_literature_row",
        "loaded_analytical_concentration_mol_L": "",
        "free_MEA_concentration_mol_L": "",
        "n_MEA_mol_L": "",
        "n_H2O_mol_L": "",
        "n_MEAH_plus_mol_L": "",
        "n_MEACOO_minus_mol_L": "",
        "n_CO2_mol_L": "",
        "n_HCO3_minus_mol_L": "",
        "n_CO3_2_minus_mol_L": "",
        "n_H3O_plus_mol_L": "",
        "n_OH_minus_mol_L": "",
        "analytical_balance_reconstructed_mol_L": "",
        "analytical_balance_residual_mol_L": "",
        "source_profile_rho_mol_L": "",
        "volume_basis": "unresolved",
        "prepared_to_loaded_volume_ratio": "unresolved",
        "difference_to_5M_mol_L": 0.0 if nominal_mol_L == 5.0 else nominal_mol_L - 5.0,
        "difference_to_5M_relative_percent": 0.0 if nominal_mol_L == 5.0 else _relative_error_percent(nominal_mol_L, 5.0),
        "within_campaign_composition_uncertainty": "not_assessed",
        "admission_decision": "basis_unresolved",
        "admission_reasons": reasons,
        "source_uncertainties": "Luo2015: preparation balance +/- 1e-5 kg, temperature +/- 0.1 C, loading +/- 0.01 mol/mol, density regression relative total standard deviation 0.06% (1 M) and 0.17% (5 M); Putta2016: row-specific concentration uncertainty unavailable",
        "source_locators": "Putta2016 printed p. 340 Table 1; printed p. 340 Section 2; printed pp. 347-348 Figures 6 and 7",
        "source_identities": "Putta2016; Luo2015",
        "source_attachment_sha256": "Putta2016=fac3789d1ff6baa53e226638a2505ee3f3ff10433e4af89ef0a8e27785771e99; Luo2015=16f655236141fb72b4a68992e7f5c4c97ca5c5946304dafe03011fc3da60ee44",
    }


def _state_row(config: dict, source: pd.Series, identity: dict) -> dict[str, object]:
    apparent_flows = source[["Fl_CO2", "Fl_MEA", "Fl_H2O"]].to_numpy(float)
    _, composition = tabulated_epcsaft_reactive_chemical_equilibrium(
        apparent_flows, float(source["Tl"]), diagnostics={}
    )
    concentrations_mol_m3 = dict(zip(SPECIES_9, composition * float(source["rho_mol_l"]), strict=True))
    for species, column in PROFILE_SPECIES_COLUMNS.items():
        if abs(concentrations_mol_m3[species] - float(source[column])) > 1.0e-8:
            raise ValueError(f"retained equilibrium reconstruction disagrees with profile column {column}")
        concentrations_mol_m3[species] = float(source[column])
    analytical_components_mol_L = [
        concentrations_mol_m3[species] / 1000.0 for species in ("MEA", "MEAH+", "MEACOO-")
    ]
    co2_components_mol_L = [concentrations_mol_m3[species] / 1000.0 for species in CO2_SPECIES]
    analytical_mol_L = sum(analytical_components_mol_L)
    balance_mol_L = sum(analytical_components_mol_L)
    balance_residual_mol_L = balance_mol_L - analytical_mol_L
    loading = sum(co2_components_mol_L) / analytical_mol_L
    temperature_K = float(source["Tl"])
    prepared_mass_fraction = float(config["nccc_case"]["nominal_mea_mass_fraction"])
    prepared_density_kg_m3 = _density(
        config["density_observations"]["unloaded"], temperature_K, 0.0
    )
    loaded_density_kg_m3 = _density(
        config["density_observations"]["loaded"], temperature_K, loading
    )
    prepared_concentration_mol_L = (
        prepared_density_kg_m3 * prepared_mass_fraction / MEAN_MOLAR_MASS_KG_PER_MOL / 1000.0
    )
    nominal_uncertainty_percent = float(
        config["nccc_case"]["nominal_mea_composition_uncertainty_relative_expanded_percent"]
    )
    prepared_concentration_uncertainty_mol_L = (
        prepared_concentration_mol_L * nominal_uncertainty_percent / 100.0
    )
    difference_to_5M = analytical_mol_L - 5.0
    reasons = [
        "nccc_preparation_temperature_unreported",
        "prepared_to_loaded_solution_volume_ratio_unreported",
        "retained_loaded_analytical_concentration_is_a_profile_state_not_a_literature_measurement",
        "exact_discrete_1M_or_5M_label_not_met",
    ]
    if not DOMAIN_TEMPERATURE_K[0] <= temperature_K <= DOMAIN_TEMPERATURE_K[1]:
        reasons.append("temperature_outside_293.15_to_323.15_K_domain")
    if not DOMAIN_LOADING[0] <= loading < DOMAIN_LOADING[1]:
        reasons.append("loading_outside_0_to_0.5_domain")
    if abs(difference_to_5M) <= 1.0e-12:
        reasons.append("exact_5M_label_would_still_require_source_basis_admission")

    expected_position_1 = identity["retained_position_1"]["mea_molarity_mol_L"]
    if abs(float(source["Position"]) - 1.0) <= 1.0e-12 and abs(analytical_mol_L - expected_position_1) > 1.0e-15:
        raise ValueError(
            "Position 1 analytical concentration no longer matches the immutable Issue 16 identity"
        )

    return {
        "record_kind": "retained_case3c_state",
        "record_id": f"Case3C_position_{float(source['Position']):g}",
        "case_id": "3C",
        "position": float(source["Position"]),
        "temperature_K": temperature_K,
        "temperature_source": "retained_reactive_case3c/film_states.csv: Tl",
        "preparation_temperature_K": "unresolved",
        "density_conversion_temperature_K": temperature_K,
        "loading_mol_CO2_per_mol_MEA": loading,
        "loading_source": "retained nine-species concentrations; CO2 + MEACOO- + HCO3- + CO3^2- over analytical MEA",
        "nominal_mea_mass_fraction": prepared_mass_fraction,
        "nominal_mea_mass_fraction_uncertainty_relative_expanded_percent": nominal_uncertainty_percent,
        "prepared_density_kg_m3": prepared_density_kg_m3,
        "prepared_density_uncertainty_kg_m3": config["density_observations"]["unloaded_density_uncertainty_kg_m3"],
        "prepared_density_source": "Amundsen2009 30 wt% unloaded density; bilinear source interpolation",
        "prepared_concentration_mol_L": prepared_concentration_mol_L,
        "prepared_concentration_uncertainty_composition_only_mol_L": prepared_concentration_uncertainty_mol_L,
        "loaded_density_kg_m3": loaded_density_kg_m3,
        "loaded_density_uncertainty_kg_m3": config["density_observations"]["loaded_density_uncertainty_kg_m3"],
        "loaded_density_source": "Amundsen2009 30 wt% loaded density; bilinear source interpolation",
        "loaded_analytical_concentration_mol_L": analytical_mol_L,
        "free_MEA_concentration_mol_L": concentrations_mol_m3["MEA"] / 1000.0,
        "n_MEA_mol_L": concentrations_mol_m3["MEA"] / 1000.0,
        "n_H2O_mol_L": concentrations_mol_m3["H2O"] / 1000.0,
        "n_MEAH_plus_mol_L": concentrations_mol_m3["MEAH+"] / 1000.0,
        "n_MEACOO_minus_mol_L": concentrations_mol_m3["MEACOO-"] / 1000.0,
        "n_CO2_mol_L": concentrations_mol_m3["CO2"] / 1000.0,
        "n_HCO3_minus_mol_L": concentrations_mol_m3["HCO3-"] / 1000.0,
        "n_CO3_2_minus_mol_L": concentrations_mol_m3["CO3^2-"] / 1000.0,
        "n_H3O_plus_mol_L": concentrations_mol_m3["H3O+"] / 1000.0,
        "n_OH_minus_mol_L": concentrations_mol_m3["OH-"] / 1000.0,
        "analytical_balance_reconstructed_mol_L": balance_mol_L,
        "analytical_balance_residual_mol_L": balance_residual_mol_L,
        "source_profile_rho_mol_L": float(source["rho_mol_l"]) / 1000.0,
        "volume_basis": "1 L of retained loaded liquid solution",
        "prepared_to_loaded_volume_ratio": "unresolved",
        "difference_to_5M_mol_L": difference_to_5M,
        "difference_to_5M_relative_percent": _relative_error_percent(analytical_mol_L, 5.0),
        "within_campaign_composition_uncertainty": abs(difference_to_5M) <= analytical_mol_L * nominal_uncertainty_percent / 100.0,
        "admission_decision": "basis_unresolved",
        "admission_reasons": "; ".join(reasons),
        "source_uncertainties": "Morgan2018: nominal amine composition +/- 7.3% expanded (approximately 95%, k=2); Amundsen2009: +/- 0.5 kg/m3 unloaded density and +/- 2.0 kg/m3 loaded density combined estimates; NCCC Case 3C-specific preparation-temperature and volume uncertainty unavailable",
        "source_locators": "Morgan2018 printed p. 10468 Section 2.2 and printed pp. 10469-10470 Section 2.4/Table 4; Amundsen2009 printed pp. 3096-3099; retained profile",
        "source_identities": "Morgan2018; Amundsen2009; retained Case 3C profile; Issue 16 identity",
        "source_attachment_sha256": "Morgan2018=bf9cfa2877c31a1ed8346951e149b6af7fc7e8413e3f62fde22a76e6c31b5ddb; Amundsen2009=a6525dde4e8b0e74902ebbe1d8c6e3f246cee36be9ae5409e32669750f409d4e",
    }


def _write_table(rows: list[dict[str, object]]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    global TABLE, SUMMARY
    if args.output_dir is not None:
        TABLE = args.output_dir / TABLE.name
        SUMMARY = args.output_dir / SUMMARY.name

    config, profile, case, identity = _load_inputs()
    rows = [_source_label_row(config, nominal) for nominal in DISCRETE_LABELS_MOL_L]
    for position in POSITIONS:
        candidates = profile.iloc[(profile["Position"] - position).abs().argsort()[:1]]
        if len(candidates) != 1 or abs(float(candidates.iloc[0]["Position"]) - position) > 1.0e-12:
            raise ValueError(f"retained profile is missing Position {position:g}")
        rows.append(_state_row(config, candidates.iloc[0], identity))

    state_rows = [row for row in rows if row["record_kind"] == "retained_case3c_state"]
    max_balance_residual = max(abs(float(row["analytical_balance_residual_mol_L"])) for row in state_rows)
    position_1 = next(row for row in state_rows if abs(float(row["position"]) - 1.0) <= 1.0e-12)
    summary = {
        "claim_label": config["claim_label"],
        "analysis": "nccc_validation",
        "issue": 33,
        "source_table_path": "analyses/nccc_validation/results/final/tables/issue33_concentration_basis.csv",
        "rows": len(rows),
        "reported_case3c_positions": [float(row["position"]) for row in state_rows],
        "gates": {
            "all_requested_positions_present": len(state_rows) == len(POSITIONS),
            "analytical_balance_max_abs_residual_mol_L": max_balance_residual,
            "analytical_balance_tolerance_mol_L": BALANCE_TOLERANCE_MOL_L,
            "analytical_balance_pass": max_balance_residual <= BALANCE_TOLERANCE_MOL_L,
            "position_1_analytical_concentration_mol_L": position_1["loaded_analytical_concentration_mol_L"],
            "position_1_exact_value_preserved": abs(
                float(position_1["loaded_analytical_concentration_mol_L"])
                - float(identity["retained_position_1"]["mea_molarity_mol_L"])
            ) <= 1.0e-15,
            "position_1_not_rounded_to_5M": abs(
                float(position_1["loaded_analytical_concentration_mol_L"]) - 5.0
            ) > 1.0e-12,
            "exact_discrete_1M_or_5M_admission": False,
            "capture_or_kinetic_tuning_performed": False,
        },
        "conclusions": [
            "Prepared concentration is a mass-fraction/density conversion on an unloaded-solution basis.",
            "Loaded analytical concentration is the conserved sum of MEA, MEAH+, and MEACOO- on the loaded-solution volume basis.",
            "Free MEA is the molecular MEA species concentration and remains separate from both prepared and analytical concentration.",
            "The retained Position 1 calculation is 4.889309897097635 mol/L (4.8893098971 mol/L at the requested display precision), not exact 5 M.",
            "Morgan2018 supplies a 7.3% campaign-level expanded uncertainty for nominal NCCC amine composition; it does not establish a Case 3C-specific prepared-to-loaded volume ratio.",
        ],
        "limitations": [
            "Putta/Luo do not report all source-local preparation temperatures, loaded analytical concentrations, or paired volume bases needed to admit exact 1 M/5 M literature inputs.",
            "NCCC Case 3C preparation temperature and prepared-to-loaded solution-volume ratio are absent from the retained case record.",
            "Amundsen density values are used as source-backed interpolation evidence; the retained ePC-SAFT profile density is retained separately and does not redefine a literature label.",
            "All source and retained state rows are reported, but every mapping row remains basis_unresolved for exact discrete absorber admission.",
        ],
        "immutable_inputs": config["immutable_inputs"],
        "retained_engine_identity": identity["engine"],
        "source_records": [
            {
                "id": source["id"],
                "doi": source["doi"],
                "zotero_attachment_key": source["zotero_attachment_key"],
                "attachment_sha256": source["attachment_sha256"],
            }
            for source in config["source_records"]
        ],
    }
    _write_table(rows)
    summary["source_table_sha256"] = _sha256(TABLE)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {TABLE}")
    print(f"Wrote {SUMMARY}")
    print(json.dumps(summary["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
