from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses/reactive_film_evidence"
FILM_SCRIPT = ROOT / "analyses/bvp_derivative_trials/scripts/run_chemical_potential_film.py"
AARD_SOURCE = ROOT / "analyses/nccc_validation/results/final/tables/issue34_rate_observation_comparisons.csv"
ANCHORS = ANALYSIS / "inputs/diffusion_anchor_assumptions.csv"
DUGAS = ANALYSIS / "inputs/dugas2011_table1_mea.csv"
RAMEZANI = ANALYSIS / "inputs/ramezani2021_si_rows.csv"
TABLES = ANALYSIS / "results/final/tables"


def _write(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_film_module():
    spec = importlib.util.spec_from_file_location("chemical_potential_film", FILM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {FILM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_aard() -> None:
    with AARD_SOURCE.open(newline="") as stream:
        source_rows = [row for row in csv.DictReader(stream) if row["metric"] == "AARD_percent"]
    if len(source_rows) != 20:
        raise AssertionError("expected 20 Putta2016 Table 4 aggregate AARD cells")
    rows = []
    for row in source_rows:
        rows.append(
            {
                "model": row["model"],
                "apparatus_or_dataset": row["apparatus_or_dataset"],
                "AARD_percent": f"{float(row['value']):.1f}",
                "source_locator": "Putta2016 printed p.349 Table 4",
                "evidence_scope": "aggregate_model_comparison_not_row_level_validation",
            }
        )
    _write(rows, TABLES / "putta2016_table4_aard_plot.csv")


def _build_chemical_potential() -> None:
    module = _load_film_module()
    result = module.run(11)
    rows = []
    for drive in ("absorption", "desorption"):
        block = result[drive]["chemical_potential"]
        rows.extend(
            [
                {
                    "drive": drive,
                    "closure": "ePC-SAFT chemical-potential force",
                    "co2_flux_mol_m2_s": f"{block['co2_flux_mol_m2_s']:.17e}",
                    "flux_sign_convention": "positive absorption; negative desorption",
                    "mobility_closure": result["mobility"],
                    "calculation_status": result["state_disposition"],
                    "claim_scope": "thermodynamic_force_isolation_non_predictive",
                },
                {
                    "drive": drive,
                    "closure": "constrained ideal-log-composition-force reference",
                    "co2_flux_mol_m2_s": f"{block['constrained_ideal_log_force_reference_co2_flux_mol_m2_s']:.17e}",
                    "flux_sign_convention": "positive absorption; negative desorption",
                    "mobility_closure": result["mobility"],
                    "calculation_status": result["state_disposition"],
                    "claim_scope": "thermodynamic_force_isolation_non_predictive",
                },
            ]
        )
    _write(rows, TABLES / "chemical_potential_isolation_plot.csv")

    case = module._load_case()
    bulk_state = module.epcsaft_liquid_transport_state(case.temperature_K, case.pressure_Pa, case.composition)
    independent, dependent = module._coordinate_indices(bulk_state)
    basis, derivative, _constrained, projection = module._constrained_block(bulk_state)
    interface_values = module._interface_coordinates(case, bulk_state, 1.01, independent, dependent)
    absorption = result["absorption"]["chemical_potential"]
    def matrix_text(matrix):
        return ";".join(
            ",".join(f"{value:.17e}" for value in row) for row in matrix
        )
    trace_rows = [
        {
            "definition_id": "integrated_boundary_response",
            "quantity_name": "CO2 flux under 1% fugacity-ratio drive",
            "chemical_value": f"{absorption['co2_flux_mol_m2_s']:.17e}",
            "reference_value": f"{absorption['constrained_ideal_log_force_reference_co2_flux_mol_m2_s']:.17e}",
            "relative_delta_percent": f"{100.0 * (absorption['co2_flux_mol_m2_s'] / absorption['constrained_ideal_log_force_reference_co2_flux_mol_m2_s'] - 1.0):.12f}",
            "force_direction": "CO2-only interface coordinate: +0.01 absorption; -0.01 desorption",
            "normalization": "integrated seven-coordinate response; provisional unit mobility / 1e-4 m",
            "reproduction_status": "reproduced",
            "prior_reported_delta_percent": "",
            "explanation": "Panel A quantity: solve the path-integrated projected derivative map for the interface coordinate.",
        },
        {
            "definition_id": "local_projected_CO2_same_gradient",
            "quantity_name": "projected CO2 flux-like response for common q",
            "chemical_value": "",
            "reference_value": "",
            "relative_delta_percent": "",
            "force_direction": "q=zeros(7); q[CO2 independent coordinate]=1e-4",
            "normalization": "jc=-P @ derivative @ q; ji=-P @ basis @ q; compare CO2 components",
            "reproduction_status": "reproduced_rounded_prior_metric",
            "prior_reported_delta_percent": "-4.75",
            "explanation": "Independent local state and same-gradient comparison; this is the meaningful 4.75% diagnostic, not a dimensional flux prediction.",
        },
    ]
    for row in trace_rows:
        row.update(
            {
                "case_id": "3C",
                "position": "1",
                "temperature_K": f"{case.temperature_K:.12f}",
                "pressure_Pa": f"{case.pressure_Pa:.12f}",
                "species_order": ";".join(module.SPECIES),
                "independent_species": ";".join(module.SPECIES[index] for index in independent),
                "dependent_species": ";".join(module.SPECIES[index] for index in dependent),
                "bulk_composition": ";".join(f"{value:.17e}" for value in case.composition),
                "coordinate_or_gradient": ";".join(f"{value:.17e}" for value in interface_values),
                "derivative_shape": str(derivative.shape),
                "basis_shape": str(basis.shape),
                "projection_shape": str(projection.shape),
                "basis_matrix": matrix_text(basis),
                "derivative_matrix": matrix_text(derivative),
                "projection_matrix": matrix_text(projection),
                "mobility_closure": result["mobility"],
            }
        )
    local_composition = np.asarray([1.0, 20.0, 70.0, 3.0, 2.0, 0.5, 0.25, 0.5, 0.5])
    local_composition /= np.sum(local_composition)
    local_state = module.epcsaft_liquid_transport_state(313.15, 101325.0, local_composition)
    local_independent, local_dependent = module._coordinate_indices(local_state)
    local_basis, local_derivative, _local_constrained, local_projection = module._constrained_block(local_state)
    local_q = np.zeros(7)
    local_q[int(np.flatnonzero(local_independent == module.CO2_INDEX)[0])] = 1.0e-4
    local_chemical = -local_projection @ local_derivative @ local_q
    local_reference = -local_projection @ local_basis @ local_q
    local_trace = trace_rows[-1]
    local_trace.update(
        {
            "case_id": "local_reference_state",
            "position": "not applicable",
            "temperature_K": "313.150000000000",
            "pressure_Pa": "101325.000000000000",
            "species_order": ";".join(module.SPECIES),
            "independent_species": ";".join(module.SPECIES[index] for index in local_independent),
            "dependent_species": ";".join(module.SPECIES[index] for index in local_dependent),
            "bulk_composition": ";".join(f"{value:.17e}" for value in local_composition),
            "coordinate_or_gradient": ";".join(f"{value:.17e}" for value in local_q),
            "derivative_shape": str(local_derivative.shape),
            "basis_shape": str(local_basis.shape),
            "projection_shape": str(local_projection.shape),
            "basis_matrix": matrix_text(local_basis),
            "derivative_matrix": matrix_text(local_derivative),
            "projection_matrix": matrix_text(local_projection),
            "chemical_value": f"{local_chemical[module.CO2_INDEX]:.17e}",
            "reference_value": f"{local_reference[module.CO2_INDEX]:.17e}",
            "relative_delta_percent": f"{100.0 * (local_chemical[module.CO2_INDEX] / local_reference[module.CO2_INDEX] - 1.0):.12f}",
            "mobility_closure": result["mobility"],
        }
    )
    trace_rows[-1] = local_trace
    trace_fields = [
        "definition_id", "case_id", "position", "temperature_K", "pressure_Pa", "species_order",
        "independent_species", "dependent_species", "bulk_composition", "coordinate_or_gradient",
        "force_direction", "derivative_shape", "basis_shape", "projection_shape", "mobility_closure",
        "basis_matrix", "derivative_matrix", "projection_matrix",
        "normalization", "quantity_name", "chemical_value", "reference_value", "relative_delta_percent",
        "prior_reported_delta_percent", "reproduction_status", "explanation",
    ]
    path = TABLES / "chemical_potential_definition_trace.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=trace_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(trace_rows)


def _copy_input(name: str, target: str) -> None:
    with (ANALYSIS / "inputs" / name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AssertionError(f"empty retained input: {name}")
    _write(rows, TABLES / target)


def _build_viscosity_sensitivity() -> None:
    with RAMEZANI.open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["table_id"] == "S4"
            and row["property_or_metric"] == "viscosity"
            and row["T_K"] == "313.15"
        ]
    if len(rows) != 25:
        raise AssertionError("expected 25 exact Ramezani S4 rows at 313.15 K")
    reference = {
        row["co2_loading_mol_per_mol_mea"]: float(row["value"])
        for row in rows
        if row["sugar_wt_pct"] == "0"
    }
    output = []
    for row in rows:
        value = float(row["value"])
        output.append(
            {
                "T_K": row["T_K"],
                "co2_loading_mol_per_mol_mea": row["co2_loading_mol_per_mol_mea"],
                "sugar_wt_pct": row["sugar_wt_pct"],
                "viscosity_mPa_s": row["value"],
                "normalized_inverse_viscosity": f"{reference[row['co2_loading_mol_per_mol_mea']] / value:.8f}",
                "normalization": "no-sugar viscosity at same T and CO2 loading",
                "source_locator": row["source_locator"],
                "evidence_label": "verified",
                "claim_scope": "fixed_chemistry_viscosity_proxy_non_predictive",
            }
        )
    _write(output, TABLES / "ramezani_viscosity_sensitivity_plot.csv")


def main() -> None:
    _build_aard()
    _build_chemical_potential()
    _copy_input("diffusion_anchor_assumptions.csv", "diffusion_anchor_plot.csv")
    _copy_input("dugas2011_table1_mea.csv", "dugas2011_table1_mea_plot.csv")
    _build_viscosity_sensitivity()
    _copy_input("ganesan2026_si_rows.csv", "ganesan2026_si_rows_plot.csv")
    _copy_input("ganesan2026_si_model_choices.csv", "ganesan2026_si_model_choices_plot.csv")
    print("retained reactive-film plot data generated")


if __name__ == "__main__":
    main()
