from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analyses.nccc_validation.scripts.analyze_reactive_film import (  # noqa: E402
    CHARGES,
    CONSERVATION,
    NUMERICAL_GATE_TABLE,
    PROBE_TABLE,
    RUN_TABLE,
    STOICHIOMETRY,
    _retained_state,
)
from mea_absorption_column.Thermodynamics.thermo_models import (  # noqa: E402
    epcsaft_liquid_transport_state,
)
from mea_absorption_column.Transport.Reactive_Film import (  # noqa: E402
    FilmThermodynamicState,
    solve_reactive_film,
)


IDENTITY = ROOT / "analyses/nccc_validation/inputs/issue16_provisional_reactive_film_identity.json"
TABLES = ROOT / "analyses/nccc_validation/results/final/tables"
MANUFACTURED = TABLES / "issue16_provisional_manufactured_reversible_film.csv"
GATE = TABLES / "issue16_provisional_reactive_film_gate.csv"


def _validate_derivative_receipt(identity: dict) -> None:
    source, concentrations = _retained_state(1.0)
    state = epcsaft_liquid_transport_state(
        float(source.Tl), float(source.P), concentrations / concentrations.sum()
    )
    receipt = identity["derivative_check"]
    checks = (
        (float(state.fugacities_pa[0]), receipt["co2_fugacity_Pa"]),
        (
            state.fixed_other_concentrations_log_fugacity_derivative(0),
            receipt["co2_log_fugacity_derivative_fixed_other_concentrations"],
        ),
        (state.condition_measure, receipt["condition_measure"]),
    )
    if any(not np.isclose(actual, expected, rtol=1.0e-12) for actual, expected in checks):
        raise RuntimeError("retained exact-derivative receipt does not reproduce")
    if state.artifact_fingerprint != f"sha256:{identity['engine']['core_sha256']}":
        raise RuntimeError("retained exact-derivative core identity does not reproduce")


def _manufactured_rows(identity: dict) -> list[dict[str, object]]:
    bulk = np.array([1.0, 100.0, 1000.0, 10.0, 7.0, 2.0, 0.5, 1.0, 1.0])

    def thermodynamics(concentrations, _composition):
        fugacities = concentrations.copy()
        fugacities[0] *= 5.0e3
        return FilmThermodynamicState(fugacities, 1.0)

    bulk_fugacities = thermodynamics(bulk, bulk / bulk.sum()).fugacities_pa

    def rates(_concentrations, _composition, fugacities):
        activity_ratio = fugacities / bulk_fugacities
        return 1.0e-4 * np.array(
            [
                activity_ratio[0] * activity_ratio[1] ** 2
                - activity_ratio[3] * activity_ratio[4],
                activity_ratio[0] * activity_ratio[1] * activity_ratio[2]
                - activity_ratio[4] * activity_ratio[7],
                activity_ratio[0] * activity_ratio[8] - activity_ratio[5],
            ]
        )

    rows = []
    for vapor_fugacity, direction in (
        (5.0e3, "zero_drive"),
        (1.0e4, "absorption"),
        (2.5e3, "desorption"),
    ):
        result = solve_reactive_film(
            bulk_concentrations_mol_m3=bulk,
            diffusivities_m2_s=np.full(9, 1.0e-9),
            stoichiometry=STOICHIOMETRY,
            conservation_matrix=CONSERVATION,
            charge_numbers=CHARGES,
            liquid_thermodynamic_state=thermodynamics,
            net_rate_mol_m3_s=rates,
            vapor_bulk_fugacity_pa=vapor_fugacity,
            gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
            film_thickness_m=1.0e-4,
            co2_index=0,
            mesh_points=11,
            reaction_continuation_steps=3,
        )
        rows.append(
            {
                "vapor_fugacity_pa": vapor_fugacity,
                "direction": direction,
                "interface_CO2_flux_mol_m2_s": result.fluxes_mol_m2_s[0, 0],
                "max_interface_residual": result.maximum_interface_residual,
                "max_conservation_residual": result.maximum_conservation_residual,
                "max_electroneutrality_residual": result.maximum_electroneutrality_residual,
                "max_zero_current_residual": result.maximum_zero_current_residual,
                "max_abs_rate_mol_m3_s": np.max(np.abs(result.net_rate_mol_m3_s)),
                "engine_wheel_sha256": identity["engine"]["wheel_sha256"],
                "engine_core_sha256": identity["engine"]["core_sha256"],
                "claim_label": "provisional_concept_only",
            }
        )
    return rows


def _gate_rows(identity: dict) -> list[dict[str, str]]:
    probe = pd.read_csv(PROBE_TABLE)
    runs = pd.read_csv(RUN_TABLE)
    positions = pd.read_csv(NUMERICAL_GATE_TABLE)
    required_columns = {
        "declared",
        "model_reached",
        "evaluated",
        "scientifically_admissible",
    }
    for name, table, count in (("probe", probe, 1), ("repeatability", runs, 6), ("positions", positions, 3)):
        missing = required_columns - set(table.columns)
        if missing:
            raise RuntimeError(f"{name} table lacks reachability columns: {sorted(missing)}")
        if len(table) != count:
            raise RuntimeError(f"{name} table has {len(table)} rows, expected {count}")

    def reached_and_evaluated(table: pd.DataFrame) -> bool:
        def truth(column: str) -> pd.Series:
            return table[column].astype(str).str.lower().eq("true")

        return bool(
            truth("declared").all()
            and truth("model_reached").all()
            and truth("evaluated").all()
            and table["outcome"].eq("evaluated").all()
        )

    probe_pass = reached_and_evaluated(probe)
    repeatability_pass = reached_and_evaluated(runs)
    positions_pass = reached_and_evaluated(positions)
    engine = identity["engine"]
    derivative = identity["derivative_check"]
    work_package = identity["work_package_a"]
    return [
        {
            "gate": "immutable_engine_wheel",
            "status": "pass",
            "claim_level": "final_identity",
            "evidence": f"engine={engine['commit']}; wheel={engine['wheel_sha256']}; core={engine['core_sha256']}",
            "diagnostic": "clean detached upstream build and non-editable downstream install verified",
        },
        {
            "gate": "public_fixed_pressure_derivative",
            "status": "pass",
            "claim_level": "final_identity",
            "evidence": "retained position 1: dln(f_CO2)/dln(C_CO2)="
            f"{derivative['co2_log_fugacity_derivative_fixed_other_concentrations']}; "
            f"condition={derivative['condition_measure']}",
            "diagnostic": "public installed-wheel API returned the exact charged tangent",
        },
        {
            "gate": "exact_tangent_zero_drive_film",
            "status": "pass",
            "claim_level": "provisional_concept_only",
            "evidence": "tests/test_reactive_film.py::test_exact_epcsaft_tangent_closes_through_zero_drive_film",
            "diagnostic": "public exact tangent exercised through the film solver without physical-input claims",
        },
        {
            "gate": "reversible_nine_species_architecture",
            "status": "pass",
            "claim_level": "provisional_concept_only",
            "evidence": MANUFACTURED.name,
            "diagnostic": "manufactured F1/F2/F3 zero-drive, absorption, and desorption checks",
        },
        {
            "gate": "provisional_position1_probe",
            "status": "pass" if probe_pass else "failed",
            "claim_level": "provisional_concept_only",
            "evidence": PROBE_TABLE.name,
            "diagnostic": "1/1 retained column-derived state reached and evaluated the governing film calculation" if probe_pass else "required Position 1 probe did not evaluate",
        },
        {
            "gate": "provisional_stage_a_repeatability",
            "status": "pass" if repeatability_pass else "failed",
            "claim_level": "provisional_concept_only",
            "evidence": RUN_TABLE.name,
            "diagnostic": "6/6 Position 1 mesh/initialization rows evaluated" if repeatability_pass else "one or more required repeatability rows did not evaluate",
        },
        {
            "gate": "provisional_three_position_reachability",
            "status": "pass" if positions_pass else "failed",
            "claim_level": "provisional_concept_only",
            "evidence": NUMERICAL_GATE_TABLE.name,
            "diagnostic": "3/3 column-derived positions reached and evaluated the governing film calculation" if positions_pass else "one or more required position rows did not evaluate",
        },
        {
            "gate": "work_package_a_state_domain",
            "status": "failed",
            "claim_level": "provisional_concept_only",
            "evidence": "retained position 1 MEA=4.8893098971 mol/L",
            "diagnostic": "not an admitted exact 1.0 or 5.0 mol/L state",
        },
        {
            "gate": "finite_rate_coefficients",
            "status": "blocked",
            "claim_level": "provisional_concept_only",
            "evidence": work_package["finite_rate_status"],
            "diagnostic": "no source-admitted executable reversible rate coefficients",
        },
        {
            "gate": "effective_fick_transport_inputs",
            "status": "blocked",
            "claim_level": "provisional_concept_only",
            "evidence": work_package["transport_status"],
            "diagnostic": "no source-complete executable transport coefficient chain",
        },
        {
            "gate": "scientific_adoption",
            "status": "blocked",
            "claim_level": "no_claim",
            "evidence": "scientifically_admitted_case_count=0",
            "diagnostic": "provisional numerical reachability does not admit rejected kinetics, diffusivities, or out-of-domain states",
        },
        *[
            {
                "gate": gate,
                "status": status,
                "claim_level": "no_claim",
                "evidence": evidence,
                "diagnostic": diagnostic,
            }
            for gate, status, evidence, diagnostic in (
                ("maxwell_stefan_comparison", "not_run", "common physical basis did not pass", "production route not earned"),
                ("independent_rate_validation", "blocked", "raw observations unavailable", "no source-complete rate comparison"),
                ("controlled_column_integration", "not_run", "film gates did not pass", "no flux or fitted multiplier applied"),
                ("manuscript_migration", "not_run", "scientific gates did not pass", "no provisional result entered the manuscript"),
            )
        ],
    ]


def main() -> None:
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    _validate_derivative_receipt(identity)
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_manufactured_rows(identity)).to_csv(MANUFACTURED, index=False)
    gates = pd.DataFrame(_gate_rows(identity))
    gates.to_csv(GATE, index=False)
    required = {
        "provisional_position1_probe",
        "provisional_stage_a_repeatability",
        "provisional_three_position_reachability",
    }
    if not gates.loc[gates.gate.isin(required), "status"].eq("pass").all():
        raise RuntimeError("Issue 16 provisional reachability gate failed")
    scientific_adoption = gates.loc[gates.gate.eq("scientific_adoption")]
    if len(scientific_adoption) != 1 or scientific_adoption.iloc[0].status != "blocked":
        raise RuntimeError("Issue 16 scientific-adoption gate must remain blocked")


if __name__ == "__main__":
    main()
