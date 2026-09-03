from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import signal
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analyses.nccc_validation.scripts.analyze_enhancement_consistency import (  # noqa: E402
    _load_profile,
)
from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import (  # noqa: E402
    SPECIES_9,
    bundle_reactive_chemical_equilibrium,
)
from mea_absorption_column.Thermodynamics.thermo_models import (  # noqa: E402
    IONIC_CHARGE_BY_SPECIES,
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    epcsaft_cache_stats,
    epcsaft_liquid_transport_state,
)
from mea_absorption_column.Thermodynamics.reactive_bundle import validate_reactive_bundle  # noqa: E402
from mea_absorption_column.Transport.Reactive_Film import (  # noqa: E402
    FilmThermodynamicState,
    ReactiveFilmSolveError,
    solve_reactive_film,
)


FINAL = ROOT / "analyses/nccc_validation/results/final"
IDENTITY = ROOT / "analyses/nccc_validation/inputs/issue16_reactive_film_identity.json"
RUN_TABLE = FINAL / "tables/issue16_exact_reactive_film_runs.csv"
PROFILE_TABLE = FINAL / "tables/issue16_exact_reactive_film_profile.csv"
SUMMARY = FINAL / "tables/issue16_exact_reactive_film_summary.json"
NUMERICAL_GATE_TABLE = FINAL / "tables/issue16_exact_reactive_film_numerical_gate.csv"
NUMERICAL_GATE_SUMMARY = (
    FINAL / "tables/issue16_exact_reactive_film_numerical_gate_summary.json"
)
DIFFUSIVITY_POLICY = (
    ROOT / "analyses/reactive_film_evidence/inputs/onsager_diffusivity_closure_policy.json"
)
CO2, MEA, H2O = 0, 1, 2
STOICHIOMETRY = np.asarray(
    (
        (-1, -1, -1),
        (-2, -1, 0),
        (0, -1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 0, 1),
        (0, 0, 0),
        (0, 1, 0),
        (0, 0, -1),
    ),
    dtype=float,
)
CHARGES = np.asarray([IONIC_CHARGE_BY_SPECIES[species] for species in SPECIES_9])
CONSERVATION = np.asarray(
    (
        (1, 2, 0, 2, 3, 1, 1, 0, 0),
        (0, 7, 2, 8, 6, 1, 0, 3, 1),
        (0, 1, 0, 1, 1, 0, 0, 0, 0),
        (2, 1, 1, 1, 3, 3, 3, 1, 1),
        (0, 0, 0, 1, -1, -1, -2, 1, -1),
    ),
    dtype=float,
)


class Issue16InputBlocker(RuntimeError):
    code = "work_package_a_inputs_not_admitted"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wheel_path() -> Path:
    direct_url = json.loads(
        importlib.metadata.distribution("epcsaft").read_text("direct_url.json") or "{}"
    )["url"]
    return Path(unquote(urlparse(direct_url).path))


def _raise_case_timeout(_signal_number, _frame) -> None:
    raise TimeoutError("film case exceeded configured time limit")


def _failure_record(error: Exception) -> dict[str, object]:
    if isinstance(error, Issue16InputBlocker):
        return {
            "outcome": "input_preflight_failure",
            "stopped_by": "input_preflight",
            "next_probe": "obtain source-admitted exact-1-or-5 M states, rate coefficients, and diffusion inputs",
            "claim_strength": "boundary_at_state",
        }
    if isinstance(error, TimeoutError):
        return {
            "outcome": "campaign_timeout",
            "stopped_by": "campaign_watchdog",
            "next_probe": "run a separately authorized longer numerical experiment",
            "claim_strength": "not_established",
        }
    if isinstance(error, ReactiveFilmSolveError):
        return {
            "outcome": "numerical_convergence_failure",
            "stopped_by": "solver",
            "next_probe": "inspect the retained solver diagnostic and nearest admitted state",
            "claim_strength": "boundary_at_state",
        }
    return {
        "outcome": "physical_invalidity",
        "stopped_by": "physical_check",
        "next_probe": "inspect the typed provider or domain diagnostic",
        "claim_strength": "boundary_at_state",
    }


def _retained_state(position: float = 1.0) -> tuple[pd.Series, np.ndarray]:
    profile = _load_profile()
    source = profile.iloc[(profile.Position - float(position)).abs().argmin()]
    if abs(float(source.Position) - float(position)) > 1.0e-12:
        raise ValueError(f"Position {position:g} is absent from the retained profile")
    apparent_flows = source[["Fl_CO2", "Fl_MEA", "Fl_H2O"]].to_numpy(float)
    _, composition = bundle_reactive_chemical_equilibrium(
        apparent_flows, float(source.Tl), P=float(source.P), diagnostics={}
    )
    concentrations = composition * float(source.rho_mol_l)
    return source, concentrations


def _load_diffusivity_policy() -> dict:
    policy = json.loads(DIFFUSIVITY_POLICY.read_text(encoding="utf-8"))
    if policy["species_order"] != list(SPECIES_9):
        raise ValueError("retained Onsager diffusivity policy species order changed")
    return policy


def _require_admitted_work_package_a_inputs(
    source: pd.Series, concentrations: np.ndarray
) -> None:
    mea_molarity = float(np.sum(concentrations[[1, 3, 4]]) / 1000.0)
    loading = float(np.sum(concentrations[[0, 4, 5, 6]]) / np.sum(concentrations[[1, 3, 4]]))
    blockers = []
    if not 293.15 <= float(source.Tl) <= 323.15:
        blockers.append(f"temperature {float(source.Tl):.12g} K is outside 293.15--323.15 K")
    if not any(abs(mea_molarity - admitted) <= 1.0e-12 for admitted in (1.0, 5.0)):
        blockers.append(f"MEA molarity {mea_molarity:.12g} mol/L is not exactly 1 or 5 mol/L")
    if not 0.0 <= loading < 0.5:
        blockers.append(f"loading {loading:.12g} is outside [0, 0.5)")
    blockers.extend(
        (
            "F1/F2 coefficients have rejected source-unit consistency and F3 has no admitted primary coefficient",
            "all Work Package A numeric diffusion candidates are rejected",
        )
    )
    raise Issue16InputBlocker("; ".join(blockers))


def _run(
    mesh_points: int,
    initial_flux_factor: float,
    position: float = 1.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    source, bulk = _retained_state(position)
    _require_admitted_work_package_a_inputs(source, bulk)
    temperature = float(source.Tl)
    pressure = float(source.P)
    diffusivity_policy = _load_diffusivity_policy()
    diffusivities = np.asarray(diffusivity_policy["species_diffusivities_m2_s"], dtype=float)

    def thermodynamic_state(
        _concentrations: np.ndarray, composition: np.ndarray
    ) -> FilmThermodynamicState:
        state = epcsaft_liquid_transport_state(temperature, pressure, composition)
        return FilmThermodynamicState(
            state.fugacities_pa,
            state.fixed_other_concentrations_log_fugacity_derivative(CO2),
            state.log_composition_basis,
            state.chemical_potential_derivatives_over_rt,
        )

    bulk_fugacities = epcsaft_liquid_transport_state(
        temperature, pressure, bulk / bulk.sum()
    ).fugacities_pa

    def rate(
        _concentrations: np.ndarray, _composition: np.ndarray, fugacities
    ) -> np.ndarray:
        activity_ratio = fugacities / bulk_fugacities
        return 1.0e-4 * np.asarray(
            (
                activity_ratio[CO2] * activity_ratio[MEA] ** 2
                - activity_ratio[3] * activity_ratio[4],
                activity_ratio[CO2] * activity_ratio[MEA] * activity_ratio[H2O]
                - activity_ratio[4] * activity_ratio[7],
                activity_ratio[CO2] * activity_ratio[8] - activity_ratio[5],
            )
        )

    started = time.perf_counter()
    result = solve_reactive_film(
        bulk_concentrations_mol_m3=bulk,
        diffusivities_m2_s=diffusivities,
        stoichiometry=STOICHIOMETRY,
        conservation_matrix=CONSERVATION,
        charge_numbers=CHARGES,
        liquid_thermodynamic_state=thermodynamic_state,
        net_rate_mol_m3_s=rate,
        vapor_bulk_fugacity_pa=float(source.fv_CO2),
        gas_transfer_coefficient_mol_m2_s_pa=float(source.kv_CO2),
        film_thickness_m=float(source.Dl_CO2 / source.kl_CO2),
        co2_index=CO2,
        mesh_points=mesh_points,
        initial_flux_factor=initial_flux_factor,
        reaction_continuation_steps=16,
        solver_tolerance=1.0e-6,
        transport_model="onsager",
        diffusivity_metadata=diffusivity_policy,
    )
    runtime = time.perf_counter() - started
    flux = float(result.fluxes_mol_m2_s[CO2, 0])
    record = {
        "outcome": "evaluated",
        "attempted": True,
        "stopped_by": "none",
        "next_probe": "none",
        "claim_strength": "result",
        "claim_label": "provisional_concept_only",
        "diagnostic": "",
        "error_type": "",
        "error_code": "",
        "Position": float(source.Position),
        "temperature_K": temperature,
        "pressure_Pa": pressure,
        "mesh_points": mesh_points,
        "initial_flux_factor": initial_flux_factor,
        "wall_time_seconds": runtime,
        "bulk_liquid_CO2_fugacity_Pa": float(
            result.liquid_species_fugacity_pa[CO2, -1]
        ),
        "interface_liquid_CO2_fugacity_Pa": float(
            result.liquid_species_fugacity_pa[CO2, 0]
        ),
        "bulk_vapor_CO2_fugacity_Pa": float(source.fv_CO2),
        "interface_CO2_flux_mol_m2_s": flux,
        "predicted_flux_mol_s_m": flux * float(source.a_eA),
        "retained_column_flux_mol_s_m": float(source.Nl_CO2),
        "maximum_interface_residual": result.maximum_interface_residual,
        "maximum_conservation_residual": result.maximum_conservation_residual,
        "maximum_invariant_source_residual": result.maximum_invariant_source_residual,
        "maximum_electroneutrality_residual": result.maximum_electroneutrality_residual,
        "maximum_zero_current_residual": result.maximum_zero_current_residual,
        "solver_message": result.solver_message,
        "diffusivity_policy_id": result.diffusivity_policy_id,
        "diffusivity_uncertainty_factor": result.diffusivity_uncertainty_factor,
    }
    profile = pd.DataFrame(
        {
            "coordinate_m": result.coordinate_m,
            "claim_label": "provisional_concept_only",
            **{
                f"C_{species}_mol_m3": result.concentrations_mol_m3[index]
                for index, species in enumerate(SPECIES_9)
            },
            **{
                f"f_{species}_Pa": result.liquid_species_fugacity_pa[index]
                for index, species in enumerate(SPECIES_9)
            },
            **{
                f"N_{species}_mol_m2_s": result.fluxes_mol_m2_s[index]
                for index, species in enumerate(SPECIES_9)
            },
            **{
                f"r_{reaction}_mol_m3_s": result.net_rate_mol_m3_s[index]
                for index, reaction in enumerate(("F1", "F2", "F3"))
            },
        }
    )
    return record, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="run mesh and initialization checks"
    )
    parser.add_argument(
        "--numerical-gate",
        action="store_true",
        help="run Positions 0, 0.5, and 1 with the direct gas-film boundary closure",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse evaluated rows from the retained run table",
    )
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=180.0,
        help="retain a typed failed row when one film case exceeds this duration",
    )
    args = parser.parse_args()
    if args.case_timeout_s <= 0.0:
        parser.error("--case-timeout-s must be positive")
    if args.full and args.numerical_gate:
        parser.error("--full and --numerical-gate are separate retained studies")
    if args.numerical_gate:
        cases = [(position, 21, 1.0) for position in (0.0, 0.5, 1.0)]
    elif args.full:
        cases = [(1.0, mesh, factor) for mesh in (21, 42) for factor in (0.5, 1.0, 2.0)]
    else:
        cases = [(1.0, 21, 1.0)]
    run_table = NUMERICAL_GATE_TABLE if args.numerical_gate else RUN_TABLE
    summary_path = NUMERICAL_GATE_SUMMARY if args.numerical_gate else SUMMARY
    retained = {}
    if args.resume and run_table.exists():
        existing = pd.read_csv(run_table)
        retained = {
            (
                float(row.Position),
                int(row.mesh_points),
                float(row.initial_flux_factor),
            ): row.to_dict()
            for _, row in existing.loc[existing.outcome.eq("evaluated")].iterrows()
        }
    rows: list[dict[str, object]] = []
    reference_profile = pd.DataFrame()
    for position, mesh, factor in cases:
        if (position, mesh, factor) in retained:
            rows.append(retained[(position, mesh, factor)])
            continue
        try:
            started = time.perf_counter()
            signal.signal(signal.SIGALRM, _raise_case_timeout)
            signal.setitimer(signal.ITIMER_REAL, args.case_timeout_s)
            row, profile = _run(mesh, factor, position)
            if (
                position == 1.0
                and mesh == max(case[1] for case in cases)
                and factor == 1.0
            ):
                reference_profile = profile
        except Exception as error:
            classification = _failure_record(error)
            row = {
                **classification,
                "diagnostic": f"{type(error).__name__}: {error}",
                "error_type": type(error).__name__,
                "error_code": getattr(
                    error,
                    "code",
                    "case_timeout" if isinstance(error, TimeoutError) else "unclassified",
                ),
                "attempted": True,
                "claim_label": "provisional_concept_only",
                "wall_time_seconds": time.perf_counter() - started,
                "Position": position,
                "mesh_points": mesh,
                "initial_flux_factor": factor,
            }
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        rows.append(row)

    table = pd.DataFrame(rows)
    evaluated = table.loc[table.outcome.eq("evaluated")]
    flux_spread = (
        float(
            evaluated.interface_CO2_flux_mol_m2_s.max()
            / evaluated.interface_CO2_flux_mol_m2_s.min()
            - 1.0
        )
        if len(evaluated) and evaluated.Position.nunique() == 1
        else None
    )
    wheel = _wheel_path()
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    diffusivity_policy = _load_diffusivity_policy()
    bundle = validate_reactive_bundle(str(MEA_THERMODYNAMICS_EPCSAFT_DATASET))["bundle"]
    retained_source, retained_bulk = _retained_state(1.0)
    runtime_state = epcsaft_liquid_transport_state(
        float(retained_source.Tl), float(retained_source.P), retained_bulk / retained_bulk.sum()
    )
    summary = {
        "issue": "https://github.com/tannerpolley/MEA-Absorption-Column/issues/16",
        "species": list(SPECIES_9),
        "claim_label": "provisional_concept_only",
        "reaction_basis": "Work Package A F1/F2/F3 reversible stoichiometry; manufactured relative-fugacity rates are limited to the separate architecture check and are not admitted in physical runs",
        "transport_basis": "isothermal constrained Onsager film with public ePC-SAFT exact fixed-T,P tangent; explicit effective-Fick comparison is separate; compact diffusivity policy is estimated and non-predictive; rejected physical inputs stop at preflight",
        "solver_formulation": "reduced electroneutral and zero-current coordinates with direct gas-film closure and exact CO2-direction boundary derivative",
        "case_count": len(cases),
        "evaluated_case_count": len(evaluated),
        "failed_case_count": int(len(table) - len(evaluated)),
        "profile_status": "generated" if not reference_profile.empty else "not_generated_input_preflight",
        "interface_flux_relative_spread": flux_spread,
        "maximum_interface_residual": float(evaluated.maximum_interface_residual.max())
        if len(evaluated)
        else None,
        "maximum_conservation_residual": float(
            evaluated.maximum_conservation_residual.max()
        )
        if len(evaluated)
        else None,
        "maximum_invariant_source_residual": float(
            evaluated.maximum_invariant_source_residual.max()
        )
        if len(evaluated)
        else None,
        "maximum_electroneutrality_residual": float(
            evaluated.maximum_electroneutrality_residual.max()
        )
        if len(evaluated)
        else None,
        "maximum_zero_current_residual": float(
            evaluated.maximum_zero_current_residual.max()
        )
        if len(evaluated)
        else None,
        "parameter_document_sha256": bundle["parameter_document_sha256"],
        "reaction_system_sha256": identity["bundle"]["reaction_system_sha256"],
        "reactive_bundle_id": bundle["bundle_id"],
        "engine_wheel_sha256": _sha256(wheel),
        "engine_core_sha256": runtime_state.artifact_fingerprint.removeprefix("sha256:"),
        "diffusivity_policy_id": diffusivity_policy["policy_id"],
        "diffusivity_uncertainty_factor": diffusivity_policy["relative_uncertainty_factor"],
        "epcsaft_runtime_counters": epcsaft_cache_stats(),
        "failed_rows": table.loc[
            ~table.outcome.eq("evaluated"),
            [
                "Position",
                "mesh_points",
                "initial_flux_factor",
                "outcome",
                "error_type",
                "error_code",
                "diagnostic",
                "attempted",
                "stopped_by",
                "claim_label",
                "wall_time_seconds",
                "next_probe",
                "claim_strength",
            ],
        ].to_dict(orient="records"),
        "claim_boundary": "Provisional reversible architecture check only; manufactured rate scales and out-of-domain placeholder inputs cannot support column-wide, WWC, kinetic, Maxwell-Stefan, predictive, or manuscript claims.",
    }
    run_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(run_table, index=False)
    if not args.numerical_gate and not reference_profile.empty:
        reference_profile.to_csv(PROFILE_TABLE, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if len(evaluated) != len(cases):
        raise RuntimeError("one or more retained reactive-film cases failed")
    if (
        summary["maximum_interface_residual"] > 1.0e-7
        or summary["maximum_conservation_residual"] > 1.0e-7
        or summary["maximum_invariant_source_residual"] > 1.0e-12
        or summary["maximum_electroneutrality_residual"] > 1.0e-12
        or summary["maximum_zero_current_residual"] > 1.0e-12
    ):
        raise RuntimeError("retained reactive-film residual gate failed")
    if args.full and flux_spread > 5.0e-3:
        raise RuntimeError(
            "retained reactive-film mesh/initialization flux spread exceeded 0.5%"
        )


if __name__ == "__main__":
    main()
