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
    REACTIVE_SUMMARY,
    REACTIVE_TABLE,
    _load_profile,
)
from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import (  # noqa: E402
    SPECIES_9,
    tabulated_epcsaft_reactive_chemical_equilibrium,
)
from mea_absorption_column.Thermodynamics.thermo_models import (  # noqa: E402
    IONIC_CHARGE_BY_SPECIES,
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    epcsaft_cache_stats,
    epcsaft_liquid_transport_state,
)
from mea_absorption_column.Transport.Reactive_Film import (  # noqa: E402
    FilmThermodynamicState,
    ReactiveFilmSolveError,
    solve_reactive_film,
)


FINAL = ROOT / "analyses/nccc_validation/results/final"
IDENTITY = ROOT / "analyses/nccc_validation/inputs/issue16_provisional_reactive_film_identity.json"
PROBE_TABLE = FINAL / "tables/issue16_provisional_reactive_film_probe.csv"
PROBE_SUMMARY = FINAL / "tables/issue16_provisional_reactive_film_probe_summary.json"
RUN_TABLE = FINAL / "tables/issue16_provisional_reactive_film_runs.csv"
PROFILE_TABLE = FINAL / "tables/issue16_provisional_reactive_film_profile.csv"
SUMMARY = FINAL / "tables/issue16_provisional_reactive_film_summary.json"
NUMERICAL_GATE_TABLE = FINAL / "tables/issue16_provisional_reactive_film_numerical_gate.csv"
NUMERICAL_GATE_SUMMARY = (
    FINAL / "tables/issue16_provisional_reactive_film_numerical_gate_summary.json"
)
PARAMETERS = Path(MEA_THERMODYNAMICS_EPCSAFT_DATASET) / "parameters.json"
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
    if isinstance(error, TimeoutError):
        return {
            "outcome": "campaign_timeout",
            "stopped_by": "campaign_watchdog",
            "next_probe": (
                "inspect the retained public density-anchor diagnostics and the "
                "last branch-certified state reached before timeout"
            ),
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
    _, composition = tabulated_epcsaft_reactive_chemical_equilibrium(
        apparent_flows, float(source.Tl), diagnostics={}
    )
    concentrations = composition * float(source.rho_mol_l)
    return source, concentrations


def _assess_work_package_a_inputs(
    source: pd.Series, concentrations: np.ndarray
) -> dict[str, object]:
    work_package = json.loads(IDENTITY.read_text(encoding="utf-8"))["work_package_a"]
    domain = work_package["domain"]
    mea_molarity = float(np.sum(concentrations[[1, 3, 4]]) / 1000.0)
    loading = float(np.sum(concentrations[[0, 4, 5, 6]]) / np.sum(concentrations[[1, 3, 4]]))
    temperature_ok = domain["temperature_K"][0] <= float(source.Tl) <= domain["temperature_K"][1]
    mea_label_ok = any(
        abs(mea_molarity - admitted) <= 1.0e-12
        for admitted in domain["mea_molarity_mol_L_exact"]
    )
    loading_ok = domain["loading_min_inclusive"] <= loading < domain["loading_max_exclusive"]
    limitations = []
    if not temperature_ok:
        limitations.append(
            f"temperature {float(source.Tl):.12g} K is outside "
            f"{domain['temperature_K'][0]:g}--{domain['temperature_K'][1]:g} K"
        )
    if not mea_label_ok:
        limitations.append(
            f"calculated analytical MEA {mea_molarity:.12g} mol/L does not equal "
            "a discrete 1 or 5 mol/L Work Package A source label"
        )
    if not loading_ok:
        limitations.append(
            f"loading {loading:.12g} is outside "
            f"[{domain['loading_min_inclusive']:g}, {domain['loading_max_exclusive']:g})"
        )
    limitations.extend(
        (work_package["finite_rate_status"], work_package["transport_status"])
    )
    return {
        "temperature_domain_status": "within_common_domain" if temperature_ok else "outside_common_domain",
        "mea_source_label_status": "exact_source_label_match" if mea_label_ok else "source_label_mismatch_unadjudicated_basis",
        "loading_domain_status": "within_common_domain" if loading_ok else "outside_common_domain",
        "f1_f2_coefficient_status": work_package["f1_f2_coefficient_status"],
        "f3_coefficient_status": work_package["f3_coefficient_status"],
        "diffusivity_provenance_status": work_package["transport_status"],
        "scientifically_admissible": False,
        "input_limitations": "; ".join(limitations),
    }


def _run(
    mesh_points: int,
    initial_flux_factor: float,
    position: float = 1.0,
    *,
    source: pd.Series | None = None,
    bulk: np.ndarray | None = None,
    assessment: dict[str, object] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    if source is None or bulk is None:
        source, bulk = _retained_state(position)
    assessment = assessment or _assess_work_package_a_inputs(source, bulk)
    temperature = float(source.Tl)
    pressure = float(source.P)
    thermodynamic_state_requests = 0
    thermodynamic_state_evaluations = 0
    thermodynamic_state_cache: dict[
        tuple[tuple[int, ...], str, bytes], FilmThermodynamicState
    ] = {}
    diffusivities = np.asarray(
        [source.Dl_CO2, source.Dl_MEA, source.Dl_MEA, *([source.Dl_ion] * 6)],
        dtype=float,
    )

    def thermodynamic_state(
        _concentrations: np.ndarray, composition: np.ndarray
    ) -> FilmThermodynamicState:
        nonlocal thermodynamic_state_requests, thermodynamic_state_evaluations
        thermodynamic_state_requests += 1
        key = (composition.shape, composition.dtype.str, composition.tobytes())
        if key in thermodynamic_state_cache:
            return thermodynamic_state_cache[key]
        state = epcsaft_liquid_transport_state(temperature, pressure, composition)
        thermodynamic_state_evaluations += 1
        film_state = FilmThermodynamicState(
            state.fugacities_pa,
            state.fixed_other_concentrations_log_fugacity_derivative(CO2),
            state.log_fugacity_derivative,
        )
        thermodynamic_state_cache[key] = film_state
        return film_state

    bulk_fugacities = thermodynamic_state(bulk, bulk / bulk.sum()).fugacities_pa

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

    def rate_jacobian(
        _concentrations: np.ndarray,
        _composition: np.ndarray,
        fugacities: np.ndarray,
        log_fugacity_jacobian: np.ndarray,
    ) -> np.ndarray:
        activity_ratio = fugacities / bulk_fugacities
        forward_f1 = activity_ratio[CO2] * activity_ratio[MEA] ** 2
        reverse_f1 = activity_ratio[3] * activity_ratio[4]
        forward_f2 = (
            activity_ratio[CO2] * activity_ratio[MEA] * activity_ratio[H2O]
        )
        reverse_f2 = activity_ratio[4] * activity_ratio[7]
        forward_f3 = activity_ratio[CO2] * activity_ratio[8]
        reverse_f3 = activity_ratio[5]
        return 1.0e-4 * np.vstack(
            (
                forward_f1
                * (log_fugacity_jacobian[CO2] + 2.0 * log_fugacity_jacobian[MEA])
                - reverse_f1
                * (log_fugacity_jacobian[3] + log_fugacity_jacobian[4]),
                forward_f2
                * (
                    log_fugacity_jacobian[CO2]
                    + log_fugacity_jacobian[MEA]
                    + log_fugacity_jacobian[H2O]
                )
                - reverse_f2
                * (log_fugacity_jacobian[4] + log_fugacity_jacobian[7]),
                forward_f3
                * (log_fugacity_jacobian[CO2] + log_fugacity_jacobian[8])
                - reverse_f3 * log_fugacity_jacobian[5],
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
        liquid_thermodynamic_derivative_state=thermodynamic_state,
        net_rate_mol_m3_s=rate,
        net_rate_jacobian=rate_jacobian,
        vapor_bulk_fugacity_pa=float(source.fv_CO2),
        gas_transfer_coefficient_mol_m2_s_pa=float(source.kv_CO2),
        film_thickness_m=float(source.Dl_CO2 / source.kl_CO2),
        co2_index=CO2,
        mesh_points=mesh_points,
        initial_flux_factor=initial_flux_factor,
        reaction_continuation_steps=1,
        solver_tolerance=1.0e-6,
    )
    runtime = time.perf_counter() - started
    flux = float(result.fluxes_mol_m2_s[CO2, 0])
    record = {
        "outcome": "evaluated",
        "attempted": True,
        "declared": True,
        "model_reached": True,
        "evaluated": True,
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
        "ode_jacobian": "exact_epcsaft_activity_chain_rule",
        "thermodynamic_state_request_count": thermodynamic_state_requests,
        "thermodynamic_state_evaluation_count": thermodynamic_state_evaluations,
        "thermodynamic_state_cache_hit_count": (
            thermodynamic_state_requests - thermodynamic_state_evaluations
        ),
        "solver_iteration_count": result.solver_iterations,
        "solver_mesh_point_count": result.solver_mesh_points,
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
        **assessment,
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
        run_table = NUMERICAL_GATE_TABLE
        summary_path = NUMERICAL_GATE_SUMMARY
    elif args.full:
        cases = [(1.0, mesh, factor) for mesh in (21, 42) for factor in (0.5, 1.0, 2.0)]
        run_table = RUN_TABLE
        summary_path = SUMMARY
    else:
        cases = [(1.0, 21, 1.0)]
        run_table = PROBE_TABLE
        summary_path = PROBE_SUMMARY
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
        assessment: dict[str, object] = {
            "temperature_domain_status": "not_evaluated",
            "mea_source_label_status": "not_evaluated",
            "loading_domain_status": "not_evaluated",
            "f1_f2_coefficient_status": "not_evaluated",
            "f3_coefficient_status": "not_evaluated",
            "diffusivity_provenance_status": "not_evaluated",
            "scientifically_admissible": False,
            "input_limitations": "input assessment did not complete",
        }
        model_reached = False
        try:
            started = time.perf_counter()
            signal.signal(signal.SIGALRM, _raise_case_timeout)
            signal.setitimer(signal.ITIMER_REAL, args.case_timeout_s)
            source, bulk = _retained_state(position)
            assessment = _assess_work_package_a_inputs(source, bulk)
            model_reached = True
            row, profile = _run(
                mesh,
                factor,
                position,
                source=source,
                bulk=bulk,
                assessment=assessment,
            )
            if (
                args.full
                and position == 1.0
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
                "declared": True,
                "model_reached": model_reached,
                "evaluated": False,
                "scientifically_admissible": False,
                "claim_label": "provisional_concept_only",
                "wall_time_seconds": time.perf_counter() - started,
                "Position": position,
                "mesh_points": mesh,
                "initial_flux_factor": factor,
                **assessment,
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
    summary = {
        "issue": "https://github.com/tannerpolley/MEA-Absorption-Column/issues/16",
        "species": list(SPECIES_9),
        "claim_label": "provisional_concept_only",
        "reaction_basis": "Work Package A F1/F2/F3 reversible stoichiometry with a provisional relative-fugacity rate scale; the rates are not scientifically admitted MEA kinetics",
        "transport_basis": "provisional isothermal effective-Fick film using retained column diffusivities, CO2-only interface flux, and the public ePC-SAFT exact fixed-T,P tangent; rejected source inputs remain scientific-adoption limitations",
        "solver_formulation": "reduced electroneutral and zero-current coordinates with direct gas-film closure, exact CO2-direction boundary derivative, and exact ePC-SAFT activity-rate ODE Jacobian",
        "case_count": len(cases),
        "declared_case_count": int(table.declared.astype(bool).sum()),
        "model_reached_case_count": int(table.model_reached.astype(bool).sum()),
        "evaluated_case_count": len(evaluated),
        "scientifically_admitted_case_count": int(table.scientifically_admissible.astype(bool).sum()),
        "failed_case_count": int(len(table) - len(evaluated)),
        "profile_status": (
            "generated"
            if not reference_profile.empty
            else "not_generated" if args.full else "not_requested"
        ),
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
        "parameter_document_sha256": _sha256(PARAMETERS),
        "engine_wheel_sha256": _sha256(wheel),
        "reactive_table_sha256": _sha256(REACTIVE_TABLE),
        "reactive_table_summary_sha256": _sha256(REACTIVE_SUMMARY),
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
        "claim_boundary": "Provisional numerical method-development result only; manufactured rate scales and scientifically inadmissible retained inputs cannot support column-wide, wetted-wall, kinetic, Maxwell-Stefan, predictive, or manuscript claims.",
    }
    run_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(run_table, index=False)
    if args.full and not reference_profile.empty:
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
