from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    epcsaft_cache_stats,
    epcsaft_phi_co2,
    ionic_liquid_composition,
)
from mea_absorption_column.Transport.Reactive_Film import (  # noqa: E402
    ReactiveFilmSolveError,
    solve_reactive_film,
)


FINAL = ROOT / "analyses/nccc_validation/results/final"
RUN_TABLE = FINAL / "tables/retained_reactive_case3c_film_runs.csv"
PROFILE_TABLE = FINAL / "tables/retained_reactive_case3c_film_profile.csv"
SUMMARY = FINAL / "tables/retained_reactive_case3c_film_summary.json"
NUMERICAL_GATE_TABLE = FINAL / "tables/retained_reactive_case3c_film_numerical_gate.csv"
NUMERICAL_GATE_SUMMARY = FINAL / "tables/retained_reactive_case3c_film_numerical_gate_summary.json"
PARAMETERS = Path(MEA_THERMODYNAMICS_EPCSAFT_DATASET) / "parameters.json"
CO2, MEA, H2O = 0, 1, 2
STOICHIOMETRY = np.asarray((-1, -2, 0, 1, 1, 0, 0, 0, 0), dtype=float)
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


def _retained_state(position: float = 1.0) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    profile = _load_profile()
    source = profile.iloc[(profile.Position - float(position)).abs().argmin()]
    if abs(float(source.Position) - float(position)) > 1.0e-12:
        raise ValueError(f"Position {position:g} is absent from the retained profile")
    apparent_flows = source[["Fl_CO2", "Fl_MEA", "Fl_H2O"]].to_numpy(float)
    _, composition = tabulated_epcsaft_reactive_chemical_equilibrium(
        apparent_flows, float(source.Tl), diagnostics={}
    )
    concentrations = composition * float(source.rho_mol_l)
    diffusivities = np.asarray(
        [source.Dl_CO2, source.Dl_MEA, source.Dl_MEA, *([source.Dl_ion] * 6)],
        dtype=float,
    )
    return source, concentrations, diffusivities


def _run(
    mesh_points: int,
    initial_flux_factor: float,
    position: float = 1.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    source, bulk, diffusivities = _retained_state(position)
    temperature = float(source.Tl)
    pressure = float(source.P)

    def fugacity(_concentrations: np.ndarray, composition: np.ndarray) -> float:
        ionic = ionic_liquid_composition(composition)
        phi_co2 = epcsaft_phi_co2(
            temperature,
            pressure,
            ionic,
            phase="liq",
            cache=False,
            mixture_kind="ionic",
        )
        return float(ionic[CO2] * phi_co2 * pressure)

    k_mea = 2.003e4 * math.exp(-4742.0 / temperature)
    k_water = 4.147 * math.exp(-3110.0 / temperature)

    def rate(concentrations: np.ndarray, *_unused) -> float:
        k2 = k_mea * concentrations[MEA] + k_water * concentrations[H2O]
        return float(k2 * concentrations[MEA] * concentrations[CO2])

    started = time.perf_counter()
    result = solve_reactive_film(
        bulk_concentrations_mol_m3=bulk,
        diffusivities_m2_s=diffusivities,
        stoichiometry=STOICHIOMETRY,
        conservation_matrix=CONSERVATION,
        liquid_co2_fugacity_pa=fugacity,
        net_rate_mol_m3_s=rate,
        vapor_bulk_fugacity_pa=float(source.fv_CO2),
        gas_transfer_coefficient_mol_m2_s_pa=float(source.kv_CO2),
        film_thickness_m=float(source.Dl_CO2 / source.kl_CO2),
        co2_index=CO2,
        mesh_points=mesh_points,
        initial_flux_factor=initial_flux_factor,
        reaction_continuation_steps=16,
        solver_tolerance=1.0e-6,
    )
    runtime = time.perf_counter() - started
    flux = float(result.fluxes_mol_m2_s[CO2, 0])
    record = {
        "outcome": "evaluated",
        "diagnostic": "",
        "Position": float(source.Position),
        "temperature_K": temperature,
        "pressure_Pa": pressure,
        "mesh_points": mesh_points,
        "initial_flux_factor": initial_flux_factor,
        "runtime_s": runtime,
        "bulk_liquid_CO2_fugacity_Pa": float(result.liquid_co2_fugacity_pa[-1]),
        "interface_liquid_CO2_fugacity_Pa": float(result.liquid_co2_fugacity_pa[0]),
        "bulk_vapor_CO2_fugacity_Pa": float(source.fv_CO2),
        "interface_CO2_flux_mol_m2_s": flux,
        "predicted_flux_mol_s_m": flux * float(source.a_eA),
        "retained_column_flux_mol_s_m": float(source.Nl_CO2),
        "maximum_interface_residual": result.maximum_interface_residual,
        "maximum_conservation_residual": result.maximum_conservation_residual,
        "maximum_invariant_source_residual": result.maximum_invariant_source_residual,
        "solver_message": result.solver_message,
    }
    profile = pd.DataFrame(
        {
            "coordinate_m": result.coordinate_m,
            "liquid_CO2_fugacity_Pa": result.liquid_co2_fugacity_pa,
            "CO2_flux_mol_m2_s": result.fluxes_mol_m2_s[CO2],
            "net_carbamate_rate_mol_m3_s": result.net_rate_mol_m3_s,
            **{
                f"C_{species}_mol_m3": result.concentrations_mol_m3[index]
                for index, species in enumerate(SPECIES_9)
            },
        }
    )
    return record, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run mesh and initialization checks")
    parser.add_argument(
        "--numerical-gate",
        action="store_true",
        help="run Positions 0, 0.5, and 1 with the direct gas-film boundary closure",
    )
    parser.add_argument("--resume", action="store_true", help="reuse evaluated rows from the retained run table")
    args = parser.parse_args()
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
            (float(row.Position), int(row.mesh_points), float(row.initial_flux_factor)): row.to_dict()
            for _, row in existing.loc[existing.outcome.eq("evaluated")].iterrows()
        }
    rows: list[dict[str, object]] = []
    reference_profile = pd.DataFrame()
    for position, mesh, factor in cases:
        if (position, mesh, factor) in retained:
            rows.append(retained[(position, mesh, factor)])
            continue
        try:
            row, profile = _run(mesh, factor, position)
            if position == 1.0 and mesh == max(case[1] for case in cases) and factor == 1.0:
                reference_profile = profile
        except Exception as error:
            row = {
                "outcome": "numerical_convergence_failure" if isinstance(error, ReactiveFilmSolveError) else "domain_or_runtime_failure",
                "diagnostic": f"{type(error).__name__}: {error}",
                "Position": position,
                "mesh_points": mesh,
                "initial_flux_factor": factor,
            }
        rows.append(row)

    table = pd.DataFrame(rows)
    evaluated = table.loc[table.outcome.eq("evaluated")]
    flux_spread = (
        float(evaluated.interface_CO2_flux_mol_m2_s.max() / evaluated.interface_CO2_flux_mol_m2_s.min() - 1.0)
        if len(evaluated) and evaluated.Position.nunique() == 1
        else None
    )
    wheel = _wheel_path()
    summary = {
        "issue": "https://github.com/tannerpolley/MEA-Absorption-Column/issues/16",
        "species": list(SPECIES_9),
        "reaction_basis": "Luo pseudo-second-order concentration-basis forward carbamate rate: (k_MEA C_MEA + k_H2O C_H2O) C_MEA C_CO2",
        "transport_basis": "isothermal effective-Fick film; CO2-only interface flux; shared ePC-SAFT liquid fugacity callback",
        "solver_formulation": "direct gas-film flux boundary condition with a CO2-direction hybrid boundary Jacobian",
        "case_count": len(cases),
        "evaluated_case_count": len(evaluated),
        "failed_case_count": int(len(table) - len(evaluated)),
        "interface_flux_relative_spread": flux_spread,
        "maximum_interface_residual": float(evaluated.maximum_interface_residual.max()) if len(evaluated) else math.nan,
        "maximum_conservation_residual": float(evaluated.maximum_conservation_residual.max()) if len(evaluated) else math.nan,
        "maximum_invariant_source_residual": float(evaluated.maximum_invariant_source_residual.max()) if len(evaluated) else math.nan,
        "parameter_document_sha256": _sha256(PARAMETERS),
        "engine_wheel_sha256": _sha256(wheel),
        "reactive_table_sha256": _sha256(REACTIVE_TABLE),
        "reactive_table_summary_sha256": _sha256(REACTIVE_SUMMARY),
        "epcsaft_runtime_counters": epcsaft_cache_stats(),
        "failed_rows": table.loc[~table.outcome.eq("evaluated"), ["Position", "mesh_points", "initial_flux_factor", "outcome", "diagnostic"]].to_dict(orient="records"),
        "claim_boundary": "Three-state numerical closure check only; the forward-only Stage A rate cannot support column-wide, WWC, reversible-kinetics, or Maxwell-Stefan validation claims." if args.numerical_gate else "Single retained Case 3C bulk state and effective-Fick Stage A only; no column-wide, WWC, reversible-kinetics, or Maxwell-Stefan validation claim.",
    }
    run_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(run_table, index=False)
    if not args.numerical_gate and not reference_profile.empty:
        reference_profile.to_csv(PROFILE_TABLE, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if len(evaluated) != len(cases):
        raise RuntimeError("one or more retained reactive-film cases failed")
    if summary["maximum_interface_residual"] > 1.0e-7 or summary["maximum_conservation_residual"] > 1.0e-7:
        raise RuntimeError("retained reactive-film residual gate failed")
    if args.full and flux_spread > 5.0e-3:
        raise RuntimeError("retained reactive-film mesh/initialization flux spread exceeded 0.5%")


if __name__ == "__main__":
    main()
