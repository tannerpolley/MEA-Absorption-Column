from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mea_absorption_column.Run_Model import run_model


ROOT = Path(__file__).parents[3]
BUNDLE = ROOT / "src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle"
CASE_FILES = {
    "C": (ROOT / "src/mea_absorption_column/data/C_cases_campaign_inputs.csv", "mole"),
    "K": (ROOT / "src/mea_absorption_column/data/NCCC_2014_model_inputs_mass.csv", "mass"),
}
CHARGES = np.asarray((0, 0, 0, 1, -1, -1, -2, 1, -1), dtype=float)
CO2_COMPONENT = np.asarray((1, 0, 0, 0, 1, 1, 1, 0, 0), dtype=float)
_WORKER_STATE_PATHS = {}


def _self_check():
    from mea_absorption_column.BVP.ABS_Column import _reactive_film_linearized_fluxes

    vapor, liquid = _reactive_film_linearized_fluxes(
        0.5, 1000.0, 75.0, ([0.0, 1.0], [1.0e-6, 2.0e-6], [100.0, 200.0])
    )
    assert abs(liquid - 0.095625) < 1.0e-12 and abs(vapor + liquid) < 1.0e-12


def _central_diffusivities(temperature_k: float) -> np.ndarray:
    co2 = 0.5 * 6.28e-7 * math.exp(-15230.0 / (8.314462618 * temperature_k))
    return np.asarray((co2, 8.8e-10, 8.8e-10, 8.4e-10, 6.8e-10, 6.8e-10,
                       6.8e-10, math.sqrt(3.4e-18), math.sqrt(3.4e-18)))


def _reactive_state(candidate, temperature, pressure, amounts, initial_state=None):
    from mea_absorption_column.Thermodynamics.reactive_bundle import solve_homogeneous_reactive_state

    try:
        return solve_homogeneous_reactive_state(
            str(candidate), temperature, pressure, amounts, initial_state=initial_state
        )
    except Exception as direct_error:
        state = None
        steps = max(1, math.ceil(abs(temperature - 313.15) / 4.0))
        try:
            for continuation_temperature in np.linspace(313.15, temperature, steps + 1):
                state = solve_homogeneous_reactive_state(
                    str(candidate), continuation_temperature, pressure, amounts,
                    initial_state=state,
                )
            return state
        except Exception:
            raise direct_error


def _film_node(job):
    candidate, position, values = job
    import mea_absorption_column.Thermodynamics.thermo_models as thermo
    from mea_absorption_column.Transport.Reactive_Film import EquilibriumManifoldState, solve_equilibrium_manifold_film

    candidate = Path(candidate)
    thermo.MEA_THERMODYNAMICS_EPCSAFT_DATASET = candidate
    thermo.epcsaft_dataset_mixture.cache_clear()
    apparent = np.asarray(values["apparent_composition"], dtype=float)
    temperature = values["temperature_K"]
    pressure = values["pressure_Pa"]
    previous_path = _WORKER_STATE_PATHS.get(float(position), {})
    previous_bulk = previous_path.get(0.0)
    target_loading = float(apparent[0] / apparent[1])
    try:
        bulk = _reactive_state(candidate, temperature, pressure, apparent, previous_bulk)
    except Exception as direct_exc:
        bulk = None
        anchor = min(0.2, target_loading)
        steps = max(1, math.ceil(abs(target_loading - anchor) / 0.01))
        for loading in np.linspace(anchor, target_loading, steps + 1):
            trial = np.asarray((loading * apparent[1], apparent[1], apparent[2]))
            trial /= trial.sum()
            try:
                bulk = _reactive_state(candidate, temperature, pressure, trial, bulk)
            except Exception as exc:
                raise RuntimeError(
                    f"film bulk state failed at z={position:.6g}, T={temperature:.6g} K, "
                    f"loading={target_loading:.6g}: {exc}"
                ) from direct_exc
    accepted = {0.0: bulk}
    manifold_states = {}

    def state(log_loading):
        log_loading = float(log_loading)
        target_key = round(log_loading, 12)
        if target_key in manifold_states:
            return manifold_states[target_key]
        if target_key in previous_path:
            trial = (apparent[0] * math.exp(log_loading), apparent[1], apparent[2])
            try:
                solved = _reactive_state(
                    candidate, temperature, pressure, trial, previous_path[target_key]
                )
                accepted[target_key] = solved
                tangent = thermo.epcsaft_liquid_transport_state(
                    temperature, pressure, solved["composition"]
                )
                manifold_states[target_key] = EquilibriumManifoldState(
                    solved["composition"], solved["density_mol_m3"], tangent.fugacities_pa,
                    solved["chemical_potentials_over_rt"], tangent.log_composition_basis,
                    tangent.chemical_potential_derivatives_over_rt,
                )
                return manifold_states[target_key]
            except Exception:
                pass
        solved = bulk
        steps = max(1, math.ceil(abs(log_loading) / 0.025))
        for point in np.linspace(0.0, log_loading, steps + 1)[1:]:
            key = round(float(point), 12)
            if key in accepted:
                solved = accepted[key]
                continue
            trial = (apparent[0] * math.exp(point), apparent[1], apparent[2])
            try:
                solved = _reactive_state(candidate, temperature, pressure, trial, solved)
            except Exception:
                solved = _reactive_state(candidate, temperature, pressure, trial)
            accepted[key] = solved
        tangent = thermo.epcsaft_liquid_transport_state(temperature, pressure, solved["composition"])
        manifold_states[target_key] = EquilibriumManifoldState(
            solved["composition"], solved["density_mol_m3"], tangent.fugacities_pa,
            solved["chemical_potentials_over_rt"], tangent.log_composition_basis,
            tangent.chemical_potential_derivatives_over_rt,
        )
        return manifold_states[target_key]

    try:
        result = solve_equilibrium_manifold_film(
            state_at_log_loading=state,
            species_diffusivities_m2_s=_central_diffusivities(temperature),
            co2_component_coefficients=CO2_COMPONENT,
            vapor_bulk_fugacity_pa=values["vapor_fugacity_Pa"],
            gas_transfer_coefficient_mol_m2_s_pa=values["kg_mol_m2_s_Pa"],
            film_thickness_m=values["film_thickness_m"],
            co2_index=0,
            charge_numbers=CHARGES,
            quadrature_points=5,
            maximum_quadrature_points=65,
            quadrature_tolerance=1.0e-3,
            profile_points=9,
        )
    except Exception as exc:
        raise RuntimeError(
            f"film solve failed at z={position:.6g}, T={temperature:.6g} K, "
            f"loading={target_loading:.6g}, vapor_fugacity={values['vapor_fugacity_Pa']:.6g} Pa: {exc}"
        ) from exc
    bulk_fugacity = float(state(0.0).fugacities_pa[0])
    _WORKER_STATE_PATHS[float(position)] = accepted
    drive = values["vapor_fugacity_Pa"] - bulk_fugacity
    return {
        "position": position,
        "temperature_K": temperature,
        "loading_mol_CO2_per_mol_MEA": float(apparent[0] / apparent[1]),
        "film_flux_mol_m2_s": result.co2_component_flux_mol_m2_s,
        "film_conductance_mol_m2_s_Pa": result.co2_component_flux_mol_m2_s / drive,
        "reactive_bulk_fugacity_Pa": bulk_fugacity,
        "quadrature_relative_change": result.quadrature_relative_change,
        "maximum_interface_residual": result.maximum_interface_residual,
        "maximum_tangent_directional_error": result.maximum_tangent_directional_error,
        "current_column_flux_mol_m2_s": values["current_column_flux_mol_m2_s"],
    }


def _sample_jobs(candidate: Path, profiles, positions):
    def at(frame, column, z):
        return float(np.interp(z, frame.index.to_numpy(float), frame[column].to_numpy(float)))

    jobs = []
    for z in positions:
        apparent = np.asarray([at(profiles["x"], name, z) for name in ("x_CO2", "x_MEA", "x_H2O")])
        jobs.append((str(candidate), float(z), {
            "apparent_composition": (apparent / apparent.sum()).tolist(),
            "temperature_K": at(profiles["T"], "Tl", z),
            "pressure_Pa": at(profiles["transport"], "P", z),
            "vapor_fugacity_Pa": at(profiles["CO2"], "fv_CO2", z),
            "kg_mol_m2_s_Pa": at(profiles["CO2"], "kv_CO2", z),
            "film_thickness_m": at(profiles["Prop_l"], "Dl_CO2", z) / at(profiles["transport"], "kl_CO2", z),
            "current_column_flux_mol_m2_s": at(profiles["CO2"], "Nl_CO2", z) / at(profiles["CO2"], "a_eA", z),
        }))
    return jobs


def _run_case(case_id, dataframe, data_type, candidate, positions, iterations, workers, relaxation):
    started = time.perf_counter()
    run = list(dataframe.index).index(case_id)
    common = dict(mesh_points=11, tol=0.5, bc_tol=0.001, max_nodes=500, return_profiles=True, return_internal_profile=True)
    baseline = run_model(dataframe, method="scipy-bvp", thermo_model="epcsaft_ionic",
                         data_type=data_type, run=run,
                         return_details=True, solver_settings=common)
    current = baseline
    used_conductance = used_bulk_fugacity = None
    rows = []
    pools = [ProcessPoolExecutor(max_workers=1) for _ in positions[:workers]]
    try:
        for iteration in range(1, iterations + 1):
            jobs = _sample_jobs(candidate, current["_profiles"], positions)
            futures = [pools[index % len(pools)].submit(_film_node, job) for index, job in enumerate(jobs)]
            nodes = []
            for job, future in zip(jobs, futures, strict=True):
                try:
                    node = future.result()
                    node["fresh_worker_retry"] = False
                except Exception:
                    with ProcessPoolExecutor(max_workers=1) as retry_pool:
                        node = retry_pool.submit(_film_node, job).result()
                    node["fresh_worker_retry"] = True
                nodes.append(node)
            conductance = np.asarray([row["film_conductance_mol_m2_s_Pa"] for row in nodes])
            bulk_fugacity = np.asarray([row["reactive_bulk_fugacity_Pa"] for row in nodes])
            conductance_change = 0.0 if used_conductance is None else float(
                np.max(np.abs(conductance - used_conductance) / np.maximum(np.abs(conductance), 1.0e-30))
            )
            bulk_fugacity_change = 0.0 if used_bulk_fugacity is None else float(
                np.max(np.abs(bulk_fugacity - used_bulk_fugacity) / np.maximum(np.abs(bulk_fugacity), 1.0))
            )
            used_conductance = conductance if used_conductance is None else used_conductance + relaxation * (conductance - used_conductance)
            used_bulk_fugacity = bulk_fugacity if used_bulk_fugacity is None else used_bulk_fugacity + relaxation * (bulk_fugacity - used_bulk_fugacity)
            settings = {
                **common,
                "co2_mass_transfer_model": "reactive_film_linearization",
                "reactive_film_linearization": (
                    positions.tolist(), used_conductance.tolist(), used_bulk_fugacity.tolist()
                ),
                "initial_guess_scaled": current["_raw_solution_scaled"],
            }
            previous_capture = current["capture_pct"]
            current = run_model(dataframe, method="scipy-bvp", thermo_model="epcsaft_ionic",
                                data_type=data_type, run=run,
                                return_details=True, solver_settings=settings)
            rows.extend(
                {
                    "case_id": case_id,
                    "outer_iteration": iteration,
                    "column_capture_pct": current["capture_pct"],
                    "column_boundary_residual_norm": current["boundary_residual_norm"],
                    "conductance_change_relative": conductance_change,
                    "bulk_fugacity_change_relative": bulk_fugacity_change,
                    **node,
                }
                for node in nodes
            )
            print(json.dumps({
                "case_id": case_id,
                "outer_iteration": iteration,
                "capture_pct": current["capture_pct"],
                "conductance_change_relative": conductance_change,
                "bulk_fugacity_change_relative": bulk_fugacity_change,
            }), flush=True)
            if not current["success"]:
                break
            if (
                abs(current["capture_pct"] - previous_capture) < 0.05
                and max(conductance_change, bulk_fugacity_change) < 0.02
                and iteration > 1
            ):
                break
    finally:
        for pool in pools:
            pool.shutdown()
    observed_column = next(name for name in dataframe.columns if " ".join(name.split()) == "CO2 %")
    observed = float(dataframe.loc[case_id, observed_column])
    final_rows = [row for row in rows if row["outer_iteration"] == max(
        (item["outer_iteration"] for item in rows), default=0
    )]
    final_conductance_change = max(
        (row["conductance_change_relative"] for row in final_rows), default=float("nan")
    )
    final_bulk_fugacity_change = max(
        (row["bulk_fugacity_change_relative"] for row in final_rows), default=float("nan")
    )
    summary = {
        "case_id": case_id,
        "observed_capture_pct": observed,
        "baseline_capture_pct": baseline["capture_pct"],
        "baseline_capture_error_pp": baseline["capture_pct"] - observed,
        "film_capture_pct": current["capture_pct"],
        "film_capture_error_pp": current["capture_pct"] - observed,
        "capture_change_percentage_points": current["capture_pct"] - baseline["capture_pct"],
        "success": current["success"],
        "message": current["message"],
        "boundary_residual_norm": current["boundary_residual_norm"],
        "boundary_residual_components": current["boundary_residual_components"],
        "outer_iterations": max((row["outer_iteration"] for row in rows), default=0),
        "final_capture_change_pp": abs(current["capture_pct"] - previous_capture),
        "final_conductance_change_relative": final_conductance_change,
        "final_bulk_fugacity_change_relative": final_bulk_fugacity_change,
        "outer_iteration_converged": (
            len({row["outer_iteration"] for row in rows}) > 1
            and abs(current["capture_pct"] - previous_capture) < 0.05
            and max(final_conductance_change, final_bulk_fugacity_change) < 0.02
        ),
        "fresh_worker_retries": sum(row["fresh_worker_retry"] for row in rows),
        "maximum_temperature_K": max(row["temperature_K"] for row in rows),
        "runtime_s": time.perf_counter() - started,
    }
    profile_rows = []
    profiles = current["_profiles"]
    z_grid = profiles["T"].index.to_numpy(float)
    for z in z_grid:
        profile_rows.append({
            "case_id": case_id,
            "position": z,
            "liquid_temperature_K": float(profiles["T"].loc[z, "Tl"]),
            "vapor_temperature_K": float(profiles["T"].loc[z, "Tv"]),
            "vapor_co2_fugacity_Pa": float(profiles["CO2"].loc[z, "fv_CO2"]),
            "liquid_co2_flux_mol_m2_s": float(
                profiles["CO2"].loc[z, "Nl_CO2"] / profiles["CO2"].loc[z, "a_eA"]
            ),
        })
    return summary, rows, profile_rows


def main():
    _self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="+", default=["K18", "K19", "1C", "2C", "3C", "4C", "5C", "6C"])
    parser.add_argument("--film-nodes", type=int, default=5)
    parser.add_argument("--outer-iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--relaxation", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    positions = np.linspace(0.0, 1.0, args.film_nodes)
    summaries, nodes, profile_rows = [], [], []
    for case_id in args.case_ids:
        family = "K" if case_id.startswith("K") else "C"
        case_file, data_type = CASE_FILES[family]
        dataframe = pd.read_csv(case_file, index_col=0)
        summary, case_nodes, case_profile = _run_case(
            case_id, dataframe, data_type, BUNDLE, positions,
            args.outer_iterations, args.workers, args.relaxation
        )
        summaries.append(summary)
        nodes.extend(case_nodes)
        profile_rows.extend(case_profile)
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(summaries).to_csv(args.output_dir / "column_comparison.csv", index=False)
            pd.DataFrame(nodes).to_csv(args.output_dir / "film_nodes.csv", index=False)
            pd.DataFrame(profile_rows).to_csv(args.output_dir / "axial_profiles.csv", index=False)
    if args.output_dir:
        bundle = json.loads((BUNDLE / "bundle.json").read_text(encoding="utf-8"))
        provenance = {
            "parameter_document_sha256": hashlib.sha256((BUNDLE / "parameters.json").read_bytes()).hexdigest(),
            "reaction_system_sha256": hashlib.sha256((BUNDLE / "reaction-system.json").read_bytes()).hexdigest(),
            "parameter_source_commit": bundle["parameter_source_commit"],
            "engine_source_commit": bundle["engine_source_commit"],
            "engine_wheel_sha256": bundle["engine_wheel_sha256"],
            "film_nodes": args.film_nodes,
            "outer_iterations": args.outer_iterations,
            "relaxation": args.relaxation,
        }
        (args.output_dir / "run_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
