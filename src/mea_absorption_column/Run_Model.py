import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import json
import platform
import sys
from importlib import metadata

from mea_absorption_column.BVP.Methods.Single_Shoot_Solve import DEFAULT_SINGLE_SHOOT_SETTINGS, single_shoot_solve
from mea_absorption_column.BVP.Methods.Scipy_BVP_Solve import DEFAULT_SCIPY_BVP_SETTINGS, scipy_BVP_solve
from mea_absorption_column.BVP.Methods.Segmented_Scipy_BVP_Solve import (
    external_profile_from_stacked_solution,
    segmented_scipy_BVP_solve,
)
from mea_absorption_column.BVP.Methods.Finite_Difference_Solve import (
    DEFAULT_FINITE_DIFFERENCE_SETTINGS,
    finite_difference_solve,
)
from mea_absorption_column.BVP.robust_core import make_solver_diagnostics
from mea_absorption_column.intercooling import build_bed_stack_spec
from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.misc.Save_Run_Outputs import save_run_outputs, write_profile_csvs
from mea_absorption_column.misc.Get_Temperature_Enthalpy import (
    get_liquid_enthalpy, get_vapor_enthalpy, get_liquid_temperature, get_vapor_temperature
)
from mea_absorption_column.misc.Scaling import scaling
from mea_absorption_column.Thermodynamics.thermo_models import MEA_THERMODYNAMICS_EPCSAFT_DATASET, epcsaft_cache_stats

np.set_printoptions(suppress=True)


def run_model(df,
              method='single',
              data_type='mole',
              run=0,
              show_info=False,
              save_run_results=False,
              plot_temperature=False,
              thermo_model='ideal_henry',
              solver_settings=None,
              return_details=False,
              staged_beds='auto',
              intercooler_settings=None,
              ):

    solver_settings_for_run = dict(solver_settings or {})
    inputs, X, case_metadata = convert_data(
        df,
        run=run,
        type=data_type,
        return_metadata=True,
        vapor_composition_mode=solver_settings_for_run.get("vapor_composition_mode", "legacy_ratio"),
        gas_flow_basis=solver_settings_for_run.get("gas_flow_basis", "reported_total_wet"),
    )

    L_G, Fv_T, alpha, w_MEA_unloaded, y_CO2, Tl_z, Tv_0, P, beds = X[:9]

    # Simulate the Absorption Column from start to finish given
    # the inlet concentrations of the top liquid and bottom vapor streams
    method_aliases = {
        'single': 'single',
        'shooting': 'single',
        'collocation': 'scipy-bvp',
        'scipy-bvp': 'scipy-bvp',
        'finite': 'finite',
        'finite-difference': 'finite',
    }
    method = method_aliases.get(method, method)
    beds_count = int(case_metadata["beds"])
    intercoolers_count = int(case_metadata["intercoolers"])
    use_staged_beds = (
        bool(staged_beds)
        if staged_beds != "auto"
        else method == "scipy-bvp" and (beds_count > 1 or intercoolers_count > 0)
    )
    intercooler_settings = dict(intercooler_settings or {})
    requested_intercooler_model = intercooler_settings.get(
        "model",
        solver_settings_for_run.get("intercooler_model", "liquid_temperature_reset"),
    )
    if (
        method == "scipy-bvp"
        and use_staged_beds
        and intercoolers_count > 0
        and requested_intercooler_model == "pumparound_temperature_approach"
        and "thermal_state_mode" not in solver_settings_for_run
    ):
        solver_settings_for_run["thermal_state_mode"] = "temperature"
    thermal_state_mode = solver_settings_for_run.get("thermal_state_mode", "enthalpy")
    if (
        method == "scipy-bvp"
        and not use_staged_beds
        and solver_settings_for_run.get("transform_mode") == "case_bounded_flow_pressure"
    ):
        solver_settings_for_run["transform_mode"] = "positive_flow_pressure"
    if (
        method == "scipy-bvp"
        and not use_staged_beds
        and solver_settings_for_run.get("seed_from_shooting", False)
        and "initial_guess_scaled" not in solver_settings_for_run
    ):
        shooting_seed_settings = {
            **solver_settings_for_run,
            "seed_from_shooting": False,
            "return_internal_profile": True,
            "continuation_stage": "shooting_seed",
            "continuation_path": "shooting->scipy-bvp",
        }
        shooting_seed = run_model(
            df,
            method="single",
            data_type=data_type,
            run=run,
            show_info=False,
            save_run_results=False,
            plot_temperature=False,
            thermo_model=thermo_model,
            solver_settings=shooting_seed_settings,
            return_details=True,
            staged_beds=False,
            intercooler_settings=intercooler_settings,
        )
        if shooting_seed.get("success") and shooting_seed.get("_raw_solution_scaled") is not None:
            solver_settings_for_run["initial_guess_scaled"] = shooting_seed["_raw_solution_scaled"]
            solver_settings_for_run["continuation_stage"] = "shooting_seeded_scipy_bvp"
            solver_settings_for_run["continuation_path"] = "shooting->scipy-bvp"

    if method == 'single':
        solving_function = single_shoot_solve
    elif method == 'scipy-bvp':
        solving_function = scipy_BVP_solve
    elif method == 'finite':
        solving_function = finite_difference_solve
    else:
        raise ValueError('Wrong method chosen, choose from the available')

    Fl_b, Fv_a, Tl_b, Tv_a, z, H, A, P, packing = inputs

    Fl_CO2_b, Fl_MEA_b, Fl_H2O_b = Fl_b
    Fv_CO2_a, Fv_H2O_a, Fv_N2_a, Fv_O2_a = Fv_a

    Fl_MEA_a = Fl_MEA_b
    Fv_N2_b, Fv_O2_b = Fv_N2_a, Fv_O2_a

    # Guesses
    CO2_cap_guess = float(solver_settings_for_run.get("co2_capture_guess_pct", 95.0)) # Guess for the percentage of CO2 transferred from Vapor to Liquid
    H2O_cap_guess = float(solver_settings_for_run.get("h2o_capture_guess_pct", -100.0)) # Guess for the percentage of H2O transferred from Vapor to Liquid

    Fv_CO2_b_guess = (1 - (CO2_cap_guess/100))*Fv_CO2_a # 0.000126993312499947
    Fv_H2O_b_guess = (1 - (H2O_cap_guess/100))*Fv_H2O_a # 5
    Tv_b_guess = 335.

    Fl_CO2_a_guess = Fl_CO2_b + (Fv_CO2_a - Fv_CO2_b_guess) # 3.55511974035339
    Fl_H2O_a_guess = Fl_H2O_b + (Fv_H2O_a - Fv_H2O_b_guess) # 55.2093581436551
    Tl_a_guess = 325.

    # Convert from Temperature to Enthalpy

    Fl_a_guess = [Fl_CO2_a_guess, Fl_MEA_a, Fl_H2O_a_guess]
    Hlt_a_guess = get_liquid_enthalpy(Fl_a_guess, Tl_a_guess)
    Hlf_a_guess = Hlt_a_guess * sum(Fl_a_guess)

    Hlt_b = get_liquid_enthalpy(Fl_b, Tl_b)
    Hlf_b = Hlt_b * sum(Fl_b)

    Hvt_a = get_vapor_enthalpy(Fv_a, Tv_a)
    Hvf_a = Hvt_a * sum(Fv_a)

    Fv_b_guess = [Fv_CO2_b_guess, Fv_H2O_b_guess, Fv_N2_b, Fv_O2_b]
    Hvt_b_guess = get_vapor_enthalpy(Fv_b_guess, Tv_b_guess)
    Hvf_b_guess = Hvt_b_guess * sum(Fv_b_guess)

    P_a = P
    P_b = P

    # Scaling

    if thermal_state_mode == "temperature":
        thermal_a = Tl_a_guess
        thermal_vapor_a = Tv_a
        thermal_b = Tl_b
        thermal_vapor_b = Tv_b_guess
    else:
        thermal_a = Hlf_a_guess
        thermal_vapor_a = Hvf_a
        thermal_b = Hlf_b
        thermal_vapor_b = Hvf_b_guess

    Y_a_unscaled = np.array([Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a,
                           thermal_a, thermal_vapor_a, P_a])

    scales = scaling(z, Y_a_unscaled)
    if thermal_state_mode == "temperature":
        scales[4] = 400.0
        scales[5] = 400.0

    Y_a_scaled = np.array([Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a,
                           thermal_a, thermal_vapor_a, P_a]) / scales

    Y_b_scaled = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess,
                           thermal_b, thermal_vapor_b, P_b]) / scales
    eq_scales = scales

    const_flow = Fl_MEA_b, Fv_N2_a, Fv_O2_a

    solver_diagnostics = make_solver_diagnostics()
    epcsaft_cache_start = epcsaft_cache_stats()
    guard_rhs = bool(solver_settings_for_run.get('guard_rhs', True))
    model_options = {
        'thermo_model': thermo_model,
        'chemical_equilibrium_model': solver_settings_for_run.get(
            'chemical_equilibrium_model',
            _default_chemical_equilibrium_model(thermo_model),
        ),
        'co2_mass_transfer_model': solver_settings_for_run.get(
            'co2_mass_transfer_model', 'enhancement_factor'
        ),
        'reactive_film_linearization': solver_settings_for_run.get('reactive_film_linearization'),
        'solver_diagnostics': solver_diagnostics,
        'guard_invalid_states': guard_rhs,
        'strict_domain_guards': bool(solver_settings_for_run.get('strict_domain_guards', guard_rhs)),
        'mass_transfer_factor': float(solver_settings_for_run.get('mass_transfer_factor', 1.0)),
        'heat_transfer_factor': float(solver_settings_for_run.get('heat_transfer_factor', 1.0)),
        'eta_psi': float(solver_settings_for_run.get('eta_psi', 1.0)),
        'thermal_state_mode': thermal_state_mode,
        'co2_flux_mode': solver_settings_for_run.get('co2_flux_mode', 'bidirectional'),
        'epcsaft_fugacity_blend': float(solver_settings_for_run.get('epcsaft_fugacity_blend', 1.0)),
        'vapor_composition_mode': case_metadata.get('vapor_composition_mode', 'legacy_ratio'),
        'gas_flow_basis': case_metadata.get('gas_flow_basis', 'reported_total_wet'),
        'gas_velocity_area_exponent': float(solver_settings_for_run.get('gas_velocity_area_exponent', 0.0) or 0.0),
        'gas_velocity_area_reference_m_s': solver_settings_for_run.get('gas_velocity_area_reference_m_s'),
        'gas_velocity_area_bounds': solver_settings_for_run.get('gas_velocity_area_bounds', (0.1, 3.0)),
    }
    solver_diagnostics["_strict_domain_guards"] = bool(model_options["strict_domain_guards"])
    parameters = scales, eq_scales, const_flow, H, A, packing, model_options
    stack_spec = build_bed_stack_spec(
        beds=beds_count if use_staged_beds else 1,
        intercoolers=intercoolers_count if use_staged_beds else 0,
        single_bed_height_m=case_metadata["single_bed_height_m"],
        liquid_feed_temperature_K=Tl_z,
        target_temperatures_K=intercooler_settings.get("target_temperatures_K"),
        intercooler_strength=float(intercooler_settings.get("strength", solver_settings_for_run.get("intercooler_strength", 1.0))),
        intercooler_model=requested_intercooler_model,
    )
    if (
        method == "scipy-bvp"
        and use_staged_beds
        and thermo_model == "ideal_henry"
        and solver_settings_for_run.get("seed_from_collapsed", False)
        and "initial_guess_scaled" not in solver_settings_for_run
    ):
        collapsed_seed_settings = {
            **solver_settings_for_run,
            "seed_from_collapsed": False,
            "return_internal_profile": True,
            "continuation_stage": "collapsed_henry_seed",
            "continuation_path": "collapsed_henry",
        }
        collapsed_seed = run_model(
            df,
            method=method,
            data_type=data_type,
            run=run,
            show_info=False,
            save_run_results=False,
            plot_temperature=False,
            thermo_model="ideal_henry",
            solver_settings=collapsed_seed_settings,
            return_details=True,
            staged_beds=False,
            intercooler_settings=intercooler_settings,
        )
        if collapsed_seed.get("success") and collapsed_seed.get("_raw_solution_scaled") is not None:
            solver_settings_for_run["initial_guess_scaled"] = collapsed_seed["_raw_solution_scaled"]
            solver_settings_for_run.setdefault("continuation_stage", "staged_from_collapsed_henry")
            solver_settings_for_run.setdefault("continuation_path", "collapsed_henry->staged_henry")
    if (
        method == "scipy-bvp"
        and use_staged_beds
        and thermo_model == "epcsaft_neutral"
        and solver_settings_for_run.get("seed_from_henry", True)
        and "initial_guess_scaled" not in solver_settings_for_run
    ):
        seed_settings = {
            **solver_settings_for_run,
            "seed_from_henry": False,
            "return_internal_profile": True,
        }
        henry_seed = run_model(
            df,
            method=method,
            data_type=data_type,
            run=run,
            show_info=False,
            save_run_results=False,
            plot_temperature=False,
            thermo_model="ideal_henry",
            solver_settings=seed_settings,
            return_details=True,
            staged_beds=use_staged_beds,
            intercooler_settings=intercooler_settings,
        )
        if henry_seed.get("success") and henry_seed.get("_raw_solution_scaled") is not None:
            solver_settings_for_run["initial_guess_scaled"] = henry_seed["_raw_solution_scaled"]
            solver_settings_for_run["continuation_stage"] = "henry_seeded_epcsaft"
            solver_settings_for_run["continuation_path"] = "ideal_henry->epcsaft_neutral"

    if show_info:
        print(f'''
Run #{run + 1:03d}:
    L/G Ratio: {L_G:.2f}, alpha: {alpha:.2f}, y_CO2: {y_CO2:.2f}
              ''')

    # Starts the time tracker for the total computation time for one simulation run
    start = time.time()

    if method == "scipy-bvp" and use_staged_beds:
        Y_scaled, z_new, solving_type, success, message = segmented_scipy_BVP_solve(
            Y_a_scaled,
            Y_b_scaled,
            z,
            parameters,
            stack_spec=stack_spec,
            settings=solver_settings_for_run,
        )
    else:
        Y_scaled, z_new, solving_type, success, message = solving_function(
            Y_a_scaled, Y_b_scaled, z, parameters, settings=solver_settings_for_run
        )

    raw_Y_scaled = np.array(Y_scaled)
    if use_staged_beds and raw_Y_scaled.shape[0] > 7:
        Y_scaled_for_outputs = external_profile_from_stacked_solution(raw_Y_scaled, stack_spec.beds)
    else:
        Y_scaled_for_outputs = raw_Y_scaled
    z_outputs = np.linspace(z[0], z[-1], Y_scaled_for_outputs.shape[1])

    Y = []
    for i in range(len(Y_scaled_for_outputs)):
        Y.append(Y_scaled_for_outputs[i] * scales[i])
    Y = np.array(Y)

    # Ends the time tracker for the total computation time for one simulation run
    end = time.time()
    total_time = end - start

    # Collects data from the final integration output

    # CO2
    Fl_CO2_a_sim = Y[0, 0]
    Fl_CO2_b_sim = Y[0, -1]
    Fv_CO2_a_sim = Y[2, 0]
    Fv_CO2_b_sim = Y[2, -1]

    # H2O
    Fl_H2O_a_sim = Y[1, 0]
    Fl_H2O_b_sim = Y[1, -1]
    Fv_H2O_a_sim = Y[3, 0]
    Fv_H2O_b_sim = Y[3, -1]

    Fl_a = [Fl_CO2_a_sim, Fl_MEA_a, Fl_H2O_a_sim]
    Fl_b = [Fl_CO2_b_sim, Fl_MEA_b, Fl_H2O_b_sim]
    x_a = [Fl_a[i] / sum(Fl_a) for i in range(len(Fl_a))]
    x_b = [Fl_b[i] / sum(Fl_b) for i in range(len(Fl_b))]

    Fv_a = [Fv_CO2_a_sim, Fv_H2O_a_sim, Fv_N2_a, Fv_O2_a]
    Fv_b = [Fv_CO2_b_sim, Fv_H2O_b_sim, Fv_N2_b, Fv_O2_b]
    y_a = [Fv_a[i] / sum(Fv_a) for i in range(len(Fv_a))]
    y_b = [Fv_b[i] / sum(Fv_b) for i in range(len(Fv_b))]

    # Temperature
    if thermal_state_mode == "temperature":
        Tl_a_sim = Y[4, 0]
        Tl_b_sim = Y[4, -1]
        Tv_a_sim = Y[5, 0]
        Tv_b_sim = Y[5, -1]
    else:
        Hl_a_sim = Y[4, 0] / sum(Fl_a)
        Hl_b_sim = Y[4, -1] / sum(Fl_b)
        Hv_a_sim = Y[5, 0] / sum(Fv_a)
        Hv_b_sim = Y[5, -1] / sum(Fv_b)

        Tl_a_sim = (get_liquid_temperature(x_a, Hl_a_sim))
        Tl_b_sim = (get_liquid_temperature(x_b, Hl_b_sim))
        Tv_a_sim = (get_vapor_temperature(y_a, Hv_a_sim))
        Tv_b_sim = (get_vapor_temperature(y_b, Hv_b_sim))

    # Computes the relative error between the solution that the shooter found to the actual inlet concentration for the
    # relevant liquid species
    Fl_CO2_rel_err = abs(Fl_CO2_b - Fl_CO2_b_sim) / Fl_CO2_b * 100
    Fl_H2O_rel_err = abs(Fl_H2O_b - Fl_H2O_b_sim) / Fl_H2O_b * 100
    Tl_rel_err = abs(Tl_b - Tl_b_sim) / Tl_b * 100

    Fv_CO2_rel_err = abs(Fv_CO2_a - Fv_CO2_a_sim) / Fv_CO2_a * 100
    Fv_H2O_rel_err = abs(Fv_H2O_a - Fv_H2O_a_sim) / Fv_H2O_a * 100
    Tv_rel_err = abs(Tv_a - Tv_a_sim) / Tv_a * 100

    # Report capture against the known inlet vapor feed.  Failed BVP iterates can
    # miss the bottom boundary badly; using the simulated inlet in the denominator
    # turns those diagnostics into meaningless trillion-percent captures.
    CO2_cap = (Fv_CO2_a - Fv_CO2_b_sim) / Fv_CO2_a * 100

    # Prints out relevant info such as simulation time, relative errors, CO2% captured, if max iterations were reached,
    # and number of Nan's counted

    if show_info:
        if success:
            result = 'A solution was found'
        else:
            result = 'No solution was found'
        print(
            f'''
    Method: {solving_type} 
    Result: {result}
    Message: {message}
    CO2 % Cap: {CO2_cap:.2f}% 
    Time: {total_time:0>{4}.1f} sec
    Liquid to Vapor Water Ratio: {Fl_H2O_b / Fv_H2O_a:.2f}

    Vapor:
        Boundary Check - % Error: [Simulated, Actual]
        CO2 = {Fv_CO2_rel_err:0>{5}.2f}% [{Fv_CO2_a_sim:.3f}, {Fv_CO2_a:.3f}]
        H2O = {Fv_H2O_rel_err:0>{5}.2f}% [{Fv_H2O_a_sim:.3f}, {Fv_H2O_a:.3f}]
        T  = {Tv_rel_err:0>{5}.2f}% [{Tv_a_sim:.3f}, {Tv_a:.3f}]
        
        Guess Check: [Simulated, Guess]
        CO2: {Fv_CO2_b_sim:.3f} | {Fv_CO2_b_guess:.3f}
        H2O: {Fv_H2O_b_sim:.3f} | {Fv_H2O_b_guess:.3f}
        T: {Tv_b_sim:.3f} | {Tv_b_guess:.3f}
    
    Liquid:
        Boundary Check - % Error: [Simulated, Actual]
        CO2 = {Fl_CO2_rel_err:0>{5}.2f}% [{Fl_CO2_b_sim:.3f}, {Fl_CO2_b:.3f}]
        H2O = {Fl_H2O_rel_err:0>{5}.2f}% [{Fl_H2O_b_sim:.3f}, {Fl_H2O_b:.3f}]
        T  = {Tl_rel_err:0>{5}.2f}% [{Tl_b_sim:.3f}, {Tl_b:.3f}]
        
        Guess Check: [Simulated, Guess]
        CO2: {Fl_CO2_a_sim:.3f} | {Fl_CO2_a_guess:.3f}
        H2O: {Fl_H2O_a_sim:.3f} | {Fl_H2O_a_guess:.3f}
        T: {Tl_a_sim:.3f} | {Tl_a_guess:.3f}
''')

    # Stores output data into text files (concentrations, mole fractions, and temperatures) (can also plot)
    return_profiles = bool(solver_settings_for_run.get("return_profiles"))
    return_internal_profile = bool(solver_settings_for_run.get("return_internal_profile"))
    profile_csv_dir = solver_settings_for_run.get("profile_csv_dir")
    needs_profile_outputs = bool(
        save_run_results
        or plot_temperature
        or return_profiles
        or profile_csv_dir
        or _has_temperature_taps(df)
    )
    output_message_suffix = ""
    if (
        (use_staged_beds and stack_spec.beds > 1 and not save_run_results and not plot_temperature and not return_profiles and not profile_csv_dir)
        or (return_internal_profile and not save_run_results and not plot_temperature and not return_profiles and not profile_csv_dir)
        or (not success and not save_run_results and not plot_temperature and not return_profiles and not profile_csv_dir)
        or (not needs_profile_outputs)
    ):
        dfs_dict = {}
    else:
        try:
            dfs_dict = save_run_outputs(Y_scaled_for_outputs, z_outputs, parameters,
                                  save_run_results=save_run_results,
                                  plot_temperature=plot_temperature,
                                  profile_metadata=case_metadata,
                                  include_coordinate_columns=bool(profile_csv_dir),
                                  )
        except Exception as exc:
            dfs_dict = _fallback_temperature_profile(
                Y,
                z_outputs,
                thermal_state_mode=thermal_state_mode,
                Fl_MEA=Fl_MEA_a,
                Fv_N2=Fv_N2_a,
                Fv_O2=Fv_O2_a,
            )
            if dfs_dict:
                output_message_suffix = (
                    f"; profile output used temperature-only fallback after: {exc}"
                )
            else:
                output_message_suffix = f"; profile output generation failed: {exc}"

    method_key = {
        'single': 'Shooting',
        'scipy-bvp': 'Collocation BVP',
        'finite': 'Finite Difference',
    }

    if plot_temperature:
        dfs_dict['T'].plot(kind='line', y=['Tl', 'Tv'])
        plt.plot(x, df.iloc[run, -5:], 'kx', label='NCCC Data')
        plt.ylabel('Temperature [K]')
        plt.legend()
        plt.title(f'{method_key[method]} Method | Simulation Time: {total_time:.1f} sec \n $\\frac{{L}}{{G}}$ Ratio: {L_G:.2f}, $\\alpha$: {alpha:.2f}, $y_{{CO2}}$: {y_CO2:.2f}, CO2%: {CO2_cap:.2f}')
        plt.show()

    if return_details:
        target_capture = _target_capture_pct(df, run)
        temperature_rmse = _temperature_rmse(df, run, dfs_dict)
        boundary_residual_norm = float(np.linalg.norm([
            Fl_CO2_rel_err,
            Fl_H2O_rel_err,
            Tl_rel_err,
            Fv_CO2_rel_err,
            Fv_H2O_rel_err,
            Tv_rel_err,
        ]))
        boundary_residual_components = {
            "Fl_CO2_pct": float(Fl_CO2_rel_err),
            "Fl_H2O_pct": float(Fl_H2O_rel_err),
            "Tl_pct": float(Tl_rel_err),
            "Fv_CO2_pct": float(Fv_CO2_rel_err),
            "Fv_H2O_pct": float(Fv_H2O_rel_err),
            "Tv_pct": float(Tv_rel_err),
        }
        settings = _effective_solver_settings(method, solver_settings_for_run)
        capture_error_pct = None if target_capture is None else float(CO2_cap - target_capture)
        method_success, gated_message = _apply_method_success_gates(
            method=method,
            solver_success=bool(success),
            message=str(message),
            boundary_residual_norm=boundary_residual_norm,
            capture_error_pct=capture_error_pct,
            settings=solver_settings_for_run,
        )
        cache_stats = _epcsaft_cache_delta(epcsaft_cache_start, epcsaft_cache_stats())
        co2_conservation_relative_residual = abs(
            (Fl_CO2_b + Fv_CO2_a) - (Fl_CO2_a_sim + Fv_CO2_b_sim)
        ) / max(abs(Fl_CO2_b + Fv_CO2_a), np.finfo(float).tiny)
        h2o_conservation_relative_residual = abs(
            (Fl_H2O_b + Fv_H2O_a) - (Fl_H2O_a_sim + Fv_H2O_b_sim)
        ) / max(abs(Fl_H2O_b + Fv_H2O_a), np.finfo(float).tiny)
        result = {
            'case_id': str(df.index[run]),
            'method': method,
            'thermo_model': thermo_model,
            'chemical_equilibrium_model': model_options.get('chemical_equilibrium_model', 'legacy'),
            'co2_mass_transfer_model': model_options['co2_mass_transfer_model'],
            'success': method_success,
            'message': f"{gated_message}{output_message_suffix}",
            'runtime_s': float(total_time),
            'capture_pct': float(CO2_cap),
            'capture_error_pct': capture_error_pct,
            'temperature_rmse_K': temperature_rmse,
            'boundary_residual_norm': boundary_residual_norm,
            'boundary_residual_components': json.dumps(boundary_residual_components, sort_keys=True),
            'co2_conservation_relative_residual': float(co2_conservation_relative_residual),
            'h2o_conservation_relative_residual': float(h2o_conservation_relative_residual),
            'mesh_points': int(settings.get('mesh_points', len(z))),
            'tol': settings.get('tol'),
            'bc_tol': settings.get('bc_tol'),
            'max_nodes': settings.get('max_nodes'),
            'co2_capture_guess_pct': CO2_cap_guess,
            'h2o_capture_guess_pct': H2O_cap_guess,
            'epcsaft_fugacity_blend': float(solver_settings_for_run.get('epcsaft_fugacity_blend', 1.0)),
            'epcsaft_dataset': str(MEA_THERMODYNAMICS_EPCSAFT_DATASET),
            'eta_psi': float(solver_settings_for_run.get('eta_psi', 1.0)),
            'gas_flow_basis': case_metadata.get('gas_flow_basis', 'reported_total_wet'),
            'beds': beds_count,
            'intercoolers': intercoolers_count,
            'staged_beds': bool(use_staged_beds),
            'intercooler_model': stack_spec.model if stack_spec.intercoolers else 'none',
            'thermal_state_mode': thermal_state_mode,
            'intercooler_assumption': (
                f"{stack_spec.assumption};strength={stack_spec.intercoolers[0].strength:g}"
                if stack_spec.intercoolers
                else 'none'
            ),
            'continuation_stage': solver_settings_for_run.get('continuation_stage', 'direct'),
            'continuation_success': bool(method_success and solver_settings_for_run.get('continuation_stage', 'direct') != 'failed'),
            'invalid_state_count': int(solver_diagnostics.get('invalid_state_count', 0)),
            'guard_penalty_count': int(solver_diagnostics.get('guard_penalty_count', 0)),
            'domain_guard_counts': _format_domain_guard_counts(solver_diagnostics.get('domain_guard_counts', {})),
            'first_failed_domain': solver_diagnostics.get('first_failed_domain', ''),
            'jacobian_status': solver_diagnostics.get('jacobian_status', ''),
            'solver_rhs_calls': int(solver_diagnostics.get('solver_rhs_calls', 0)),
            'solver_rhs_node_evaluations': int(solver_diagnostics.get('solver_rhs_node_evaluations', 0)),
            'solver_boundary_calls': int(solver_diagnostics.get('solver_boundary_calls', 0)),
            'solver_jacobian_calls': int(solver_diagnostics.get('solver_jacobian_calls', 0)),
            'solver_iterations': int(solver_diagnostics.get('solver_iterations', 0)),
            'solver_final_nodes': int(solver_diagnostics.get('solver_final_nodes', 0)),
            'solver_mesh_nodes_added': int(solver_diagnostics.get('solver_mesh_nodes_added', 0)),
            'solver_max_rms_residual': float(solver_diagnostics.get('solver_max_rms_residual', float('nan'))),
            'dense_grid_points': int(solver_diagnostics.get('dense_grid_points', 0)),
            'dense_ode_residual_max': float(solver_diagnostics.get('dense_ode_residual_max', float('nan'))),
            'dense_boundary_residual_max': float(solver_diagnostics.get('dense_boundary_residual_max', float('nan'))),
            'scaling_mode': solver_settings_for_run.get('scaling_mode', 'legacy_flow_enthalpy'),
            'transform_mode': solver_settings_for_run.get('transform_mode', 'bounded_guarded_raw_state'),
            'continuation_path': solver_settings_for_run.get('continuation_path', 'none'),
            'epcsaft_cache_hits': int(cache_stats.get('epcsaft_cache_hits', 0)),
            'epcsaft_cache_misses': int(cache_stats.get('epcsaft_cache_misses', 0)),
            'epcsaft_direct_density_solve_s': float(cache_stats.get('epcsaft_direct_density_solve_s', 0.0)),
            'epcsaft_rho_guess_hits': int(cache_stats.get('epcsaft_rho_guess_hits', 0)),
            'epcsaft_rho_guess_misses': int(cache_stats.get('epcsaft_rho_guess_misses', 0)),
            'epcsaft_chemistry_cache_hits': int(solver_diagnostics.get('epcsaft_chemistry_cache_hits', 0)),
            'epcsaft_chemistry_cache_misses': int(solver_diagnostics.get('epcsaft_chemistry_cache_misses', 0)),
            'epcsaft_chemistry_solve_s': float(solver_diagnostics.get('epcsaft_chemistry_solve_s', 0.0)),
            'epcsaft_chemistry_max_mass_residual': float(solver_diagnostics.get('epcsaft_chemistry_max_mass_residual', 0.0)),
            'epcsaft_chemistry_max_reaction_residual': float(solver_diagnostics.get('epcsaft_chemistry_max_reaction_residual', 0.0)),
            'epcsaft_chemistry_max_charge_residual': float(solver_diagnostics.get('epcsaft_chemistry_max_charge_residual', 0.0)),
            'epcsaft_chemistry_accepted_best_effort_count': int(solver_diagnostics.get('epcsaft_chemistry_accepted_best_effort_count', 0)),
            'epcsaft_chemistry_failed_count': int(solver_diagnostics.get('epcsaft_chemistry_failed_count', 0)),
            'epcsaft_chemistry_last_iterations': int(solver_diagnostics.get('epcsaft_chemistry_last_iterations', 0)),
            'epcsaft_chemistry_last_native_success': bool(solver_diagnostics.get('epcsaft_chemistry_last_native_success', False)),
            'epcsaft_chemistry_last_message': solver_diagnostics.get('epcsaft_chemistry_last_message', ''),
            'epcsaft_chemistry_table_hits': int(solver_diagnostics.get('epcsaft_chemistry_table_hits', 0)),
            'epcsaft_chemistry_interpolation_fallback_count': int(
                solver_diagnostics.get('epcsaft_chemistry_interpolation_fallback_count', 0)
            ),
            'epcsaft_chemistry_max_mea_mass_fraction_deviation': float(
                solver_diagnostics.get('epcsaft_chemistry_max_mea_mass_fraction_deviation', 0.0)
            ),
            'profile_png': solver_settings_for_run.get('profile_png', ''),
            'profile_csv_dir': '',
            'profile_csv_status': '',
            'profile_csv_files': '',
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'package_versions': _package_versions(),
            '_case_metadata': case_metadata,
        }
        if profile_csv_dir:
            profile_status = "clean" if method_success else "diagnostic"
            if dfs_dict:
                export_metadata = {
                    **case_metadata,
                    "case_source": solver_settings_for_run.get("case_source", ""),
                    "method": method,
                    "thermo_model": thermo_model,
                    "success": bool(method_success),
                    "message": result["message"],
                    "runtime_s": result["runtime_s"],
                    "runtime_label": _format_runtime_label(result["runtime_s"]),
                    "eta_psi": result["eta_psi"],
                    "intercooler_model": result["intercooler_model"],
                    "intercooler_assumption": result["intercooler_assumption"],
                    "profile_status": profile_status,
                    "position_orientation": "global_normalized_bottom_to_top",
                }
                result.update(write_profile_csvs(dfs_dict, profile_csv_dir, export_metadata))
            else:
                result.update(
                    {
                        "profile_csv_dir": str(profile_csv_dir),
                        "profile_csv_status": "empty",
                        "profile_csv_files": "",
                    }
                )
        if return_internal_profile:
            result["_raw_solution_scaled"] = raw_Y_scaled
        if return_profiles:
            result["_profiles"] = dfs_dict
        return result

    return CO2_cap, success
    # return CO2_cap


def _target_capture_pct(df, run):
    for column in ('CO2 %', 'CO2  %'):
        if column in df.columns:
            value = df.iloc[run][column]
            return None if value is None or np.isnan(value) else float(value)
    return None


def _temperature_rmse(df, run, dfs_dict):
    tap_columns = [column for column in df.columns if _is_temperature_tap(column)]
    if not tap_columns or 'T' not in dfs_dict or 'Tl' not in dfs_dict['T']:
        return None
    positions = np.asarray([float(column) for column in tap_columns], dtype=float)
    observed = df.iloc[run][tap_columns].astype(float).to_numpy()
    profile = dfs_dict['T']['Tl'].sort_index()
    predicted = np.interp(positions, profile.index.to_numpy(dtype=float), profile.to_numpy(dtype=float))
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def _has_temperature_taps(df):
    return any(_is_temperature_tap(column) for column in df.columns)


def _fallback_temperature_profile(Y, z, thermal_state_mode, Fl_MEA, Fv_N2, Fv_O2):
    temperatures = []
    for i in range(Y.shape[1]):
        if thermal_state_mode == "temperature":
            Tl = _finite_or_nan(Y[4, i])
            Tv = _finite_or_nan(Y[5, i])
        else:
            Fl = np.asarray([Y[0, i], Fl_MEA, Y[1, i]], dtype=float)
            Fv = np.asarray([Y[2, i], Y[3, i], Fv_N2, Fv_O2], dtype=float)
            try:
                x = Fl / np.sum(Fl)
                Hl = float(Y[4, i]) / float(np.sum(Fl))
                Tl = _finite_or_nan(get_liquid_temperature(x, Hl))
            except Exception:
                Tl = np.nan
            try:
                y = Fv / np.sum(Fv)
                Hv = float(Y[5, i]) / float(np.sum(Fv))
                Tv = _finite_or_nan(get_vapor_temperature(y, Hv))
            except Exception:
                Tv = np.nan
        temperatures.append({"Tl": Tl, "Tv": Tv})
    profile = pd.DataFrame(temperatures, index=np.asarray(z, dtype=float))
    profile.index.name = "Position"
    if not np.isfinite(profile[["Tl", "Tv"]].to_numpy(dtype=float)).any():
        return {}
    return {"T": profile}


def _finite_or_nan(value):
    value = float(np.asarray(value, dtype=float).reshape(-1)[0])
    return value if np.isfinite(value) else np.nan


def _is_temperature_tap(column):
    try:
        float(column)
        return True
    except (TypeError, ValueError):
        return False


def _package_versions():
    names = ['mea-absorption-column', 'numpy', 'pandas', 'scipy', 'matplotlib']
    versions = []
    for name in names:
        try:
            versions.append(f'{name}={metadata.version(name)}')
        except metadata.PackageNotFoundError:
            versions.append(f'{name}=uninstalled')
    return ';'.join(versions)


def _format_runtime_label(runtime_s):
    runtime_s = float(runtime_s)
    if runtime_s < 60.0:
        return f"{runtime_s:.2f} s"
    minutes, seconds = divmod(runtime_s, 60.0)
    return f"{int(minutes)} min {seconds:.1f} s"


def _format_domain_guard_counts(counts):
    if not counts:
        return ""
    return ";".join(f"{key}={int(value)}" for key, value in sorted(counts.items()))


def _epcsaft_cache_delta(start, end):
    return {
        key: end.get(key, 0) - start.get(key, 0)
        for key in {
            "epcsaft_cache_hits",
            "epcsaft_cache_misses",
            "epcsaft_direct_density_solve_s",
            "epcsaft_rho_guess_hits",
            "epcsaft_rho_guess_misses",
        }
    }


def _default_chemical_equilibrium_model(thermo_model):
    normalized = (thermo_model or "").lower()
    if normalized in {
        "epcsaft_reactive_six",
        "epcsaft_reactive_six_concentration",
        "epcsaft_reactive_six_activity",
        "epcsaft_reactive_six_activity_converted",
        "epcsaft_reactive_six_activity_rebased",
        "epcsaft_reactive_nine",
        "epcsaft_reactive_nine_activity",
        "epcsaft_reactive_nine_activity_converted",
        "epcsaft_reactive_nine_activity_rebased",
        "epcsaft_reactive_nine_tabulated",
        "epcsaft_full_species_activity",
        "epcsaft_full_species_activity_converted",
        "epcsaft_full_species_activity_rebased",
    }:
        return normalized
    return "legacy"


def _apply_method_success_gates(
    method,
    solver_success,
    message,
    boundary_residual_norm,
    capture_error_pct,
    settings,
):
    method_success = bool(solver_success)
    gate_messages = []
    boundary_rejected = False
    if boundary_residual_norm > float(settings.get("success_boundary_residual_max", 1.0)):
        method_success = False
        boundary_rejected = True
        gate_messages.append(f"Rejected by strict boundary residual gate: {boundary_residual_norm:.6g}")
    if (
        method in {"single", "finite"}
        and capture_error_pct is not None
        and abs(float(capture_error_pct)) > float(settings.get("success_capture_error_max_pct", 10.0))
    ):
        method_success = False
        gate_messages.append(f"Rejected by strict capture gate: {capture_error_pct:.6g} pct")
    if (
        method == "scipy-bvp"
        and "success_capture_error_max_pct" in settings
        and capture_error_pct is not None
        and abs(float(capture_error_pct)) > float(settings["success_capture_error_max_pct"])
    ):
        method_success = False
        gate_messages.append(f"Rejected by collocation capture gate: {capture_error_pct:.6g} pct")
    if (
        method == "scipy-bvp"
        and not method_success
        and not boundary_rejected
        and "max_runtime_s" not in str(message)
        and settings.get("accept_low_residual_final_iterate", True)
        and boundary_residual_norm <= float(settings.get("accept_boundary_residual_max", 10.0))
        and (
            capture_error_pct is None
            or abs(float(capture_error_pct))
            <= float(
                settings.get(
                    "accept_capture_error_max_pct",
                    settings.get("success_capture_error_max_pct", 5.0),
                )
            )
        )
    ):
        method_success = True
        gate_messages.append(
            "Accepted low-residual collocation final iterate despite solver status"
        )
    if gate_messages:
        return method_success, f"{message}; {'; '.join(gate_messages)}"
    return method_success, message


def _effective_solver_settings(method, solver_settings):
    defaults = {
        'single': DEFAULT_SINGLE_SHOOT_SETTINGS,
        'scipy-bvp': DEFAULT_SCIPY_BVP_SETTINGS,
        'finite': DEFAULT_FINITE_DIFFERENCE_SETTINGS,
    }.get(method, {})
    return {**defaults, **(solver_settings or {})}
