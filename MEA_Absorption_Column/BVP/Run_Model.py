import numpy as np
import time
from collections import defaultdict

from MEA_Absorption_Column.BVP.Methods.Single_Shoot_Solve import single_shoot_solve
from MEA_Absorption_Column.BVP.Methods.Multiple_Shoot_Solve_2 import multiple_shoot_solve
from MEA_Absorption_Column.BVP.Methods.Collocation_Solve import orthogonal_collocation
from MEA_Absorption_Column.BVP.Methods.Scipy_BVP_Solve import scipy_BVP_solve
from MEA_Absorption_Column.BVP.Methods.Finite_Difference_Solve import finite_difference_solve

from MEA_Absorption_Column.Convert_Data.Convert_SRP_Data import convert_SRP_data
from MEA_Absorption_Column.Convert_Data.Convert_Data import convert_mass_data, convert_mole_data
from MEA_Absorption_Column.misc.Save_Run_Outputs import save_run_outputs
from MEA_Absorption_Column.misc.Get_Temperature_Enthalpy import get_liquid_enthalpy, get_vapor_enthalpy, \
    get_liquid_temperature

np.set_printoptions(suppress=True)


def run_model(df, method='single', data_type='mass', run=0, show_info=True, save_run_results=True, plot_temperature=False):
    # ---- SRP Data Runs ---
    # Create an input list for the values that are used in the simulation
    if data_type == 'mass':
        X = df.iloc[run, :].to_numpy()
        inputs = convert_mass_data(X)
    elif data_type == 'mole':
        X = df.iloc[run, :].to_numpy()
        inputs = convert_mole_data(X)


    # ---- SRP Data Runs ---
    # Create an input list for the values that are used in the simulation
    elif data_type == 'SRP':
        # Tl_b, Tv_a, L, G, alpha, w_MEA, y_H2O, y_CO2
        X = df.iloc[run, :8].to_numpy()
        # X = [314, 320, 29.0, 3.52, 0.279, 0.325, 0.013, 0.100]
        X = [314, 322.717468719381, 29.0, 3.653690241, 0.279, 0.325, 0.0856390806700391, 0.0598166656728878]
        parameters = df.iloc[run, 8:].to_dict()
        # Convert the parameters to a nested dictionary based on type (VLE, Surface Tension, Viscosity)
        df_param = defaultdict(dict)
        for k, v in parameters.items():
            k1, k2 = k.split('-')
            df_param[k1][k2] = v
        df_param = dict(df_param)

        inputs = convert_SRP_data(X, mass=False)


    else:
        raise ValueError('Data source must be either NCCC or SRP')
    # print(inputs)

    # L, G = sum(inputs[0]), sum(inputs[1])
    L_G = X[0]
    alpha = X[2]

    # Simulate the Absorption Column from start to finish given
    # the inlet concentrations of the top liquid and bottom vapor streams
    if method == 'single':
        solving_function = single_shoot_solve
    elif method == 'multiple':
        solving_function = multiple_shoot_solve
    elif method == 'collocation':
        solving_function = orthogonal_collocation
    elif method == 'scipy':
        solving_function = scipy_BVP_solve
    elif method == 'finite':
        solving_function = finite_difference_solve
    else:
        raise ValueError('Wrong method chosen, choose from the available')

    Fl_CO2_a_guess = 3.55511974035339
    Fl_H2O_a_guess = 55.2093581436551
    Tl_a_guess = 325.

    Fv_CO2_b_guess = 0.000126993312499947
    Fv_H2O_b_guess = 5
    Tv_b_guess = 335.

    Fl_CO2_scaling = 2.0
    Fl_H2O_scaling = 60.
    Fv_CO2_scaling = 1.
    Fv_H2O_scaling = 5.
    Hl_scaling = 2800000.
    Hv_scaling = 40000.
    P_scaling = 100000.

    scales = np.array(
        [Fl_CO2_scaling, Fl_H2O_scaling, Fv_CO2_scaling, Fv_H2O_scaling, Hl_scaling, Hv_scaling, P_scaling])

    Fl_b, Fv_a, Tl_b, Tv_a, z, A, P, packing = inputs
    Fl_CO2_b, Fl_MEA_b, Fl_H2O_b = Fl_b
    Fv_CO2_a, Fv_H2O_a, Fv_N2_a, Fv_O2_a = Fv_a

    Fl_MEA_a = Fl_MEA_b
    Fv_N2_b, Fv_O2_b = Fv_N2_a, Fv_O2_a

    # Convert from Temperature to Enthalpy
    Fl_a_guess = [Fl_CO2_a_guess, Fl_MEA_a, Fl_H2O_a_guess]
    Hlt_a_guess = get_liquid_enthalpy(Fl_a_guess, Tl_a_guess)
    Hlf_a_guess = Hlt_a_guess * sum(Fl_a_guess)

    Hlt_b = get_liquid_enthalpy(Fl_b, Tl_b)
    Hlf_b = Hlt_b * sum(Fl_b)

    Hvt_a = get_vapor_enthalpy(Fv_a, Tv_a)
    Hvf_a = Hvt_a * sum(Fv_a)

    Fv_b_guess = [Fv_CO2_b_guess, Fv_H2O_b_guess, Fv_N2_b, Fv_N2_b]
    Hvt_b_guess = get_vapor_enthalpy(Fv_b_guess, Tv_b_guess)
    Hvf_b_guess = Hvt_b_guess * sum(Fv_b_guess)

    P_a = P
    P_b = P

    Y_a_scaled = np.array([Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a,
                           Hlf_a_guess, Hvf_a, P_a]) / scales

    Y_b_scaled = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess,
                           Hlf_b, Hvf_b_guess, P_b]) / scales

    const_flow = Fl_MEA_b, Fv_N2_a, Fv_O2_a
    parameters = scales, const_flow, A, packing

    guesses = [Fl_CO2_a_guess, Fl_H2O_a_guess, Tl_a_guess]

    if show_info:
        print(f'Run #{run + 1:03d} ---')

    # Starts the time tracker for the total computation time for one simulation run
    start = time.time()

    Y_scaled, z_new, solving_type, success, message = solving_function(Y_a_scaled, Y_b_scaled, z, parameters)

    Y = []
    for i in range(len(Y_scaled)):
        Y.append(Y_scaled[i] * scales[i])
    Y = np.array(Y)

    # Ends the time tracker for the total computation time for one simulation run
    end = time.time()
    total_time = end - start

    # Collects data from the final integration output

    Fl_b, Fv_a, Tl_b, Tv_a, z, A, P, packing = inputs

    Fl_CO2_b_act, Fl_H2O_b_act = Fl_b[0], Fl_b[2]
    Tl_b_act = Tl_b

    Fv_H2O_a = Fv_a[1]

    Fv_CO2_b, Fv_H2O_b = Y[2, -1], Y[3, -1]
    CO2_cap = abs(Fv_a[0] - Fv_CO2_b) / Fv_a[0] * 100

    Fl_CO2_b_sim, Fl_H2O_b_sim, Tl_b_sim = Y[0, -1], Y[1, -1], Y[4, -1]
    Fl = [Y[0, -1], Fl_b[1], Y[1, -1]]
    x = [Fl[i] / sum(Fl) for i in range(len(Fl))]

    Hl_b = Y[4, -1] / sum(Fl)
    Tl_b_sim = (get_liquid_temperature(x, Hl_b))

    # Computes the relative error between the solution that the shooter found to the actual inlet concentration for the relevant liquid species
    CO2_rel_err = abs(Fl_CO2_b_act - Fl_CO2_b_sim) / Fl_CO2_b_act * 100
    H2O_rel_err = abs(Fl_H2O_b_act - Fl_H2O_b_sim) / Fl_H2O_b_act * 100
    Tl_rel_err = abs(Tl_b_act - Tl_b_sim) / Tl_b_act * 100

    # Prints out relevant info such as simulation time, relative errors, CO2% captured, if max iterations were reached, and number of Nan's counted

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
    L/G Ratio: {L_G:.2f}
    Lean Loading: {alpha:.2f}
    CO2 % Cap: {CO2_cap:.2f}% 
    Time: {total_time:0>{4}.1f} sec
    Liquid to Vapor Water Ratio: {Fl_H2O_b_act / Fv_H2O_a:.2f}
    % Error: [Simulated, Actual]
        CO2 = {CO2_rel_err:0>{5}.2f}% [{Fl_CO2_b_sim:.3f}, {Fl_CO2_b_act:.3f}]
        H2O = {H2O_rel_err:0>{5}.2f}% [{Fl_H2O_b_sim:.3f}, {Fl_H2O_b_act:.3f}]
        Tl  = {Tl_rel_err:0>{5}.2f}% [{Tl_b_sim:.3f}, {Tl_b_act:.3f}]
''')

    # Stores output data into text files (concentrations, mole fractions, and temperatures) (can also plot)
    save_run_outputs(Y_scaled, z, parameters, save_run_results=save_run_results, plot_temperature=plot_temperature)

    return CO2_cap, message
