import numpy as np
import time
import matplotlib.pyplot as plt

from mea_absorption_column.BVP.Methods.Single_Shoot_Solve import single_shoot_solve
from mea_absorption_column.BVP.Methods.Scipy_BVP_Solve import scipy_BVP_solve
from mea_absorption_column.BVP.Methods.Finite_Difference_Solve import finite_difference_solve
from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.misc.Save_Run_Outputs import save_run_outputs
from mea_absorption_column.misc.Get_Temperature_Enthalpy import (
    get_liquid_enthalpy, get_vapor_enthalpy, get_liquid_temperature, get_vapor_temperature
)
from mea_absorption_column.misc.Scaling import scaling

np.set_printoptions(suppress=True)


def run_model(df,
              method='single',
              data_type='mole',
              run=0,
              show_info=False,
              save_run_results=False,
              plot_temperature=False
              ):

    inputs, X = convert_data(df, run=run, type=data_type)

    L_G, Fv_T, alpha, w_MEA_unloaded, y_CO2, Tl_z, Tv_0, P, beds = X[:9]

    # Simulate the Absorption Column from start to finish given
    # the inlet concentrations of the top liquid and bottom vapor streams
    if method == 'single':
        solving_function = single_shoot_solve
    elif method == 'collocation':
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
    CO2_cap_guess = 95 # Guess for the percentage of CO2 transferred from Vapor to Liquid
    H2O_cap_guess = -100 # Guess for the percentage of H2O transferred from Vapor to Liquid

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

    Fv_b_guess = [Fv_CO2_b_guess, Fv_H2O_b_guess, Fv_N2_b, Fv_N2_b]
    Hvt_b_guess = get_vapor_enthalpy(Fv_b_guess, Tv_b_guess)
    Hvf_b_guess = Hvt_b_guess * sum(Fv_b_guess)

    P_a = P
    P_b = P

    # Scaling

    Y_a_unscaled = np.array([Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a,
                           Hlf_a_guess, Hvf_a, P_a])

    scales = scaling(z, Y_a_unscaled)

    Y_a_scaled = np.array([Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a,
                           Hlf_a_guess, Hvf_a, P_a]) / scales

    Y_b_scaled = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess,
                           Hlf_b, Hvf_b_guess, P_b]) / scales
    eq_scales = scales

    const_flow = Fl_MEA_b, Fv_N2_a, Fv_O2_a

    parameters = scales, eq_scales, const_flow, H, A, packing

    if show_info:
        print(f'''
Run #{run + 1:03d}:
    L/G Ratio: {L_G:.2f}, alpha: {alpha:.2f}, y_CO2: {y_CO2:.2f}
              ''')

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

    CO2_cap = abs(Fv_CO2_a_sim - Fv_CO2_b_sim) / Fv_CO2_a_sim * 100

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
    dfs_dict = save_run_outputs(Y_scaled, z, parameters,
                          save_run_results=save_run_results,
                          plot_temperature=plot_temperature,
                          )

    method_key = {
        'single': 'Shooting',
        'collocation': 'Collocation',
        'finite-difference': 'Finite Difference',
    }

    if plot_temperature:
        dfs_dict['T'].plot(kind='line', y=['Tl', 'Tv'])
        plt.plot(x, df.iloc[run, -5:], 'kx', label='NCCC Data')
        plt.ylabel('Temperature [K]')
        plt.legend()
        plt.title(f'{method_key[method]} Method | Simulation Time: {total_time:.1f} sec \n $\\frac{{L}}{{G}}$ Ratio: {L_G:.2f}, $\\alpha$: {alpha:.2f}, $y_{{CO2}}$: {y_CO2:.2f}, CO2%: {CO2_cap:.2f}')
        plt.show()

    return CO2_cap, success
    # return CO2_cap
