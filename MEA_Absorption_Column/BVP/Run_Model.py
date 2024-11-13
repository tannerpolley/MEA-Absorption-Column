import numpy as np
import time
from collections import defaultdict

from MEA_Absorption_Column.Parameters import n
from MEA_Absorption_Column.BVP.Methods.Single_Shoot_Solve import single_shoot_solve
from MEA_Absorption_Column.BVP.Methods.Multiple_Shoot_Solve_2 import multiple_shoot_solve
from MEA_Absorption_Column.BVP.Methods.Collocation_Solve import collocation_solve
from MEA_Absorption_Column.BVP.Methods.Scipy_BVP_Solve import scipy_BVP_solve
from MEA_Absorption_Column.BVP.Methods.Finite_Difference_Solve import finite_difference_solve

from MEA_Absorption_Column.Convert_Data.Convert_SRP_Data import convert_SRP_data
from MEA_Absorption_Column.Convert_Data.Convert_NCCC_Data import convert_NCCC_data
from MEA_Absorption_Column.misc.Save_Run_Outputs import save_run_outputs


np.set_printoptions(suppress=True)


def run_model(df, method='single', data_source='NCCC', run=0, show_info=True, save_run_results=True):

    # ---- SRP Data Runs ---
    # Create an input list for the values that are used in the simulation
    if data_source == 'NCCC':
        # Tl_z, Tv_0, L, G, alpha, w_MEA, y_CO2, y_H2O, H
        X = df.iloc[run, :9].to_numpy()
        # X = [314, 320, 29.0, 3.52, 0.279, 0.325, 0.013, 0.100]
        parameters = df.iloc[run, 9:].to_dict()
        # Convert the parameters to a nested dictionary based on type (VLE, Surface Tension, Viscosity)
        df_param = defaultdict(dict)
        for k, v in parameters.items():
            k1, k2 = k.split('-')
            df_param[k1][k2] = v
        df_param = dict(df_param)

        inputs = convert_NCCC_data(X, case='18')


    # ---- SRP Data Runs ---
    # Create an input list for the values that are used in the simulation
    elif data_source == 'SRP':
        # Tl_z, Tv_0, L, G, alpha, w_MEA, y_CO2, y_H2O
        X = df.iloc[run, :8].to_numpy()
        X = [314, 320, 29.0, 3.52, 0.279, 0.325, 0.013, 0.100]
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
    # Determine Scaling values
    Fl_CO2_scaling = 1.0
    Fl_H2O_scaling = 25
    Fv_CO2_scaling = .2
    Fv_H2O_scaling = .2
    Tl_scaling = 300
    Tv_scaling = 300

    scales = [Fl_CO2_scaling, Fl_H2O_scaling, Fv_CO2_scaling, Fv_H2O_scaling, Tl_scaling, Tv_scaling]
    # scales = [1, 1, 1, 1, 1, 1]

    if show_info:
        print(f'Run #{run + 1:03d} --- ', end='')

    # Starts the time tracker for the total computation time for one simulation run
    start = time.time()

    # Simulate the Absorption Column from start to finish given
    # the inlet concentrations of the top liquid and bottom vapor streams
    if method == 'single':
        solving_function = single_shoot_solve
    elif method == 'multiple':
        solving_function = multiple_shoot_solve
    elif method == 'collocation':
        solving_function = collocation_solve
    elif method == 'scipy':
        solving_function = scipy_BVP_solve
    elif method == 'finite':
        solving_function = finite_difference_solve
    else:
        raise ValueError('Wrong method chosen, choose from the available')

    if data_source == 'NCCC':
        Fl_CO2_0_guess = 3.554499396
        Fl_H2O_0_guess = 69.68405525
        Tl_0_guess = 323.646741247888

        Fl_CO2_scaling = 1.5
        Fl_H2O_scaling = 70
        Fv_CO2_scaling = 1
        Fv_H2O_scaling = 5
        Tl_scaling = 300
        Tv_scaling = 300

        scales = [Fl_CO2_scaling, Fl_H2O_scaling, Fv_CO2_scaling, Fv_H2O_scaling, Tl_scaling, Tv_scaling]
    elif data_source == 'SRP':
        Fl_CO2_0_guess = 1.3
        Fl_H2O_0_guess = 25.0
        Tl_0_guess = 320.0

        Fl_CO2_scaling = 1.0
        Fl_H2O_scaling = 25
        Fv_CO2_scaling = .2
        Fv_H2O_scaling = .3
        Tl_scaling = 300
        Tv_scaling = 300

        scales = [Fl_CO2_scaling, Fl_H2O_scaling, Fv_CO2_scaling, Fv_H2O_scaling, Tl_scaling, Tv_scaling]
    else:
        raise ValueError('Data source must be either NCCC or SRP')
    guesses = [Fl_CO2_0_guess, Fl_H2O_0_guess, Tl_0_guess]

    Y_scaled, solving_type, success, message = solving_function(inputs, guesses, df_param, scales)

    Y = []
    for i in range(len(Y_scaled)):
        Y.append(Y_scaled[i]*scales[i])
    Y = np.array(Y)
    # if not success:
    #     print('Integration Failed:', message)
    #     return Y, 0, 0

    # Ends the time tracker for the total computation time for one simulation run
    end = time.time()
    total_time = end - start

    # Collects data from the final integration output

    Fl_z, Fv_0, Tl_z, Tv_0, z, A, P = inputs

    Fl_CO2_z_act, Fl_H2O_z_act = Fl_z[0], Fl_z[2]
    Tl_z_act = Tl_z

    Fv_CO2_z, Fv_H2O_z = Y[2, -1], Y[3, -1]
    CO2_cap = abs(Fv_0[0] - Fv_CO2_z) / Fv_0[0] * 100

    Fl_CO2_z_sim, Fl_H2O_z_sim, Tl_z_sim = Y[0, -1], Y[1, -1], Y[4, -1]

    # Computes the relative error between the solution that the shooter found to the actual inlet concentration for the relevant liquid species
    CO2_rel_err = abs(Fl_CO2_z_act - Fl_CO2_z_sim) / Fl_CO2_z_act * 100
    H2O_rel_err = abs(Fl_H2O_z_act - Fl_H2O_z_sim) / Fl_H2O_z_act * 100
    Tl_rel_err = abs(Tl_z_act - Tl_z_sim) / Tl_z_act * 100

    # Prints out relevant info such as simulation time, relative errors, CO2% captured, if max iterations were reached, and number of Nan's counted
    if show_info:
        print(f'''
        - CO2 % Cap: {CO2_cap:.2f}% - Time: {total_time:0>{4}.1f} sec
        - % Error: [Simulated, Actual]
            - CO2 = {CO2_rel_err:0>{5}.2f}% [{Fl_CO2_z_sim:.3f}, {Fl_CO2_z_act:.3f}]
            - H2O = {H2O_rel_err:0>{5}.2f}% [{Fl_H2O_z_sim:.3f}, {Fl_H2O_z_act:.3f}]
            - Tl = {Tl_rel_err:0>{5}.2f}% [{Tl_z_sim:.3f}, {Tl_z_act:.3f}]
        - {solving_type} - {message}
        ''')
    # Stores output data into text files (concentrations, mole fractions, and temperatures) (can also plot)
    if save_run_results:
        save_run_outputs(Y_scaled, Fl_z[1], Fv_0[2], Fv_0[3], scales, z, A, P, df_param, n)

    return CO2_cap, message
