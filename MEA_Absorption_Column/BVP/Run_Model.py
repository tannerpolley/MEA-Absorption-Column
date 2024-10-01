import numpy as np
import time
from MEA_Absorption_Column.Parameters import n
from MEA_Absorption_Column.BVP.Simulate_Abs_Column import simulate_abs_column
from MEA_Absorption_Column.Convert_Data.Convert_SRP_Data import convert_SRP_data
from MEA_Absorption_Column.misc.Save_Run_Outputs import save_run_outputs
from collections import defaultdict

np.set_printoptions(suppress=True)


def run_model(df, run=0, show_info=True, save_run_results=True):

    X = df.iloc[run, :8].to_numpy()
    X = [314, 320, 29.0, 3.52, 0.279, 0.325, 0.013, 0.100]

    # Grab the parameters for each run
    parameters = df.iloc[run, 8:].to_dict()

    # Convert the parameters to a nested dictionary based on type (VLE, Surface Tension, Viscosity)
    df_param = defaultdict(dict)
    for k, v in parameters.items():
        k1, k2 = k.split('-')
        df_param[k1][k2] = v
    df_param = dict(df_param)

    # ---- SRP Data Runs ---
    # Tl_z, Tv_0, L, G, alpha, y_CO2, y_H2O, H
    # X2 = [314, 320, 29.0, 3.52, 0.279, 0.013,	0.177, 6]
    # Create an input list for the values that are used in the simulation
    inputs = convert_SRP_data(X, n, mass=False)

    # Determine Scaling values
    Fl_CO2_0_scaling = 1.5
    Fl_H2O_0_scaling = 30
    Tl_0_scaling = 360

    scales = [Fl_CO2_0_scaling, Fl_H2O_0_scaling, Tl_0_scaling]

    if show_info:
        print(f'Run #{run + 1:03d} --- ', end='')

    # Starts the time tracker for the total computation time for one simulation run
    start = time.time()

    # Simulate the Absorption Column from start to finish given the inlet concentrations of the top liquid and bottom vapor streams
    # This function simulates either with solving for BC's or assuming them
    Y, shooter_message, success, message = simulate_abs_column(inputs, df_param, scales)

    if not success:
        print('Integration Failed:', message)
        return 0, 0

    # Ends the time tracker for the total computation time for one simulation run
    end = time.time()
    total_time = end - start

    # Collects data from the final integration output

    Fl_z, Fv_0, Tl_z, Tv_0, z, A, P = inputs

    Fv_CO2_z, Fv_H2O_z = Y[2, -1], Y[3, -1]
    CO2_cap = abs(Fv_0[0] - Fv_CO2_z) / Fv_0[0] * 100

    Fl_CO2_z, Fl_H2O_z, Tl_z_sim = Y[0, -1], Y[1, -1], Y[4, -1]

    # Computes the relative error between the solution that the shooter found to the actual inlet concentration for the relevant liquid species
    CO2_rel_err = abs(Fl_z[0] - Fl_CO2_z) / Fl_z[0] * 100
    H2O_rel_err = abs(Fl_z[2] - Fl_H2O_z) / Fl_z[2] * 100
    Tl_rel_err = abs(Tl_z - Tl_z_sim) / Tl_z * 100

    # Prints out relevant info such as simulation time, relative errors, CO2% captured, if max iterations were reached, and number of Nan's counted
    if show_info:
        print(f'CO2 % Cap: {CO2_cap:.2f}% - Time: {total_time:0>{4}.1f} sec - % Error: CO2 = {CO2_rel_err:0>{5}.2f}% [{Fl_CO2_z:.3f}, {Fl_z[0]:.3f}], H2O = {H2O_rel_err:0>{5}.2f}%, Tl = {Tl_rel_err:0>{5}.2f}% - {shooter_message}')

    # Stores output data into text files (concentrations, mole fractions, and temperatures) (can also plot)
    if save_run_results:
        save_run_outputs(Y, Fl_z[1], Fv_0[2], Fv_0[3], z, A, P, df_param, n)

    return CO2_cap, shooter_message
