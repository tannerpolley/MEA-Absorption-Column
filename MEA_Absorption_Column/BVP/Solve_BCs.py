from scipy.optimize import root
from MEA_Absorption_Column.BVP.Shooter import shooter
import numpy as np


def solve_bcs(inputs, df_param, scales):
    Fl_z, Fv_0, Tl_z, Tv_0, z = inputs[:-2]

    Fl_CO2_z, Fl_MEA_z, Fl_H2O_z = Fl_z
    Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0

    Fl_CO2_0_guess = Fl_CO2_z + Fv_CO2_0 * .70
    Fl_H2O_0_guess = Fl_H2O_z
    Tl_0_guess = 318

    # Fl_CO2_0_guess = 1.34
    # Fl_H2O_0_guess = 24.5
    # Tl_0_guess = 316.5

    Yv_0_guess = np.array([Fl_CO2_0_guess / scales[0], Fl_H2O_0_guess / scales[1], Tl_0_guess / scales[2]])

    shoot = True

    if shoot:

        method = 'Krylov'
        display = False

        if method == 'df-sane':

            options = {'ftol': .1,
                       'fatol': .1,
                       'maxfev': 50,
                       'line_search': 'cruz',
                       'disp': display,
                       'sigma_0': .1
                       }

        elif method == 'Krylov':

            options = {'fatol': .1,
                       'maxiter': 50,
                       'line_search': 'armijo',
                       'disp': display,
                       }

        else:
            options = {}

        root_output = root(shooter,
                           Yv_0_guess,
                           args=(inputs, df_param, scales),
                           method=method,
                           options=options)

        solved_initials, solved, term_msg, n_eval = root_output.x, root_output.success, root_output.message, root_output.nit

        shooter_message = f'Solved? {solved}, with {n_eval:02d} obj function evaluations'

        Fl_CO2_0, Fl_H2O_0, Tl_0 = solved_initials

        Y_0 = [Fl_CO2_0 * scales[0], Fl_H2O_0 * scales[1], Fv_CO2_0, Fv_H2O_0, Tl_0 * scales[2], Tv_0]

    else:
        shooter_message = 'No shooting'

        Y_0 = [Fl_CO2_0_guess, Fl_H2O_0_guess, Fv_CO2_0, Fv_H2O_0, Tl_0_guess, Tv_0]

    return Y_0, shooter_message
