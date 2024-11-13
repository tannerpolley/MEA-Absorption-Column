from scipy.integrate import solve_ivp
from scipy.optimize import root
from MEA_Absorption_Column.BVP.ABS_Column import abs_column
import numpy as np


def single_shoot_solve(inputs, guesses, df_param, scales):
    Fl_z, Fv_0, Tl_z, Tv_0, z = inputs[:-2]

    Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0
    Fl_CO2_0_guess, Fl_H2O_0_guess, Tl_0_guess = guesses

    shoot = True

    if shoot:

        def shooter(X, inputs, df_param, scales):

            Fl_CO2_0, Fl_H2O_0, Tl_0 = X

            Fl_z, Fv_0, Tl_z, Tv_0, z, A, P = inputs

            Fl_CO2_z, Fl_MEA_z, Fl_H2O_z = Fl_z
            Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0

            Y_0_scaled = [Fl_CO2_0, Fl_H2O_0,
                          Fv_CO2_0 / scales[2], Fv_H2O_0 / scales[3],
                          Tl_0, Tv_0 / scales[5]]

            run_type = 'shooting'

            result = solve_ivp(abs_column, [z[0], z[-1]], Y_0_scaled,
                               args=(scales, Fl_MEA_z, Fv_N2_0, Fv_O2_0, P, A, df_param, run_type),
                               method='Radau', t_eval=z)

            Y_scaled = result.y

            Fl_CO2_z_sim, Fl_H2O_z_sim, Tl_z_sim = Y_scaled[0, -1], Y_scaled[1, -1], Y_scaled[4, -1]

            eq1 = Fl_CO2_z_sim - Fl_CO2_z / scales[0]
            eq2 = Fl_H2O_z_sim - Fl_H2O_z / scales[1]
            eq3 = Tl_z_sim - Tl_z / scales[4]

            eqs = [eq1, eq2, eq3]

            return eqs

        Y_0_guess = np.array([Fl_CO2_0_guess / scales[0],
                              Fl_H2O_0_guess / scales[1],
                              Tl_0_guess / scales[4]])

        method = 'Krylov'
        display = False

        options = {'fatol': .1, 'maxiter': 50, 'line_search': 'armijo', 'disp': display, }
        # options = {}

        root_output = root(shooter, Y_0_guess, args=(inputs, df_param, scales), method=method, options=options)

        solved_initials_scaled, success, message, n_eval = root_output.x, root_output.success, root_output.message, root_output.nit

        Fl_CO2_0_scaled, Fl_H2O_0_scaled, Tl_0_scaled = solved_initials_scaled
        Y_0_scaled = [Fl_CO2_0_scaled, Fl_H2O_0_scaled,
                      Fv_CO2_0 / scales[2], Fv_H2O_0 / scales[3],
                      Tl_0_scaled, Tv_0 / scales[5]]

    else:
        message = 'No shooting'
        success = 'No shooting'

        Y_0_scaled = [Fl_CO2_0_guess / scales[0], Fl_H2O_0_guess / scales[1],
                      Fv_CO2_0 / scales[2], Fv_H2O_0 / scales[3],
                      Tl_0_guess / scales[4], Tv_0 / scales[5]]

    run_type = 'simulating'

    Fl_z, Fv_0, Tl_z, Tv_0, z, A, P = inputs

    obj = solve_ivp(abs_column, [z[0], z[-1]], Y_0_scaled,
                    args=(scales, Fl_z[1], Fv_0[2], Fv_0[3], P, A, df_param, run_type),
                    method='BDF', t_eval=z,
                    # vectorized=False,
                    # options={'first_step': None, 'max_step': 1e10, 'atol': 1e-5}
                    )

    Y_scaled = obj.y

    return Y_scaled, 'Single Shooting Method', success, message
