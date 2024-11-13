from scipy.integrate import solve_ivp
from scipy.optimize import root, minimize
from MEA_Absorption_Column.BVP.ABS_Column import abs_column
import numpy as np


def multiple_shoot_solve(inputs, guesses, df_param, scales):
    Fl_z, Fv_0, Tl_z, Tv_0, z = inputs[:-2]

    Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0
    Fl_CO2_0_guess, Fl_H2O_0_guess, Tl_0_guess = guesses

    shoot = True
    n_segments = 2
    z_segments = np.linspace(z[0], z[-1], n_segments + 1)

    if shoot:

        def shooter(X, inputs, df_param, scales):
            print()
            print(X)
            print()

            Fl_CO2_0, Fl_H2O_0, Tl_0 = X[0:3]

            Fl_z, Fv_0, Tl_z, Tv_0, z, A, P = inputs

            Fl_CO2_z, Fl_MEA_z, Fl_H2O_z = Fl_z
            Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0

            Y_i_scaled = [Fl_CO2_0, Fl_H2O_0,
                          Fv_CO2_0 / scales[2], Fv_H2O_0 / scales[3],
                          Tl_0, Tv_0 / scales[5]]

            run_type = 'shooting'

            error = 0
            for i in range(n_segments):

                z_span = [z_segments[i], z_segments[i + 1]]

                sol1 = solve_ivp(abs_column, z_span, Y_i_scaled,
                                 args=(scales, Fl_MEA_z, Fv_N2_0, Fv_O2_0, P, A, df_param, run_type),
                                 method='Radau', t_eval=[z_segments[i + 1]])

                Y_i_scaled = sol1.y

                if i < n_segments - 1:

                    Fl_CO2_i, Fl_H2O_i, Tl_i = X[(i + 1) * 3:(i + 2) * 3]
                    error += (Y_i_scaled[0, -1] - Fl_CO2_i / scales[0])**2
                    error += (Y_i_scaled[1, -1] - Fl_H2O_i / scales[1])**2
                    error += (Y_i_scaled[4, -1] - Tl_i / scales[4])**2

                else:
                    error += (Y_i_scaled[0, -1] - Fl_CO2_z / scales[0])**2
                    error += (Y_i_scaled[1, -1] - Fl_H2O_z / scales[1])**2
                    error += (Y_i_scaled[4, -1] - Tl_z / scales[4])**2

                Y_i_scaled = Y_i_scaled[:, -1]
            print(error)
            return error

        Y_guess = [Fl_CO2_0_guess / scales[0],
                   Fl_H2O_0_guess / scales[1],
                   Tl_0_guess / scales[4]] * n_segments

        Fl_CO2_modifier = np.linspace(0, .90, n_segments + 1)
        Tl_modifier = lambda z: .1784 * z ** 3 - 1.632 * z ** 2 + 2.0197 * z + 323.27

        for i in range(n_segments):
            Y_guess[i * 3] = (Fl_CO2_0_guess - (Fv_CO2_0 * Fl_CO2_modifier[i + 1])) / scales[0]
            Y_guess[(i * 3) + 2] = (Tl_modifier(z_segments[i])) / scales[4]
        Y_guess = np.array(Y_guess)

        print(Y_guess)

        bounds = []
        for i in range(n_segments):
            bounds.append((.8, 1.4))
            bounds.append((1.0, 1.1))
            bounds.append((1., 1.25))


        root_output = minimize(shooter, Y_guess, args=(inputs, df_param, scales),
                               method='SLSQP', bounds=bounds)

        solved_initials_scaled, success, message, n_eval = root_output.x, root_output.success, root_output.message, root_output.nit

        Fl_CO2_0_scaled, Fl_H2O_0_scaled, Tl_0_scaled = solved_initials_scaled[:3]
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

    return Y_scaled, 'Multiple Shooting Method', success, message
