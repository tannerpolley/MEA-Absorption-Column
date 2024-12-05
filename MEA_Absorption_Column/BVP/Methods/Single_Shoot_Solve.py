from scipy.integrate import solve_ivp
from MEA_Absorption_Column.BVP.Methods.Integration_Methods import runge_kutta
from MEA_Absorption_Column.BVP.Methods.Integration_Methods import scipy_integrate
from MEA_Absorption_Column.BVP.Methods.Integration_Methods import eulers
from MEA_Absorption_Column.Thermodynamics.Get_Temperature_Enthalpy import get_liquid_enthalpy, get_vapor_enthalpy, get_liquid_temperature, get_vapor_temperature
from scipy.optimize import root
from MEA_Absorption_Column.BVP.ABS_Column import abs_column
import numpy as np


def single_shoot_solve(inputs, guesses, df_param, scales):
    Fl_z, Fv_0, Tl_z, Tv_0, z, A, P, packing = inputs

    Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0
    Fl_CO2_0_guess, Fl_H2O_0_guess, Tl_0_guess = guesses

    shoot = False

    if shoot:

        def shooter(X, inputs, df_param, scales):

            Fl_CO2_0, Fl_H2O_0, Tl_0 = X

            Fl_z, Fv_0, Tl_z, Tv_0, z, A, P, packing = inputs

            Fl_CO2_z, Fl_MEA_z, Fl_H2O_z = Fl_z
            Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0

            Y_0_scaled = [Fl_CO2_0, Fl_H2O_0,
                          Fv_CO2_0 / scales[2], Fv_H2O_0 / scales[3],
                          Tl_0, Tv_0 / scales[5], P / scales[6]]

            run_type = 'shooting'

            result = solve_ivp(abs_column, [z[0], z[-1]], Y_0_scaled,
                               args=(scales, Fl_MEA_z, Fv_N2_0, Fv_O2_0, A, packing, df_param, run_type),
                               method='Radau', t_eval=z)

            Y_scaled = result.y

            Fl_CO2_z_sim, Fl_H2O_z_sim, Tl_z_sim = Y_scaled[0, -1], Y_scaled[1, -1], Y_scaled[4, -1]

            eq1 = Fl_CO2_z_sim - Fl_CO2_z / scales[0]
            eq2 = Fl_H2O_z_sim - Fl_H2O_z / scales[1]
            eq3 = Tl_z_sim - Tl_z / scales[4]

            eqs = [eq1, eq2, eq3]
            # print(eqs)
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
                      Tl_0_scaled, Tv_0 / scales[5], P / scales[6]]

    else:
        message = 'No shooting'
        success = 'No shooting'

        Fl_0_guess = [Fl_CO2_0_guess, Fl_z[1], Fl_H2O_0_guess]

        Hlf_T_0_guess = get_liquid_enthalpy(Fl_0_guess, Tl_0_guess)*sum(Fl_0_guess)
        Hvf_T_0 = get_vapor_enthalpy(Fv_0, Tv_0)*sum(Fv_0)

        Y_0_scaled = [Fl_CO2_0_guess / scales[0], Fl_H2O_0_guess / scales[1],
                      Fv_CO2_0 / scales[2], Fv_H2O_0 / scales[3],
                      Hlf_T_0_guess / scales[4], Hvf_T_0 / scales[5],
                      P / scales[6]]

    run_type = 'simulating'

    Fl_z, Fv_0, Tl_z, Tv_0, z, A, P, packing = inputs
    # obj = solve_ivp(abs_column, [z[0], z[-1]], Y_0_scaled,
    #                 args=(scales, Fl_z[1], Fv_0[2], Fv_0[3], A, packing, df_param, run_type),
    #                 method='Radau', t_eval=z,
    #                 vectorized=False,
    #                 # options={'first_step': 1e-10,
    #                 #          'max_step': 1e-5,
    #                 #          'rtol': 1e-0,
    #                 #          'atol': 1e-0}
    #                 )

    # Y_scaled = obj.y
    # success = obj.success
    # message = obj.message
    # z = obj.t

    # integrater = scipy_integrate
    integrater = runge_kutta
    # integrater = eulers

    Y_scaled, z, success, message = integrater(abs_column, Y_0_scaled, z, scales,
                                               args=(
                                                   scales,
                                                   Fl_z[1], Fv_0[2], Fv_0[3],
                                                   A, packing, df_param, run_type)
                                               )

    Y_scaled_new = np.empty_like(Y_scaled)

    Y_scaled_new[0] = Y_scaled[0]
    Y_scaled_new[1] = Y_scaled[1]
    Y_scaled_new[2] = Y_scaled[2]
    Y_scaled_new[3] = Y_scaled[3]
    Y_scaled_new[6] = Y_scaled[6]

    Tl = []
    Tv = []
    for i in range(len(Y_scaled[4])):
        Fl = [Y_scaled[0, i], Fl_z[1], Y_scaled[2, i]]
        x = [Fl[i]/sum(Fl) for i in range(len(Fl))]
        Hl_T = Y_scaled[4, i]/sum(Fl)
        Tl.append(get_liquid_temperature(x, Hl_T))

        Fv = [Y_scaled[3, i], Y_scaled[4, i], Fv_0[2], Fv_0[3]]
        y = [Fv[i]/sum(Fv) for i in range(len(Fv))]
        Hv_T = Y_scaled[4, i]/sum(Fv)
        Tv.append(get_vapor_temperature(y, Hv_T))

    Y_scaled_new[4] = np.array(Tl)
    Y_scaled_new[5] = np.array(Tv)

    print(Y_scaled_new)

    return Y_scaled_new, z, 'Single Shooting Method', success, message
