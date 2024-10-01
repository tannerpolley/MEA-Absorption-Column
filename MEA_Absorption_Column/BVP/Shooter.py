from scipy.integrate import solve_ivp
from MEA_Absorption_Column.BVP.ABS_Column import abs_column


def shooter(X, inputs, df_param, scales):

    Fl_CO2_0, Fl_H2O_0, Tl_0 = X

    Fl_z, Fv_0, Tl_z, Tv_0, z, A, P = inputs

    Fl_CO2_z, Fl_MEA_z, Fl_H2O_z = Fl_z
    Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0

    Y_0 = [Fl_CO2_0*scales[0], Fl_H2O_0*scales[1], Fv_CO2_0, Fv_H2O_0, Tl_0*scales[2], Tv_0]

    run_type = 'shooting'

    result = solve_ivp(abs_column,
                  [z[0], z[-1]],
                  Y_0,
                  args=(Fl_MEA_z, Fv_N2_0, Fv_O2_0, P, A, df_param, run_type),
                  method='Radau', t_eval=z)

    Y = result.y

    Fl_CO2_z_sim, Fl_H2O_z_sim, Tl_z_sim = Y[0, -1], Y[1, -1], Y[4, -2]

    eq1 = Fl_CO2_z_sim/scales[0] - Fl_CO2_z/scales[0]
    eq2 = Fl_H2O_z_sim/scales[1] - Fl_H2O_z/scales[1]
    eq3 = Tl_z_sim/scales[2] - Tl_z/scales[2]

    eqs = [eq1, eq2, eq3]

    return eqs
