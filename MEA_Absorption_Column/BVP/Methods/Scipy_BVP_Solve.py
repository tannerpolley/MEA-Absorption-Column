import numpy as np
from scipy.integrate import solve_bvp
from MEA_Absorption_Column.Parameters import column_params
from MEA_Absorption_Column.BVP.ABS_Column import abs_column
from MEA_Absorption_Column.misc.Polynomial_Fit import polynomial_fit
from MEA_Absorption_Column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium
EPS = np.finfo(float).eps


def scipy_BVP_solve(Y_a_scaled, Y_b_scaled, z, parameters):
    Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a, Hlf_a_guess, Hvf_a, P_a = Y_a_scaled
    Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess, Hlf_b, Hvf_b_guess, P_b = Y_b_scaled

    scales = parameters[0]

    bcs_1 = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_a, Fv_H2O_a, Hlf_b, Hvf_a, P_a]) / scales

    # Define the system of differential equations for the absorption column
    def column_odes(z, w):

        differentials = [abs_column(z[i], w[:, i], parameters) for i in range(np.shape(w)[1])]
        del chemical_equilibrium.cache
        arr = np.array(differentials).T
        has_nan = np.isnan(arr).any()
        # for i in range(np.shape(arr)[1]):
        #     if  w[3, i] < 0:
        #         print(f'{z[i]:.4f}, {w[3, i]:.4f}, {arr[3, i]:.4f}, {arr[3, i-1]:.4f}')
        return np.array(differentials).T

    # Define the boundary conditions
    def boundary_conditions(bottom, top):
        # Enforce the boundary conditions at the bottom (vapor) and top (liquid)
        Fl_CO2_a_bc, Fl_H2O_a_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Hlf_a_bc, Hvf_a_bc, P_a_bc = bottom
        Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_b_bc, Fv_H2O_b_bc, Hlf_b_bc, Hvf_b_bc, P_b_bc = top

        bcs_2 = np.array([Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Hlf_b_bc, Hvf_a_bc, P_a_bc]) / scales

        # Boundary conditions at the bottom for vapor and at the top for liquid
        return bcs_1 - bcs_2

    def fun_jac(x, y):

        fun = column_odes
        n, m = y.shape
        f0 = fun(x, y)

        dtype = y.dtype

        df_dy = np.empty((n, n, m), dtype=dtype)
        h = EPS ** 0.5 * (1 + np.abs(y))
        for i in range(n):
            y_new = y.copy()
            y_new2 = y.copy()
            y_new[i] += h[i]
            y_new2[i] -= h[i]
            hi = y_new[i] - y[i]
            f_new = fun(x, y_new)
            f_new2 = fun(x, y_new2)

            df_dy[:, i, :] = (f_new - f_new2) / (2*hi)

        return df_dy


    # Initial guess for the solution (constant profiles as initial guess)

    m = len(Y_a_scaled)
    n = 51 # mesh points
    # z_2 = np.linspace(0, .8, n)
    # z_2 = np.append(z_2, np.linspace(.81, 1, (3 * n) - 3))
    # n = len(z_2)
    z_2 = np.linspace(z[0], z[-1], n)
    w_guess_scaled = np.zeros((m, n))
    import pandas as pd
    import matplotlib.pyplot as plt
    keys = {
        'Fl_CO2': 'Fl',
        'Fl_H2O': 'Fl',
        'Fv_CO2': 'Fv',
        'Fv_H2O': 'Fv',
        'Hlf': 'ql',
        'Hvf': 'qv',
        'P': 'transport',
    }
    filename = r'C:\Users\Tanner\Documents\git\IDAES_MEA_Flowsheet_Tanner\Simulation_Results\Profiles_IDAES.xlsx'
    # filename2 = 'data/test.csv'
    # df_2 = pd.read_csv(filename2)

    for i, (k, v) in enumerate(keys.items()):
        # df = pd.read_excel(filename, sheet_name=v)
        # y = df[k].to_numpy()

        # y = df_2.iloc[:, i].to_numpy()[::-1]



        y = polynomial_fit(z_2, Y_a_scaled[i]*scales[i], i)
        # plt.plot(z_2, y)
        # plt.show()
        # print(y)
        w_guess_scaled[i] = y/scales[i]


    # Solve the BVP

    sol = solve_bvp(column_odes, boundary_conditions, z_2, w_guess_scaled,
                    # fun_jac=fun_jac,
                    max_nodes=1000,
                    tol=5e-1,
                    bc_tol=1e-3,
                    verbose=0,
                    )
    Y_scaled = sol.sol(z)
    z = sol.x

    success = sol.success
    message = sol.message

    return Y_scaled, z, 'Scipy BVP Method', success, message

