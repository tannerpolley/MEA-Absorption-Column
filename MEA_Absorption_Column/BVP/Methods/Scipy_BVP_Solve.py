import numpy as np
from scipy.integrate import solve_bvp
from MEA_Absorption_Column.BVP.ABS_Column import abs_column


def scipy_BVP_solve(Y_a_scaled, Y_b_scaled, z, parameters):
    Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a, Hlf_a_guess, Hvf_a, P_a = Y_a_scaled
    Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess, Hlf_b, Hvf_b_guess, P_b = Y_b_scaled

    bcs_1 = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_a, Fv_H2O_a, Hlf_b, Hvf_a, P_a])

    # Define the system of differential equations for the absorption column
    def column_odes(z, w):
        differentials = [abs_column(z, w[:, i], parameters) for i in range(len(w.T))]
        return np.array(differentials).T

    # Define the boundary conditions
    def boundary_conditions(bottom, top):
        # Enforce the boundary conditions at the bottom (vapor) and top (liquid)
        Fl_CO2_a_bc, Fl_H2O_a_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Hlf_a_bc, Hvf_a_bc, P_a_bc = bottom
        Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_b_bc, Fv_H2O_b_bc, Hlf_b_bc, Hvf_b_bc, P_b_bc = top

        bcs_2 = np.array([Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Hlf_b_bc, Hvf_a_bc, P_a_bc])

        # Boundary conditions at the bottom for vapor and at the top for liquid
        return bcs_1 - bcs_2

    # Initial guess for the solution (constant profiles as initial guess)

    m = len(Y_a_scaled)
    n = 21 # mesh points
    z_2 = np.linspace(z[0], z[-1], n)
    w_guess_scaled = np.zeros((m, n))

    peak_values = np.array([1, 80, 1, 9.25, -3620000, 42500, 1]) / parameters[0]
    # print(peak_values)

    for i in range(m):
        if i == 0 or i == 2 or i == 6:
            w_guess_scaled[i] = np.linspace(Y_a_scaled[i], Y_b_scaled[i], n)
        else:
            start = [z[0], Y_a_scaled[i]]
            stop = [z[-1], Y_b_scaled[i]]
            control = [z[-1] * .8, peak_values[i] * 1.05]

            w_guess_scaled[i] = quadratic_arc(start, stop, control, num_points=n)

    # Solve the BVP

    sol = solve_bvp(column_odes, boundary_conditions, z_2, w_guess_scaled,
                    max_nodes=1000, tol=2e-1,
                    verbose=0)
    Y_scaled = sol.sol(z)
    z = sol.x

    success = sol.success
    message = sol.message

    return Y_scaled, z, 'Scipy BVP Method', success, message

import matplotlib.pyplot as plt
def quadratic_arc(start, end, control, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve = (1 - t)[:, None] ** 2 * np.array(start) + \
            2 * t[:, None] * (1 - t)[:, None] * np.array(control) + \
            t[:, None] ** 2 * np.array(end)

    # plt.plot(t, curve.T[1])
    # plt.show()
    return curve.T[1]
