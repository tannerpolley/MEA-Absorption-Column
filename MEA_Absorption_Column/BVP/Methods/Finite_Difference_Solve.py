import numpy as np
from scipy.optimize import root
from MEA_Absorption_Column.BVP.ABS_Column import abs_column


def finite_difference_solve(inputs, guesses, df_param, scales):
    # Parameters
    Fl_z, Fv_0, Tl_z, Tv_0, z, A, P = inputs

    Fl_CO2_z, Fl_MEA_z, Fl_H2O_z = Fl_z
    Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0

    Fl_CO2_0_guess, Fl_H2O_0_guess, Tl_0_guess = guesses

    run_type = 'simulating'

    # Parameters (example values, typically column-specific data and parameters)
    stages = len(z)  # Number of stages in the column (z-axis)
    dz = z[1] - z[0]

    F_l_CO2_0 = Fl_CO2_z  # Liquid phase flow rate of CO2 at the top
    F_l_H2O_0 = Fl_H2O_z  # Liquid phase flow rate of H2O at the top

    F_v_CO2_0 = Fv_CO2_0  # Vapor phase flow rate of CO2 at the bottom
    F_v_H2O_0 = Fv_H2O_0  # Vapor phase flow rate of H2O at the bottom

    T_v0 = Tv_0  # Initial vapor temperature at the bottom in K
    T_l0 = Tl_z  # Initial liquid temperature at the top in K

    # Initial guess for the solution (constant profiles)
    w_guess = np.zeros((6, stages))
    w_guess[0, :] = F_l_CO2_0  # CO2 liquid flow rate profile
    w_guess[1, :] = F_l_H2O_0  # H2O liquid flow rate profile
    w_guess[2, :] = F_v_CO2_0  # CO2 vapor flow rate profile
    w_guess[3, :] = F_v_H2O_0  # H2O vapor flow rate profile
    w_guess[4, :] = T_l0  # Liquid temperature profile
    w_guess[5, :] = T_v0  # Vapor temperature profile

    # Flatten the initial guess to use with root solver
    w_guess_flat = w_guess.flatten()

    # Finite difference function that uses evaluate_differential
    def finite_difference_eqs(w_flat):
        # Reshape w_flat back into a 2D array for easier indexing
        w = w_flat.reshape(6, stages)

        # Initialize the system of equations
        equations = np.zeros_like(w)

        # Interior points: use central difference
        for i in range(1, stages - 1):
            Y = w[:, i]

            # Calculate the differentials using the function
            diffs = abs_column(0, Y, Fl_MEA_z, Fv_N2_0, Fv_O2_0, P, A, df_param, run_type)

            for j in range(len(diffs)):
                # Apply central differences and compare with the differential function results
                equations[j, i] = (w[j, i + 1] - w[j, i - 1]) / (2 * dz) - diffs[j]

        # Boundary conditions
        equations[0, -1] = w[0, -1] - F_l_CO2_0  # CO2 liquid flow rate at the top
        equations[1, -1] = w[1, -1] - F_l_H2O_0  # H2O liquid flow rate at the top
        equations[2, 0] = w[2, 0] - F_v_CO2_0  # CO2 vapor flow rate at the bottom
        equations[3, 0] = w[3, 0] - F_v_H2O_0  # H2O vapor flow rate at the bottom
        equations[4, -1] = w[4, -1] - T_l0  # Liquid temperature at the top
        equations[5, 0] = w[5, 0] - T_v0  # Vapor temperature at the bottom

        diffs = lambda i: abs_column(0, [w[0, i], w[1, i], w[2, i], w[3, i], w[4, i], w[5, i]],
                                     Fl_MEA_z, Fv_N2_0, Fv_O2_0, P, A, df_param, run_type)

        # Forward and backward differences at boundaries
        equations[0, 0] = (w[0, 1] - w[0, 0]) / dz - diffs(0)[0]
        equations[1, 0] = (w[1, 1] - w[1, 0]) / dz - diffs(0)[1]
        equations[2, -1] = (w[2, -1] - w[2, -2]) / dz - diffs(-1)[2]
        equations[3, -1] = (w[3, -1] - w[3, -2]) / dz - diffs(-1)[3]
        equations[4, 0] = (w[4, 1] - w[4, 0]) / dz - diffs(0)[4]
        equations[5, -1] = (w[5, -1] - w[5, -2]) / dz - diffs(-1)[5]

        return equations.flatten()

    # Solve the system of equations
    solution = root(finite_difference_eqs, w_guess_flat)

    Y = solution.x.reshape(6, stages)
    success = solution.success
    message = solution.message

    return Y, 'Finite Difference Method', success, message
