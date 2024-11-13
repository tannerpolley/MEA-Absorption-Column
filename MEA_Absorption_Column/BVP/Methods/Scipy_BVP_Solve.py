import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from MEA_Absorption_Column.BVP.ABS_Column import abs_column

def scipy_BVP_solve(inputs, guesses, df_param, scales):

    Fl_z, Fv_0, Tl_z, Tv_0, z, A, P = inputs

    Fl_CO2_z, Fl_MEA_z, Fl_H2O_z = Fl_z
    Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0

    Fl_CO2_0_guess, Fl_H2O_0_guess, Tl_0_guess = guesses

    run_type = 'simulating'

    # Parameters (example values, typically column-specific data and parameters)
    stages = len(z)  # Number of stages in the column (z-axis)

    # Define the system of differential equations for the absorption column
    def column_odes(z, w):
        differentials = np.zeros_like(w.T)
        for i in range(len(differentials)):
            print(w[:, i])
            # Flow rate changes
            differentials_i = np.array(
                abs_column(0, w[:, i], scales, Fl_MEA_z, Fv_N2_0, Fv_O2_0, P, A, df_param, run_type))

            dF_CO2_l_dz = differentials_i[0]
            dF_H2O_l_dz = differentials_i[1]
            dF_CO2_v_dz = -differentials_i[2]
            dF_H2O_v_dz = -differentials_i[3]

            # Liquid and vapor temperature changes
            dT_l_dz = differentials_i[4]
            dT_v_dz = -differentials_i[5]
            differentials[i] = np.array([dF_CO2_l_dz, dF_H2O_l_dz, dF_CO2_v_dz, dF_H2O_v_dz, dT_l_dz, dT_v_dz])
        print()
        return differentials.T

    # Define the boundary conditions
    def boundary_conditions(bottom, top):
        # Enforce the boundary conditions at the bottom (vapor) and top (liquid)
        Fl_CO2_0_bc, Fl_H2O_0_bc, Fv_CO2_0_bc, Fv_H2O_0_bc, Tl_0_bc, Tv_0_bc = bottom
        Fl_CO2_z_bc, Fl_H2O_z_bc, Fv_CO2_z_bc, Fv_H2O_z_bc, Tl_z_bc, Tv_z_bc = top

        # Boundary conditions at the bottom for vapor and at the top for liquid
        return np.array([
            Fl_CO2_z_bc/scales[0] - Fl_CO2_z/scales[0],  # CO2 liquid flow rate at the top
            Fl_H2O_z_bc/scales[1] - Fl_H2O_z/scales[1],  # H2O liquid flow rate at the top
            Fv_CO2_0_bc/scales[2] - Fv_CO2_0/scales[2],  # CO2 vapor flow rate at the bottom
            Fv_H2O_0_bc/scales[3] - Fv_H2O_0/scales[3],  # H2O vapor flow rate at the bottom
            Tl_z_bc/scales[4] - Tl_z/scales[4],  # Liquid temperature at the top
            Tv_0_bc/scales[5] - Tv_0/scales[5],  # Vapor temperature at the bottom
        ])

    # Initial guess for the solution (constant profiles as initial guess)
    z = np.linspace(0, 6, stages)
    w_guess = np.zeros((6, stages))
    w_guess[0] = Fl_CO2_0_guess/scales[0]  # Initial guess for CO2 liquid flow rate profile
    w_guess[1] = Fl_H2O_0_guess/scales[1]  # Initial guess for H2O liquid flow rate profile
    w_guess[2] = Fv_CO2_0/scales[2]  # Initial guess for CO2 vapor flow rate profile
    w_guess[3] = Fv_H2O_0/scales[3]  # Initial guess for H2O vapor flow rate profile
    w_guess[4] = Tl_0_guess/scales[4]  # Initial guess for liquid temperature profile
    w_guess[5] = Tv_0/scales[5]  # Initial guess for vapor temperature profile

    w_guess[0, -1] = Fl_CO2_z/scales[0]
    w_guess[1, -1] = Fl_H2O_z/scales[1]
    w_guess[4, -1] = Tl_z/scales[4]

    # Solve the BVP
    solution = solve_bvp(column_odes, boundary_conditions, z, w_guess, max_nodes=2000, tol=1e-1, bc_tol=1e-1)

    Y_scaled = solution.y

    success = solution.success
    message = solution.message


    return Y_scaled, 'Scipy BVP Method', success, message


