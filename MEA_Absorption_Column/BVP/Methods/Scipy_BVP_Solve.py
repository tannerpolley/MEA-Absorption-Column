import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from MEA_Absorption_Column.BVP.ABS_Column import abs_column

def scipy_BVP_solve(Y_a_scaled, Y_b_scaled, z, parameters):

    Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a, Hlf_a_guess, Hvf_a, P_a = Y_a_scaled
    Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess, Hlf_b, Hvf_b_guess, P_b = Y_b_scaled

    scales = parameters[0]

    # Parameters (example values, typically column-specific data and parameters)


    # Define the system of differential equations for the absorption column
    def column_odes(z, w):
        differentials = np.zeros_like(w.T)
        for i in range(len(differentials)):
            # Flow rate changes
            differentials[i] = abs_column(z, w[:, i], parameters)

        return differentials.T

    # Define the boundary conditions
    def boundary_conditions(bottom, top):
        # Enforce the boundary conditions at the bottom (vapor) and top (liquid)
        Fl_CO2_a_bc, Fl_H2O_a_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Hlf_a_bc, Hvf_a_bc, P_a_bc = bottom
        Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_b_bc, Fv_H2O_b_bc, Hlf_b_bc, Hvf_b_bc, P_b_bc = top

        # Boundary conditions at the bottom for vapor and at the top for liquid
        return np.array([
            Fl_CO2_b_bc - Fl_CO2_b,  # CO2 liquid flow rate at the top
            Fl_H2O_b_bc- Fl_H2O_b,  # H2O liquid flow rate at the top
            Fv_CO2_a_bc - Fv_CO2_a,  # CO2 vapor flow rate at the bottom
            Fv_H2O_a_bc - Fv_H2O_a,  # H2O vapor flow rate at the bottom
            Hlf_b_bc - Hlf_b,  # Liquid temperature at the top
            Hvf_a_bc - Hvf_a,  # Vapor temperature at the bottom
            P_a_bc - P_a
        ])

    # Initial guess for the solution (constant profiles as initial guess)

    m = len(Y_a_scaled)
    w_guess_scaled = np.zeros((m, len(z)))

    print(len(z))

    for i in range(m):
        w_guess_scaled[i] = np.linspace(Y_a_scaled[i], Y_b_scaled[i], len(z))



    # Solve the BVP
    solution = solve_bvp(column_odes, boundary_conditions, z, w_guess_scaled, max_nodes=2000)

    Y_scaled = solution.y
    z = solution.x
    print(len(z))

    success = solution.success
    message = solution.message


    return Y_scaled, z, 'Scipy BVP Method', success, message


