import numpy as np
from scipy.integrate import solve_bvp
from MEA_Absorption_Column.BVP.ABS_Column import abs_column
from MEA_Absorption_Column.misc.Polynomial_Fit import polynomial_fit
from MEA_Absorption_Column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium
from MEA_Absorption_Column.misc.special_functions import jac


def scipy_BVP_solve(Y_a_scaled, Y_b_scaled, z, parameters):
    Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a, Tl_a_guess, Tv_a, P_a = Y_a_scaled
    Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess, Tl_b, Tv_b_guess, P_b = Y_b_scaled

    scales = parameters[0]

    bcs_1 = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_a, Fv_H2O_a, Tl_b, Tv_a, P_a]) / scales

    # Define the system of differential equations for the absorption column
    def odes(z, w):

        differentials = [abs_column(z[i], w[:, i], parameters) for i in range(np.shape(w)[1])]
        del chemical_equilibrium.cache

        return np.array(differentials).T

    # Define the boundary conditions
    def boundary_conditions(bottom, top):
        # Enforce the boundary conditions at the bottom (vapor) and top (liquid)
        Fl_CO2_a_bc, Fl_H2O_a_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Tl_a_bc, Tv_a_bc, P_a_bc = bottom
        Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_b_bc, Fv_H2O_b_bc, Tl_b_bc, Tv_b_bc, P_b_bc = top

        bcs_2 = np.array([Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Tl_b_bc, Tv_a_bc, P_a_bc]) / scales

        # Boundary conditions at the bottom for vapor and at the top for liquid
        return bcs_1 - bcs_2
    
    def fun_jac(x, y):
        return jac(odes, x, y)

    # Initial guess for the solution (constant profiles as initial guess)
    m = len(Y_a_scaled)
    n = 51 # mesh points
    z_2 = np.linspace(z[0], z[-1], n)
    w_guess_scaled = np.array([polynomial_fit(z_2, Y_a_scaled[i]*scales[i], i)/scales[i] for i in range(m)])

    # Solve the BVP
    sol = solve_bvp(odes, boundary_conditions, z_2, w_guess_scaled,
                    fun_jac=fun_jac,
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

