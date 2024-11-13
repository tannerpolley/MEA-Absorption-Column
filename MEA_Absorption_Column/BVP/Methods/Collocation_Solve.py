import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from MEA_Absorption_Column.BVP.ABS_Column import abs_column


def collocation_solve(inputs, guesses, df_param, scales):
    Fl_z, Fv_0, Tl_z, Tv_0, z, A, P = inputs

    Fl_CO2_z, Fl_MEA_z, Fl_H2O_z = Fl_z
    Fv_CO2_0, Fv_H2O_0, Fv_N2_0, Fv_O2_0 = Fv_0

    Fl_CO2_0_guess, Fl_H2O_0_guess, Tl_0_guess = guesses

    run_type = 'simulating'

    # Parameters (example values, typically column-specific data and parameters)
    stages = len(z)  # Number of stages in the column (z-axis)
    # stages = 11

    # Parameters (example values, typically column-specific data and parameters)
    collocation_points = 3  # Number of collocation points per stage

    # Define collocation coefficients for polynomial approximation (Lagrange or Radau basis)
    collocation_times = np.array([0, 1 / 2, 1])  # Example for Radau points
    collocation_weights = np.array([1 / 6, 2 / 3, 1 / 6])  # Weights for a simple 3-point Radau

    # Function to evaluate the collocation residuals for a given stage
    def collocation_residuals(w):
        F_CO2_l_c, F_H2O_l_c, F_CO2_v_c, F_H2O_v_c, T_l_c, T_v_c = w

        # Apply absorption dynamics (simplified)
        residuals = []

        # Iterate through collocation points
        for i in range(collocation_points):
            Y = F_CO2_l_c[i], F_H2O_l_c[i], F_CO2_v_c[i], F_H2O_v_c[i], T_l_c[i], T_v_c[i]
            differentials = abs_column(i, Y, scales, Fl_MEA_z, Fv_N2_0, Fv_O2_0, P, A, df_param, run_type)

            # Flow rate changes
            dF_CO2_l_dz = differentials[0]
            dF_H2O_l_dz = differentials[1]
            dF_CO2_v_dz = -differentials[2]
            dF_H2O_v_dz = -differentials[3]

            # Liquid and vapor temperature changes
            dT_l_dz = differentials[4]
            dT_v_dz = -differentials[5]

            # Enforce collocation dynamics (residuals for each variable at each collocation point)
            residuals.extend([
                F_CO2_l_c[i] - (F_CO2_l_c[0] + collocation_weights[i] * dF_CO2_l_dz),
                F_H2O_l_c[i] - (F_H2O_l_c[0] + collocation_weights[i] * dF_H2O_l_dz),
                F_CO2_v_c[i] - (F_CO2_v_c[0] + collocation_weights[i] * dF_CO2_v_dz),
                F_H2O_v_c[i] - (F_H2O_v_c[0] + collocation_weights[i] * dF_H2O_v_dz),
                T_l_c[i] - (T_l_c[0] + collocation_weights[i] * dT_l_dz),
                T_v_c[i] - (T_v_c[0] + collocation_weights[i] * dT_v_dz),
            ])
        return residuals

    # Objective function that aggregates the residuals across all stages
    def objective(w, stages):
        continuity_penalty = 0
        for stage in range(stages):
            start_idx = stage * 6 * collocation_points
            end_idx = (stage + 1) * 6 * collocation_points

            # Extract variables for each collocation point in the stage
            w_stage = w[start_idx:end_idx].reshape(collocation_points, 6)

            # Calculate residuals for this stage
            residuals = collocation_residuals(w_stage.T)

            # Sum of squared residuals for continuity
            continuity_penalty += sum(r ** 2 for r in residuals)
        print(continuity_penalty)
        return continuity_penalty

    # Initial guess for w (each collocation point has [F_CO2_liquid, F_H2O_liquid, F_CO2_vapor, F_H2O_vapor, T_liquid, T_vapor])
    w0 = np.ones(6 * collocation_points * stages)
    w0[0] = Fl_CO2_0_guess / scales[0]
    w0[1] = Fl_H2O_0_guess / scales[1]
    w0[2] = Fv_CO2_0 / scales[2]
    w0[3] = Fv_H2O_0 / scales[3]
    w0[4] = Tl_0_guess / scales[4]
    w0[5] = Tv_0 / scales[5]

    # Set the boundary conditions at the end of the column
    w0[6 * (collocation_points * stages - 1)] = Fl_CO2_z / scales[0]
    w0[6 * (collocation_points * stages - 1) + 1] = Fl_H2O_z / scales[1]
    w0[6 * (collocation_points * stages - 1) + 4] = Tl_z / scales[4]

    # Constraints for initial and final boundary conditions
    constraints = [
        {'type': 'eq', 'fun': lambda w: w[6 * (collocation_points * stages - 1)] - Fl_CO2_z / scales[0]},
        {'type': 'eq', 'fun': lambda w: w[6 * (collocation_points * stages - 1) + 1] - Fl_H2O_z / scales[1]},
        {'type': 'eq', 'fun': lambda w: w[2] - Fv_CO2_0 / scales[2]},
        {'type': 'eq', 'fun': lambda w: w[3] - Fv_H2O_0 / scales[3]},
        {'type': 'eq', 'fun': lambda w: w[6 * (collocation_points * stages - 1) + 4] - Tl_z / scales[4]},
        {'type': 'eq', 'fun': lambda w: w[5] - Tv_0 / scales[5]}
    ]

    bounds = []
    for i in range(collocation_points * stages):
        bounds.append((.1, 10))
        bounds.append((.1, 10))
        bounds.append((.1, 10))
        bounds.append((.1, 10))
        bounds.append((.75, 1.25))
        bounds.append((.75, 1.25))

    # Solve the collocation-based optimization problem
    result = minimize(objective, w0, args=(stages,),
                      bounds=bounds, constraints=constraints,
                      method='trust-constr', tol=.3e-0)

    # Extract and reshape results for visualization
    w_opt = result.x
    F_CO2_liquids = w_opt[0::6 * collocation_points]
    F_H2O_liquids = w_opt[1::6 * collocation_points]
    F_CO2_vapors = w_opt[2::6 * collocation_points]
    F_H2O_vapors = w_opt[3::6 * collocation_points]
    liquid_temperatures = w_opt[4::6 * collocation_points]
    vapor_temperatures = w_opt[5::6 * collocation_points]

    Y_scaled = np.column_stack([F_CO2_liquids, F_H2O_liquids, F_CO2_vapors, F_H2O_vapors,
                                liquid_temperatures, vapor_temperatures]).T

    # print(Y_scaled)
    #
    # Y_0_scaled = [Y_scaled[0, 0], Y_scaled[1, 0], Fv_CO2_0 / scales[2], Fv_CO2_0 / scales[2],
    #               Y_scaled[4, 0], Tv_0 / scales[5]]
    #
    # obj = solve_ivp(abs_column, [z[0], z[-1]], Y_0_scaled,
    #                 args=(scales, Fl_z[1], Fv_0[2], Fv_0[3], P, A, df_param, run_type),
    #                 method='BDF', t_eval=z,
    #                 # vectorized=False,
    #                 # options={'first_step': None, 'max_step': 1e10, 'atol': 1e-5}
    #                 )
    #
    # Y_scaled = obj.y

    success = result.success
    message = result.message

    return Y_scaled, 'Collocation Method', success, message
