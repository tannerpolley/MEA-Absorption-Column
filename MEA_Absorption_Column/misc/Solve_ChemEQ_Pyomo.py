from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, log, NonNegativeReals, RangeSet, \
    Param, Reals
import numpy as np


def solve_ChemEQ(Fl, Tl):

    # Given data (these need to be defined before running the model)
    Fl += [0, 0, 0]
    Fl = np.array(Fl)
    nT_0 = np.sum(Fl)
    x_j0 = Fl / np.sum(Fl)
    a1, b1, c1, d1 = 164.039636, -707.0056712, -26.40136817, 0
    a2, b2, c2, d2 = 366.061867998774, -13326.25411, -55.68643292, 0
    K1 = np.exp(a1 + b1 / Tl + c1 * np.log(Tl) + d1 * Tl)
    K2 = np.exp(a2 + b2 / Tl + c2 * np.log(Tl) + d2 * Tl)
    K = [K1, K2]  # Example values for K_i, length should be 2

    eps_g = np.array([4, 0.25])
    x_g = np.array([1.01400382e-8, 0.026814247, 0.843592377, 0.043197789, 0.039920478, 0.003277311])

    v_ij = np.array([[-1, -2, 0, 1, 1, 0], [-1, -1, -1, 1, 0, 1]])
    # Initialize the model
    model = ConcreteModel()

    # Sets
    model.i = RangeSet(1, 2)  # Index for i from 1 to 2
    model.j = RangeSet(1, 6)  # Index for j from 1 to 6

    # Parameters
    model.K = Param(model.i, initialize={i + 1: K[i] for i in range(2)})
    model.x_j0 = Param(model.j, initialize={j + 1: x_j0[j] for j in range(6)})
    # model.x_j0 = Param(model.j, initialize={j + 1: x_j0[j] for j in range(6)})
    model.v = Param(model.i, model.j, initialize={(i + 1, j + 1): v_ij[i][j] for i in range(2) for j in range(6)})

    # Variables
    model.epsilon = Var(model.i,
                        initialize={i + 1: eps_g[i] for i in range(2)},
                        within=NonNegativeReals,
                        bounds=(0.01, 5))

    # Define scaling factors for each x_tilde variable (adjust based on expected magnitude)
    scaling_factors = np.array([1e-8, .01, 1, .05, .05, .005])
    eps_scales = np.array([2, .1])

    # Parameters for scaling factors
    model.scaling_factors = Param(model.j, initialize={j + 1: scaling_factors[j] for j in range(6)})
    model.scaling_factors_eps = Param(model.i, initialize={i + 1: eps_scales[i] for i in range(2)})

    # Define scaled x_tilde variables
    model.x_scaled = Var(model.j,
                         initialize={j + 1: x_g[j] / scaling_factors[j] for j in range(6)},
                         within=NonNegativeReals,
                         bounds=(0.0, None))

    model.eps_scaled = Var(model.i,
                        initialize={i + 1: eps_g[i] / eps_scales[i] for i in range(2)},
                        within=NonNegativeReals,
                        bounds=(0.0, None))

    # Small constant to avoid log(0)
    small_constant = 0


    # Objective Function
    def objective_rule(model):
        term1 = -sum(log(model.K[i]) * (model.eps_scaled[i] * model.scaling_factors_eps[i])/nT_0 for i in model.i)

        term2 = sum(model.x_scaled[j] * model.scaling_factors[j] * log(
            model.x_scaled[j] * model.scaling_factors[j] + small_constant)
                    for j in model.j)

        total_x_scaled = sum(model.x_scaled[j] * model.scaling_factors[j] for j in model.j)
        term3 = -total_x_scaled * log(total_x_scaled + small_constant)

        return term1 + term2 + term3


    model.objective = Objective(rule=objective_rule, sense='minimize')


    # Redefine the x_tilde constraint to use scaled variables
    def x_tilde_constraint_rule(model, j):
        return model.x_scaled[j] * model.scaling_factors[j] == model.x_j0[j] + sum(
            model.v[i, j] * (model.eps_scaled[i] * model.scaling_factors_eps[i]) for i in model.i)/nT_0


    model.x_tilde_constraint = Constraint(
        model.j,
        rule=x_tilde_constraint_rule)


    def positivity_constraint_rule(model, j):
        # Ensure that the computed x_tilde[j] is non-negative (redundant with variable domain, but ensures clarity)
        return model.x_scaled[j] * model.scaling_factors[j] >= 0


    model.positivity_constraint = Constraint(
        model.j,
        rule=positivity_constraint_rule)

    # Solve the model with IPOPT for non-linear optimization
    solver = SolverFactory('ipopt', executable=r'C:\Users\Tanner\anaconda3\Library\bin\ipopt.exe')
    solver.options['tol'] = 1e-6  # Tolerance for convergence
    # solver.options['mu_strategy'] = 'adaptive'  # Adaptive strategy for feasibility
    solver.options['acceptable_tol'] = 1e-6  # More lenient feasibility tolerance
    solution = solver.solve(model, tee=False)

    # Display results

    ξ = [(model.eps_scaled[i].value * model.scaling_factors_eps[i]) for i in model.i]

    # print('obj:', model.objective.expr())
    # print('Extent:', ξ)
    # print('Flow Rate:', Fl)

    Fl_true = [Fl[j] + np.sum([v_ij[i, j]*ξ[i] for i in range(len(ξ))]) for j in range(len(Fl))]
    x_true = [Fl_true[i]/np.sum(Fl_true) for i in range(len(Fl_true))]

    return Fl_true, x_true

if __name__ == '__main__':
    Fl = [3.5544993956757, 9.045716779, 69.6840552543039]
    Tl = 333.646741247888

    Fl_true, x_true = solve_ChemEQ(Fl, Tl)
    print(Fl_true)
    print(x_true)