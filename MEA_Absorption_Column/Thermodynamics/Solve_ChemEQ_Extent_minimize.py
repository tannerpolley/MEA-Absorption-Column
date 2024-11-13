import numpy as np
from scipy.optimize import minimize


def solve_ChemEQ(Fl, Tl):

    # Constants and initial guesses provided
    a1, b1, c1, d1 = 164.039636, -707.0056712, -26.40136817, 0
    a2, b2, c2, d2 = 366.061867998774, -13326.25411, -55.68643292, 0

    Fl = Fl + [0, 0, 0]
    Fl_0_T = sum(Fl)
    x_j0 = [Fl[i] / sum(Fl) for i in range(len(Fl))]

    # Compute log(K) values
    log_K1 = a1 + b1 / Tl + c1 * np.log(Tl) + d1 * Tl
    log_K2 = a2 + b2 / Tl + c2 * np.log(Tl) + d2 * Tl
    log_K = [log_K1, log_K2]  # K_i values

    # Stoichiometric coefficients
    v_ij = np.array([[-1, -2, 0, 1, 1, 0], [-1, -1, -1, 1, 0, 1]])

    x_scales = np.array([1e-8, .01, 1, .05, .05, .005])
    eps_scales = np.array([2, .1])
    # scales = np.concatenate([eps_scales, x_scales])
    # Initial guesses
    epsilon_initial = np.array([3.2, 0.27]) / eps_scales  # scaled initial guesses for epsilon
    x_tilde_initial = np.array([1.01400382e-8, 0.026814247, 0.843592377, 0.043197789, 0.039920478, 0.003277311]) / x_scales

    # Initial combined variable (concatenate epsilon and x_tilde for minimize)
    initial_guess = np.concatenate([epsilon_initial, x_tilde_initial])

    # Objective function
    def objective(z, eps_scales, x_scales):
        epsilon = z[:2]*eps_scales # first two elements are epsilon
        x_tilde = z[2:]*x_scales  # remaining elements are x_tilde

        term1 = np.sum(log_K * epsilon/Fl_0_T)
        term2 = np.sum(x_tilde * np.log(x_tilde + 1e-10))  # small term to avoid log(0)
        term3 = np.sum(x_tilde) * np.log(np.sum(x_tilde) + 1e-10)
        obj = -term1 + term2 - term3
        return obj

    # Constraints
    def equality_constraint(z, eps_scales, x_scales):
        epsilon = z[:2]*eps_scales # first two elements are epsilon
        x_tilde = z[2:]*x_scales  # remaining elements are x_tilde
        return [x_tilde[j] - (x_j0[j] + np.sum([v_ij[i, j] * epsilon[i] for i in range(len(epsilon))])/Fl_0_T) for j
                in range(len(x_j0))]

    def inequality_constraint(z, eps_scales, x_scales):
        epsilon = z[:2]*eps_scales # first two elements are epsilon
        x_tilde = z[2:]*x_scales  # remaining elements are x_tilde
        return [x_tilde[j] for j in range(len(x_j0))]

    # Set up constraints for minimize
    constraints = [
        {'type': 'eq', 'fun': equality_constraint, 'args': (eps_scales, x_scales)},  # equality constraint
        {'type': 'ineq', 'fun': inequality_constraint, 'args': (eps_scales, x_scales)}
    ]

    # Bounds (assuming positive values for epsilon and x_tilde)
    bounds = [(0, None)] * len(initial_guess)
    # Optimization
    result = minimize(objective, initial_guess,
                      method='SLSQP', bounds=bounds, constraints=constraints, args=(eps_scales, x_scales)
                      )

    eps = result.x[:2]*eps_scales

    Fl_true = [Fl[j] + np.sum([v_ij[i, j]*eps[i] for i in range(len(eps))]) for j in range(len(Fl))]
    x_true = [Fl_true[i]/np.sum(Fl_true) for i in range(len(Fl_true))]
    return Fl_true, x_true

if __name__ == '__main__':
    Fl = [3.5544993956757, 9.045716779, 69.6840552543039]
    Tl = 333.646741247888

    Fl_true, x_true = solve_ChemEQ(Fl, Tl)
    print(Fl_true)
    print(x_true)
