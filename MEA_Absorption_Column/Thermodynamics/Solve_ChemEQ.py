import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d

# From Akula Appendix of Model Development, Validation, and Part-Load Optimization of a
# MEA-Based Post-Combustion CO2 Capture Process Under SteadyState Flexible Capture Operation

from MEA_Absorption_Column.Properties.Density import liquid_density


def solve_ChemEQ(Fl, Tl):

    # Constants and initial guesses provided
    a1, b1, c1, d1 = 164.039636, -707.0056712, -26.40136817, 0
    a2, b2, c2, d2 = 366.061867998774, -13326.25411, -55.68643292, 0

    Fl = Fl + [0, 0, 0]
    Fl_0_T = sum(Fl)
    x_0 = [Fl[i] / Fl_0_T for i in range(len(Fl))]
    alpha = x_0[0]/x_0[1]
    rho_mol_l, _, _ = liquid_density(Tl, x_0[:3])
    Cl_0 = [x_0[i]*rho_mol_l for i in range(len(x_0))]

    # Compute log(K) values
    log_K1 = a1 + b1 / Tl + c1 * np.log(Tl) + d1 * Tl
    log_K2 = a2 + b2 / Tl + c2 * np.log(Tl) + d2 * Tl
    log_K = [log_K1, log_K2]  # K_i values

    # Stoichiometric coefficients
    v_ij = np.array([[-1, -2, 0, 1, 1, 0], [-1, -1, -1, 1, 0, 1]])

    if alpha > .35:
        guesses = np.array([0.000475252, 1256.753612, 39538.22586,
                            2024.631764, 1871.027883, 153.6038808
                            ])
    elif .35 > alpha > .20:
        guesses = np.array([9.44138E-05,	2582.680049,	39385.02661,	1137.300895,	1079.024729,	58.27616665])

    else:
        guesses = np.array([2.19063E-06, 3541.63341, 39325.97789, 742.3266307, 730.0176094, 12.30902129])
    scales = np.array([1e-5, 1e2, 4e4, 1e3, 1e3, 1e2])
    guesses_scaled = guesses/scales

    def root_solve(guesses_scaled, Cl_0, scales):
        guesses = guesses_scaled*scales
        Cl_CO2_0 = Cl_0[0]
        Cl_MEA_0 = Cl_0[1]
        Cl_H2O_0 = Cl_0[2]
        Cl_CO2 = guesses[0]
        Cl_MEA = guesses[1]
        Cl_H2O = guesses[2]
        Cl_MEAH = guesses[3]
        Cl_MEACOO = guesses[4]
        Cl_HCO3 = guesses[5]

        Kee1 = (Cl_MEAH * Cl_MEACOO) / (Cl_CO2 * Cl_MEA ** 2)  # carbamate
        Kee2 = (Cl_MEAH * Cl_HCO3) / (Cl_CO2 * Cl_MEA * Cl_H2O)  # bicarbonate

        eq1 = Kee1 - np.exp(log_K[0])
        eq2 = Kee2 - np.exp(log_K[1])
        eq3 = Cl_CO2_0 - (Cl_CO2 + Cl_MEAH)
        eq4 = Cl_MEA_0 - (Cl_MEA + Cl_MEAH + Cl_MEACOO)
        eq5 = Cl_H2O_0 - (Cl_H2O + Cl_MEAH - Cl_MEACOO)
        eq6 = Cl_MEAH - (Cl_MEACOO + Cl_HCO3)
        eqs = [eq1, eq2, eq3, eq4, eq5, eq6]
        return eqs

    result = root(root_solve, guesses_scaled, args=(Cl_0, scales), tol=1e-10)

    Cl_true_scaled, solution = result.x, result.message
    Cl_true = Cl_true_scaled*scales

    x_true = [Cl_true[i]/sum(Cl_true) for i in range(len(Cl_true))]

    return np.array(Cl_true), np.array(x_true)


if __name__ == '__main__':
    Fl = [3.5544993956757, 9.045716779, 69.6840552543039]
    Tl = 333.646741247888
    Fl_true, x_true = solve_ChemEQ(Fl, Tl)
    print(Fl_true)
    print(x_true)
