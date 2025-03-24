import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d

# From Akula Appendix of Model Development, Validation, and Part-Load Optimization of a
# MEA-Based Post-Combustion CO2 Capture Process Under SteadyState Flexible Capture Operation

from MEA_Absorption_Column.Properties.Thermophysical_Properties import density


def chemical_equilibrium(Fl, Tl):

    Fl = Fl + [0, 0, 0]
    Fl_0_T = sum(Fl)
    x_0 = [Fl[i] / Fl_0_T for i in range(len(Fl))]
    alpha = x_0[0]/x_0[1]

    # Stoichiometric coefficients

    if not hasattr(chemical_equilibrium, "cache"):
        chemical_equilibrium.cache = {}

    use_previous = True

    if "prev_value" in chemical_equilibrium.cache and use_previous:
        guesses = chemical_equilibrium.cache["prev_value"]
        # print(zi, guesses)
        # use old_val in your calculation...
    else:
        if alpha > .55:
            guesses = np.array([np.float64(0.2898922682406258), np.float64(49.55833510066053), np.float64(37034.44479911041),
                                np.float64(2977.227954765714), np.float64(2281.7371099910897), np.float64(670.8767135873576)])
        elif .55 > alpha > .5:
            guesses = np.array([0.000475252, 1256.753612, 39538.22586,
                                2024.631764, 1871.027883, 153.6038808
                                ])
        
        elif .5 > alpha > .35:
            guesses = np.array([0.000475252, 1256.753612, 39538.22586,
                                2024.631764, 1871.027883, 153.6038808
                                ])
        elif .35 > alpha > .20:
            guesses = np.array([1.44138E-05, 2582.680049, 39385.02661, 1137.300895, 1079.024729, 58.27616665])

        else:
            guesses = np.array([2.19063E-06, 3541.63341, 39325.97789, 742.3266307, 730.0176094, 12.30902129])

    # print(Fl, Tl)
    rho_mol_l, _, _ = density(Tl, x_0[:3], 0, phase='liquid')
    Cl_0 = [x_0[i] * rho_mol_l for i in range(len(x_0))]

    # Constants and initial guesses provided
    # a1, b1, c1, d1 = 164.039636, -707.0056712, -26.40136817, 0
    # a2, b2, c2, d2 = 366.061867998774, -13326.25411, -55.68643292, 0
    
    a1, b1, c1, d1 = 233.4, -3410, -36.8, 0
    a2, b2, c2, d2 = 176.72, -2909, -28.46, 0.0

    # a1, b1, c1, d1 = 234.3, -1204.1, -36.9, -.008
    # a2, b2, c2, d2 = 176.72, -1582.5, -29.2, 0.013

    # Compute log(K) values
    log_K1 = a1 + b1 / Tl + c1 * np.log(Tl) + d1 * Tl
    log_K2 = a2 + b2 / Tl + c2 * np.log(Tl) + d2 * Tl
    log_K = np.array([log_K1, log_K2]) # K_i values

    v_ij = np.array([[-1, -2, 0, 1, 1, 0], [-1, -1, -1, 1, 0, 1]])

    scales = np.array(guesses)
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
        
        Cl = guesses
        #
        Kee1 = float(np.prod([Cl[i]**v_ij[0, i] for i in range(len(Cl))]))
        Kee2 = float(np.prod([Cl[i]**v_ij[1, i] for i in range(len(Cl))]))

        eq1 = (Kee1 - np.exp(log_K[0])) / Kee1
        eq2 = (Kee2 - np.exp(log_K[1])) / Kee2
        eq3 = (Cl_CO2_0 - (Cl_CO2 + Cl_MEAH)) / Cl_CO2_0
        eq4 = (Cl_MEA_0 - (Cl_MEA + Cl_MEAH + Cl_MEACOO)) / Cl_MEA_0
        eq5 = (Cl_H2O_0 - (Cl_H2O + Cl_MEAH - Cl_MEACOO)) / Cl_H2O_0
        eq6 = (Cl_MEAH - (Cl_MEACOO + Cl_HCO3)) / Cl_MEAH
        eqs = np.array([eq1, eq2, eq3, eq4, eq5, eq6])

        return eqs

    result = root(root_solve, guesses_scaled, args=(Cl_0, scales), tol=1e-10)

    Cl_true_scaled, solution, success = result.x, result.message, result.success

    Cl_true = Cl_true_scaled*scales

    chemical_equilibrium.cache["prev_value"] = Cl_true

    x_true = [Cl_true[i]/sum(Cl_true) for i in range(len(Cl_true))]

    return np.array(Cl_true), np.array(x_true)


if __name__ == '__main__':
    Fl = [3.112461691790208, 4.489846767160833, 33.584951199639164]
    # Fl = [6.45947872, 11.22461692, 88.15075214]
    alpha = 0.6932222530521641


    def get_mole_fraction(CO2_loading, amine_concentration=.3):
        MW_MEA = 61.084
        MW_H2O = 18.02

        x_MEA_unloaded = amine_concentration / (MW_MEA / MW_H2O + amine_concentration * (1 - MW_MEA / MW_H2O))
        x_H2O_unloaded = 1 - x_MEA_unloaded

        n_MEA = 100 * x_MEA_unloaded
        n_H2O = 100 * x_H2O_unloaded

        n_CO2 = n_MEA * CO2_loading
        n_tot = n_MEA + n_H2O + n_CO2
        x_CO2, x_MEA, x_H2O = n_CO2 / n_tot, n_MEA / n_tot, n_H2O / n_tot
        return x_CO2, x_MEA, x_H2O

    x = get_mole_fraction(alpha)
    Tl = 330
    rho_mol_l, _, _ = density(Tl, x, 0, phase='liquid')
    Cl_0 = [x[i] * rho_mol_l for i in range(len(x))]

    print(x, alpha)


    Cl_true, x_true = chemical_equilibrium(Fl, Tl)
    print(Cl_true)
    print(x_true)
