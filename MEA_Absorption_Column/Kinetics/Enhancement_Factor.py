from numpy import exp, array
from scipy.optimize import root


def enhancement_factor(Tl, Cl, Cl_MEA_true, kl_CO2, Dl_CO2):
    k2b = 3.95e10 * exp(-6864 / Tl)  # Greer -
    k2c = 10 ** (
                10.49 - 2200 / Tl) / 1000  # Hakita - Reaction rate of CO2 in aqueous MEA-AMP solution: Experiment and modeling
    k2d = exp(10 - 904.6 / Tl) / 1000  # Freguia - Modeling of CO2 Capture by Aqueous Monoethanolamine
    k2e = 3.1732e9 * exp(-4936.6 / Tl) * Cl_MEA_true * 1e-6  # Putta, Svendsen, Knuutila 2017 Eqn. 42

    # Akula Model Development, Validation, and Optimization of an MEA-Based
    # Post-Combustion CO2 Capture Process under Part-Load and Variable
    # Capture Operations
    k_MEA = 2.003e10 * exp(-4742 / Tl)
    k_H2O = 4.147e6 * exp(-3110 / Tl)
    k2f = (k_MEA * Cl[1] + k_H2O * Cl[2]) * 5e-8  # kmol -> mol added fudge factor, still in development

    # print(k2e, k2f)

    # Choose the rate constant to use
    k_rxn = k2e

    # Can be seen from Eq 8: Kvamsdal - Effects of the Temperature Bulge in CO2 Absorption from Flue Gas by Aqueous Monoethanolamine
    # Also from Moore 2021
    # Akula Model Development, Validation, and Optimization of an MEA-Based Post-Combustion CO2 Capture Process under Part-Load and Variable Capture Operations
    Ha = (k_rxn * Cl_MEA_true * Dl_CO2) ** .5 / kl_CO2

    E = Ha

    kinetics = [k_rxn, Ha, E]

    return E, kinetics


def enhancement_factor_2(Tl, y_CO2, P, Cl_true,
                         H_CO2_mix, kl_CO2, kv_CO2,
                         Dl_CO2, Dl_MEA, Dl_ion):
    Cl_CO2_true, Cl_MEA_true, Cl_MEAH_true, Cl_MEACOO_true = Cl_true[0], Cl_true[1], Cl_true[3], Cl_true[4]
    k2 = 3.1732e9 * exp(-4936.6 / Tl) * Cl_MEA_true * 1e-6  # Putta, Svendsen, Knuutila 2017 Eqn. 42

    Ha = (k2 * Cl_MEA_true * Dl_CO2) ** .5 / kl_CO2
    Dl_MEAH = Dl_ion
    Dl_MEACOO = Dl_ion

    def solve(x):
        E, Cl_MEA_int = x
        k = E * kl_CO2 / kv_CO2
        KH = k / (k + H_CO2_mix)
        Cl_MEAH_int = 1 + Dl_MEA * Cl_MEA_true * ((1 - Cl_MEA_int) / (2 * Dl_MEAH * Cl_MEAH_true))
        Cl_MEACOO_int = 1 + Dl_MEA * Cl_MEA_true * ((1 - Cl_MEA_int) / (2 * Dl_MEACOO * Cl_MEACOO_true))
        Cl_CO2_bulk = Cl_CO2_true * (H_CO2_mix / KH + 1) / (y_CO2 * P / KH + Cl_CO2_true)
        Cl_CO2_bulk_eq = Cl_CO2_bulk * Cl_MEAH_int * Cl_MEACOO_int / Cl_MEA_int
        sing_CO2_ratio = (1-Cl_CO2_bulk_eq)/(1-Cl_CO2_bulk)
        instant_E_minus_one = Dl_MEA*Cl_MEA_true*Cl_CO2_bulk/(2*Dl_CO2*Cl_CO2_true)

        eq1 = E - Ha * Cl_MEA_int ** (1 / 2) * sing_CO2_ratio
        eq2 = ((E - 1)*(1 - Cl_CO2_bulk)) - (instant_E_minus_one*(1 - Cl_MEA_int))

        return eq1, eq2

    E, Cl_MEA_int = root(solve, array((30, .9))).x

    kinetics = [k2, Ha, E]

    return E, kinetics
