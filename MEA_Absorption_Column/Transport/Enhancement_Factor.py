from numpy import exp, array
from scipy.optimize import root


def enhancement_factor(Tl, y_CO2, P, Cl_true,
                       H_CO2_mix, kl_CO2, kv_CO2,
                       Dl_CO2, Dl_MEA, Dl_ion):

    enable_enhancement_factor = True

    Cl_CO2_true, Cl_MEA_true, Cl_MEAH_true, Cl_MEACOO_true = Cl_true[0], Cl_true[1], Cl_true[3], Cl_true[4]

    # k2b = 3.95e10 * exp(-6864 / Tl)  # Greer -
    # k2c = 10 ** (
    #         10.49 - 2200 / Tl) / 1000  # Hakita - Reaction rate of CO2 in aqueous MEA-AMP solution: Experiment and modeling
    # k2d = exp(10 - 904.6 / Tl) / 1000  # Freguia - Modeling of CO2 Capture by Aqueous Monoethanolamine
    k_MEA = 2.003e10 * exp(-4742 / Tl)
    k_H2O = 4.147e6 * exp(-3110 / Tl)
    k2 = (k_MEA * Cl_MEA_true + k_H2O * Cl_true[2]) * 1e-6 # 5e-8  # kmol -> mol added fudge factor, still in development Akula Model Development, Validation, and Optimization of an MEA-Based

    k2 = 3.1732e9 * exp(-4936.6 / Tl) * Cl_MEA_true * 1e-6  # Putta, Svendsen, Knuutila 2017 Eqn. 42

    Ha = (k2 * Cl_MEA_true * Dl_CO2) ** .5 / kl_CO2

    Dl_MEAH = Dl_ion
    Dl_MEACOO = Dl_ion

    def solve(x):

        E, Υ_MEA_int = x
        KH = E * kl_CO2 / kv_CO2 / (E * kl_CO2 / kv_CO2 + H_CO2_mix)
        Cl_CO2_int = (y_CO2 * P / KH + Cl_CO2_true)/(H_CO2_mix / KH + 1)
        Υ_CO2_bulk = Cl_CO2_true / Cl_CO2_int

        Υ_MEAH = 1 + Dl_MEA * Cl_MEA_true * ((1 - Υ_MEA_int) / (2 * Dl_MEAH * Cl_MEAH_true))
        Υ_MEACOO = 1 + Dl_MEA * Cl_MEA_true * ((1 - Υ_MEA_int) / (2 * Dl_MEACOO * Cl_MEACOO_true))
        Υ_CO2_int = Υ_CO2_bulk * Υ_MEAH * Υ_MEACOO / Υ_MEA_int**2

        E_inst = 1 + Dl_MEA * Cl_MEA_true / (2 * Dl_CO2 * Cl_CO2_int)

        eq1 = E - Ha * Υ_MEA_int ** (1 / 2) * (1 - Υ_CO2_int) / (1 - Υ_CO2_bulk)
        eq2 = E - (1 + (E_inst - 1) * (1 - Υ_MEA_int)/(1 - Υ_CO2_bulk))

        return eq1, eq2

    if enable_enhancement_factor:

        E, Cl_MEA_int = root(solve, array((30, .9))).x
    else:
        E = Ha

    KH = E * kl_CO2 / kv_CO2 / (E * kl_CO2 / kv_CO2 + H_CO2_mix)

    enhance_factor = [k2, Ha, E, KH]

    return KH, enhance_factor
