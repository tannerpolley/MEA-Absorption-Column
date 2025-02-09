from numpy import exp, array
from scipy.optimize import root


def enhancement_factor(Tl, Cl_true, y_CO2, P,
                       H_CO2_mix, kl_CO2, kv_CO2,
                       Dl_CO2, Dl_MEA, Dl_ion, E_type='explicit'):
    enable_enhancement_factor = True

    Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true = Cl_true
    Cl_CO2_true = Cl_CO2_true / 1.04542981654115
    Dl_MEAH = Dl_ion
    Dl_MEACOO = Dl_ion

    k_MEA = 2.003e4 * exp(-4742.0 / Tl)
    k_H2O = 4.147 * exp(-3110 / Tl)
    k2 = (k_MEA * Cl_MEA_true + k_H2O * Cl_H2O_true)  # Pseudo Second Order from IDAES Luo

    Ha = (k2 * Cl_MEA_true * Dl_CO2) ** .5 / kl_CO2

    if enable_enhancement_factor:

        if E_type == 'implicit':

            def solve(x):

                E, Υ_MEA_int = x
                KH = E * kl_CO2 / kv_CO2 / (E * kl_CO2 / kv_CO2 + H_CO2_mix)
                Cl_CO2_int = (y_CO2 * P / KH + Cl_CO2_true) / (H_CO2_mix / KH + 1)
                Υ_CO2_bulk = Cl_CO2_true / Cl_CO2_int

                Υ_MEAH = 1 + Dl_MEA * Cl_MEA_true * ((1 - Υ_MEA_int) / (2 * Dl_MEAH * Cl_MEAH_true))
                Υ_MEACOO = 1 + Dl_MEA * Cl_MEA_true * ((1 - Υ_MEA_int) / (2 * Dl_MEACOO * Cl_MEACOO_true))
                Υ_CO2_int = Υ_CO2_bulk * Υ_MEAH * Υ_MEACOO / Υ_MEA_int ** 2

                E_inst = 1 + Dl_MEA * Cl_MEA_true / (2 * Dl_CO2 * Cl_CO2_int)

                eq1 = E - Ha * Υ_MEA_int ** (1 / 2) * (1 - Υ_CO2_int) / (1 - Υ_CO2_bulk)
                eq2 = E - (1 + (E_inst - 1) * (1 - Υ_MEA_int) / (1 - Υ_CO2_bulk))

                return eq1, eq2

            E, Cl_MEA_int = root(solve, array((65., .9))).x

        elif E_type == 'explicit':

            R_plus = (Dl_MEA * Cl_MEA_true) / (2 * Dl_MEAH * Cl_MEAH_true)
            R_minus = (Dl_MEA * Cl_MEA_true) / (2 * Dl_MEACOO * Cl_MEACOO_true)
            E_hat = (Dl_MEA * Cl_MEA_true) / (2 * Dl_CO2 * Cl_CO2_true)
            E = 1 + (Ha - 1) / (Ha * (R_plus + R_minus + 2) / E_hat + 1)

        else:
            raise ValueError('E_type must be explicit or explicit')

    else:
        E = Ha

    Psi = E * kl_CO2 / kv_CO2
    KH = Psi / (Psi + H_CO2_mix)

    enhance_factor = [k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, KH, Psi]

    return E, Psi, KH, enhance_factor
