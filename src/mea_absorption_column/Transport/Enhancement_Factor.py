import numpy as np
from numpy import exp, array
from scipy.optimize import least_squares

from mea_absorption_column.BVP.robust_core import record_domain_guard
from .domain_guards import DomainGuardError, require_positive

CO2_CONCENTRATION_DIVISOR = 1.04542981654115

def enhancement_factor(Tl, Cl_true, y_CO2, P,
                       H_CO2_mix, kl_CO2, kv_CO2,
                       Dl_CO2, Dl_MEA, Dl_ion, E_type='explicit', diagnostics=None, eta_psi=1.0,
                       co2_concentration_divisor=CO2_CONCENTRATION_DIVISOR):
    enable_enhancement_factor = True

    Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true = Cl_true[:6]
    require_positive("enhancement_factor", diagnostics, co2_concentration_divisor=co2_concentration_divisor)
    Cl_CO2_true = Cl_CO2_true / co2_concentration_divisor
    Dl_MEAH = Dl_ion
    Dl_MEACOO = Dl_ion
    require_positive(
        "enhancement_factor",
        diagnostics,
        Tl=Tl,
        P=P,
        H_CO2_mix=H_CO2_mix,
        kl_CO2=kl_CO2,
        kv_CO2=kv_CO2,
        Dl_CO2=Dl_CO2,
        Dl_MEA=Dl_MEA,
        Dl_ion=Dl_ion,
        Cl_CO2_true=Cl_CO2_true,
        Cl_MEA_true=Cl_MEA_true,
        Cl_H2O_true=Cl_H2O_true,
        Cl_MEAH_true=Cl_MEAH_true,
        Cl_MEACOO_true=Cl_MEACOO_true,
    )

    k2, Ha = hatta_expression(Tl, Cl_MEA_true, Cl_H2O_true, Dl_CO2, kl_CO2)

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

                return np.nan_to_num(array((eq1, eq2), dtype=float), nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)

            solution = least_squares(
                solve,
                array((max(1.0, float(Ha)), .9)),
                bounds=(array((1.0, 1.0e-8)), array((1.0e4, 1.0))),
                max_nfev=100,
            )
            if not solution.success or not np.all(np.isfinite(solution.x)):
                record_domain_guard(
                    diagnostics,
                    "enhancement_factor",
                    f"implicit subsolve fallback: {solution.message}",
                )
                E = _explicit_enhancement_factor(
                    Ha=Ha,
                    Dl_MEA=Dl_MEA,
                    Cl_MEA_true=Cl_MEA_true,
                    Dl_MEAH=Dl_MEAH,
                    Cl_MEAH_true=Cl_MEAH_true,
                    Dl_MEACOO=Dl_MEACOO,
                    Cl_MEACOO_true=Cl_MEACOO_true,
                    Dl_CO2=Dl_CO2,
                    Cl_CO2_true=Cl_CO2_true,
                )
            else:
                E, Cl_MEA_int = solution.x

        elif E_type == 'explicit':

            E = _explicit_enhancement_factor(
                Ha=Ha,
                Dl_MEA=Dl_MEA,
                Cl_MEA_true=Cl_MEA_true,
                Dl_MEAH=Dl_MEAH,
                Cl_MEAH_true=Cl_MEAH_true,
                Dl_MEACOO=Dl_MEACOO,
                Cl_MEACOO_true=Cl_MEACOO_true,
                Dl_CO2=Dl_CO2,
                Cl_CO2_true=Cl_CO2_true,
            )

        else:
            raise ValueError('E_type must be explicit or explicit')

    else:
        E = Ha

    eta_psi = float(eta_psi)
    Psi, Psi_H = film_resistance_expression(E, kl_CO2, kv_CO2, H_CO2_mix, eta_psi)
    require_positive("enhancement_factor", diagnostics, Ha=Ha, E=E, Psi=Psi, Psi_H=Psi_H, eta_psi=eta_psi)

    enhance_factor = [k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, Psi_H, Psi, eta_psi]

    return E, Psi, Psi_H, enhance_factor


def hatta_expression(Tl, Cl_MEA_true, Cl_H2O_true, Dl_CO2, kl_CO2):
    # Luo kinetics used by the conventional film, m³/(mol s).
    k2 = 2.003e4 * exp(-4742.0 / Tl) * Cl_MEA_true + 4.147 * exp(-3110 / Tl) * Cl_H2O_true
    return k2, (k2 * Cl_MEA_true * Dl_CO2) ** .5 / kl_CO2


def _explicit_enhancement_factor(*args, **kwargs):
    E = explicit_enhancement_expression(*args, **kwargs)
    if not np.isfinite(E):
        E = 1.0
    return float(bounded_enhancement_expression(E))


def bounded_enhancement_expression(E, minimum=min, maximum=max):
    return minimum(maximum(E, 1.0), 1.0e4)


def film_resistance_expression(E, kl_CO2, kv_CO2, H_CO2_mix, eta_psi):
    psi = E * kl_CO2 / kv_CO2
    return psi, psi / (psi + H_CO2_mix) * eta_psi


def explicit_enhancement_expression(
    Ha,
    Dl_MEA,
    Cl_MEA_true,
    Dl_MEAH,
    Cl_MEAH_true,
    Dl_MEACOO,
    Cl_MEACOO_true,
    Dl_CO2,
    Cl_CO2_true,
    maximum=max,
):
    floor = 1.0e-30
    R_plus = (Dl_MEA * Cl_MEA_true) / (2 * maximum(Dl_MEAH * Cl_MEAH_true, floor))
    R_minus = (Dl_MEA * Cl_MEA_true) / (2 * maximum(Dl_MEACOO * Cl_MEACOO_true, floor))
    E_hat = (Dl_MEA * Cl_MEA_true) / (2 * maximum(Dl_CO2 * Cl_CO2_true, floor))
    denominator = Ha * (R_plus + R_minus + 2) / maximum(E_hat, floor) + 1
    return 1 + (Ha - 1) / denominator
