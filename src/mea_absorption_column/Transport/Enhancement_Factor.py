import numpy as np
from numpy import exp, array
from scipy.optimize import least_squares

from mea_absorption_column.BVP.robust_core import record_domain_guard
from .domain_guards import require_positive


def enhancement_factor(Tl, Cl_true, y_CO2, P,
                       H_CO2_mix, kl_CO2, kv_CO2,
                       Dl_CO2, Dl_MEA, Dl_ion, E_type='explicit', diagnostics=None, eta_psi=1.0,
                       amine_id='MEA'):
    if str(amine_id).upper() == 'MDEA':
        return _mdea_enhancement_factor(
            Tl, Cl_true, y_CO2, P, H_CO2_mix, kl_CO2, kv_CO2,
            Dl_CO2, Dl_MEA, diagnostics, eta_psi,
        )
    enable_enhancement_factor = True

    Cl_CO2_true, Cl_MEA_true, Cl_H2O_true, Cl_MEAH_true, Cl_MEACOO_true, Cl_HCO3_true = Cl_true[:6]
    Cl_CO2_true = Cl_CO2_true / 1.04542981654115
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

    # Based On Luo from IDAES
    k_MEA = 2.003e4 * exp(-4742.0 / Tl) # m^6/(mol^2*s)
    k_H2O = 4.147 * exp(-3110 / Tl) # m^6/(mol^2*s)
    k2 = (k_MEA * Cl_MEA_true + k_H2O * Cl_H2O_true)  # m^3/(mol*s)

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
    Psi = E * kl_CO2 / kv_CO2
    Psi_H = Psi / (Psi + H_CO2_mix) * eta_psi
    require_positive("enhancement_factor", diagnostics, Ha=Ha, E=E, Psi=Psi, Psi_H=Psi_H, eta_psi=eta_psi)

    enhance_factor = [k2, Cl_MEA_true, Dl_CO2, kl_CO2, Ha, E, Psi_H, Psi, eta_psi]

    return E, Psi, Psi_H, enhance_factor


def _mdea_enhancement_factor(
    Tl, Cl_true, y_CO2, P, H_CO2_mix, kl_CO2, kv_CO2,
    Dl_CO2, Dl_MDEA, diagnostics, eta_psi,
):
    cl_co2, cl_mdea = np.asarray(Cl_true, dtype=float)[:2]
    require_positive(
        "enhancement_factor",
        diagnostics,
        Tl=Tl,
        P=P,
        H_CO2_mix=H_CO2_mix,
        kl_CO2=kl_CO2,
        kv_CO2=kv_CO2,
        Dl_CO2=Dl_CO2,
        Dl_MDEA=Dl_MDEA,
        Cl_CO2_true=cl_co2,
        Cl_MDEA_true=cl_mdea,
    )
    # Camacho et al. (2009), Eq. 23; published units are m3/(kmol s).
    k2 = np.exp(22.4 - 6243.5 / float(Tl)) / 1000.0
    ha = np.sqrt(k2 * cl_mdea * Dl_CO2) / kl_CO2
    interfacial_co2 = max(float(y_CO2) * float(P) / float(H_CO2_mix), 1.0e-30)
    instantaneous = 1.0 + Dl_MDEA * cl_mdea / max(Dl_CO2 * interfacial_co2, 1.0e-30)
    # Camacho Table IV shows E bounded closely by 1 + Ha and Ei.
    enhancement = float(np.clip(min(1.0 + ha, instantaneous), 1.0, 1.0e4))
    eta_psi = float(eta_psi)
    psi = enhancement * kl_CO2 / kv_CO2
    psi_h = psi / (psi + H_CO2_mix) * eta_psi
    require_positive(
        "enhancement_factor", diagnostics,
        Ha=ha, E=enhancement, Psi=psi, Psi_H=psi_h, eta_psi=eta_psi,
    )
    payload = [k2, cl_mdea, Dl_CO2, kl_CO2, ha, enhancement, psi_h, psi, eta_psi]
    return enhancement, psi, psi_h, payload


def _explicit_enhancement_factor(
    Ha,
    Dl_MEA,
    Cl_MEA_true,
    Dl_MEAH,
    Cl_MEAH_true,
    Dl_MEACOO,
    Cl_MEACOO_true,
    Dl_CO2,
    Cl_CO2_true,
):
    floor = 1.0e-30
    R_plus = (Dl_MEA * Cl_MEA_true) / (2 * max(Dl_MEAH * Cl_MEAH_true, floor))
    R_minus = (Dl_MEA * Cl_MEA_true) / (2 * max(Dl_MEACOO * Cl_MEACOO_true, floor))
    E_hat = (Dl_MEA * Cl_MEA_true) / (2 * max(Dl_CO2 * Cl_CO2_true, floor))
    denominator = Ha * (R_plus + R_minus + 2) / max(E_hat, floor) + 1
    E = 1 + (Ha - 1) / denominator
    if not np.isfinite(E):
        E = 1.0
    return float(np.clip(E, 1.0, 1.0e4))
