import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, root

# From Akula Appendix of Model Development, Validation, and Part-Load Optimization of a
# MEA-Based Post-Combustion CO2 Capture Process Under SteadyState Flexible Capture Operation

from mea_absorption_column.BVP.robust_core import record_domain_guard
from mea_absorption_column.Properties.Thermophysical_Properties import density
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    ensure_epcsaft_importable,
)


SPECIES_6 = ("CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-")
REACTIONS_6 = (
    {"CO2": -1.0, "MEA": -2.0, "MEAH+": 1.0, "MEACOO-": 1.0},
    {"CO2": -1.0, "MEA": -1.0, "H2O": -1.0, "MEAH+": 1.0, "HCO3-": 1.0},
)
REACTION_NAMES_6 = ("carbamate", "bicarbonate")
EPCSAFT_CHEMISTRY_CACHE_T_DIGITS = int(os.environ.get("MEA_EPCSAFT_CHEMISTRY_CACHE_T_DIGITS", "2"))
EPCSAFT_CHEMISTRY_CACHE_X_DIGITS = int(os.environ.get("MEA_EPCSAFT_CHEMISTRY_CACHE_X_DIGITS", "6"))
EPCSAFT_CHEMISTRY_CACHE_P_ROUND_PA = float(os.environ.get("MEA_EPCSAFT_CHEMISTRY_CACHE_P_ROUND_PA", "10.0"))


def chemical_equilibrium(Fl, Tl):

    Fl = Fl + [0, 0, 0]
    Fl_0_T = sum(Fl)
    x_0 = [Fl[i] / Fl_0_T for i in range(len(Fl))]
    alpha = x_0[0]/x_0[1]

    # Stoichiometric coefficients

    if not hasattr(chemical_equilibrium, "cache"):
        chemical_equilibrium.cache = {}

    use_previous = True

    if (
        "prev_value" in chemical_equilibrium.cache
        and use_previous
        and np.all(np.isfinite(chemical_equilibrium.cache["prev_value"]))
        and np.all(np.asarray(chemical_equilibrium.cache["prev_value"]) > 0.0)
    ):
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

    scales = np.maximum(np.array(guesses, dtype=float), 1.0e-12)
    guesses_scaled = np.clip(guesses / scales, 1.0e-12, np.inf)

    def root_solve(guesses_scaled, Cl_0, scales):
        guesses = np.maximum(guesses_scaled*scales, 1.0e-30)
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
        Kee1 = float(np.exp(np.sum(v_ij[0] * np.log(Cl))))
        Kee2 = float(np.exp(np.sum(v_ij[1] * np.log(Cl))))

        eq1 = (Kee1 - np.exp(log_K[0])) / max(abs(Kee1), 1.0e-30)
        eq2 = (Kee2 - np.exp(log_K[1])) / max(abs(Kee2), 1.0e-30)
        eq3 = (Cl_CO2_0 - (Cl_CO2 + Cl_MEAH)) / max(abs(Cl_CO2_0), 1.0e-30)
        eq4 = (Cl_MEA_0 - (Cl_MEA + Cl_MEAH + Cl_MEACOO)) / max(abs(Cl_MEA_0), 1.0e-30)
        eq5 = (Cl_H2O_0 - (Cl_H2O + Cl_MEAH - Cl_MEACOO)) / max(abs(Cl_H2O_0), 1.0e-30)
        eq6 = (Cl_MEAH - (Cl_MEACOO + Cl_HCO3)) / max(abs(Cl_MEAH), 1.0e-30)
        eqs = np.array([eq1, eq2, eq3, eq4, eq5, eq6])

        return np.nan_to_num(eqs, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)

    total_concentration = max(float(sum(Cl_0[:3])), 1.0)
    lower = np.full(6, 1.0e-16)
    upper = np.maximum(total_concentration * 2.0 / scales, guesses_scaled * 20.0)
    fast_result = root(root_solve, guesses_scaled, args=(Cl_0, scales), tol=1.0e-10)
    fast_Cl = np.asarray(fast_result.x, dtype=float) * scales
    if (
        fast_result.success
        and np.all(np.isfinite(fast_Cl))
        and np.all(fast_Cl > 0.0)
        and np.linalg.norm(root_solve(fast_result.x, Cl_0, scales)) < 1.0e-5
    ):
        result = fast_result
    else:
        result = least_squares(
            root_solve,
            np.clip(guesses_scaled, lower * 10.0, upper * 0.5),
            args=(Cl_0, scales),
            bounds=(lower, upper),
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
            max_nfev=60,
        )

    Cl_true_scaled, solution, success = result.x, result.message, result.success

    Cl_true = np.maximum(Cl_true_scaled*scales, 1.0e-30)

    chemical_equilibrium.cache["prev_value"] = Cl_true

    x_true = [Cl_true[i]/sum(Cl_true) for i in range(len(Cl_true))]

    return np.array(Cl_true), np.array(x_true)


def chemical_equilibrium_with_model(
    Fl,
    Tl,
    *,
    model="legacy",
    P=101325.0,
    diagnostics=None,
):
    normalized_model = (model or "legacy").lower()
    if normalized_model in {"legacy", "legacy_concentration", "local"}:
        return chemical_equilibrium(Fl, Tl)
    if normalized_model in {
        "epcsaft_reactive_six",
        "epcsaft_reactive_six_concentration",
        "epcsaft_six_concentration",
    }:
        return epcsaft_reactive_chemical_equilibrium(
            Fl,
            Tl,
            P=P,
            standard_state="concentration",
            calibrate_activity_to_legacy=False,
            diagnostics=diagnostics,
        )
    if normalized_model in {
        "epcsaft_reactive_six_activity",
        "epcsaft_six_activity",
    }:
        return epcsaft_reactive_chemical_equilibrium(
            Fl,
            Tl,
            P=P,
            standard_state="mole_fraction_activity",
            calibrate_activity_to_legacy=False,
            diagnostics=diagnostics,
        )
    if normalized_model in {
        "epcsaft_reactive_six_activity_converted",
        "epcsaft_six_activity_converted",
    }:
        return epcsaft_reactive_chemical_equilibrium(
            Fl,
            Tl,
            P=P,
            standard_state="mole_fraction_activity",
            log_k_basis="concentration_to_mole_fraction",
            calibrate_activity_to_legacy=False,
            diagnostics=diagnostics,
        )
    if normalized_model in {
        "epcsaft_reactive_six_activity_rebased",
        "epcsaft_six_activity_rebased",
    }:
        return epcsaft_reactive_chemical_equilibrium(
            Fl,
            Tl,
            P=P,
            standard_state="mole_fraction_activity",
            calibrate_activity_to_legacy=True,
            diagnostics=diagnostics,
        )
    raise ValueError(
        "Choose legacy, epcsaft_reactive_six_concentration, "
        "epcsaft_reactive_six_activity, epcsaft_reactive_six_activity_converted, "
        "or epcsaft_reactive_six_activity_rebased."
    )


def epcsaft_reactive_chemical_equilibrium(
    Fl,
    Tl,
    *,
    P=101325.0,
    standard_state="concentration",
    log_k_basis="native",
    calibrate_activity_to_legacy=False,
    diagnostics=None,
):
    apparent_x = _apparent_liquid_mole_fraction(Fl)
    pressure = float(P)
    temperature = float(Tl)
    cache_key = _epcsaft_chemistry_cache_key(
        apparent_x,
        temperature,
        pressure,
        standard_state,
        log_k_basis,
        calibrate_activity_to_legacy,
    )
    cache = getattr(epcsaft_reactive_chemical_equilibrium, "cache", {})
    cached = cache.get(cache_key)
    if cached is not None:
        _increment_diagnostic(diagnostics, "epcsaft_chemistry_cache_hits")
        return cached[0].copy(), cached[1].copy()
    _increment_diagnostic(diagnostics, "epcsaft_chemistry_cache_misses")

    epcsaft = _epcsaft_module()
    initial_x = _initial_epcsaft_chemistry_guess(apparent_x, temperature)
    mixture = _epcsaft_six_mixture(epcsaft, temperature, initial_x)
    if calibrate_activity_to_legacy:
        log_k = _log_k_from_activity_state(mixture, temperature, pressure, initial_x)
    else:
        log_k = _epcsaft_reaction_log_constants(
            apparent_x,
            temperature,
            pressure,
            standard_state=standard_state,
            log_k_basis=log_k_basis,
        )
    reactions = [
        epcsaft.ReactionDefinition(
            reaction,
            value,
            name=name,
            standard_state=standard_state,
        )
        for reaction, value, name in zip(REACTIONS_6, log_k, REACTION_NAMES_6)
    ]
    options = epcsaft.ReactiveSpeciationOptions(
        max_iterations=60,
        tolerance=1.0e-8,
        mass_tolerance=1.0e-7,
        charge_tolerance=1.0e-7,
        reaction_tolerance=1.0e-7,
        damping=0.7,
        return_best_effort=True,
    )
    started = time.perf_counter()
    result = epcsaft.solve_reactive_speciation(
        species=list(SPECIES_6),
        mixture_factory=lambda x, T, P: mixture,
        T=temperature,
        P=pressure,
        balances={
            "amine_total": {"MEA": 1.0, "MEAH+": 1.0, "MEACOO-": 1.0},
            "carbon_total": {"CO2": 1.0, "MEACOO-": 1.0, "HCO3-": 1.0},
            "water_total": {"H2O": 1.0},
        },
        totals={
            "amine_total": float(apparent_x[1]),
            "carbon_total": float(apparent_x[0]),
            "water_total": float(apparent_x[2]),
        },
        reactions=reactions,
        initial_x=initial_x,
        options=options,
    )
    _increment_diagnostic(
        diagnostics,
        "epcsaft_chemistry_solve_s",
        time.perf_counter() - started,
    )
    if not result.success:
        record_domain_guard(
            diagnostics,
            "chemical_equilibrium",
            f"ePC-SAFT reactive speciation did not converge: {result.message}",
        )
        raise RuntimeError(f"ePC-SAFT reactive speciation failed: {result.message}")
    x_true = np.asarray([float(result.x[name]) for name in SPECIES_6], dtype=float)
    x_true = np.maximum(x_true, 1.0e-30)
    x_true = x_true / float(np.sum(x_true))
    rho_mol_l, _, _ = density(temperature, apparent_x[:3], pressure, phase="liquid")
    Cl_true = x_true * float(rho_mol_l)
    cache[cache_key] = (Cl_true.copy(), x_true.copy())
    epcsaft_reactive_chemical_equilibrium.cache = cache
    return Cl_true, x_true


def legacy_log_constants(temperature_K):
    T = float(temperature_K)
    a1, b1, c1, d1 = 233.4, -3410.0, -36.8, 0.0
    a2, b2, c2, d2 = 176.72, -2909.0, -28.46, 0.0
    return (
        float(a1 + b1 / T + c1 * np.log(T) + d1 * T),
        float(a2 + b2 / T + c2 * np.log(T) + d2 * T),
    )


def _epcsaft_reaction_log_constants(
    apparent_x,
    temperature,
    pressure,
    *,
    standard_state,
    log_k_basis,
):
    log_k = np.asarray(legacy_log_constants(temperature), dtype=float)
    if str(log_k_basis).lower() in {"native", "concentration"}:
        return tuple(float(value) for value in log_k)
    if str(log_k_basis).lower() != "concentration_to_mole_fraction":
        raise ValueError(f"Unknown ePC-SAFT reaction log-K basis: {log_k_basis!r}")
    if str(standard_state).lower() not in {"mole_fraction_activity", "ideal_mole_fraction"}:
        return tuple(float(value) for value in log_k)

    rho_mol_l, _, _ = density(float(temperature), np.asarray(apparent_x[:3], dtype=float), float(pressure), phase="liquid")
    rho_mol_l = max(float(rho_mol_l), 1.0e-30)
    converted = []
    for reaction, value in zip(REACTIONS_6, log_k):
        stoich_sum = float(sum(reaction.values()))
        converted.append(float(value - stoich_sum * math.log(rho_mol_l)))
    return tuple(converted)


def _apparent_liquid_mole_fraction(Fl):
    flows = np.asarray(Fl[:3], dtype=float)
    flows = np.maximum(flows, 1.0e-30)
    total = float(np.sum(flows))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Liquid apparent flows must have a positive finite sum.")
    return flows / total


def _initial_epcsaft_chemistry_guess(apparent_x, temperature):
    try:
        if not hasattr(epcsaft_reactive_chemical_equilibrium, "legacy_guess_cache"):
            epcsaft_reactive_chemical_equilibrium.legacy_guess_cache = {}
        legacy_Cl, legacy_x = chemical_equilibrium(list(apparent_x[:3]), float(temperature))
        return np.asarray(legacy_x, dtype=float)
    except Exception:
        seed = np.asarray([apparent_x[0], apparent_x[1], apparent_x[2], 1.0e-8, 1.0e-8, 1.0e-8], dtype=float)
        seed = np.maximum(seed, 1.0e-14)
        return seed / float(np.sum(seed))


def _epcsaft_module():
    ensure_epcsaft_importable()
    import epcsaft

    return epcsaft


def _epcsaft_six_mixture(epcsaft, temperature, x):
    dataset = Path(MEA_THERMODYNAMICS_EPCSAFT_DATASET)
    user_options = None
    user_options_path = dataset / "user_options.json"
    if user_options_path.exists():
        user_options = json.loads(user_options_path.read_text(encoding="utf-8"))
    return epcsaft.ePCSAFTMixture.from_dataset(
        str(dataset),
        list(SPECIES_6),
        np.asarray(x, dtype=float),
        float(temperature),
        user_options=user_options,
    )


def _log_k_from_activity_state(mixture, temperature, pressure, x):
    state = mixture.state(T=float(temperature), P=float(pressure), x=np.asarray(x, dtype=float), phase="liq")
    gamma = state.activity_coefficient(species=list(SPECIES_6))
    values = []
    for reaction in REACTIONS_6:
        total = 0.0
        for species, coefficient in reaction.items():
            idx = SPECIES_6.index(species)
            activity = max(float(x[idx]) * float(gamma[species]), 1.0e-300)
            total += float(coefficient) * math.log(activity)
        values.append(float(total))
    return tuple(values)


def _epcsaft_chemistry_cache_key(
    apparent_x,
    temperature,
    pressure,
    standard_state,
    log_k_basis,
    calibrate_activity_to_legacy,
):
    pressure_increment = max(EPCSAFT_CHEMISTRY_CACHE_P_ROUND_PA, 1.0e-12)
    return (
        str(standard_state),
        str(log_k_basis),
        bool(calibrate_activity_to_legacy),
        float(np.round(float(temperature), EPCSAFT_CHEMISTRY_CACHE_T_DIGITS)),
        float(np.round(float(pressure) / pressure_increment) * pressure_increment),
        tuple(float(np.round(value, EPCSAFT_CHEMISTRY_CACHE_X_DIGITS)) for value in apparent_x),
    )


def _increment_diagnostic(diagnostics, key, amount=1):
    if diagnostics is None:
        return
    diagnostics[key] = diagnostics.get(key, 0) + amount


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
