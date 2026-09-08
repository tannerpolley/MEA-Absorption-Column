import csv
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import least_squares, root

# From Akula Appendix of Model Development, Validation, and Part-Load Optimization of a
# MEA-Based Post-Combustion CO2 Capture Process Under SteadyState Flexible Capture Operation

from mea_absorption_column.BVP.robust_core import record_domain_guard
from mea_absorption_column.Properties.Thermophysical_Properties import density
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    ensure_epcsaft_importable,
)
from mea_absorption_column.Thermodynamics.reactive_bundle import (
    solve_homogeneous_reactive_state,
)


SPECIES_9 = ("CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-")


def _set_diagnostic_max(diagnostics, key, value):
    if diagnostics is None:
        return
    if not np.isfinite(value):
        raise ValueError(f"Non-finite reactive equilibrium diagnostic: {key}")
    diagnostics[key] = max(float(diagnostics.get(key, 0.0)), float(value))


def chemical_equilibrium(Fl, Tl):

    Fl = Fl + [0, 0, 0]
    Fl_0_T = sum(Fl)
    x_0 = [Fl[i] / Fl_0_T for i in range(len(Fl))]
    alpha = x_0[0]/x_0[1]

    # Stoichiometric coefficients

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

    # The legacy constants are tabulated on a kmol/m3-compatible concentration basis.
    # The absorber state uses mol/m3, so reactions with sum(nu) = -1 need K/1000.
    log_K1 = a1 + b1 / Tl + c1 * np.log(Tl) + d1 * Tl
    log_K2 = a2 + b2 / Tl + c2 * np.log(Tl) + d2 * Tl
    log_K = np.array([log_K1, log_K2]) - np.log(1000.0) # K_i values

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

    Cl_true_scaled = result.x

    Cl_true = np.maximum(Cl_true_scaled*scales, 1.0e-30)


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
    if normalized_model in {
        "epcsaft_reactive_nine_tabulated",
        "epcsaft_nine_tabulated",
    }:
        return tabulated_epcsaft_reactive_chemical_equilibrium(
            Fl,
            Tl,
            diagnostics=diagnostics,
        )
    if normalized_model in {
        "epcsaft_reactive_nine",
        "epcsaft_reactive_nine_bundle",
        "epcsaft_reactive_nine_activity",
        "epcsaft_nine_activity",
        "epcsaft_full_species_activity",
    }:
        return bundle_reactive_chemical_equilibrium(Fl, Tl, P=P, diagnostics=diagnostics)
    if normalized_model in {
        "epcsaft_reactive_nine_activity_rebased",
        "epcsaft_nine_activity_rebased",
        "epcsaft_full_species_activity_rebased",
    }:
        return epcsaft_reactive_chemical_equilibrium(
            Fl,
            Tl,
            P=P,
            standard_state="mole_fraction_activity",
            species_set="nine",
            calibrate_activity_to_legacy=True,
            diagnostics=diagnostics,
        )
    if normalized_model in {
        "epcsaft_reactive_nine_activity_converted",
        "epcsaft_nine_activity_converted",
        "epcsaft_full_species_activity_converted",
    }:
        return epcsaft_reactive_chemical_equilibrium(
            Fl,
            Tl,
            P=P,
            standard_state="mole_fraction_activity",
            log_k_basis="concentration_to_mole_fraction",
            species_set="nine",
            calibrate_activity_to_legacy=False,
            diagnostics=diagnostics,
        )
    raise ValueError(
        "Choose legacy, epcsaft_reactive_six_concentration, "
        "epcsaft_reactive_six_activity, epcsaft_reactive_six_activity_converted, "
        "epcsaft_reactive_six_activity_rebased, or epcsaft_reactive_nine_activity_rebased."
    )


def bundle_reactive_chemical_equilibrium(Fl, Tl, *, P=101325.0, diagnostics=None):
    started = time.perf_counter()
    result = solve_homogeneous_reactive_state(
        str(MEA_THERMODYNAMICS_EPCSAFT_DATASET), float(Tl), float(P), Fl
    )
    _increment_diagnostic(diagnostics, "epcsaft_chemistry_solve_s", time.perf_counter() - started)
    for name in (
        "balance_inf_norm",
        "reaction_affinity_inf_norm",
        "pressure_relative_inf_norm",
        "kkt_stationarity_inf_norm",
    ):
        _set_diagnostic_max(diagnostics, f"epcsaft_chemistry_{name}", result["evidence"][name])
    composition = np.asarray(result["composition"], dtype=float)
    return composition * float(result["density_mol_m3"]), composition.copy()


def _reactive_speciation_table(path_text):
    path = Path(path_text)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "evaluated"]
    if len(rows) < 3:
        raise ValueError(f"Reactive ePC-SAFT table needs at least three evaluated states: {path}")
    species_columns = tuple(f"x_{name}" for name in (
        "carbon-dioxide",
        "monoethanolamine",
        "water",
        "protonated-monoethanolamine",
        "carbamate-anion",
        "bicarbonate-anion",
        "carbonate-anion",
        "hydronium-cation",
        "hydroxide-anion",
    ))
    points = np.asarray(
        [(float(row["temperature_k"]), float(row["loading"])) for row in rows],
        dtype=float,
    )
    mole_fractions = np.asarray(
        [[float(row[column]) for column in species_columns] for row in rows],
        dtype=float,
    )
    nitrogen_fraction = mole_fractions[:, 1] + mole_fractions[:, 3] + mole_fractions[:, 4]
    amounts_per_mol_mea = mole_fractions / nitrogen_fraction[:, None]
    return (
        LinearNDInterpolator(points, amounts_per_mol_mea),
        NearestNDInterpolator(points, amounts_per_mol_mea),
        LinearNDInterpolator(points, np.log(amounts_per_mol_mea[:, 0])),
    )


def tabulated_epcsaft_reactive_chemical_equilibrium(Fl, Tl, *, diagnostics=None):
    table_path = os.environ.get("MEA_EPCSAFT_REACTIVE_TABLE")
    if not table_path:
        raise RuntimeError(
            "MEA_EPCSAFT_REACTIVE_TABLE must name the certified reactive ePC-SAFT table."
        )
    apparent = _apparent_liquid_mole_fraction(Fl)
    loading = float(apparent[0] / apparent[1])
    linear, nearest, log_co2 = _reactive_speciation_table(table_path)
    amounts = np.asarray(linear(float(Tl), loading), dtype=float)
    if not np.all(np.isfinite(amounts)):
        amounts = np.asarray(nearest(float(Tl), loading), dtype=float)
        _increment_diagnostic(diagnostics, "epcsaft_chemistry_interpolation_fallback_count")
    else:
        amounts[0] = math.exp(float(log_co2(float(Tl), loading)))
        _increment_diagnostic(diagnostics, "epcsaft_chemistry_table_hits")
    x_true = np.maximum(amounts, 1.0e-30)
    x_true /= float(np.sum(x_true))
    mea_mass_fraction = (
        float(Fl[1]) * 0.061080535833333255
        / (float(Fl[1]) * 0.061080535833333255 + float(Fl[2]) * 0.018015221250000022)
    )
    _set_diagnostic_max(
        diagnostics,
        "epcsaft_chemistry_max_mea_mass_fraction_deviation",
        abs(mea_mass_fraction - 0.3),
    )
    rho_mol_l, _, _ = density(float(Tl), apparent[:3], 0.0, phase="liquid")
    return x_true * float(rho_mol_l), x_true


def epcsaft_reactive_chemical_equilibrium(
    Fl,
    Tl,
    *,
    P=101325.0,
    standard_state="concentration",
    log_k_basis="native",
    species_set="six",
    calibrate_activity_to_legacy=False,
    diagnostics=None,
):
    ensure_epcsaft_importable()
    raise RuntimeError(
        "The legacy epcsaft_reactive_* modes are intentionally unavailable after the "
        "ePC-SAFT 0.2 API cutover. The new typed chemical-equilibrium API requires "
        "independently sourced, dimensionless reaction constants with an explicit "
        "standard-state conversion; the archived locally rebased constants do not meet "
        "that admission contract. Use epcsaft_ionic for the supported fugacity-only lane."
    )

def _apparent_liquid_mole_fraction(Fl):
    flows = np.asarray(Fl[:3], dtype=float)
    flows = np.maximum(flows, 1.0e-30)
    total = float(np.sum(flows))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Liquid apparent flows must have a positive finite sum.")
    return flows / total


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
