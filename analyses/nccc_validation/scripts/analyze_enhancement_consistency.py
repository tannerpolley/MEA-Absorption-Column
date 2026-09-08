from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import (  # noqa: E402
    SPECIES_9,
    tabulated_epcsaft_reactive_chemical_equilibrium,
)
from mea_absorption_column.Thermodynamics.epcsaft_v02 import (  # noqa: E402
    fugacity_coefficients,
    mixture,
    molar_density_value,
    state,
)
from mea_absorption_column.Thermodynamics.thermo_models import (  # noqa: E402
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
)


ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "analyses/nccc_validation/inputs/retained_reactive_case3c"
PROFILE = INPUTS / "film_states.csv"
REACTIVE_TABLE = INPUTS / "speciation_table.csv"
REACTIVE_SUMMARY = INPUTS / "speciation_table_summary.json"
FINAL = ROOT / "analyses/nccc_validation/results/final"
RESULT_TABLE = FINAL / "tables/retained_reactive_case3c_enhancement_formulations.csv"
SUMMARY = FINAL / "tables/retained_reactive_case3c_enhancement_summary.json"
FILM_RUNS = FINAL / "tables/retained_reactive_case3c_film_runs.csv"
COMPARISON_TABLE = FINAL / "tables/retained_reactive_case3c_enhancement_film_comparison.csv"
ISSUE_URL = "https://github.com/tannerpolley/MEA-Absorption-Column/issues/17"
CO2_DIVISOR = 1.04542981654115
CHARGES = np.asarray((0, 0, 0, 1, -1, -1, -2, 1, -1), dtype=float)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def explicit_enhancement(
    hatta: float,
    c_co2: float,
    c_mea: float,
    c_meah: float,
    c_meacoo: float,
    d_co2: float,
    d_mea: float,
    d_ion: float,
) -> float:
    r_plus = d_mea * c_mea / (2.0 * d_ion * c_meah)
    r_minus = d_mea * c_mea / (2.0 * d_ion * c_meacoo)
    e_inf_minus_one = d_mea * c_mea / (2.0 * d_co2 * c_co2)
    return 1.0 + (hatta - 1.0) / (
        1.0 + hatta * (r_plus + r_minus + 2.0) / e_inf_minus_one
    )


def kinetic_state(temperature_k: float, c_mea: float, c_water: float, model: str) -> tuple[float, float]:
    if model == "Luo":
        k_mea = 2.003e4 * math.exp(-4742.0 / temperature_k)
        k_water = 4.147 * math.exp(-3110.0 / temperature_k)
    elif model == "Putta":
        k_mea = 3.1732e3 * math.exp(-4936.6 / temperature_k)
        k_water = 1.0882e2 * math.exp(-3900.0 / temperature_k)
    else:
        raise ValueError(f"Unknown kinetic model: {model}")
    return k_mea * c_mea + k_water * c_water, k_mea


def local_fugacity_slope(
    model,
    composition: np.ndarray,
    temperature_k: float,
    pressure_pa: float,
    relative_step: float,
) -> tuple[float, float, float]:
    points = []
    for sign in (-1.0, 1.0):
        varied = composition.copy()
        varied[0] *= 1.0 + sign * relative_step
        varied[1:] *= (1.0 - varied[0]) / float(np.sum(varied[1:]))
        liquid = state(
            model,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            composition=varied,
            phase="liquid",
        )
        density = molar_density_value(liquid)
        phi_co2 = fugacity_coefficients(liquid)[0]
        points.append((varied[0] * density, varied[0] * phi_co2 * pressure_pa))
    (c_minus, f_minus), (c_plus, f_plus) = points
    slope = (f_plus - f_minus) / (c_plus - c_minus)
    return slope, c_minus, c_plus


def implicit_enhancement(
    *,
    hatta: float,
    c_co2: float,
    c_mea: float,
    c_meah: float,
    c_meacoo: float,
    d_co2: float,
    d_mea: float,
    d_ion: float,
    k_liquid: float,
    k_vapor: float,
    vapor_pressure_co2: float,
    vapor_fugacity_co2: float,
    liquid_fugacity_co2: float,
    interface_slope: float,
    relation: str,
) -> dict[str, float | str]:
    def residual(values: np.ndarray) -> np.ndarray:
        enhancement, y_mea_i = values
        psi = enhancement * k_liquid / k_vapor
        if relation == "legacy_henry":
            resistance_fraction = psi / (psi + interface_slope)
            c_co2_i = (
                vapor_pressure_co2 / resistance_fraction + c_co2
            ) / (interface_slope / resistance_fraction + 1.0)
        elif relation == "epcsaft_local":
            c_co2_i = c_co2 + (vapor_fugacity_co2 - liquid_fugacity_co2) / (
                psi + interface_slope
            )
        else:
            raise ValueError(f"Unknown interface relation: {relation}")
        y_co2_b = c_co2 / c_co2_i
        y_meah_i = 1.0 + d_mea * c_mea * (1.0 - y_mea_i) / (2.0 * d_ion * c_meah)
        y_meacoo_i = 1.0 + d_mea * c_mea * (1.0 - y_mea_i) / (2.0 * d_ion * c_meacoo)
        y_co2_star = y_co2_b * y_meah_i * y_meacoo_i / y_mea_i**2
        e_infinity = 1.0 + d_mea * c_mea / (2.0 * d_co2 * c_co2_i)
        denominator = 1.0 - y_co2_b
        return np.asarray(
            (
                enhancement
                - hatta * math.sqrt(y_mea_i) * (1.0 - y_co2_star) / denominator,
                enhancement
                - 1.0
                - (e_infinity - 1.0) * (1.0 - y_mea_i) / denominator,
            ),
            dtype=float,
        )

    solutions = []
    attempts = []
    for guess in ((max(1.01, hatta), 0.9), (10.0, 0.5), (100.0, 0.1)):
        solved = least_squares(
            residual,
            np.asarray(guess, dtype=float),
            bounds=(np.asarray((1.0, 1.0e-8)), np.asarray((1.0e5, 1.0))),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=2000,
        )
        norm = float(np.max(np.abs(residual(solved.x))))
        attempts.append(
            {
                "initial_E": guess[0],
                "initial_y_MEA_interface": guess[1],
                "success": bool(solved.success),
                "E": float(solved.x[0]),
                "y_MEA_interface": float(solved.x[1]),
                "residual_inf": norm,
            }
        )
        if solved.success and np.all(np.isfinite(solved.x)) and norm <= 1.0e-7:
            solutions.append((float(solved.x[0]), float(solved.x[1]), norm))
    if len(solutions) != 3:
        return {
            "outcome": "numerical_convergence_failure",
            "diagnostic": f"{len(solutions)} of 3 initial guesses met the residual tolerance",
            "initial_guess_results_json": json.dumps(attempts, separators=(",", ":")),
        }
    spread = (max(value[0] for value in solutions) - min(value[0] for value in solutions)) / max(
        value[0] for value in solutions
    )
    if spread > 1.0e-3:
        return {
            "outcome": "numerical_convergence_failure",
            "diagnostic": f"initial-guess enhancement spread {spread:.6g} exceeded 0.001",
            "initial_guess_results_json": json.dumps(attempts, separators=(",", ":")),
        }
    enhancement = float(np.mean([value[0] for value in solutions]))
    y_mea_i = float(np.mean([value[1] for value in solutions]))
    psi = enhancement * k_liquid / k_vapor
    if relation == "legacy_henry":
        fraction = psi / (psi + interface_slope)
        c_co2_i = (vapor_pressure_co2 / fraction + c_co2) / (interface_slope / fraction + 1.0)
    else:
        c_co2_i = c_co2 + (vapor_fugacity_co2 - liquid_fugacity_co2) / (psi + interface_slope)
    return {
        "outcome": "evaluated",
        "diagnostic": "",
        "E": enhancement,
        "y_MEA_interface": y_mea_i,
        "C_CO2_interface_mol_m3": c_co2_i,
        "residual_inf": max(value[2] for value in solutions),
        "initial_guess_relative_spread": spread,
        "initial_guess_results_json": json.dumps(attempts, separators=(",", ":")),
    }


def _load_profile() -> pd.DataFrame:
    return pd.read_csv(PROFILE)


def _base_row(source: pd.Series, formulation: str) -> dict[str, object]:
    return {
        "Position": float(source.Position),
        "height_m": float(source.height_m),
        "formulation": formulation,
        "outcome": "evaluated",
        "diagnostic": "",
        "temperature_K": float(source.Tl),
        "pressure_Pa": float(source.P),
        "f_CO2_vapor_Pa": float(source.fv_CO2),
        "f_CO2_liquid_Pa": float(source.fl_CO2),
        "driving_force_Pa": float(source.DF_CO2),
        "current_E": float(source.E),
        "current_flux_mol_s_m": float(source.Nl_CO2),
    }


def _finish_row(
    row: dict[str, object],
    *,
    enhancement: float,
    hatta: float,
    k_observed: float,
    k_liquid: float,
    k_vapor: float,
    area: float,
    interface_slope: float,
) -> None:
    psi = enhancement * k_liquid / k_vapor
    psi_h = psi / (psi + interface_slope)
    flux = k_vapor * area * float(row["driving_force_Pa"]) * psi_h
    row.update(
        {
            "k_observed_per_s": k_observed,
            "Ha": hatta,
            "E": enhancement,
            "Psi": psi,
            "interface_slope_Pa_m3_mol": interface_slope,
            "Psi_H": psi_h,
            "predicted_flux_mol_s_m": flux,
            "E_ratio_to_current": enhancement / float(row["current_E"]),
            "flux_ratio_to_current": flux / float(row["current_flux_mol_s_m"]),
        }
    )


def generate() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    os.environ["MEA_EPCSAFT_REACTIVE_TABLE"] = str(REACTIVE_TABLE)
    profile = _load_profile()
    rows: list[dict[str, object]] = []
    slope_checks = []
    for _, source in profile.iterrows():
        apparent_flows = source[["Fl_CO2", "Fl_MEA", "Fl_H2O"]].to_numpy(float)
        _, composition = tabulated_epcsaft_reactive_chemical_equilibrium(
            apparent_flows, float(source.Tl), diagnostics={}
        )
        temperature_k = float(source.Tl)
        pressure_pa = float(source.P)
        model = mixture(str(MEA_THERMODYNAMICS_EPCSAFT_DATASET), SPECIES_9, temperature_k)
        bulk_state = state(
            model,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            composition=composition,
            phase="liquid",
        )
        eos_density = molar_density_value(bulk_state)
        eos_concentrations = composition * eos_density
        slope_1, _, _ = local_fugacity_slope(model, composition, temperature_k, pressure_pa, 1.0e-3)
        slope_2, _, _ = local_fugacity_slope(model, composition, temperature_k, pressure_pa, 5.0e-4)
        slope_difference = abs(slope_1 - slope_2) / max(abs(slope_2), 1.0e-30)
        slope_checks.append(slope_difference)
        local_slope_valid = (
            math.isfinite(slope_1)
            and slope_1 > 0.0
            and math.isfinite(slope_2)
            and slope_2 > 0.0
            and slope_difference <= 0.02
        )

        c = {
            "CO2": float(source.Cl_CO2_true),
            "MEA": float(source.Cl_MEA_true),
            "H2O": float(source.Cl_H2O_true),
            "MEAH": float(source.Cl_MEAH_true),
            "MEACOO": float(source.Cl_MEACOO_true),
        }
        d_co2, d_mea, d_ion = float(source.Dl_CO2), float(source.Dl_MEA), float(source.Dl_ion)
        k_liquid, k_vapor, area = float(source.kl_CO2), float(source.kv_CO2), float(source.a_eA)
        legacy_h = float(source.H_CO2_mix)

        for formulation, kinetics, divisor in (
            ("idaes_explicit_luo_current", "Luo", CO2_DIVISOR),
            ("idaes_explicit_luo_uncorrected", "Luo", 1.0),
            ("idaes_explicit_putta_uncorrected", "Putta", 1.0),
        ):
            row = _base_row(source, formulation)
            k_observed, _ = kinetic_state(temperature_k, c["MEA"], c["H2O"], kinetics)
            hatta = math.sqrt(k_observed * c["MEA"] * d_co2) / k_liquid
            enhancement = explicit_enhancement(
                hatta,
                c["CO2"] / divisor,
                c["MEA"],
                c["MEAH"],
                c["MEACOO"],
                d_co2,
                d_mea,
                d_ion,
            )
            _finish_row(
                row,
                enhancement=enhancement,
                hatta=hatta,
                k_observed=k_observed,
                k_liquid=k_liquid,
                k_vapor=k_vapor,
                area=area,
                interface_slope=legacy_h,
            )
            rows.append(row)

        k_observed, _ = kinetic_state(temperature_k, c["MEA"], c["H2O"], "Luo")
        hatta = math.sqrt(k_observed * c["MEA"] * d_co2) / k_liquid
        legacy = _base_row(source, "gaspar_implicit_luo_legacy_henry")
        legacy_result = implicit_enhancement(
            hatta=hatta,
            c_co2=c["CO2"] / CO2_DIVISOR,
            c_mea=c["MEA"],
            c_meah=c["MEAH"],
            c_meacoo=c["MEACOO"],
            d_co2=d_co2,
            d_mea=d_mea,
            d_ion=d_ion,
            k_liquid=k_liquid,
            k_vapor=k_vapor,
            vapor_pressure_co2=float(source.y_CO2) * pressure_pa,
            vapor_fugacity_co2=float(source.fv_CO2),
            liquid_fugacity_co2=float(source.fl_CO2),
            interface_slope=legacy_h,
            relation="legacy_henry",
        )
        legacy.update(legacy_result)
        if legacy_result["outcome"] == "evaluated":
            _finish_row(
                legacy,
                enhancement=float(legacy_result["E"]),
                hatta=hatta,
                k_observed=k_observed,
                k_liquid=k_liquid,
                k_vapor=k_vapor,
                area=area,
                interface_slope=legacy_h,
            )
        rows.append(legacy)

        local = _base_row(source, "gaspar_implicit_luo_epcsaft_local")
        local["local_slope_relative_step_difference"] = slope_difference
        local["eos_density_mol_m3"] = eos_density
        local["eos_C_CO2_bulk_mol_m3"] = float(eos_concentrations[0])
        local["charge_residual"] = float(np.dot(CHARGES, composition))
        if not local_slope_valid:
            local.update(
                {
                    "outcome": "not_established",
                    "diagnostic": "local ePC-SAFT fugacity slope failed positivity or refinement check",
                }
            )
        else:
            local_c = {
                "CO2": float(eos_concentrations[0]),
                "MEA": float(eos_concentrations[1]),
                "H2O": float(eos_concentrations[2]),
                "MEAH": float(eos_concentrations[3]),
                "MEACOO": float(eos_concentrations[4]),
            }
            local_k, _ = kinetic_state(temperature_k, local_c["MEA"], local_c["H2O"], "Luo")
            local_hatta = math.sqrt(local_k * local_c["MEA"] * d_co2) / k_liquid
            local_result = implicit_enhancement(
                hatta=local_hatta,
                c_co2=local_c["CO2"],
                c_mea=local_c["MEA"],
                c_meah=local_c["MEAH"],
                c_meacoo=local_c["MEACOO"],
                d_co2=d_co2,
                d_mea=d_mea,
                d_ion=d_ion,
                k_liquid=k_liquid,
                k_vapor=k_vapor,
                vapor_pressure_co2=float(source.y_CO2) * pressure_pa,
                vapor_fugacity_co2=float(source.fv_CO2),
                liquid_fugacity_co2=float(source.fl_CO2),
                interface_slope=slope_2,
                relation="epcsaft_local",
            )
            local.update(local_result)
            if local_result["outcome"] == "evaluated":
                _finish_row(
                    local,
                    enhancement=float(local_result["E"]),
                    hatta=local_hatta,
                    k_observed=local_k,
                    k_liquid=k_liquid,
                    k_vapor=k_vapor,
                    area=area,
                    interface_slope=slope_2,
                )
        rows.append(local)

    result = pd.DataFrame(rows)
    result["interface_concentration_ratio"] = (
        result.C_CO2_interface_mol_m3 / result.eos_C_CO2_bulk_mol_m3
    )
    implicit = result.formulation.str.startswith("gaspar_implicit") & result.outcome.eq("evaluated")
    result["physical_check_pass"] = (
        result.outcome.eq("evaluated")
        & result.E.ge(1.0)
        & result.predicted_flux_mol_s_m.ge(0.0)
        & (~implicit | result.y_MEA_interface.gt(0.0) & result.y_MEA_interface.le(1.0))
    )
    current = result[result.formulation == "idaes_explicit_luo_current"]
    reproduction_error = np.max(np.abs(current.E / current.current_E - 1.0))
    evaluated = result[result.outcome == "evaluated"]
    aggregates = (
        evaluated.groupby("formulation")
        .agg(
            state_count=("Position", "size"),
            median_E=("E", "median"),
            median_E_ratio_to_current=("E_ratio_to_current", "median"),
            median_flux_ratio_to_current=("flux_ratio_to_current", "median"),
            minimum_E=("E", "min"),
            maximum_E=("E", "max"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    wheel_url = json.loads(
        importlib.metadata.distribution("epcsaft").read_text("direct_url.json") or "{}"
    ).get("url", "")
    wheel_path = Path(unquote(urlparse(wheel_url).path))
    position_one = result.loc[result.Position.eq(1.0)].copy()
    comparison = position_one[
        ["Position", "height_m", "formulation", "E", "predicted_flux_mol_s_m", "current_flux_mol_s_m", "flux_ratio_to_current", "outcome", "diagnostic"]
    ].rename(columns={"formulation": "method_id"})
    comparison.insert(3, "method_class", "enhancement_correlation")
    film_runs = pd.read_csv(FILM_RUNS)
    film = film_runs.loc[
        film_runs.outcome.eq("evaluated")
        & film_runs.mesh_points.eq(film_runs.mesh_points.max())
        & film_runs.initial_flux_factor.eq(1.0)
    ].iloc[0]
    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [
                    {
                        "Position": 1.0,
                        "height_m": float(position_one.height_m.iloc[0]),
                        "method_id": "nonlinear_reactive_film_luo_epcsaft",
                        "method_class": "mechanistic_reaction_diffusion",
                        "E": math.nan,
                        "predicted_flux_mol_s_m": float(film.predicted_flux_mol_s_m),
                        "current_flux_mol_s_m": float(film.retained_column_flux_mol_s_m),
                        "flux_ratio_to_current": float(
                            film.predicted_flux_mol_s_m / film.retained_column_flux_mol_s_m
                        ),
                        "outcome": str(film.outcome),
                        "diagnostic": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    summary = {
        "issue": ISSUE_URL,
        "scientific_question": "Does enhancement/interface inconsistency explain the retained Case 3C transfer decrease?",
        "state_count": len(profile),
        "formulation_count": int(result.formulation.nunique()),
        "evaluated_row_count": int((result.outcome == "evaluated").sum()),
        "failed_row_count": int((result.outcome != "evaluated").sum()),
        "maximum_current_E_relative_reproduction_error": float(reproduction_error),
        "maximum_local_slope_relative_step_difference": float(max(slope_checks)),
        "local_interface_concentration_ratio_median": float(
            result.loc[
                result.formulation.eq("gaspar_implicit_luo_epcsaft_local"),
                "interface_concentration_ratio",
            ].median()
        ),
        "local_interface_concentration_ratio_maximum": float(
            result.loc[
                result.formulation.eq("gaspar_implicit_luo_epcsaft_local"),
                "interface_concentration_ratio",
            ].max()
        ),
        "physical_check_failure_count": int(
            (result.outcome.eq("evaluated") & ~result.physical_check_pass).sum()
        ),
        "parameter_document_sha256": sha256(Path(MEA_THERMODYNAMICS_EPCSAFT_DATASET) / "parameters.json"),
        "engine_wheel_sha256": sha256(wheel_path),
        "reactive_table_sha256": sha256(REACTIVE_TABLE),
        "reactive_table_summary_sha256": sha256(REACTIVE_SUMMARY),
        "aggregates": aggregates,
        "failed_rows": result.loc[result.outcome != "evaluated", ["Position", "formulation", "outcome", "diagnostic"]].to_dict(orient="records"),
        "mechanistic_film_flux_ratio_to_current_at_position_one": float(
            film.predicted_flux_mol_s_m / film.retained_column_flux_mol_s_m
        ),
        "mechanistic_film_run_table_sha256": sha256(FILM_RUNS),
        "claim_boundary": "Correlation comparison over retained Case 3C states plus one accepted Stage A mechanistic film state at Position 1; no column-wide film, WWC, reversible-kinetics, or Maxwell-Stefan validation claim.",
    }
    return result, comparison, summary


def main() -> None:
    result, comparison, summary = generate()
    RESULT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULT_TABLE, index=False)
    comparison.to_csv(COMPARISON_TABLE, index=False)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["maximum_current_E_relative_reproduction_error"] > 1.0e-3:
        raise RuntimeError("Current explicit calculation did not reproduce the retained enhancement profile")
    if summary["failed_row_count"]:
        raise RuntimeError("one or more retained enhancement states failed")


if __name__ == "__main__":
    main()
