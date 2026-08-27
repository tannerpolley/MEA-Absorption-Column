from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_enhancement_consistency import implicit_enhancement


ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "analyses/nccc_validation/inputs/issue17_enhancement_comparison"
FIXED_STATES = INPUTS / "current_fugacity_only_fixed_states.csv"
FULL_PROFILE = INPUTS / "current_fugacity_only_full_profile.csv"
IDENTITY = INPUTS / "identity.json"
REACTIVE_STATES = ROOT / "analyses/nccc_validation/inputs/retained_reactive_case3c/film_states.csv"
TABLES = ROOT / "analyses/nccc_validation/results/final/tables"
RESULT_TABLE = TABLES / "issue17_fugacity_only_enhancement_formulations.csv"
AGGREGATE_TABLE = TABLES / "issue17_fugacity_only_enhancement_aggregates.csv"
STAGE_TABLE = TABLES / "issue17_enhancement_stage_outcomes.csv"
SUMMARY = TABLES / "issue17_fugacity_only_enhancement_summary.json"

FORMULATIONS = (
    "EF-GF-IMPLICIT",
    "EF-AOP-78-PUBLISHED-MEA",
    "EF-AOP-73-CORRECTED-MEA",
    "EF-CURRENT",
)
SOURCE = {
    "EF-GF-IMPLICIT": (
        "EF-GF-IMPLICIT",
        "Gaspar and Fosbøl (2015), DOI 10.1016/j.ces.2015.08.023, Eqs. 10, 12, 13, 17, "
        "22, and MEA/CO2 m=n=1 Eq. 23; Zotero parent PPDL8269, PDF PW5C8YSL; "
        "retained bulk-state Ha and legacy Henry relation",
    ),
    "EF-AOP-78-PUBLISHED-MEA": (
        "EF-AOP-78-PUBLISHED-MEA",
        "Allan, Ostace, and Polley (2025), DOI 10.1016/j.compchemeng.2025.109312, "
        "Eq. 78, p. 16; accepted PDF SHA-256 0c4e4a9d4303d4381651f42dcad00f2a77aa1ad4905a6297eacb3e096ed79249",
    ),
    "EF-AOP-73-CORRECTED-MEA": (
        "EF-AOP-73-CORRECTED-MEA",
        "Allan letter to Jozsef Gaspar, paragraph beginning 'However, while preparing the new manuscript', "
        "replacement Eq. 73; DOCX SHA-256 752a69bba61b398cc8951dc964b1c5b8ce22ec30d6b4e4591a8c289d129b6f71; "
        "direct m=nu_A=nu_B=nu_C=nu_D=1 specialization",
    ),
    "EF-CURRENT": (
        "EF-CURRENT",
        "src/mea_absorption_column/Transport/Enhancement_Factor.py and "
        "idaes/models_extra/column_models/enhancement_factor_model_pseudo_second_order_explicit.py",
    ),
}
UNITS = {
    "Position": "normalized packed height, bottom to top",
    "height_m": "m",
    "Tl": "K",
    "Tv": "K",
    "P": "Pa",
    "fv_CO2": "Pa",
    "fl_CO2": "Pa",
    "DF_CO2": "Pa",
    "H_CO2_mix": "Pa m^3 mol^-1",
    "Dl_species": "m^2 s^-1",
    "kl_CO2": "m s^-1",
    "kv_CO2": "mol m^-2 s^-1 Pa^-1",
    "a_eA": "m^2 m^-1 packed height",
    "rho_mol_l": "mol m^-3",
    "k2": "m^3 mol^-1 s^-1",
    "Ha": "dimensionless",
    "E": "dimensionless",
    "Psi": "Pa m^3 mol^-1",
    "Nl_CO2": "mol s^-1 m^-1 packed height",
}


def fixed_input_units(columns) -> dict[str, str]:
    units = {}
    for column in columns:
        if column in UNITS:
            units[column] = UNITS[column]
        elif column.startswith(("Fl_", "Fv_")):
            units[column] = "mol s^-1"
        elif column.startswith("Cl_"):
            units[column] = "mol m^-3"
        elif column.startswith("Dl_"):
            units[column] = "m^2 s^-1"
        elif column.startswith(("x_", "y_")):
            units[column] = "mole fraction"
        elif column in {"profile_id"}:
            units[column] = "identifier"
        elif column in {"CO2_divisor", "Ha", "E", "Psi_H", "eta_psi"}:
            units[column] = "dimensionless"
        elif column in {"Nl_CO2", "Nv_CO2"}:
            units[column] = "mol s^-1 m^-1 packed height"
        else:
            raise KeyError(f"Missing fixed-input unit for {column}")
    return units


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equation_groups(row: pd.Series) -> tuple[float, float, float, float]:
    c_co2 = float(row.Cl_CO2_true) / float(row.CO2_divisor)
    r_plus = float(row.Dl_MEA * row.Cl_MEA_true / (2.0 * row.Dl_ion * row.Cl_MEAH_true))
    r_minus = float(row.Dl_MEA * row.Cl_MEA_true / (2.0 * row.Dl_ion * row.Cl_MEACOO_true))
    q = float(row.Dl_MEA * row.Cl_MEA_true / (2.0 * row.Dl_CO2 * c_co2))
    return r_plus, r_minus, q, r_plus + r_minus + 1.0


def explicit_equation(formulation: str, hatta: float, q: float, s_mea: float) -> float:
    if formulation == "EF-AOP-78-PUBLISHED-MEA":
        return hatta * (1.0 + s_mea / q) / (1.0 + hatta * s_mea)
    if formulation == "EF-AOP-73-CORRECTED-MEA":
        return hatta * math.sqrt(1.0 + s_mea / q) / (
            1.0 + hatta * math.sqrt(s_mea) / q
        )
    if formulation == "EF-CURRENT":
        return 1.0 + (hatta - 1.0) / (1.0 + hatta * (s_mea + 1.0) / q)
    raise ValueError(f"Explicit evaluator does not own {formulation}")


def scalar_reference(formulation: str, hatta: float, q: float, r_plus: float, r_minus: float) -> float:
    """Literal scalar transcription used independently of row evaluation."""
    s = r_plus + r_minus + 1.0
    if formulation == "EF-AOP-78-PUBLISHED-MEA":
        return (hatta * (1.0 + s / q)) / (1.0 + hatta * s)
    if formulation == "EF-AOP-73-CORRECTED-MEA":
        numerator = hatta * ((1.0 + s / q) ** 0.5)
        denominator = 1.0 + (hatta / q) * (s**0.5)
        return numerator / denominator
    if formulation == "EF-CURRENT":
        return 1.0 + (hatta - 1.0) / (1.0 + hatta * (r_plus + r_minus + 2.0) / q)
    raise ValueError(f"Scalar reference does not own {formulation}")


def _branch_count(attempts_json: str) -> int:
    attempts = [item for item in json.loads(attempts_json) if item["success"]]
    branches: list[float] = []
    for attempt in attempts:
        value = float(attempt["E"])
        if not any(abs(value - branch) / max(abs(value), abs(branch), 1.0) <= 1.0e-3 for branch in branches):
            branches.append(value)
    return len(branches)


def _evaluate_row(source: pd.Series, formulation: str) -> dict[str, object]:
    started = time.perf_counter()
    row = source.to_dict()
    r_plus, r_minus, q, s_mea = equation_groups(source)
    row.update(
        {
            "formulation": formulation,
            "equation_id": SOURCE[formulation][0],
            "source_locator": SOURCE[formulation][1],
            "fixed_input_units_json": json.dumps(
                fixed_input_units(source.index), sort_keys=True, separators=(",", ":")
            ),
            "R_plus": r_plus,
            "R_minus": r_minus,
            "E_infinity_star": q + 1.0,
            "Q": q,
            "S_MEA": s_mea,
            "attempted": "yes",
            "fallback_used": False,
            "initial_guess_results_json": "",
            "initial_guess_relative_spread": 0.0,
            "implicit_branch_count": 0,
            "C_CO2_interface_mol_m3": math.nan,
            "p_CO2_interface_Pa": math.nan,
            "y_MEA_interface": math.nan,
        }
    )
    if formulation == "EF-GF-IMPLICIT":
        solved = implicit_enhancement(
            hatta=float(source.Ha),
            c_co2=float(source.Cl_CO2_true / source.CO2_divisor),
            c_mea=float(source.Cl_MEA_true),
            c_meah=float(source.Cl_MEAH_true),
            c_meacoo=float(source.Cl_MEACOO_true),
            d_co2=float(source.Dl_CO2),
            d_mea=float(source.Dl_MEA),
            d_ion=float(source.Dl_ion),
            k_liquid=float(source.kl_CO2),
            k_vapor=float(source.kv_CO2),
            vapor_pressure_co2=float(source.y_CO2 * source.P),
            vapor_fugacity_co2=float(source.fv_CO2),
            liquid_fugacity_co2=float(source.fl_CO2),
            interface_slope=float(source.H_CO2_mix),
            relation="legacy_henry",
        )
        row.update(solved)
        attempts_json = str(solved.get("initial_guess_results_json", "[]"))
        row["implicit_branch_count"] = _branch_count(attempts_json)
        if solved["outcome"] != "evaluated":
            row.update(
                {
                    "evaluation_status": "incomplete",
                    "stopped_by": "solver",
                    "wall_time_seconds": time.perf_counter() - started,
                    "claim_strength": "boundary_at_state",
                }
            )
            return row
        row["p_CO2_interface_Pa"] = float(solved["C_CO2_interface_mol_m3"]) * float(source.H_CO2_mix)
        enhancement = float(solved["E"])
        residual_inf = float(solved["residual_inf"])
        reference_error = math.nan
    else:
        enhancement = explicit_equation(formulation, float(source.Ha), q, s_mea)
        reference = scalar_reference(formulation, float(source.Ha), q, r_plus, r_minus)
        residual_inf = 0.0
        reference_error = abs(enhancement - reference) / max(abs(reference), 1.0e-300)

    psi = enhancement * float(source.kl_CO2) / float(source.kv_CO2)
    gas_resistance_fraction = psi / (psi + float(source.H_CO2_mix))
    flux = float(source.kv_CO2 * source.a_eA * source.DF_CO2 * gas_resistance_fraction)
    finite = all(math.isfinite(value) for value in (enhancement, psi, gas_resistance_fraction, flux))
    direction_pass = flux == 0.0 or math.copysign(1.0, flux) == math.copysign(1.0, float(source.DF_CO2))
    positive_pass = enhancement > 0.0
    enhancement_pass = enhancement >= 1.0
    residual_scaled = residual_inf / max(abs(enhancement), 1.0)
    residual_pass = formulation != "EF-GF-IMPLICIT" or residual_scaled <= 1.0e-8
    spread_pass = formulation != "EF-GF-IMPLICIT" or float(row["initial_guess_relative_spread"]) <= 1.0e-3
    branch_pass = formulation != "EF-GF-IMPLICIT" or int(row["implicit_branch_count"]) == 1
    physical_pass = finite and positive_pass and enhancement_pass and direction_pass
    failures = [
        reason
        for passed, reason in (
            (finite, "non-finite evaluated value"),
            (positive_pass, "enhancement is not positive"),
            (enhancement_pass, "enhancement is below the physical nonreactive limit E=1"),
            (direction_pass, "flux opposes the retained fugacity driving force"),
            (residual_pass, "scaled implicit equation residual exceeds 1e-8"),
            (spread_pass, "implicit initial-guess spread exceeds 1e-3"),
            (branch_pass, "multiple implicit branches were returned"),
        )
        if not passed
    ]
    outcome = "evaluated" if not failures else (
        "physical_invalidity" if not physical_pass else "certificate_failure"
    )
    row.update(
        {
            "evaluation_status": "evaluated",
            "outcome": outcome,
            "stopped_by": "none" if outcome == "evaluated" else (
                "physical_check" if outcome == "physical_invalidity" else "certificate_check"
            ),
            "diagnostic": "; ".join(failures),
            "wall_time_seconds": time.perf_counter() - started,
            "claim_strength": "result" if outcome == "evaluated" else "boundary_at_state",
            "E": enhancement,
            "Psi": psi,
            "gas_resistance_fraction": gas_resistance_fraction,
            "liquid_resistance_fraction": 1.0 - gas_resistance_fraction,
            "predicted_flux_mol_s_m": flux,
            "residual_inf": residual_inf,
            "scaled_equation_residual": residual_scaled,
            "scalar_reference_relative_error": reference_error,
            "finite_values_pass": finite,
            "positive_enhancement_pass": positive_pass,
            "enhancement_at_least_one_pass": enhancement_pass,
            "flux_direction_pass": direction_pass,
            "physical_acceptance_pass": physical_pass,
            "physical_acceptance_reason": "accepted" if physical_pass else "; ".join(failures),
            "E_ratio_to_current": enhancement / float(source.E),
            "E_difference_from_current": enhancement - float(source.E),
            "flux_ratio_to_current": flux / float(source.Nl_CO2),
            "flux_difference_from_current_mol_s_m": flux - float(source.Nl_CO2),
            "distance_to_local_pinch_Pa": abs(float(source.DF_CO2)),
            "current_E_relative_reproduction_error": (
                abs(enhancement - float(source.E)) / abs(float(source.E))
                if formulation == "EF-CURRENT"
                else math.nan
            ),
        }
    )
    return row


def evaluate_fixed_states() -> pd.DataFrame:
    fixed = pd.read_csv(FIXED_STATES)
    full = pd.read_csv(FULL_PROFILE)
    pd.testing.assert_frame_equal(
        fixed,
        full.iloc[::5].reset_index(drop=True),
        check_exact=True,
    )
    forward = pd.DataFrame(
        [_evaluate_row(source, formulation) for _, source in fixed.iterrows() for formulation in FORMULATIONS]
    )
    reverse = pd.DataFrame(
        [
            _evaluate_row(source, formulation)
            for _, source in fixed.iloc[::-1].iterrows()
            for formulation in FORMULATIONS
        ]
    )
    reverse = reverse[["Position", "formulation", "E", "outcome", "implicit_branch_count"]].rename(
        columns={
            "E": "reverse_E",
            "outcome": "reverse_outcome",
            "implicit_branch_count": "reverse_implicit_branch_count",
        }
    )
    result = forward.merge(reverse, on=["Position", "formulation"], validate="one_to_one")
    result["reverse_relative_E_difference"] = (
        (result.E - result.reverse_E).abs() / result.E.abs().clip(lower=1.0e-300)
    )
    result["reverse_check_pass"] = (
        result.reverse_relative_E_difference.le(1.0e-12)
        & result.outcome.eq(result.reverse_outcome)
        & result.implicit_branch_count.eq(result.reverse_implicit_branch_count)
    )
    result["E_rank_at_state"] = result.groupby("Position").E.rank(
        method="min", ascending=False
    ).astype(int)
    result["flux_rank_at_state"] = result.groupby("Position").predicted_flux_mol_s_m.rank(
        method="min", ascending=False
    ).astype(int)
    return result


def aggregate_results(result: pd.DataFrame) -> pd.DataFrame:
    records = []
    for formulation, group in result.groupby("formulation", sort=False):
        records.append(
            {
                "formulation": formulation,
                "row_count": len(group),
                "evaluated_value_count": int(group.evaluation_status.eq("evaluated").sum()),
                "physically_accepted_count": int(group.physical_acceptance_pass.sum()),
                "minimum_E": float(group.E.min()),
                "median_E": float(group.E.median()),
                "maximum_E": float(group.E.max()),
                "minimum_E_ratio_to_current": float(group.E_ratio_to_current.min()),
                "p05_E_ratio_to_current": float(group.E_ratio_to_current.quantile(0.05)),
                "median_E_ratio_to_current": float(group.E_ratio_to_current.median()),
                "p95_E_ratio_to_current": float(group.E_ratio_to_current.quantile(0.95)),
                "maximum_E_ratio_to_current": float(group.E_ratio_to_current.max()),
                "minimum_E_difference_from_current": float(group.E_difference_from_current.min()),
                "p05_E_difference_from_current": float(group.E_difference_from_current.quantile(0.05)),
                "median_E_difference_from_current": float(group.E_difference_from_current.median()),
                "p95_E_difference_from_current": float(group.E_difference_from_current.quantile(0.95)),
                "maximum_E_difference_from_current": float(group.E_difference_from_current.max()),
                "minimum_flux_ratio_to_current": float(group.flux_ratio_to_current.min()),
                "p05_flux_ratio_to_current": float(group.flux_ratio_to_current.quantile(0.05)),
                "median_flux_ratio_to_current": float(group.flux_ratio_to_current.median()),
                "p95_flux_ratio_to_current": float(group.flux_ratio_to_current.quantile(0.95)),
                "maximum_flux_ratio_to_current": float(group.flux_ratio_to_current.max()),
                "minimum_flux_difference_from_current_mol_s_m": float(
                    group.flux_difference_from_current_mol_s_m.min()
                ),
                "p05_flux_difference_from_current_mol_s_m": float(
                    group.flux_difference_from_current_mol_s_m.quantile(0.05)
                ),
                "median_flux_difference_from_current_mol_s_m": float(
                    group.flux_difference_from_current_mol_s_m.median()
                ),
                "p95_flux_difference_from_current_mol_s_m": float(
                    group.flux_difference_from_current_mol_s_m.quantile(0.95)
                ),
                "maximum_flux_difference_from_current_mol_s_m": float(
                    group.flux_difference_from_current_mol_s_m.max()
                ),
                "maximum_scaled_equation_residual": float(group.scaled_equation_residual.max()),
                "maximum_initial_guess_relative_spread": float(group.initial_guess_relative_spread.max()),
                "maximum_reverse_relative_E_difference": float(group.reverse_relative_E_difference.max()),
                "spearman_flux_ratio_vs_Ha": float(group.flux_ratio_to_current.corr(group.Ha, method="spearman")),
                "spearman_flux_ratio_vs_Q": float(group.flux_ratio_to_current.corr(group.Q, method="spearman")),
                "spearman_flux_ratio_vs_S_MEA": float(
                    group.flux_ratio_to_current.corr(group.S_MEA, method="spearman")
                ),
            }
        )
    return pd.DataFrame(records)


def disagreement_records(result: pd.DataFrame, value_column: str) -> list[dict[str, object]]:
    records = []
    for position, group in result.groupby("Position"):
        maximum = group.loc[group[value_column].idxmax()]
        minimum = group.loc[group[value_column].idxmin()]
        records.append(
            {
                "Position": float(position),
                "height_m": float(maximum.height_m),
                "maximum_formulation": str(maximum.formulation),
                "maximum_value": float(maximum[value_column]),
                "minimum_formulation": str(minimum.formulation),
                "minimum_value": float(minimum[value_column]),
                "maximum_to_minimum_ratio": float(
                    maximum[value_column] / minimum[value_column]
                ),
                "distance_to_local_pinch_Pa": float(maximum.distance_to_local_pinch_Pa),
            }
        )
    return sorted(records, key=lambda record: record["maximum_to_minimum_ratio"], reverse=True)[:5]


def ranking_sensitivity(result: pd.DataFrame) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]]]:
    state = result.drop_duplicates("Position").set_index("Position")
    sensitivity = []
    for variable in ("Ha", "Q", "S_MEA"):
        bands = pd.qcut(state[variable], 3, labels=("low", "middle", "high"))
        for band in ("low", "middle", "high"):
            positions = bands.index[bands.eq(band)]
            median_flux = (
                result.loc[result.Position.isin(positions)]
                .groupby("formulation")
                .flux_ratio_to_current.median()
                .sort_values(ascending=False)
            )
            sensitivity.append(
                {
                    "variable": variable,
                    "band": band,
                    "minimum": float(state.loc[positions, variable].min()),
                    "maximum": float(state.loc[positions, variable].max()),
                    "flux_ranking": " > ".join(median_flux.index),
                }
            )
    axial_orders = []
    reversals = []
    previous = None
    for position, group in result.groupby("Position", sort=True):
        order = " > ".join(
            group.sort_values("predicted_flux_mol_s_m", ascending=False).formulation
        )
        axial_orders.append(order)
        if previous is not None and order != previous:
            reversals.append(
                {
                    "Position": float(position),
                    "height_m": float(group.height_m.iloc[0]),
                    "previous_order": previous,
                    "new_order": order,
                }
            )
        previous = order
    return sensitivity, list(dict.fromkeys(axial_orders)), reversals


def build_summary(result: pd.DataFrame, aggregates: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    identity = json.loads(IDENTITY.read_text())
    sensitivity, unique_flux_orders, flux_rank_reversals = ranking_sensitivity(result)
    physical_invalid = result.loc[
        result.evaluation_status.eq("evaluated") & ~result.physical_acceptance_pass,
        ["Position", "height_m", "formulation", "E", "predicted_flux_mol_s_m", "diagnostic"],
    ]
    numerical_gate = (
        len(result) == 84
        and result.evaluation_status.eq("evaluated").all()
        and result.finite_values_pass.all()
        and result.positive_enhancement_pass.all()
        and result.flux_direction_pass.all()
        and result.reverse_check_pass.all()
        and result.fallback_used.eq(False).all()
        and float(result.current_E_relative_reproduction_error.max()) <= 1.0e-12
        and float(
            result.loc[result.formulation.eq("EF-GF-IMPLICIT"), "scaled_equation_residual"].max()
        )
        <= 1.0e-8
        and float(
            result.loc[result.formulation.eq("EF-GF-IMPLICIT"), "initial_guess_relative_spread"].max()
        )
        <= 1.0e-3
    )
    physical_gate = result.physical_acceptance_pass.all()
    stage4_allowed = bool(numerical_gate and physical_gate)
    stages = pd.DataFrame(
        [
            {
                "stage": 0,
                "attempted": "yes",
                "stopped_by": "none",
                "outcome": "evaluated",
                "diagnostic": "historical package/profile identity gap retained; separately named current reconstruction retained",
                "claim_strength": "result",
            },
            {
                "stage": 1,
                "attempted": "yes",
                "stopped_by": "none",
                "outcome": "evaluated",
                "diagnostic": "84 formulation/state rows evaluated on 21 identical current-reconstruction states",
                "claim_strength": "result",
            },
            {
                "stage": 2,
                "attempted": "yes",
                "stopped_by": "physical_check" if not physical_gate else "none",
                "outcome": "physical_invalidity" if not physical_gate else "evaluated",
                "diagnostic": f"{len(physical_invalid)} rows violate E >= 1" if not physical_gate else "",
                "claim_strength": "boundary_at_state" if not physical_gate else "result",
            },
            {
                "stage": 3,
                "attempted": "yes",
                "stopped_by": "none",
                "outcome": "evaluated",
                "diagnostic": "fixed-state aggregates and three table-derived figures retained",
                "claim_strength": "result",
            },
            {
                "stage": 4,
                "attempted": "no" if not stage4_allowed else "yes",
                "stopped_by": "input_preflight" if not stage4_allowed else "none",
                "outcome": "not_attempted" if not stage4_allowed else "evaluated",
                "diagnostic": "Stage 2 physical gate failed; controlled column variants were not run" if not stage4_allowed else "",
                "claim_strength": "not_established" if not stage4_allowed else "result",
            },
            {
                "stage": 5,
                "attempted": "no" if not stage4_allowed else "yes",
                "stopped_by": "input_preflight" if not stage4_allowed else "none",
                "outcome": "not_attempted" if not stage4_allowed else "evaluated",
                "diagnostic": (
                    "A retained reactive profile exists, but the required Stage 2/4 sequence did not pass; "
                    "the reactive repeat was not run"
                    if not stage4_allowed
                    else ""
                ),
                "claim_strength": "not_established" if not stage4_allowed else "result",
            },
        ]
    )
    summary = {
        "issue": "https://github.com/tannerpolley/MEA-Absorption-Column/issues/17",
        "profile_id": str(result.profile_id.iloc[0]),
        "historical_identity": identity["historical_run"],
        "current_reconstruction": identity["current_reconstruction"],
        "fixed_states_sha256": sha256(FIXED_STATES),
        "full_profile_sha256": sha256(FULL_PROFILE),
        "retained_reactive_profile_available": REACTIVE_STATES.exists(),
        "retained_reactive_profile_sha256": sha256(REACTIVE_STATES) if REACTIVE_STATES.exists() else None,
        "state_count": int(result.Position.nunique()),
        "formulation_count": int(result.formulation.nunique()),
        "admitted_row_count": len(result),
        "evaluated_value_count": int(result.evaluation_status.eq("evaluated").sum()),
        "physical_invalidity_count": len(physical_invalid),
        "maximum_current_E_relative_reproduction_error": float(
            result.current_E_relative_reproduction_error.max()
        ),
        "maximum_current_flux_relative_reproduction_error": float(
            (
                result.loc[result.formulation.eq("EF-CURRENT"), "flux_ratio_to_current"]
                - 1.0
            ).abs().max()
        ),
        "maximum_implicit_scaled_equation_residual": float(
            result.loc[result.formulation.eq("EF-GF-IMPLICIT"), "scaled_equation_residual"].max()
        ),
        "maximum_implicit_initial_guess_relative_spread": float(
            result.loc[result.formulation.eq("EF-GF-IMPLICIT"), "initial_guess_relative_spread"].max()
        ),
        "maximum_reverse_relative_E_difference": float(result.reverse_relative_E_difference.max()),
        "numerical_gate_pass": bool(numerical_gate),
        "physical_gate_pass": bool(physical_gate),
        "stage4_allowed": stage4_allowed,
        "largest_E_disagreements": disagreement_records(result, "E"),
        "largest_flux_disagreements": disagreement_records(
            result, "predicted_flux_mol_s_m"
        ),
        "ranking_sensitivity_by_Ha_Q_S_MEA_tercile": sensitivity,
        "unique_axial_flux_rank_orders": unique_flux_orders,
        "axial_flux_rank_reversal_count": len(flux_rank_reversals),
        "axial_flux_rank_reversals": flux_rank_reversals,
        "physical_invalid_rows": physical_invalid.to_dict(orient="records"),
        "aggregates": aggregates.to_dict(orient="records"),
        "regeneration_command": (
            "uv run python analyses/nccc_validation/scripts/analyze_issue17_enhancement_comparison.py && "
            "uv run python analyses/nccc_validation/scripts/render_issue17_enhancement_comparison.py"
        ),
        "controlled_column_variants": stages.loc[stages.stage.eq(4)].iloc[0].to_dict(),
        "reactive_profile_repeat": stages.loc[stages.stage.eq(5)].iloc[0].to_dict(),
        "claim_boundary": (
            "The retained calculation compares four enhancement equations on one 21-state current locked-wheel "
            "fugacity-only reconstruction. It does not recover the historical wheel identity, validate any equation "
            "against absorption-rate observations, select thermodynamic parameters, establish a new column result, "
            "or support manuscript adoption."
        ),
    }
    return summary, stages


def main() -> None:
    started = time.perf_counter()
    result = evaluate_fixed_states()
    aggregates = aggregate_results(result)
    summary, stages = build_summary(result, aggregates)
    summary["analysis_runtime_s"] = time.perf_counter() - started
    TABLES.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULT_TABLE, index=False)
    aggregates.to_csv(AGGREGATE_TABLE, index=False)
    stages.to_csv(STAGE_TABLE, index=False)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["numerical_gate_pass"]:
        raise RuntimeError("Issue 17 numerical checks failed")


if __name__ == "__main__":
    main()
