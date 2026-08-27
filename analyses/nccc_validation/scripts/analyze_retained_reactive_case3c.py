from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import (
    SPECIES_9,
    chemical_equilibrium,
    tabulated_epcsaft_reactive_chemical_equilibrium,
)
from mea_absorption_column.Thermodynamics.epcsaft_v02 import (
    fugacity_coefficients,
    mixture,
    state,
)


ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "analyses/nccc_validation/results/runs"
FINAL_TABLES = ROOT / "analyses/nccc_validation/results/final/tables"
PRIOR_PROFILE = (
    RUN_ROOT
    / "predictive_reactive_epcsaft_3c/current_baselines/profiles/C_cases_campaign_inputs/3C/scipy-bvp/epcsaft_ionic"
)
RETAINED_ROOT = RUN_ROOT / "retained_predictive_reactive_epcsaft_3c"
RETAINED_PROFILE = (
    RETAINED_ROOT
    / "column_mesh51_case3c/profiles/C_cases_campaign_inputs/3C/scipy-bvp/epcsaft_reactive_nine_tabulated"
)
PRIOR_DATASET = ROOT / "src/mea_absorption_column/data/epcsaft_datasets/MEA_CO2_H2O_ionic_fit"
RETAINED_DATASET = (
    ROOT / "src/mea_absorption_column/data/epcsaft_datasets/MEA_CO2_H2O_retained_predictive"
)
REACTIVE_TABLE = RETAINED_ROOT / "speciation_table.csv"


def _profile(path: Path, prefix: str) -> pd.DataFrame:
    sources = {
        "T.csv": ["Tl"],
        "x.csv": ["x_CO2_true"],
        "CO2.csv": ["Nl_CO2", "DF_CO2", "fv_CO2", "fl_CO2"],
        "enhance_factor.csv": ["E", "Psi", "Psi_H"],
    }
    merged = None
    for filename, columns in sources.items():
        frame = pd.read_csv(path / filename)[["Position", *columns]]
        merged = frame if merged is None else merged.merge(frame, on="Position", validate="one_to_one")
    return merged.rename(columns={name: f"{prefix}_{name}" for name in merged if name != "Position"})


def _fugacity(dataset: Path, temperature_k: float, pressure_pa: float, composition) -> tuple[float, float]:
    model = mixture(str(dataset), SPECIES_9, temperature_k)
    liquid = state(
        model,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        composition=composition,
        phase="liquid",
    )
    phi = fugacity_coefficients(liquid)[0]
    return float(composition[0] * phi * pressure_pa), float(phi)


def _zero_co2_mea_dataset(target: Path) -> None:
    document = json.loads((RETAINED_DATASET / "parameters.json").read_text(encoding="utf-8"))
    for pair in document["pairs"]:
        if {pair["component_id_a"], pair["component_id_b"]} == {
            "carbon-dioxide",
            "monoethanolamine",
        }:
            pair["coefficients"][0]["value"]["magnitude"] = 0.0
            break
    parameter_path = target / "parameters.json"
    parameter_path.write_text(json.dumps(document), encoding="utf-8")
    adjustment = json.loads(
        (RETAINED_DATASET / "temperature_adjustments.json").read_text(encoding="utf-8")
    )
    adjustment["parameter_document_sha256"] = hashlib.sha256(parameter_path.read_bytes()).hexdigest()
    (target / "temperature_adjustments.json").write_text(
        json.dumps(adjustment), encoding="utf-8"
    )


def _factorial() -> pd.DataFrame:
    os.environ["MEA_EPCSAFT_REACTIVE_TABLE"] = str(REACTIVE_TABLE)
    flows = pd.read_csv(RETAINED_PROFILE / "Fl.csv")
    temperatures = pd.read_csv(RETAINED_PROFILE / "T.csv")
    transport = pd.read_csv(RETAINED_PROFILE / "transport.csv")
    selected = np.linspace(0, len(flows) - 1, 21, dtype=int)
    rows = []
    with TemporaryDirectory() as no_adjustment_text, TemporaryDirectory() as zero_k_text:
        no_adjustment = Path(no_adjustment_text)
        zero_k = Path(zero_k_text)
        (no_adjustment / "parameters.json").write_bytes(
            (RETAINED_DATASET / "parameters.json").read_bytes()
        )
        _zero_co2_mea_dataset(zero_k)
        for index in selected:
            apparent_flows = flows.loc[index, ["Fl_CO2", "Fl_MEA", "Fl_H2O"]].to_numpy(float)
            apparent_x = apparent_flows / apparent_flows.sum()
            temperature_k = float(temperatures.loc[index, "Tl"])
            pressure_pa = float(transport.loc[index, "P"])
            _, legacy_six = chemical_equilibrium(apparent_x, temperature_k)
            legacy_nine = np.r_[legacy_six, [1.0e-30] * 3]
            legacy_nine /= legacy_nine.sum()
            _, reactive_nine = tabulated_epcsaft_reactive_chemical_equilibrium(
                apparent_flows, temperature_k, diagnostics={}
            )
            prior_legacy, prior_legacy_phi = _fugacity(
                PRIOR_DATASET, temperature_k, pressure_pa, legacy_nine
            )
            retained_legacy, retained_legacy_phi = _fugacity(
                RETAINED_DATASET, temperature_k, pressure_pa, legacy_nine
            )
            prior_reactive, prior_reactive_phi = _fugacity(
                PRIOR_DATASET, temperature_k, pressure_pa, reactive_nine
            )
            retained_reactive, retained_reactive_phi = _fugacity(
                RETAINED_DATASET, temperature_k, pressure_pa, reactive_nine
            )
            no_adjustment_reactive, _ = _fugacity(
                no_adjustment, temperature_k, pressure_pa, reactive_nine
            )
            zero_k_reactive, _ = _fugacity(zero_k, temperature_k, pressure_pa, reactive_nine)
            rows.append(
                {
                    "position": float(flows.loc[index, "Position"]),
                    "temperature_K": temperature_k,
                    "pressure_Pa": pressure_pa,
                    "loading": float(apparent_x[0] / apparent_x[1]),
                    "x_CO2_legacy": float(legacy_nine[0]),
                    "x_CO2_reactive": float(reactive_nine[0]),
                    "f_prior_parameters_legacy_speciation_Pa": prior_legacy,
                    "f_retained_parameters_legacy_speciation_Pa": retained_legacy,
                    "f_prior_parameters_reactive_speciation_Pa": prior_reactive,
                    "f_retained_parameters_reactive_speciation_Pa": retained_reactive,
                    "f_retained_without_CO2_water_T_adjustment_Pa": no_adjustment_reactive,
                    "f_retained_with_k_CO2_MEA_zero_Pa": zero_k_reactive,
                    "phi_prior_parameters_legacy_speciation": prior_legacy_phi,
                    "phi_retained_parameters_legacy_speciation": retained_legacy_phi,
                    "phi_prior_parameters_reactive_speciation": prior_reactive_phi,
                    "phi_retained_parameters_reactive_speciation": retained_reactive_phi,
                }
            )
    frame = pd.DataFrame(rows)
    frame["ratio_x_CO2_reactive_to_legacy"] = frame.x_CO2_reactive / frame.x_CO2_legacy
    frame["ratio_retained_parameters_at_legacy_speciation"] = (
        frame.f_retained_parameters_legacy_speciation_Pa
        / frame.f_prior_parameters_legacy_speciation_Pa
    )
    frame["ratio_reactive_speciation_with_prior_parameters"] = (
        frame.f_prior_parameters_reactive_speciation_Pa
        / frame.f_prior_parameters_legacy_speciation_Pa
    )
    frame["ratio_total_retained_reactive_to_prior_legacy"] = (
        frame.f_retained_parameters_reactive_speciation_Pa
        / frame.f_prior_parameters_legacy_speciation_Pa
    )
    frame["ratio_CO2_water_T_adjustment"] = (
        frame.f_retained_parameters_reactive_speciation_Pa
        / frame.f_retained_without_CO2_water_T_adjustment_Pa
    )
    frame["ratio_k_CO2_MEA_zero_to_retained"] = (
        frame.f_retained_with_k_CO2_MEA_zero_Pa
        / frame.f_retained_parameters_reactive_speciation_Pa
    )
    return frame


def _benchmark_row(path: Path, family: str, setting: float, label: str) -> dict[str, object]:
    row = pd.read_csv(path / "benchmark_results.csv").iloc[0]
    return {
        "family": family,
        "setting": setting,
        "label": label,
        "success": bool(row.success),
        "capture_pct": float(row.capture_pct),
        "capture_error_percentage_points": float(row.capture_error_pct),
        "temperature_rmse_K": float(row.temperature_rmse_K),
        "boundary_residual_norm": float(row.boundary_residual_norm),
        "invalid_state_count": int(row.invalid_state_count),
    }


def _sensitivity() -> pd.DataFrame:
    rows = []
    for mesh, folder in ((21, "column_mesh21_case3c"), (51, "column_mesh51_case3c"), (81, "column_mesh81_case3c")):
        rows.append(_benchmark_row(RETAINED_ROOT / folder, "mesh_points", mesh, f"mesh {mesh}"))
    for blend, folder in ((0.0, "blend_0_mesh21_case3c"), (0.5, "blend_0p5_mesh21_case3c"), (1.0, "column_mesh21_case3c")):
        rows.append(_benchmark_row(RETAINED_ROOT / folder, "fugacity_blend", blend, f"blend {blend:g}"))
    for factor, folder in ((1.0, "column_mesh21_case3c"), (1.5, "eta_1p5_mesh21_case3c"), (2.0, "eta_2_mesh21_case3c"), (4.0, "eta_4_mesh21_case3c")):
        rows.append(_benchmark_row(RETAINED_ROOT / folder, "eta_psi", factor, f"eta_psi {factor:g}"))
    rows.append(
        _benchmark_row(
            RETAINED_ROOT / "no_co2_water_temperature_adjustment_mesh21_case3c",
            "parameter_structure",
            0.0,
            "without CO2-water temperature adjustment",
        )
    )
    return pd.DataFrame(rows)


def main() -> None:
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    profile = _profile(PRIOR_PROFILE, "prior").merge(
        _profile(RETAINED_PROFILE, "retained"), on="Position", validate="one_to_one"
    )
    for name in ("fl_CO2", "fv_CO2", "DF_CO2", "E", "Psi", "Psi_H", "x_CO2_true", "Nl_CO2"):
        profile[f"ratio_retained_to_prior_{name}"] = profile[f"retained_{name}"] / profile[f"prior_{name}"]
    factorial = _factorial()
    sensitivity = _sensitivity()
    direct_checks = json.loads((RETAINED_ROOT / "direct_profile_liquid_checks.json").read_text())
    summary = {
        "observed_capture_pct": 89.5,
        "prior_capture_pct": float(
            pd.read_csv(RUN_ROOT / "predictive_reactive_epcsaft_3c/current_baselines/benchmark_results.csv")
            .query("thermo_model == 'epcsaft_ionic'")
            .iloc[0]
            .capture_pct
        ),
        "retained_capture_pct_mesh51": float(
            sensitivity.query("family == 'mesh_points' and setting == 51").iloc[0].capture_pct
        ),
        "profile_median_liquid_fugacity_ratio_retained_to_prior": float(
            profile.ratio_retained_to_prior_fl_CO2.median()
        ),
        "profile_median_driving_force_ratio_retained_to_prior": float(
            profile.ratio_retained_to_prior_DF_CO2.median()
        ),
        "profile_median_enhancement_ratio_retained_to_prior": float(
            profile.ratio_retained_to_prior_E.median()
        ),
        "profile_median_transfer_rate_ratio_retained_to_prior": float(
            profile.ratio_retained_to_prior_Nl_CO2.median()
        ),
        "factorial_median_x_CO2_ratio_reactive_to_legacy": float(
            factorial.ratio_x_CO2_reactive_to_legacy.median()
        ),
        "factorial_median_parameter_ratio_at_legacy_speciation": float(
            factorial.ratio_retained_parameters_at_legacy_speciation.median()
        ),
        "factorial_median_speciation_ratio_with_prior_parameters": float(
            factorial.ratio_reactive_speciation_with_prior_parameters.median()
        ),
        "factorial_median_total_fugacity_ratio": float(
            factorial.ratio_total_retained_reactive_to_prior_legacy.median()
        ),
        "factorial_CO2_water_T_adjustment_ratio_range": [
            float(factorial.ratio_CO2_water_T_adjustment.min()),
            float(factorial.ratio_CO2_water_T_adjustment.max()),
        ],
        "factorial_k_CO2_MEA_zero_ratio_range": [
            float(factorial.ratio_k_CO2_MEA_zero_to_retained.min()),
            float(factorial.ratio_k_CO2_MEA_zero_to_retained.max()),
        ],
        "maximum_direct_interpolation_fugacity_relative_error": max(
            abs(float(row["relative_fugacity_error"])) for row in direct_checks
        ),
    }
    profile.to_csv(FINAL_TABLES / "retained_reactive_case3c_profile_comparison.csv", index=False)
    factorial.to_csv(FINAL_TABLES / "retained_reactive_case3c_fugacity_factorial.csv", index=False)
    sensitivity.to_csv(FINAL_TABLES / "retained_reactive_case3c_sensitivity.csv", index=False)
    (FINAL_TABLES / "retained_reactive_case3c_diagnosis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
