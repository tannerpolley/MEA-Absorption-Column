from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mea_absorption_column.Thermodynamics.thermo_models import epcsaft_state_contribution_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "MEA_CO2_H2O_ionic_fit"
DATASET_PATH = (
    REPO_ROOT / "src" / "mea_absorption_column" / "data" / "epcsaft_datasets" / DATASET_NAME
)
RUN_ROOT = REPO_ROOT / "analyses" / "nccc_validation" / "results" / "runs" / "epcsaft_electrolyte_config_matrix"
FINAL_DIR = REPO_ROOT / "analyses" / "nccc_validation" / "results" / "final"
TABLE_DIR = FINAL_DIR / "tables"
REPORT_DIR = FINAL_DIR / "reports"
RESULT_TABLE = TABLE_DIR / "epcsaft_electrolyte_column_config_matrix.csv"
CONFIG_TABLE = TABLE_DIR / "epcsaft_electrolyte_config_user_options.csv"
PURE_TABLE = TABLE_DIR / "epcsaft_electrolyte_pure_parameters.csv"
BINARY_TABLE = TABLE_DIR / "epcsaft_electrolyte_binary_parameters.csv"
REL_PERM_TABLE = TABLE_DIR / "epcsaft_electrolyte_relative_permittivity_parameters.csv"
REPORT = REPORT_DIR / "epcsaft_electrolyte_column_config_matrix.md"
IONIC_X = np.array([1.0e-8, 0.055, 0.888, 0.028, 0.027, 0.001], dtype=float)
IONIC_X = IONIC_X / IONIC_X.sum()


def _advanced_born_options(*, ssm=True, ds=True, mu_mode="numerical") -> dict:
    return {
        "elec_model": {
            "rel_perm": {"rule": "empirical", "differential_mode": "numerical"},
            "born_model": {
                "d_Born_mode": 3,
                "solvation_shell_model": bool(ssm),
                "dielectric_saturation": bool(ds),
                "mu_born_model": {
                    "differential_mode": mu_mode,
                    "comp_dep_delta_d": bool(ssm or ds),
                },
            },
        }
    }


def _preset_source(name: str, note: str) -> str:
    return f"{note}; package preset source=epcsaft reference data: epcsaft_parameters/{name}/user_options.json"


def _configurations() -> list[dict]:
    return [
        {
            "config": "2005_Cameretti_constant_DH_no_Born",
            "family": "dated_preset",
            "rel_perm_rule": "constant",
            "born_mode": "off",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "not_applicable",
            "dh_variant": "default",
            "user_options": {"elec_model": {"rel_perm": {"rule": "constant"}, "include_born_model": False}},
            "expected_success": True,
            "source_note": _preset_source(
                "2005_Cameretti",
                "Cameretti-2005-like package user-option pattern: constant dielectric with Born disabled",
            ),
        },
        {
            "config": "2008_Held_constant_DH_no_Born",
            "family": "dated_preset",
            "rel_perm_rule": "constant",
            "born_mode": "off",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "not_applicable",
            "dh_variant": "default",
            "user_options": {"elec_model": {"rel_perm": {"rule": "constant"}, "include_born_model": False}},
            "expected_success": True,
            "source_note": _preset_source(
                "2008_Held",
                "Held-2008-like package user-option pattern: constant dielectric with Born disabled",
            ),
        },
        {
            "config": "2014_Held_constant_DH_no_Born",
            "family": "dated_preset",
            "rel_perm_rule": "constant",
            "born_mode": "off",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "not_applicable",
            "dh_variant": "default",
            "user_options": {"elec_model": {"rel_perm": {"rule": "constant"}, "include_born_model": False}},
            "expected_success": True,
            "source_note": _preset_source(
                "2014_Held",
                "Held-2014-like package user-option pattern: constant dielectric with Born disabled",
            ),
        },
        {
            "config": "2019_Bulow_linear_DH_no_Born",
            "family": "dated_preset",
            "rel_perm_rule": "linear",
            "born_mode": "off",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "not_applicable",
            "dh_variant": "default",
            "user_options": {"elec_model": {"include_born_model": False}},
            "expected_success": True,
            "source_note": _preset_source(
                "2019_Bulow",
                "Bulow-2019 package preset only disables Born; linear dielectric is inherited from current defaults",
            ),
        },
        {
            "config": "2020_Bulow_linear_base_Born",
            "family": "dated_preset",
            "rel_perm_rule": "linear",
            "born_mode": "d_Born_mode_1_base",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "auto",
            "dh_variant": "default",
            "user_options": {"elec_model": {"born_model": {"d_Born_mode": 1}}},
            "expected_success": True,
            "source_note": _preset_source(
                "2020_Bulow",
                "Bulow-2020 package preset supplies d_Born_mode=1; Born inclusion and other toggles are inherited",
            ),
        },
        {
            "config": "2025_Figiel_empirical_fitted_Born_SSM_DS",
            "family": "dated_preset",
            "rel_perm_rule": "empirical",
            "born_mode": "fitted_param",
            "ssm": True,
            "ds": True,
            "mu_born_mode": "numerical",
            "dh_variant": "default",
            "user_options": _advanced_born_options(ssm=True, ds=True, mu_mode="numerical"),
            "expected_success": True,
            "source_note": _preset_source(
                "2025_Figiel",
                "Figiel-2025-like advanced Born pattern: empirical dielectric, fitted d_Born, SSM, DS, numerical mu_born",
            ),
        },
        {
            "config": "mode_relperm_combined_no_Born",
            "family": "mode_coverage",
            "rel_perm_rule": "combined",
            "born_mode": "off",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "not_applicable",
            "dh_variant": "default",
            "user_options": {"elec_model": {"rel_perm": {"rule": "combined"}, "include_born_model": False}},
            "expected_success": True,
            "source_note": "Mode-coverage run for supported rel_perm.rule=combined",
        },
        {
            "config": "mode_relperm_linear_saltfraction_no_Born",
            "family": "mode_coverage",
            "rel_perm_rule": "linear-saltfraction",
            "born_mode": "off",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "not_applicable",
            "dh_variant": "default",
            "user_options": {"elec_model": {"rel_perm": {"rule": "linear-saltfraction"}, "include_born_model": False}},
            "expected_success": True,
            "source_note": "Mode-coverage run for supported rel_perm.rule=linear-saltfraction",
        },
        {
            "config": "mode_relperm_aqueous_organic_no_Born",
            "family": "mode_coverage",
            "rel_perm_rule": "aqueous-organic",
            "born_mode": "off",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "not_applicable",
            "dh_variant": "default",
            "user_options": {"elec_model": {"rel_perm": {"rule": "aqueous-organic"}, "include_born_model": False}},
            "expected_success": True,
            "source_note": "Mode-coverage run for supported rel_perm.rule=aqueous-organic",
        },
        {
            "config": "mode_sigma_radius_classic_Born",
            "family": "mode_coverage",
            "rel_perm_rule": "empirical",
            "born_mode": "d_Born_mode_0_sigma",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "auto",
            "dh_variant": "default",
            "user_options": {
                "elec_model": {
                    "rel_perm": {"rule": "empirical", "differential_mode": "numerical"},
                    "born_model": {"d_Born_mode": 0, "solvation_shell_model": False, "dielectric_saturation": False},
                }
            },
            "expected_success": True,
            "source_note": "Mode-coverage run for classic Born using segment-diameter radius",
        },
        {
            "config": "mode_fitted_Born_SSM_only",
            "family": "mode_coverage",
            "rel_perm_rule": "empirical",
            "born_mode": "fitted_param",
            "ssm": True,
            "ds": False,
            "mu_born_mode": "numerical",
            "dh_variant": "default",
            "user_options": _advanced_born_options(ssm=True, ds=False, mu_mode="numerical"),
            "expected_success": True,
            "source_note": "Mode-coverage run for fitted d_Born with solvation-shell model only",
        },
        {
            "config": "mode_fitted_Born_DS_only",
            "family": "mode_coverage",
            "rel_perm_rule": "empirical",
            "born_mode": "fitted_param",
            "ssm": False,
            "ds": True,
            "mu_born_mode": "numerical",
            "dh_variant": "default",
            "user_options": _advanced_born_options(ssm=False, ds=True, mu_mode="numerical"),
            "expected_success": True,
            "source_note": "Mode-coverage run for fitted d_Born with dielectric saturation only",
        },
        {
            "config": "mode_fitted_Born_SSM_DS_auto_mu",
            "family": "mode_coverage",
            "rel_perm_rule": "empirical",
            "born_mode": "fitted_param",
            "ssm": True,
            "ds": True,
            "mu_born_mode": "auto",
            "dh_variant": "default",
            "user_options": _advanced_born_options(ssm=True, ds=True, mu_mode="auto"),
            "expected_success": True,
            "source_note": "Mode-coverage run for fitted d_Born with SSM+DS and automatic mu_born derivatives",
        },
        {
            "config": "mode_DH_sigma_no_Born",
            "family": "mode_coverage",
            "rel_perm_rule": "linear",
            "born_mode": "off",
            "ssm": False,
            "ds": False,
            "mu_born_mode": "not_applicable",
            "dh_variant": "d_ion_mode_0",
            "user_options": {
                "elec_model": {
                    "include_born_model": False,
                    "DH_model": {"d_ion_mode": 0},
                }
            },
            "expected_success": True,
            "source_note": "Mode-coverage run for the alternate DH ion-size mode",
        },
    ]


def _clean_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _run_benchmark(config: dict) -> dict:
    run_dir = RUN_ROOT / config["config"]
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MEA_EPCSAFT_DATASET_NAME"] = DATASET_NAME
    env["MEA_EPCSAFT_USER_OPTIONS_JSON"] = _clean_json(config["user_options"])
    cmd = [
        sys.executable,
        "-m",
        "mea_absorption_column.benchmark",
        "--methods",
        "scipy-bvp",
        "--thermo-models",
        "epcsaft_ionic",
        "--c-case-ids",
        "3C",
        "--nccc-case-limit",
        "0",
        "--srp-case-limit",
        "0",
        "--staged-beds",
        "false",
        "--mesh-points",
        "21",
        "--tol",
        "1",
        "--bc-tol",
        "0.05",
        "--max-nodes",
        "200",
        "--subprocess-timeout-s",
        "75",
        "--output-dir",
        str(run_dir),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=100,
    )
    wall_s = time.perf_counter() - started
    row = {
        "config": config["config"],
        "family": config["family"],
        "expected_success": config["expected_success"],
        "run_wall_s": wall_s,
        "run_dir": str(run_dir),
        "command_returncode": completed.returncode,
    }
    result_path = run_dir / "benchmark_results.csv"
    if result_path.exists():
        result = pd.read_csv(result_path).iloc[0].to_dict()
        row.update(result)
    else:
        row.update({"success": False, "message": "benchmark_results.csv was not written"})
    return row


def _diagnostic_row(config: dict) -> dict:
    diagnostics = epcsaft_state_contribution_diagnostics(
        323.15,
        109500.0,
        IONIC_X,
        phase="liq",
        mixture_kind="ionic",
        user_options=copy.deepcopy(config["user_options"]),
    )
    ares = diagnostics["ares_terms"]
    fugacity = diagnostics["lnfugcoef_co2_terms"]
    return {
        "config": config["config"],
        "diagnostic_phi_co2": diagnostics["phi_co2"],
        "diagnostic_density_mol_m3": diagnostics["density_mol_m3"],
        "a_hc": ares.get("hc"),
        "a_disp": ares.get("disp"),
        "a_assoc": ares.get("assoc"),
        "a_ion": ares.get("ion"),
        "a_born": ares.get("born"),
        "lnphi_co2_ion": fugacity.get("ion"),
        "lnphi_co2_born": fugacity.get("born"),
    }


def _write_config_table(configs: list[dict]) -> pd.DataFrame:
    rows = []
    for config in configs:
        rows.append(
            {
                "config": config["config"],
                "family": config["family"],
                "dataset": DATASET_NAME,
                "rel_perm_rule": config["rel_perm_rule"],
                "born_mode": config["born_mode"],
                "ssm": config["ssm"],
                "ds": config["ds"],
                "mu_born_mode": config["mu_born_mode"],
                "dh_variant": config["dh_variant"],
                "expected_success": config["expected_success"],
                "user_options_json": _clean_json(config["user_options"]),
                "source_note": config["source_note"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(CONFIG_TABLE, index=False)
    return frame


def _source_note_for_component(component: str) -> str:
    if component in {"CO2", "MEA", "H2O"}:
        return "Neutral component parameters from the repo-vendored MEA ePC-SAFT ionic-fit dataset."
    if component in {"MEAH+", "MEACOO-"}:
        return "MEA ionic species values from the repo-vendored ionic-fit dataset; d_born is treated as the fitted Born diameter for SSM/DS runs."
    return "Auxiliary carbonate/water ion placeholder from the repo-vendored ionic-fit dataset; d_born=3 A is used as a reasonable hydrated-ion-scale assumption."


def _write_pure_parameter_table() -> pd.DataFrame:
    path = DATASET_PATH / "pure" / "any_solvent.csv"
    frame = pd.read_csv(path)
    frame.insert(0, "dataset", DATASET_NAME)
    frame["source_path"] = str(path)
    frame["source_note"] = frame["component"].map(_source_note_for_component)
    frame.to_csv(PURE_TABLE, index=False)
    return frame


def _write_binary_parameter_table() -> pd.DataFrame:
    rows = []
    base = DATASET_PATH / "mixed" / "binary_interaction"
    for parameter in ("k_ij", "l_ij", "k_hb_ij"):
        path = base / f"{parameter}.csv"
        matrix = pd.read_csv(path).set_index("component")
        components = list(matrix.index)
        for i, component_i in enumerate(components):
            for component_j in components[i:]:
                rows.append(
                    {
                        "dataset": DATASET_NAME,
                        "parameter": parameter,
                        "component_i": component_i,
                        "component_j": component_j,
                        "value": matrix.loc[component_i, component_j],
                        "source_path": str(path),
                        "source_note": "Repo-vendored MEA ePC-SAFT ionic-fit binary interaction matrix; upper triangle reported once.",
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(BINARY_TABLE, index=False)
    return frame


def _write_rel_perm_parameter_table() -> pd.DataFrame:
    path = DATASET_PATH / "mixed" / "rel_perm" / "parameters.csv"
    frame = pd.read_csv(path)
    frame.insert(0, "dataset", DATASET_NAME)
    frame["source_path"] = str(path)
    frame["source_note"] = "Relative-permittivity coefficient table used when rel_perm.rule=empirical or aqueous-organic."
    frame.to_csv(REL_PERM_TABLE, index=False)
    return frame


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(cell(row.get(column)) for column in columns) + " |")
    return "\n".join(rows)


def _write_report(results: pd.DataFrame, pure: pd.DataFrame, binary: pd.DataFrame) -> None:
    report_results = results.copy()
    if "diagnostic_message" not in report_results.columns:
        report_results["diagnostic_message"] = ""
    report_results["diagnostic_note"] = report_results["diagnostic_message"].fillna("").map(
        lambda value: "" if not value else str(value).split(":", 1)[0] + ": fixed-state contribution probe failed"
    )
    display_columns = [
        "config",
        "success",
        "capture_pct",
        "capture_error_pct",
        "runtime_s",
        "temperature_rmse_K",
        "boundary_residual_norm",
        "epcsaft_cache_hits",
        "epcsaft_cache_misses",
        "diagnostic_phi_co2",
        "a_ion",
        "a_born",
        "diagnostic_note",
    ]
    nonzero_binary = binary[pd.to_numeric(binary["value"], errors="coerce").fillna(0.0) != 0.0]
    pure_columns = ["component", "m", "s", "e", "e_assoc", "vol_a", "z", "dielc", "d_born", "source_note"]
    lines = [
        "# ePC-SAFT Electrolyte Column Configuration Matrix",
        "",
        "These runs use the six-species MEA absorber state with the repo-vendored `MEA_CO2_H2O_ionic_fit` ePC-SAFT dataset.",
        "The 3C C-case is intentionally small enough to keep the comparison reproducible while still running the full column solver.",
        "",
        f"- Dataset path: `{DATASET_PATH}`",
        f"- Run root: `{RUN_ROOT}`",
        f"- Successful column rows: {int(results['success'].astype(str).str.lower().eq('true').sum())}/{len(results)}",
        f"- Primary result table: `{RESULT_TABLE}`",
        f"- Pure parameter table: `{PURE_TABLE}`",
        f"- Binary interaction table: `{BINARY_TABLE}`",
        "",
        "## Column Results",
        "",
        _markdown_table(report_results[display_columns], display_columns),
        "",
        "## Pure Component Parameters",
        "",
        _markdown_table(pure[pure_columns], pure_columns),
        "",
        "## Nonzero Binary Interaction Parameters",
        "",
        _markdown_table(nonzero_binary[["parameter", "component_i", "component_j", "value", "source_note"]], ["parameter", "component_i", "component_j", "value", "source_note"])
        if not nonzero_binary.empty
        else "All reported binary interaction entries are zero.",
        "",
        "## Notes",
        "",
        "- The dated rows reproduce the ePC-SAFT package user-option patterns while holding the MEA component dataset fixed.",
        "- The SSM+DS rows use the `MEA_CO2_H2O_ionic_fit` ion Born diameters. For auxiliary carbonate/water ions, the vendored dataset uses 3 A hydrated-ion-scale assumptions.",
        "- The absorber itself is currently the six-species chemistry state (`CO2`, `MEA`, `H2O`, `MEAH+`, `MEACOO-`, `HCO3-`); the parameter tables still report the full nine-species dataset so the unused ionic species are auditable.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    configs = _configurations()
    _write_config_table(configs)
    pure = _write_pure_parameter_table()
    binary = _write_binary_parameter_table()
    _write_rel_perm_parameter_table()
    rows = []
    for config in configs:
        print(f"Running {config['config']}...")
        row = _run_benchmark(config)
        try:
            row.update(_diagnostic_row(config))
        except Exception as exc:
            row["diagnostic_message"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(RESULT_TABLE, index=False)
    _write_report(results, pure, binary)
    print(f"Wrote {RESULT_TABLE}")
    print(f"Wrote {CONFIG_TABLE}")
    print(f"Wrote {PURE_TABLE}")
    print(f"Wrote {BINARY_TABLE}")
    print(f"Wrote {REL_PERM_TABLE}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
