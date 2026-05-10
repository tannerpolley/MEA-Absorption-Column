from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    epcsaft_dataset_user_options,
    epcsaft_state_contribution_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FINAL_TABLE = (
    REPO_ROOT
    / "analyses"
    / "nccc_validation"
    / "results"
    / "final"
    / "tables"
    / "epcsaft_electrolyte_option_matrix.csv"
)
FINAL_REPORT = (
    REPO_ROOT
    / "analyses"
    / "nccc_validation"
    / "results"
    / "final"
    / "reports"
    / "epcsaft_electrolyte_option_matrix.md"
)
IONIC_X = np.array([1.0e-8, 0.055, 0.888, 0.028, 0.027, 0.001], dtype=float)
IONIC_X = IONIC_X / IONIC_X.sum()
NEUTRAL_X = np.array([0.02, 0.24, 0.74], dtype=float)


def _options(
    *,
    include_born=True,
    d_born_mode=0,
    ssm=False,
    ds=False,
    mu_mode="analytical",
    rel_perm_rule="linear",
    rel_perm_mode="analytical",
):
    return {
        "elec_model": {
            "rel_perm": {"rule": rel_perm_rule, "differential_mode": rel_perm_mode},
            "include_born_model": include_born,
            "born_model": {
                "d_Born_mode": d_born_mode,
                "solvation_shell_model": ssm,
                "dielectric_saturation": ds,
                "mu_born_model": {
                    "differential_mode": mu_mode,
                    "comp_dep_delta_d": bool(ssm or ds),
                },
            },
        }
    }


def _matrix():
    return [
        {
            "config": "neutral_reference",
            "mixture_kind": "neutral",
            "composition": NEUTRAL_X,
            "user_options": None,
            "expected_success": True,
        },
        {
            "config": "ionic_dataset_default",
            "mixture_kind": "ionic",
            "composition": IONIC_X,
            "user_options": epcsaft_dataset_user_options(),
            "expected_success": True,
        },
        {
            "config": "ionic_dh_only_born_disabled",
            "mixture_kind": "ionic",
            "composition": IONIC_X,
            "user_options": _options(include_born=False),
            "expected_success": True,
        },
        {
            "config": "ionic_classic_born_sigma_radius",
            "mixture_kind": "ionic",
            "composition": IONIC_X,
            "user_options": _options(),
            "expected_success": True,
        },
        {
            "config": "ionic_fitted_born_ssm_only",
            "mixture_kind": "ionic",
            "composition": IONIC_X,
            "user_options": _options(
                d_born_mode=3,
                ssm=True,
                ds=False,
                mu_mode="numerical",
                rel_perm_rule="empirical",
                rel_perm_mode="numerical",
            ),
            "expected_success": True,
        },
        {
            "config": "ionic_fitted_born_ds_only",
            "mixture_kind": "ionic",
            "composition": IONIC_X,
            "user_options": _options(
                d_born_mode=3,
                ssm=False,
                ds=True,
                mu_mode="numerical",
                rel_perm_rule="empirical",
                rel_perm_mode="numerical",
            ),
            "expected_success": True,
        },
        {
            "config": "ionic_fitted_born_ssm_ds_numerical",
            "mixture_kind": "ionic",
            "composition": IONIC_X,
            "user_options": _options(
                d_born_mode=3,
                ssm=True,
                ds=True,
                mu_mode="numerical",
                rel_perm_rule="empirical",
                rel_perm_mode="numerical",
            ),
            "expected_success": True,
        },
        {
            "config": "ionic_fitted_born_ssm_ds_auto",
            "mixture_kind": "ionic",
            "composition": IONIC_X,
            "user_options": _options(
                d_born_mode=3,
                ssm=True,
                ds=True,
                mu_mode="auto",
                rel_perm_rule="empirical",
                rel_perm_mode="numerical",
            ),
            "expected_success": True,
        },
        {
            "config": "unsupported_fitted_born_without_ssm_ds",
            "mixture_kind": "ionic",
            "composition": IONIC_X,
            "user_options": _options(d_born_mode="fitted_param", ssm=False, ds=False),
            "expected_success": False,
        },
    ]


def _row(entry):
    base = {
        "config": entry["config"],
        "mixture_kind": entry["mixture_kind"],
        "expected_success": entry["expected_success"],
        "success": False,
        "message": "",
        "dataset": str(MEA_THERMODYNAMICS_EPCSAFT_DATASET),
    }
    try:
        diagnostics = epcsaft_state_contribution_diagnostics(
            323.15,
            109500.0,
            entry["composition"],
            phase="liq",
            mixture_kind=entry["mixture_kind"],
            user_options=copy.deepcopy(entry["user_options"]),
        )
    except Exception as exc:
        base["message"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        return base

    ares = diagnostics["ares_terms"]
    lnfug = diagnostics["lnfugcoef_co2_terms"]
    base.update(
        {
            "success": True,
            "phi_co2": diagnostics["phi_co2"],
            "density_mol_m3": diagnostics["density_mol_m3"],
            "a_hc": ares.get("hc"),
            "a_disp": ares.get("disp"),
            "a_assoc": ares.get("assoc"),
            "a_ion": ares.get("ion"),
            "a_born": ares.get("born"),
            "lnphi_co2_ion": lnfug.get("ion"),
            "lnphi_co2_born": lnfug.get("born"),
        }
    )
    return base


def _write_report(frame: pd.DataFrame) -> None:
    successes = int(frame["success"].sum())
    expected_failures = frame[(~frame["success"]) & (~frame["expected_success"])]
    unexpected = frame[frame["success"] != frame["expected_success"]]
    matrix_columns = [
        "config",
        "success",
        "expected_success",
        "a_ion",
        "a_born",
        "lnphi_co2_ion",
        "lnphi_co2_born",
        "message",
    ]
    lines = [
        "# ePC-SAFT Electrolyte Option Matrix",
        "",
        "This diagnostic exercises the MEA six-species ePC-SAFT dataset at a Case-3C-like liquid state.",
        "It separates the neutral fugacity-coefficient path from electrolyte option paths that activate Debye-Huckel ion and Born terms.",
        "",
        f"- Dataset: `{MEA_THERMODYNAMICS_EPCSAFT_DATASET}`",
        f"- Successful configurations: {successes}/{len(frame)}",
        f"- Expected unsupported configurations: {len(expected_failures)}",
        f"- Unexpected outcomes: {len(unexpected)}",
        "",
        "## Matrix",
        "",
        _markdown_table(frame[matrix_columns], matrix_columns),
        "",
        "## Interpretation",
        "",
        "The neutral reference keeps both ion and Born residual Helmholtz contributions at zero.",
        "The ionic dataset path activates the ion term and, when Born is enabled with a supported radius model, activates the Born contribution as well.",
        "The dataset-default path uses linear relative-permittivity mixing and the classic Born radius mode. The fitted Born-diameter diagnostic rows are retained only as option-coverage checks; the unsupported fitted-without-SSM/DS row is an expected clear failure rather than a silent neutral fallback.",
    ]
    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(cell(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def main() -> None:
    rows = [_row(entry) for entry in _matrix()]
    frame = pd.DataFrame(rows)
    FINAL_TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FINAL_TABLE, index=False)
    _write_report(frame)
    print(f"Wrote {FINAL_TABLE}")
    print(f"Wrote {FINAL_REPORT}")
    if not (frame["success"] == frame["expected_success"]).all():
        raise SystemExit("ePC-SAFT electrolyte option matrix had unexpected outcomes.")


if __name__ == "__main__":
    main()
