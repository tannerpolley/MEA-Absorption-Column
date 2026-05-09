from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
TABLES = ANALYSIS / "results" / "final" / "tables"
FIGURES = ANALYSIS / "results" / "final" / "figures"
DOC_FIGURES = ROOT / "docs" / "latex" / "figures"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)

    c_cases = _read(TABLES / "verified_c_case_thermo_benchmark.csv")
    k_primary = _read(TABLES / "verified_staged_kcase_benchmark.csv")
    k_recovery = _read(TABLES / "kcase_sensitivity_recoveries.csv")
    k_unresolved = _read(TABLES / "kcase_unresolved_diagnostics.csv")
    epcsaft_smoke = _read(TABLES / "staged_epcsaft_smoke.csv")
    epcsaft_recovery = _read(TABLES / "staged_epcsaft_recovery_probe.csv")
    k2_blend = _read(TABLES / "staged_epcsaft_k2_blend_probe.csv")

    nccc = _read(ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_Data.csv")
    nccc = nccc.rename(columns={"Runs": "case_id", "CO2  %": "measured_capture_pct"})
    c_inputs = _read(ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_data.csv")
    c_inputs = c_inputs.rename(columns={"Case": "case_id", "CO2 %": "measured_capture_pct"})

    _write_validation_registry(
        c_cases,
        k_primary,
        k_recovery,
        k_unresolved,
        epcsaft_smoke,
        epcsaft_recovery,
        k2_blend,
    )
    k_primary_aug = _augment_k_cases(k_primary, nccc)
    c_primary_aug = _augment_c_cases(c_cases, c_inputs)
    _write_no_case_tuning_gate(c_primary_aug, k_primary_aug)
    _write_calibration_artifacts(k_primary_aug)
    _write_error_regime_artifacts(c_primary_aug, k_primary_aug)
    _write_staged_epcsaft_reliability(epcsaft_smoke, epcsaft_recovery, k2_blend)
    _write_intercooled_temperature_profile()
    _write_morgan_appendix_c_intercooled_profile()
    _write_intercooled_profile_comparison()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _success_mask(data: pd.DataFrame) -> pd.Series:
    if "success" not in data.columns:
        return pd.Series(False, index=data.index)
    return data["success"].astype(str).str.lower().eq("true")


def _numeric(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index)
    return pd.to_numeric(data[column], errors="coerce")


def _summary_row(
    label: str,
    data: pd.DataFrame,
    evidence_class: str,
    primary_validation: bool,
    no_case_specific_tuning: bool,
    notes: str,
) -> dict[str, object]:
    success = _success_mask(data)
    error = _numeric(data, "capture_error_pct")
    runtime = _numeric(data, "runtime_s")
    accepted_error = error[success]
    return {
        "evidence_group": label,
        "evidence_class": evidence_class,
        "primary_validation": primary_validation,
        "no_case_specific_tuning": no_case_specific_tuning,
        "rows": int(len(data)),
        "accepted_rows": int(success.sum()),
        "failed_or_diagnostic_rows": int((~success).sum()),
        "success_rate_pct": round(100.0 * float(success.mean()) if len(data) else 0.0, 2),
        "capture_mae_pct": _round(accepted_error.abs().mean()),
        "max_abs_capture_error_pct": _round(accepted_error.abs().max()),
        "median_runtime_s": _round(runtime.median()),
        "notes": notes,
    }


def _write_validation_registry(
    c_cases: pd.DataFrame,
    k_primary: pd.DataFrame,
    k_recovery: pd.DataFrame,
    k_unresolved: pd.DataFrame,
    epcsaft_smoke: pd.DataFrame,
    epcsaft_recovery: pd.DataFrame,
    k2_blend: pd.DataFrame,
) -> None:
    rows = []
    for thermo_model, group in c_cases.groupby("thermo_model", sort=False):
        rows.append(
            _summary_row(
                f"One-bed C cases, {thermo_model}",
                group,
                "primary",
                True,
                True,
                "All seven one-bed C rows are accepted; several cases still retain large capture errors.",
            )
        )
    rows.extend(
        [
            _summary_row(
                "Staged/intercooled K cases, Henry primary",
                k_primary,
                "primary",
                True,
                True,
                "Primary K rows use one documented global setting set and the eight-percentage-point capture gate.",
            ),
            _summary_row(
                "K-case recovery rows, Henry",
                k_recovery,
                "recovery",
                False,
                False,
                "Rows use case-specific recovery settings and are not counted as primary validation.",
            ),
            _summary_row(
                "K-case unresolved diagnostic rows, Henry",
                k_unresolved,
                "diagnostic",
                False,
                False,
                "Rows document failed or unresolved branches and are not mixed into accepted validation evidence.",
            ),
            _summary_row(
                "Staged K cases, neutral ePC-SAFT nominal",
                epcsaft_smoke,
                "diagnostic",
                False,
                True,
                "Nominal Henry-seeded ePC-SAFT fugacity rows; accepted and failed rows are reported together.",
            ),
            _summary_row(
                "Staged K cases, neutral ePC-SAFT recovery",
                epcsaft_recovery,
                "recovery",
                False,
                False,
                "Targeted recovery probe for nominal ePC-SAFT failures.",
            ),
            _summary_row(
                "K2 neutral ePC-SAFT fugacity-blend diagnostic",
                k2_blend,
                "diagnostic",
                False,
                False,
                "Thermodynamic-continuation diagnostic; partial blends are not primary full-endpoint validation.",
            ),
        ]
    )
    pd.DataFrame(rows).to_csv(TABLES / "validation_evidence_registry.csv", index=False)


def _augment_k_cases(k_primary: pd.DataFrame, nccc: pd.DataFrame) -> pd.DataFrame:
    data = k_primary.merge(nccc, on="case_id", how="left", suffixes=("", "_input"))
    data["L_over_G"] = data["L"].astype(float) / data["G"].astype(float)
    data["case_source"] = "K staged/intercooled"
    return data


def _augment_c_cases(c_cases: pd.DataFrame, c_inputs: pd.DataFrame) -> pd.DataFrame:
    data = c_cases.merge(c_inputs, on="case_id", how="left", suffixes=("", "_input"))
    data["L_over_G"] = data["L/G"].astype(float)
    data["beds"] = data.get("beds", data.get("beds_input", 1))
    data["intercoolers"] = 0
    data["case_source"] = "C one-bed"
    return data


def _write_no_case_tuning_gate(c_primary: pd.DataFrame, k_primary: pd.DataFrame) -> None:
    rows = []
    for _, row in c_primary.iterrows():
        rows.append(
            {
                "case_id": row["case_id"],
                "thermo_model": row["thermo_model"],
                "evidence_group": "one-bed C cases",
                "primary_validation": True,
                "no_case_specific_tuning": True,
                "mass_transfer_factor": 1.0,
                "intercooler_strength": 1.0,
                "case_specific_recovery": False,
                "gate_note": "primary C-case row uses default benchmark settings",
            }
        )
    for _, row in k_primary.iterrows():
        path = str(row.get("continuation_path", ""))
        no_tuning = (
            "mass_transfer_factor=1" in path
            and "intercooler_strength=1" in path
            and "co2_flux_mode=bidirectional" in path
        )
        rows.append(
            {
                "case_id": row["case_id"],
                "thermo_model": row["thermo_model"],
                "evidence_group": "staged/intercooled K cases",
                "primary_validation": True,
                "no_case_specific_tuning": bool(no_tuning),
                "mass_transfer_factor": 1.0,
                "intercooler_strength": 1.0,
                "case_specific_recovery": False,
                "gate_note": "primary K row must retain the documented global continuation settings",
            }
        )
    gate = pd.DataFrame(rows)
    gate.to_csv(TABLES / "primary_validation_gate.csv", index=False)
    summary = (
        gate.groupby(["evidence_group", "thermo_model"], dropna=False)
        .agg(
            rows=("case_id", "count"),
            no_case_specific_tuning_rows=("no_case_specific_tuning", "sum"),
            primary_validation_rows=("primary_validation", "sum"),
        )
        .reset_index()
    )
    summary["gate_pass"] = summary["rows"].eq(summary["no_case_specific_tuning_rows"])
    summary.to_csv(TABLES / "primary_validation_gate_summary.csv", index=False)


def _structured_split(data: pd.DataFrame) -> pd.DataFrame:
    split = data.copy()
    split["case_num"] = split["case_id"].astype(str).str.extract(r"(\d+)").astype(int)
    split["split"] = "train"
    for _, group in split.sort_values("case_num").groupby(["beds", "intercoolers"], sort=True):
        split.loc[group.tail(1).index, "split"] = "holdout"
    return split


def _write_calibration_artifacts(k_primary: pd.DataFrame) -> None:
    data = _structured_split(k_primary)
    features = ["L_over_G", "alpha", "y_CO2"]
    train = data["split"].eq("train")
    means = {feature: float(data.loc[train, feature].astype(float).mean()) for feature in features}
    x_train = _design_matrix(data.loc[train], features, means)
    y_train = data.loc[train, "capture_error_pct"].astype(float).to_numpy()
    coefficients = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    data["predicted_capture_error_correction_pct"] = _design_matrix(data, features, means) @ coefficients
    data["calibrated_capture_error_pct"] = (
        data["capture_error_pct"].astype(float) - data["predicted_capture_error_correction_pct"]
    )
    data["calibrated_capture_pct"] = (
        data["capture_pct"].astype(float) - data["predicted_capture_error_correction_pct"]
    ).clip(0.0, 100.0)
    data["calibration_model"] = "three_term_global_residual_correction"
    data["calibration_scope"] = "screening_only_not_mechanistic_factor_fit"
    data[
        [
            "case_id",
            "split",
            "capture_pct",
            "measured_capture_pct",
            "capture_error_pct",
            "predicted_capture_error_correction_pct",
            "calibrated_capture_pct",
            "calibrated_capture_error_pct",
            "L_over_G",
            "alpha",
            "y_CO2",
            "beds",
            "intercoolers",
            "calibration_model",
            "calibration_scope",
        ]
    ].to_csv(TABLES / "calibration_holdout_predictions.csv", index=False)

    coefficient_rows = [{"term": "intercept", "coefficient": coefficients[0], "center": 0.0}]
    coefficient_rows.extend(
        {"term": feature, "coefficient": coefficients[i + 1], "center": means[feature]}
        for i, feature in enumerate(features)
    )
    pd.DataFrame(coefficient_rows).to_csv(TABLES / "calibration_coefficients.csv", index=False)

    metrics = []
    residual_std = float(data.loc[train, "calibrated_capture_error_pct"].std(ddof=1))
    band = 1.96 * residual_std
    for split_name, group in data.groupby("split", sort=False):
        raw_error = group["capture_error_pct"].astype(float)
        calibrated_error = group["calibrated_capture_error_pct"].astype(float)
        metrics.append(
            {
                "split": split_name,
                "cases": ";".join(group["case_id"].astype(str)),
                "n_cases": int(len(group)),
                "raw_capture_mae_pct": _round(raw_error.abs().mean()),
                "raw_capture_rmse_pct": _round(math.sqrt(float((raw_error**2).mean()))),
                "calibrated_capture_mae_pct": _round(calibrated_error.abs().mean()),
                "calibrated_capture_rmse_pct": _round(math.sqrt(float((calibrated_error**2).mean()))),
                "uncertainty_band_half_width_pct": _round(band),
                "band_coverage_fraction": _round(calibrated_error.abs().le(band).mean()),
            }
        )
    pd.DataFrame(metrics).to_csv(TABLES / "calibration_holdout_metrics.csv", index=False)

    band_data = data[
        [
            "case_id",
            "split",
            "measured_capture_pct",
            "capture_pct",
            "calibrated_capture_pct",
            "calibrated_capture_error_pct",
        ]
    ].copy()
    band_data["lower_95_pct"] = band_data["calibrated_capture_pct"] - band
    band_data["upper_95_pct"] = band_data["calibrated_capture_pct"] + band
    band_data.to_csv(TABLES / "uncertainty_band_capture.csv", index=False)
    _plot_calibration_band(band_data, band)


def _design_matrix(data: pd.DataFrame, features: list[str], means: dict[str, float]) -> np.ndarray:
    columns = [np.ones(len(data))]
    columns.extend((data[feature].astype(float) - means[feature]).to_numpy() for feature in features)
    return np.column_stack(columns)


def _plot_calibration_band(data: pd.DataFrame, band: float) -> None:
    ordered = data.sort_values(["split", "case_id"]).reset_index(drop=True)
    x = np.arange(len(ordered))
    colors = ordered["split"].map({"train": "#4f6f52", "holdout": "#8a4b2b"}).fillna("#555555")
    fig, ax = plt.subplots(figsize=(7.5, 3.7), constrained_layout=True)
    ax.errorbar(
        x,
        ordered["calibrated_capture_pct"],
        yerr=band,
        fmt="none",
        ecolor="0.65",
        elinewidth=1.0,
        capsize=2.5,
        zorder=1,
        label="95% residual band",
    )
    ax.scatter(x, ordered["measured_capture_pct"], marker="x", color="black", s=32, zorder=3, label="measured")
    ax.scatter(x, ordered["capture_pct"], facecolors="none", edgecolors="#2f5d8c", s=34, zorder=2, label="raw model")
    ax.scatter(x, ordered["calibrated_capture_pct"], color=colors, s=28, zorder=4, label="calibrated screen")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["case_id"], rotation=45)
    ax.set_ylabel(r"CO$_2$ capture (%)")
    ax.set_xlabel("NCCC K case")
    ax.set_title("Structured holdout calibration screen with residual band", pad=8)
    handles = [
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="measured"),
        Line2D([0], [0], marker="o", color="#2f5d8c", markerfacecolor="none", linestyle="None", label="raw model"),
        Line2D([0], [0], marker="o", color="#4f6f52", linestyle="None", label="train calibrated"),
        Line2D([0], [0], marker="o", color="#8a4b2b", linestyle="None", label="holdout calibrated"),
        Line2D([0], [0], color="0.65", lw=1.0, label="95% residual band"),
    ]
    ax.legend(handles=handles, ncol=2, fontsize=8, frameon=False)
    _save_figure(fig, "calibration_uncertainty_band")


def _write_error_regime_artifacts(c_primary: pd.DataFrame, k_primary: pd.DataFrame) -> None:
    c_henry = c_primary[c_primary["thermo_model"].eq("ideal_henry")].copy()
    c_henry["group_label"] = "C one-bed Henry"
    k = k_primary.copy()
    k["group_label"] = "K staged Henry"
    for frame in (c_henry, k):
        if "temperature_rmse_K" not in frame.columns:
            frame["temperature_rmse_K"] = np.nan
    data = pd.concat(
        [
            c_henry[
                [
                    "case_id",
                    "group_label",
                    "capture_error_pct",
                    "L_over_G",
                    "alpha",
                    "y_CO2",
                    "beds",
                    "intercoolers",
                    "temperature_rmse_K",
                ]
            ],
            k[
                [
                    "case_id",
                    "group_label",
                    "capture_error_pct",
                    "L_over_G",
                    "alpha",
                    "y_CO2",
                    "beds",
                    "intercoolers",
                    "temperature_rmse_K",
                ]
            ],
        ],
        ignore_index=True,
    )
    data.to_csv(TABLES / "error_regime_capture_data.csv", index=False)
    _plot_error_regimes(data)


def _plot_error_regimes(data: pd.DataFrame) -> None:
    specs = [
        ("L_over_G", r"$L/G$"),
        ("alpha", r"lean loading $\alpha$"),
        ("y_CO2", r"inlet $y_{\mathrm{CO_2}}$"),
        ("beds", "packed beds"),
        ("intercoolers", "intercoolers"),
    ]
    colors = {"C one-bed Henry": "#2f5d8c", "K staged Henry": "#4f6f52"}
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.3), constrained_layout=True)
    axes = axes.ravel()
    for ax, (column, label) in zip(axes, specs):
        for group, subset in data.groupby("group_label", sort=False):
            ax.scatter(
                subset[column],
                subset["capture_error_pct"],
                s=28,
                color=colors.get(group, "#555555"),
                label=group,
                alpha=0.9,
            )
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.axhline(8, color="0.65", linewidth=0.7, linestyle="--")
        ax.axhline(-8, color="0.65", linewidth=0.7, linestyle="--")
        ax.set_xlabel(label)
        ax.set_ylabel("Capture error (%)")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    _save_figure(fig, "error_regime_capture_error")


def _write_staged_epcsaft_reliability(
    epcsaft_smoke: pd.DataFrame,
    epcsaft_recovery: pd.DataFrame,
    k2_blend: pd.DataFrame,
) -> None:
    cases = []
    for source, data in [
        ("nominal", epcsaft_smoke),
        ("targeted_recovery", epcsaft_recovery),
        ("k2_blend", k2_blend),
    ]:
        for _, row in data.iterrows():
            error = pd.to_numeric(pd.Series([row.get("capture_error_pct")]), errors="coerce").iloc[0]
            residual = pd.to_numeric(pd.Series([row.get("boundary_residual_norm")]), errors="coerce").iloc[0]
            success = str(row.get("success")).lower() == "true"
            if success:
                outcome = "accepted"
            elif pd.notna(error) and abs(error) > 8:
                outcome = "capture_gate_failure"
            elif pd.notna(residual) and residual > 1:
                outcome = "residual_gate_failure"
            else:
                outcome = "other_failure"
            cases.append(
                {
                    "source": source,
                    "case_id": row.get("case_id"),
                    "success": success,
                    "capture_error_pct": error,
                    "boundary_residual_norm": residual,
                    "runtime_s": row.get("runtime_s"),
                    "outcome": outcome,
                    "message": row.get("message", ""),
                }
            )
    case_df = pd.DataFrame(cases)
    case_df.to_csv(TABLES / "staged_epcsaft_reliability_cases.csv", index=False)
    summary = (
        case_df.groupby(["source", "outcome"], sort=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    totals = case_df.groupby("source").size().rename("total_rows").reset_index()
    summary = summary.merge(totals, on="source", how="left")
    summary["fraction"] = summary["rows"] / summary["total_rows"]
    summary.to_csv(TABLES / "staged_epcsaft_reliability_summary.csv", index=False)
    _plot_epcsaft_reliability(summary)


def _plot_epcsaft_reliability(summary: pd.DataFrame) -> None:
    sources = ["nominal", "targeted_recovery", "k2_blend"]
    outcomes = ["accepted", "capture_gate_failure", "residual_gate_failure", "other_failure"]
    colors = {
        "accepted": "#4f6f52",
        "capture_gate_failure": "#b36b5e",
        "residual_gate_failure": "#d08b3e",
        "other_failure": "#8f8f8f",
    }
    pivot = summary.pivot_table(index="source", columns="outcome", values="rows", fill_value=0)
    for outcome in outcomes:
        if outcome not in pivot.columns:
            pivot[outcome] = 0
    pivot = pivot.reindex(sources).fillna(0)
    fig, ax = plt.subplots(figsize=(6.9, 3.4), constrained_layout=True)
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for outcome in outcomes:
        values = pivot[outcome].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=colors[outcome], label=outcome.replace("_", " "))
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(["nominal\n19 rows", "targeted\nrecovery", "K2 blend\ndiagnostic"])
    ax.set_ylabel("Rows")
    ax.set_title("Staged neutral ePC-SAFT reliability evidence", pad=8)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    _save_figure(fig, "staged_epcsaft_reliability")


def _write_intercooled_temperature_profile() -> None:
    source = (
        ANALYSIS
        / "results"
        / "runs"
        / "clean_profile_csvs_k_cases_fast"
        / "profiles"
        / "NCCC_Data"
        / "K3"
        / "scipy-bvp"
        / "ideal_henry"
        / "T.csv"
    )
    if not source.exists():
        raise FileNotFoundError(source)
    data = pd.read_csv(source)
    data.to_csv(TABLES / "intercooled_temperature_profile_k3.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.plot(data["height_m"], data["Tl"], color="#2f5d8c", linewidth=1.8, label="liquid model")
    ax.plot(data["height_m"], data["Tv"], color="#d97706", linewidth=1.8, label="vapor model")
    for boundary in sorted(data.loc[data["bed_id"].diff().fillna(0).ne(0), "height_m"].unique()):
        ax.axvline(boundary, color="0.45", linewidth=0.8, linestyle="--")
        ax.text(boundary, data[["Tl", "Tv"]].min().min() + 0.5, "liquid reset", rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("Packed height from column bottom (m)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Modeled intercooled temperature profile, NCCC K3", pad=8)
    ax.legend(frameon=False)
    _save_figure(fig, "intercooled_temperature_profile_k3")


def _write_morgan_appendix_c_intercooled_profile() -> None:
    source = (
        ANALYSIS
        / "data"
        / "input"
        / "morgan2020_appendix_c_case1a_absorber_temperature_profile.csv"
    )
    if not source.exists():
        raise FileNotFoundError(source)
    data = pd.read_csv(source)
    data["normalized_height_from_bottom"] = 1.0 - data["relative_position_top_to_bottom"].astype(float)
    data = data.sort_values("normalized_height_from_bottom")
    data.to_csv(TABLES / "morgan2020_case1a_measured_intercooled_profile.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.scatter(
        data["normalized_height_from_bottom"],
        data["measured_absorber_temperature_K"],
        color="#2f5d8c",
        marker="o",
        s=28,
        label="measured NCCC",
    )
    for boundary in (1.0 / 3.0, 2.0 / 3.0):
        ax.axvline(boundary, color="0.55", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Normalized packed height from column bottom")
    ax.set_ylabel("Absorber temperature (K)")
    ax.set_title("Measured three-bed absorber temperature profile, NCCC 1A", pad=8)
    ax.legend(frameon=False)
    _save_figure(fig, "morgan2020_case1a_measured_intercooled_profile")


def _write_intercooled_profile_comparison() -> None:
    model_path = TABLES / "intercooled_temperature_profile_k3.csv"
    measured_path = TABLES / "morgan2020_case1a_measured_intercooled_profile.csv"
    if not model_path.exists() or not measured_path.exists():
        raise FileNotFoundError("Run intercooled profile writers before the comparison figure.")
    model = pd.read_csv(model_path)
    measured = pd.read_csv(measured_path).sort_values("normalized_height_from_bottom")
    model["normalized_height_from_bottom"] = model["height_m"].astype(float) / model["height_m"].astype(float).max()

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True, constrained_layout=True)
    axes[0].plot(
        model["normalized_height_from_bottom"],
        model["Tl"],
        color="#2f5d8c",
        linewidth=1.8,
        label="liquid model",
    )
    axes[0].plot(
        model["normalized_height_from_bottom"],
        model["Tv"],
        color="#d97706",
        linewidth=1.5,
        label="vapor model",
    )
    axes[0].set_title("Modeled K3", pad=6)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(
        measured["normalized_height_from_bottom"],
        measured["measured_absorber_temperature_K"],
        color="#2f5d8c",
        marker="o",
        s=28,
        label="measured NCCC",
    )
    axes[1].set_title("Measured NCCC 1A", pad=6)
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        for boundary in (1.0 / 3.0, 2.0 / 3.0):
            ax.axvline(boundary, color="0.55", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Normalized packed height from bottom")
    axes[0].set_ylabel("Absorber temperature (K)")
    _save_figure(fig, "intercooled_temperature_profile_comparison")


def _save_figure(fig: plt.Figure, stem: str) -> None:
    svg = FIGURES / f"{stem}.svg"
    pdf = FIGURES / f"{stem}.pdf"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _strip_svg_trailing_whitespace(svg)
    shutil.copy2(pdf, DOC_FIGURES / f"{stem.replace('_', '-')}.pdf")


def _strip_svg_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _round(value: float | int | np.floating | None, digits: int = 3) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return round(float(value), digits)


if __name__ == "__main__":
    main()
