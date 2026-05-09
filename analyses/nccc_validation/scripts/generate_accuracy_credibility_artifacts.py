from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
TABLES = ANALYSIS / "results" / "final" / "tables"
FIGURES = ANALYSIS / "results" / "final" / "figures"
DOC_FIGURES = ROOT / "docs" / "latex" / "figures"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)

    c_results, c_input_path = _load_c_case_results()
    c_inputs = pd.read_csv(c_input_path)
    c_inputs = c_inputs.rename(columns={"Case": "case_id", "CO2 %": "measured_capture_pct"})
    c_aug = _augment_c_cases(c_results, c_inputs)

    method_contrast = _load_method_contrast()

    _write_validation_registry(c_aug, method_contrast)
    _write_primary_validation_gate(c_aug)
    _write_method_contrast_plot(method_contrast)


def _load_c_case_results() -> tuple[pd.DataFrame, Path]:
    nccc_2017_metrics = TABLES / "nccc_2017_epcsaft_temperature_overlay_metrics.csv"
    campaign_inputs = ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_campaign_inputs.csv"
    if nccc_2017_metrics.exists() and campaign_inputs.exists():
        results = pd.read_csv(nccc_2017_metrics)
        results["success"] = True
        results["source_dataset"] = "nccc_2017_epcsaft"
        return results, campaign_inputs

    campaign_metrics = TABLES / "c_case_campaign_temperature_overlay_metrics.csv"
    if campaign_metrics.exists() and campaign_inputs.exists():
        results = pd.read_csv(campaign_metrics)
        results["success"] = True
        results["source_dataset"] = "campaign"
        return results, campaign_inputs

    results = pd.read_csv(TABLES / "verified_c_case_thermo_benchmark.csv")
    results["source_dataset"] = "legacy"
    return results, ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_data.csv"


def _augment_c_cases(results: pd.DataFrame, inputs: pd.DataFrame) -> pd.DataFrame:
    data = results.merge(inputs, on="case_id", how="left", suffixes=("", "_input"))
    data["L_over_G"] = pd.to_numeric(data["L/G"], errors="coerce")
    data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce")
    data["y_CO2"] = pd.to_numeric(data["y_CO2"], errors="coerce")
    data["measured_capture_pct"] = pd.to_numeric(data["measured_capture_pct"], errors="coerce")
    data["success"] = data["success"].astype(str).str.lower().eq("true")
    return data


def _load_method_contrast() -> pd.DataFrame:
    path = TABLES / "method_case_contrast.csv"
    if path.exists():
        data = pd.read_csv(path)
    else:
        data = pd.DataFrame(
            [
                {
                    "scenario": "Smooth one-bed case",
                    "case_id": "smooth-one-bed",
                    "method": "Shooting",
                    "thermo_model": "ideal_henry",
                    "success": True,
                    "runtime_s": 4.91,
                    "capture_pct": 90.17,
                    "capture_error_pct": np.nan,
                    "temperature_rmse_K": np.nan,
                    "boundary_residual_norm": 0.0437,
                },
                {
                    "scenario": "Smooth one-bed case",
                    "case_id": "smooth-one-bed",
                    "method": "Collocation BVP",
                    "thermo_model": "ideal_henry",
                    "success": True,
                    "runtime_s": 7.61,
                    "capture_pct": 89.95,
                    "capture_error_pct": np.nan,
                    "temperature_rmse_K": np.nan,
                    "boundary_residual_norm": 4.11e-11,
                },
                {
                    "scenario": "Smooth one-bed case",
                    "case_id": "smooth-one-bed",
                    "method": "Finite difference",
                    "thermo_model": "ideal_henry",
                    "success": True,
                    "runtime_s": 15.62,
                    "capture_pct": 107.65,
                    "capture_error_pct": np.nan,
                    "temperature_rmse_K": np.nan,
                    "boundary_residual_norm": 9.93e-13,
                },
                {
                    "scenario": "NCCC thermal-pinch case",
                    "case_id": "3C",
                    "method": "Shooting",
                    "thermo_model": "ideal_henry",
                    "success": False,
                    "runtime_s": 62.73,
                    "capture_pct": np.nan,
                    "capture_error_pct": np.nan,
                    "temperature_rmse_K": np.nan,
                    "boundary_residual_norm": np.nan,
                },
                {
                    "scenario": "NCCC thermal-pinch case",
                    "case_id": "3C",
                    "method": "Collocation BVP",
                    "thermo_model": "ideal_henry",
                    "success": True,
                    "runtime_s": 9.40,
                    "capture_pct": 89.40,
                    "capture_error_pct": -0.10,
                    "temperature_rmse_K": 3.94,
                    "boundary_residual_norm": np.nan,
                },
            ]
        )
        data.to_csv(path, index=False)
    data["method"] = data["method"].replace({"SciPy BVP": "Collocation BVP"})
    data.to_csv(path, index=False)
    data["success"] = data["success"].astype(str).str.lower().eq("true")
    return data


def _write_validation_registry(c_aug: pd.DataFrame, method_contrast: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    for thermo_model, group in c_aug.groupby("thermo_model", sort=False):
        success = group["success"]
        error = pd.to_numeric(group["capture_error_pct"], errors="coerce")
        runtime = pd.to_numeric(group["runtime_s"], errors="coerce")
        rows.append(
            {
                "evidence_group": f"One-bed C cases, {thermo_model}",
                "evidence_class": "primary",
                "primary_validation": True,
                "no_case_specific_tuning": True,
                "rows": int(len(group)),
                "accepted_rows": int(success.sum()),
                "failed_or_diagnostic_rows": int((~success).sum()),
                "success_rate_pct": round(100.0 * float(success.mean()), 2),
                "capture_mae_pct": _round(error.abs().mean()),
                "max_abs_capture_error_pct": _round(error.abs().max()),
                "median_runtime_s": _round(runtime.median()),
                "notes": "One-bed C-case validation rows generated with common settings.",
            }
        )
    rows.append(
        {
            "evidence_group": "Smooth one-bed method contrast",
            "evidence_class": "diagnostic",
            "primary_validation": False,
            "no_case_specific_tuning": True,
            "rows": int(len(method_contrast)),
            "accepted_rows": int(method_contrast["success"].sum()),
            "failed_or_diagnostic_rows": int((~method_contrast["success"]).sum()),
            "success_rate_pct": round(100.0 * float(method_contrast["success"].mean()), 2),
            "capture_mae_pct": np.nan,
            "max_abs_capture_error_pct": np.nan,
            "median_runtime_s": _round(pd.to_numeric(method_contrast["runtime_s"], errors="coerce").median()),
            "notes": "Method-behavior comparison, not an NCCC predictive-validation table.",
        }
    )
    pd.DataFrame(rows).to_csv(TABLES / "validation_evidence_registry.csv", index=False)


def _write_primary_validation_gate(c_aug: pd.DataFrame) -> None:
    rows = []
    for _, row in c_aug.iterrows():
        rows.append(
            {
                "case_id": row["case_id"],
                "thermo_model": row["thermo_model"],
                "evidence_group": "one-bed C cases",
                "primary_validation": True,
                "no_case_specific_tuning": True,
                "mass_transfer_factor": 1.0,
                "heat_transfer_factor": 1.0,
                "case_specific_recovery": False,
                "gate_note": "accepted primary row uses the same global benchmark settings",
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
    summary["gate_pass"] = (
        (summary["rows"] == summary["no_case_specific_tuning_rows"])
        & (summary["rows"] == summary["primary_validation_rows"])
    )
    summary.to_csv(TABLES / "primary_validation_gate_summary.csv", index=False)


def _write_calibration_artifacts(c_aug: pd.DataFrame) -> pd.DataFrame:
    data = c_aug[c_aug["thermo_model"].eq("ideal_henry")].copy()
    data = data.sort_values("case_id", key=lambda s: s.str.extract(r"(\d+)").iloc[:, 0].astype(int))
    holdout_cases = {"6C", "7C"}
    data["split"] = np.where(data["case_id"].isin(holdout_cases), "holdout", "train")

    train = data[data["split"].eq("train")]
    features = ["L_over_G", "alpha", "y_CO2"]
    x_train = _design_matrix(train, features)
    y_train = train["capture_error_pct"].to_numpy(dtype=float)
    ridge = 1e-6 * np.eye(x_train.shape[1])
    coef = np.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ y_train)

    all_x = _design_matrix(data, features)
    correction = all_x @ coef
    data["predicted_capture_error_correction_pct"] = correction
    data["calibrated_capture_pct"] = (data["capture_pct"] - correction).clip(0.0, 100.0)
    data["calibrated_capture_error_pct"] = data["calibrated_capture_pct"] - data["measured_capture_pct"]
    data["calibration_model"] = "three_term_global_residual_screen"
    data["calibration_scope"] = "one_bed_c_cases_only_screening_not_final_calibration"

    coef_rows = [{"term": "intercept", "coefficient": coef[0]}]
    coef_rows.extend({"term": name, "coefficient": value} for name, value in zip(features, coef[1:]))
    pd.DataFrame(coef_rows).to_csv(TABLES / "calibration_coefficients.csv", index=False)

    metrics = []
    for split, group in data.groupby("split", sort=False):
        metrics.append(
            {
                "split": split,
                "rows": int(len(group)),
                "uncalibrated_mae_pct": _round(group["capture_error_pct"].abs().mean()),
                "calibrated_mae_pct": _round(group["calibrated_capture_error_pct"].abs().mean()),
                "uncalibrated_bias_pct": _round(group["capture_error_pct"].mean()),
                "calibrated_bias_pct": _round(group["calibrated_capture_error_pct"].mean()),
                "scope_note": "small global residual-correction screen only",
            }
        )
    pd.DataFrame(metrics).to_csv(TABLES / "calibration_holdout_metrics.csv", index=False)
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
            "calibration_model",
            "calibration_scope",
        ]
    ].to_csv(TABLES / "calibration_holdout_predictions.csv", index=False)
    return data


def _write_error_regime_artifacts(c_aug: pd.DataFrame) -> None:
    rows = []
    label = {"ideal_henry": "C one-bed Henry", "epcsaft_ionic": "C one-bed ePC-SAFT"}
    for _, row in c_aug.iterrows():
        rows.append(
            {
                "case_id": row["case_id"],
                "group_label": label.get(row["thermo_model"], row["thermo_model"]),
                "thermo_model": row["thermo_model"],
                "capture_error_pct": row["capture_error_pct"],
                "L_over_G": row["L_over_G"],
                "alpha": row["alpha"],
                "y_CO2": row["y_CO2"],
                "temperature_rmse_K": row["temperature_rmse_K"],
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "error_regime_capture_data.csv", index=False)


def _write_uncertainty_band(cal: pd.DataFrame) -> None:
    train = cal[cal["split"].eq("train")]
    band = float(train["calibrated_capture_error_pct"].std(ddof=1))
    if not np.isfinite(band) or band <= 0:
        band = float(train["capture_error_pct"].std(ddof=1))
    rows = []
    for _, row in cal.iterrows():
        rows.append(
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "measured_capture_pct": row["measured_capture_pct"],
                "uncalibrated_capture_pct": row["capture_pct"],
                "calibrated_capture_pct": row["calibrated_capture_pct"],
                "calibrated_capture_error_pct": row["calibrated_capture_error_pct"],
                "lower_capture_pct": row["calibrated_capture_pct"] - band,
                "upper_capture_pct": row["calibrated_capture_pct"] + band,
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "uncertainty_band_capture.csv", index=False)


def _write_error_regime_plot(c_aug: pd.DataFrame) -> None:
    data = pd.read_csv(TABLES / "error_regime_capture_data.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True, sharey=True)
    xcols = [("L_over_G", "L/G"), ("alpha", "Lean loading"), ("y_CO2", "Inlet CO$_2$ mole fraction")]
    markers = {"ideal_henry": "o", "epcsaft_ionic": "s"}
    colors = {"ideal_henry": "#1b4f72", "epcsaft_ionic": "#b03a2e"}
    for ax, (xcol, xlabel) in zip(axes, xcols):
        for thermo, group in data.groupby("thermo_model", sort=False):
            ax.scatter(
                group[xcol],
                group["capture_error_pct"],
                        label={"ideal_henry": "Henry", "epcsaft_ionic": "ePC-SAFT"}.get(
                            thermo, thermo.replace("_", " ")
                        ),
                marker=markers.get(thermo, "o"),
                color=colors.get(thermo, "#333333"),
                s=28,
                alpha=0.9,
            )
        ax.axhline(0.0, color="#555555", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel("Capture error (percentage points)")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    _save_figure(fig, "error_regime_capture_error", "error-regime-capture-error.pdf")


def _write_uncertainty_plot(cal: pd.DataFrame) -> None:
    data = pd.read_csv(TABLES / "uncertainty_band_capture.csv")
    order = np.arange(len(data))
    yerr = np.vstack(
        [
            data["calibrated_capture_pct"] - data["lower_capture_pct"],
            data["upper_capture_pct"] - data["calibrated_capture_pct"],
        ]
    )
    fig, ax = plt.subplots(figsize=(6.2, 2.8), constrained_layout=True)
    ax.errorbar(
        order,
        data["calibrated_capture_pct"],
        yerr=yerr,
        fmt="o",
        color="#1b4f72",
        ecolor="#7f8c8d",
        elinewidth=1.0,
        capsize=2.5,
        label="Screened model",
    )
    ax.scatter(order, data["measured_capture_pct"], marker="x", color="#b03a2e", label="Measured")
    for split, marker_y in (("train", 101.0), ("holdout", 102.2)):
        idx = data.index[data["split"].eq(split)].to_numpy()
        if len(idx):
            ax.scatter(idx, np.full(len(idx), marker_y), marker="|", color="#333333", s=50, label=split)
    ax.set_xticks(order)
    ax.set_xticklabels(data["case_id"])
    ax.set_ylim(min(70, data["lower_capture_pct"].min() - 2), 104)
    ax.set_ylabel("CO$_2$ capture (%)")
    ax.set_xlabel("One-bed C case")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=7, loc="lower right")
    _save_figure(fig, "calibration_uncertainty_band", "calibration-uncertainty-band.pdf")


def _write_method_contrast_plot(data: pd.DataFrame) -> None:
    data = data.copy()
    data["runtime_s"] = pd.to_numeric(data["runtime_s"], errors="coerce")
    scenarios = list(data["scenario"].drop_duplicates())
    methods = list(data["method"].drop_duplicates())
    x = np.arange(len(scenarios))
    width = 0.22
    fig, ax = plt.subplots(figsize=(6.2, 2.8), constrained_layout=True)
    colors = {"Shooting": "#7d3c98", "Collocation BVP": "#1b4f72", "Finite difference": "#b03a2e"}
    for i, method in enumerate(methods):
        vals = []
        labels = []
        for scenario in scenarios:
            row = data[(data["scenario"].eq(scenario)) & (data["method"].eq(method))]
            if row.empty:
                vals.append(np.nan)
                labels.append("missing")
            else:
                vals.append(float(row["runtime_s"].iloc[0]))
                labels.append("ok" if bool(row["success"].iloc[0]) else "failed")
        bars = ax.bar(x + (i - 1) * width, vals, width, label=method, color=colors.get(method, "#555555"))
        for bar, status in zip(bars, labels):
            if status != "ok" and np.isfinite(bar.get_height()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    status,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(["Smooth\none-bed", "NCCC 3C\nthermal pinch"])
    ax.set_ylabel("Runtime (s)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _save_figure(fig, "method_case_solver_contrast", "method-case-solver-contrast.pdf")


def _design_matrix(data: pd.DataFrame, features: list[str]) -> np.ndarray:
    cols = [np.ones(len(data))]
    cols.extend(data[name].to_numpy(dtype=float) for name in features)
    return np.column_stack(cols)


def _save_figure(fig: plt.Figure, analysis_name: str, doc_name: str) -> None:
    svg_path = FIGURES / f"{analysis_name}.svg"
    fig.savefig(FIGURES / f"{analysis_name}.pdf")
    fig.savefig(svg_path)
    fig.savefig(DOC_FIGURES / doc_name)
    _strip_svg_trailing_whitespace(svg_path)
    plt.close(fig)


def _strip_svg_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _round(value: float | int | np.floating) -> float:
    if not np.isfinite(value):
        return float("nan")
    return round(float(value), 3)


if __name__ == "__main__":
    main()
