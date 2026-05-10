from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
TABLES = ANALYSIS / "results" / "final" / "tables"
FIGURES = ANALYSIS / "results" / "final" / "figures"
LATEX_TABLES = ROOT / "docs" / "latex" / "tables"

SOURCE_2014 = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2014_cases.csv"
SOURCE_2017 = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_cases.csv"
MODEL_2017 = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_model_inputs_mass.csv"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_TABLES.mkdir(parents=True, exist_ok=True)

    accepted, summary = _load_pr_backed_accepted_artifacts()

    case_table = _build_case_table()
    case_table.to_csv(TABLES / "nccc_one_bed_case_scope.csv", index=False)
    _write_latex_case_table(case_table)
    _plot_accepted_results(accepted, summary)

    print(f"Read {TABLES / 'nccc_one_bed_accepted_results.csv'}")
    print(f"Wrote {TABLES / 'nccc_one_bed_case_scope.csv'}")
    print(f"Wrote {FIGURES / 'nccc_one_bed_thermo_benchmark.pdf'}")
    return 0


def _load_pr_backed_accepted_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_path = TABLES / "nccc_one_bed_accepted_results.csv"
    summary_path = TABLES / "nccc_one_bed_accepted_summary.csv"
    if not accepted_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Figure 4 requires the PR-backed final accepted-row tables. "
            f"Missing {accepted_path if not accepted_path.exists() else summary_path}."
        )
    accepted = pd.read_csv(accepted_path)
    summary = pd.read_csv(summary_path)
    required_accepted = {
        "case_id",
        "campaign_year",
        "thermo_label",
        "capture_error_pct",
        "runtime_s",
    }
    required_summary = {
        "thermo_label",
        "accepted_rows",
        "capture_mae_pct",
        "max_abs_capture_error_pct",
        "median_runtime_s",
    }
    if not required_accepted.issubset(accepted.columns) or not required_summary.issubset(summary.columns):
        missing_accepted = sorted(required_accepted - set(accepted.columns))
        missing_summary = sorted(required_summary - set(summary.columns))
        raise ValueError(
            "Final accepted-row artifacts are missing required columns: "
            f"accepted={missing_accepted}, summary={missing_summary}"
        )
    accepted = accepted.copy()
    if "abs_capture_error_pct" not in accepted.columns:
        accepted["abs_capture_error_pct"] = accepted["capture_error_pct"].abs()
    _validate_pr_backed_accepted_artifacts(accepted, summary)
    return accepted, summary


def _validate_pr_backed_accepted_artifacts(accepted: pd.DataFrame, summary: pd.DataFrame) -> None:
    expected_cases = {"K18", "K19", "1C", "2C", "3C", "4C", "5C", "6C"}
    expected_labels = {"Henry", "ePC-SAFT"}
    actual_cases = set(accepted["case_id"].astype(str))
    actual_labels = set(accepted["thermo_label"].astype(str))
    if actual_cases != expected_cases or actual_labels != expected_labels or len(accepted) != 16:
        raise ValueError(
            "Final accepted-row table does not match the PR-backed Figure 4 scope: "
            f"cases={sorted(actual_cases)}, labels={sorted(actual_labels)}, rows={len(accepted)}"
        )

    computed = (
        accepted.groupby("thermo_label", sort=False)
        .agg(
            accepted_rows=("case_id", "count"),
            capture_mae_pct=("abs_capture_error_pct", "mean"),
            max_abs_capture_error_pct=("abs_capture_error_pct", "max"),
            median_runtime_s=("runtime_s", "median"),
        )
        .reset_index()
    )
    merged = summary.merge(computed, on="thermo_label", suffixes=("_file", "_computed"), validate="one_to_one")
    for column in ("accepted_rows", "capture_mae_pct", "max_abs_capture_error_pct", "median_runtime_s"):
        delta = (merged[f"{column}_file"] - merged[f"{column}_computed"]).abs().max()
        if delta > 1e-9:
            raise ValueError(f"Final accepted-row summary disagrees with accepted results for {column}.")


def _build_case_table() -> pd.DataFrame:
    source_2014 = pd.read_csv(SOURCE_2014)
    source_2017 = pd.read_csv(SOURCE_2017)
    model_2017 = pd.read_csv(MODEL_2017).set_index("case_no")

    rows: list[dict[str, object]] = []
    for _, row in source_2014[source_2014["case_no"].isin(["K18", "K19", "K20"])].iterrows():
        rows.append(
            {
                "campaign_year": 2014,
                "case_id": row["case_no"],
                "lean_flow_kg_h": row["absorber_lean_solvent_flow_kg_h"],
                "gas_flow_kg_h": row["absorber_flue_gas_flow_kg_h"],
                "lean_temp_C": row["absorber_lean_solvent_temp_C"],
                "gas_temp_C": row["absorber_inlet_gas_temp_C"],
                "pressure_kPa": row["absorber_pressure_kPa"],
                "alpha": row["absorber_lean_loading_mol_co2_per_mol_mea"],
                "w_MEA": row["absorber_nominal_lean_solvent_mea_weight_fraction"],
                "y_CO2": row["absorber_inlet_gas_co2_mol_pct"] / 100.0,
                "capture_pct": row["capture_gas_side_pct"],
                "temp_imputed": False,
            }
        )
    c_rows = source_2017[source_2017["case_no"].isin([f"{idx}C" for idx in range(1, 8)])]
    for _, row in c_rows.iterrows():
        case_id = row["case_no"]
        model_row = model_2017.loc[case_id]
        lean_temp_c = row["absorber_lean_solvent_temp_C"]
        temp_imputed = bool(model_row["lean_solvent_temp_imputed"])
        if pd.isna(lean_temp_c) and temp_imputed:
            lean_temp_c = float(model_row["lean_solvent_temp_imputed_C"])
        rows.append(
            {
                "campaign_year": 2017,
                "case_id": case_id,
                "lean_flow_kg_h": row["absorber_lean_solvent_flow_kg_h"],
                "gas_flow_kg_h": row["absorber_flue_gas_flow_kg_h"],
                "lean_temp_C": lean_temp_c,
                "gas_temp_C": row["absorber_inlet_gas_temp_C"],
                "pressure_kPa": row["absorber_top_pressure_kPa"],
                "alpha": row["absorber_lean_loading_mol_co2_per_mol_mea"],
                "w_MEA": row["absorber_lean_solvent_mea_weight_fraction"],
                "y_CO2": row["absorber_inlet_gas_co2_mole_fraction"],
                "capture_pct": row["absorber_capture_pct_avg"],
                "temp_imputed": temp_imputed,
            }
        )
    table = pd.DataFrame(rows)
    table["case_sort"] = table["case_id"].map(_case_sort_key)
    return table.sort_values(["campaign_year", "case_sort"]).drop(columns="case_sort")


def _write_latex_case_table(data: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \scriptsize",
        r"    \caption{NCCC one-bed no-intercooler MEA validation scope}",
        r"    \label{tab:nccc-one-bed-scope}",
        r"    \renewcommand{\arraystretch}{1.16}",
        r"    \begin{adjustbox}{max width=\textwidth}",
        r"    \begin{tabularx}{1.16\textwidth}{|c|c| *{9}{>{\centering\arraybackslash}X|}}",
        r"    \hline",
        r"    \textbf{Year} & \textbf{Case} & \makecell{\textbf{Lean}\\\textbf{flow}\\(kg/h)} & \makecell{\textbf{Gas}\\\textbf{flow}\\(kg/h)} & \makecell{\textbf{Lean}\\\textbf{temp.}\\(\si{\degreeCelsius})} & \makecell{\textbf{Gas}\\\textbf{temp.}\\(\si{\degreeCelsius})} & \makecell{\textbf{Pressure}\\(kPa)} & \makecell{\textbf{Lean}\\\textbf{loading}} & \makecell{\textbf{MEA}\\\textbf{mass frac.}} & \makecell{\textbf{Inlet}\\\textbf{$y_{\mathrm{CO_2}}$}} & \makecell{\textbf{Capture}\\(\%)} \\",
        r"    \hline",
    ]
    for _, row in data.iterrows():
        lean_temp = f"{row['lean_temp_C']:.1f}"
        if bool(row["temp_imputed"]):
            lean_temp += r"$^{\dagger}$"
        lines.append(
            "    "
            + " & ".join(
                [
                    str(int(row["campaign_year"])),
                    str(row["case_id"]),
                    f"{row['lean_flow_kg_h']:.0f}",
                    f"{row['gas_flow_kg_h']:.0f}",
                    lean_temp,
                    f"{row['gas_temp_C']:.1f}",
                    f"{row['pressure_kPa']:.1f}",
                    f"{row['alpha']:.2f}",
                    f"{row['w_MEA']:.2f}",
                    f"{row['y_CO2']:.3f}",
                    f"{row['capture_pct']:.1f}",
                ]
            )
            + r"\\"
        )
    lines.extend(
        [
            r"    \hline",
            r"    \end{tabularx}",
            r"    \end{adjustbox}",
            r"    \parbox{0.95\textwidth}{\footnotesize $^{\dagger}$Lean solvent inlet temperature was blank in the source table and was set to \SI{45.0}{\degreeCelsius} for the model input.}",
            r"\end{table}",
            "",
        ]
    )
    (LATEX_TABLES / "nccc_one_bed_case_scope.tex").write_text("\n".join(lines), encoding="utf-8")


def _plot_accepted_results(accepted: pd.DataFrame, summary: pd.DataFrame) -> None:
    colors = {"Henry": "#2f5d8c", "ePC-SAFT": "#8a4b2b"}
    markers = {2014: "s", 2017: "o"}
    accepted = accepted.copy()
    accepted["case_label"] = accepted["campaign_year"].astype(str) + " " + accepted["case_id"].astype(str)
    order = accepted.drop_duplicates("case_label")["case_label"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
    for (label, year), group in accepted.groupby(["thermo_label", "campaign_year"], sort=False):
        axes[0].plot(
            group["case_label"],
            group["capture_error_pct"],
            marker=markers.get(int(year), "o"),
            linewidth=1.7,
            markersize=5.5,
            color=colors.get(label),
            label=f"{label}, {year}",
        )
    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].set_xticks(range(len(order)), order, rotation=45, ha="right")
    axes[0].set_ylabel("Capture error, predicted - measured (p.p.)")
    axes[0].set_xlabel("Accepted one-bed NCCC case")
    axes[0].set_title("Accepted capture validation", pad=8)
    axes[0].grid(False)

    axes[1].bar(
        summary["thermo_label"],
        summary["median_runtime_s"],
        color=[colors.get(label) for label in summary["thermo_label"]],
        width=0.55,
    )
    axes[1].set_ylabel("Median runtime (s)")
    axes[1].set_title("Accepted-row runtime", pad=8)
    axes[1].grid(False)

    axes[0].legend(
        loc="upper center",
        ncol=2,
        fontsize=7.0,
        frameon=False,
    )
    fig.savefig(FIGURES / "nccc_one_bed_thermo_benchmark.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "nccc_one_bed_thermo_benchmark.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _case_sort_key(case_id: str) -> tuple[str, int]:
    if case_id.startswith("K"):
        return ("K", int(case_id[1:]))
    return (case_id[-1], int(case_id[:-1]))


if __name__ == "__main__":
    raise SystemExit(main())
