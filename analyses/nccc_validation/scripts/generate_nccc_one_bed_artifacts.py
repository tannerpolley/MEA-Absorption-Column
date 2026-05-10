from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
TABLES = ANALYSIS / "results" / "final" / "tables"
FIGURES = ANALYSIS / "results" / "final" / "figures"
LATEX_TABLES = ROOT / "docs" / "latex" / "tables"

RUN_2014 = ANALYSIS / "results" / "runs" / "nccc_2014_no_intercooler_sweep" / "benchmark_results.csv"
RUN_2017 = ANALYSIS / "results" / "runs" / "nccc_2017_no_intercooler_sweep" / "benchmark_results.csv"
SOURCE_2014 = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2014_cases.csv"
SOURCE_2017 = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_cases.csv"
MODEL_2017 = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_model_inputs_mass.csv"
ACCEPTED_ONE_BED_CASE_IDS = {
    2014: {"K18", "K19"},
    2017: {f"{idx}C" for idx in range(1, 7)},
}


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_TABLES.mkdir(parents=True, exist_ok=True)

    results = _load_one_bed_results()
    accepted = _accepted_one_bed_results(results)
    accepted["abs_capture_error_pct"] = accepted["capture_error_pct"].abs()
    accepted.to_csv(TABLES / "nccc_one_bed_accepted_results.csv", index=False)
    results.to_csv(TABLES / "nccc_one_bed_all_attempted_results.csv", index=False)

    summary = (
        accepted.groupby("thermo_label", sort=False)
        .agg(
            accepted_rows=("case_id", "count"),
            capture_mae_pct=("abs_capture_error_pct", "mean"),
            max_abs_capture_error_pct=("abs_capture_error_pct", "max"),
            median_runtime_s=("runtime_s", "median"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "nccc_one_bed_accepted_summary.csv", index=False)

    case_table = _build_case_table()
    case_table.to_csv(TABLES / "nccc_one_bed_case_scope.csv", index=False)
    _write_latex_case_table(case_table)
    _plot_accepted_results(accepted, summary)

    print(f"Wrote {TABLES / 'nccc_one_bed_accepted_results.csv'}")
    print(f"Wrote {TABLES / 'nccc_one_bed_case_scope.csv'}")
    print(f"Wrote {FIGURES / 'nccc_one_bed_thermo_benchmark.pdf'}")
    return 0


def _load_one_bed_results() -> pd.DataFrame:
    frames = []
    for year, path in ((2014, RUN_2014), (2017, RUN_2017)):
        data = pd.read_csv(path)
        data["campaign_year"] = year
        frames.append(data)
    results = pd.concat(frames, ignore_index=True)
    results = results[(results["beds"].eq(1)) & (results["intercoolers"].eq(0))].copy()
    results["thermo_label"] = results["thermo_model"].map(
        {"ideal_henry": "Henry", "epcsaft_ionic": "ePC-SAFT"}
    )
    results["case_sort"] = results["case_id"].map(_case_sort_key)
    return results.sort_values(["campaign_year", "case_sort", "thermo_model"]).drop(columns="case_sort")


def _accepted_one_bed_results(results: pd.DataFrame) -> pd.DataFrame:
    accepted_mask = results["success"].astype(bool)
    accepted_mask &= results.apply(
        lambda row: str(row["case_id"]) in ACCEPTED_ONE_BED_CASE_IDS.get(int(row["campaign_year"]), set()),
        axis=1,
    )
    return results[accepted_mask].copy()


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
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.7)

    axes[1].bar(
        summary["thermo_label"],
        summary["median_runtime_s"],
        color=[colors.get(label) for label in summary["thermo_label"]],
        width=0.55,
    )
    axes[1].set_ylabel("Median runtime (s)")
    axes[1].set_title("Accepted-row runtime", pad=8)
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.7)

    axes[0].legend(loc="upper left", fontsize=7.0, frameon=False)
    fig.savefig(FIGURES / "nccc_one_bed_thermo_benchmark.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "nccc_one_bed_thermo_benchmark.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _case_sort_key(case_id: str) -> tuple[str, int]:
    if case_id.startswith("K"):
        return ("K", int(case_id[1:]))
    return (case_id[-1], int(case_id[:-1]))


if __name__ == "__main__":
    raise SystemExit(main())
