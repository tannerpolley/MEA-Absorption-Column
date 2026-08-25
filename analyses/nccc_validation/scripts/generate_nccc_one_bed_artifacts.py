from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
TABLES = ANALYSIS / "results" / "final" / "tables"
FIGURES = ANALYSIS / "results" / "final" / "figures"
LATEX_TABLES = ROOT / "docs" / "latex" / "tables"
LATEX_FIGURES = ROOT / "docs" / "latex" / "figures"

SOURCE_2014 = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2014_cases.csv"
SOURCE_2017 = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_cases.csv"
MODEL_2017 = (
    ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_2017_model_inputs_mass.csv"
)
ATTEMPTED_RESULTS = TABLES / "nccc_one_bed_all_attempted_results.csv"
ACCEPTED_RESULTS = TABLES / "nccc_one_bed_accepted_results.csv"
ACCEPTED_SUMMARY = TABLES / "nccc_one_bed_accepted_summary.csv"

BOUNDARY_RESIDUAL_LIMIT = 1.0
RUNTIME_LIMIT_S = 90.0
EXPECTED_CASES = {"K18", "K19", "K20", "1C", "2C", "3C", "4C", "5C", "6C", "7C"}
EXPECTED_THERMO_LABELS = {"Henry", "ePC-SAFT"}


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_TABLES.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)

    attempted, accepted, summary = _derive_accepted_artifacts()
    accepted.to_csv(ACCEPTED_RESULTS, index=False)
    summary.to_csv(ACCEPTED_SUMMARY, index=False)

    case_table = _build_case_table()
    case_table.to_csv(TABLES / "nccc_one_bed_case_scope.csv", index=False)
    _write_latex_case_table(case_table)
    _write_latex_attempted_status_table(attempted)
    _plot_accepted_results(accepted, summary)

    print(f"Read {ATTEMPTED_RESULTS}")
    print(f"Wrote {ACCEPTED_RESULTS}")
    print(f"Wrote {ACCEPTED_SUMMARY}")
    print(f"Wrote {TABLES / 'nccc_one_bed_case_scope.csv'}")
    print(f"Wrote {LATEX_TABLES / 'nccc_one_bed_attempted_status.tex'}")
    print(f"Wrote {FIGURES / 'nccc_one_bed_thermo_benchmark.pdf'}")
    print(f"Wrote {LATEX_FIGURES / 'nccc-one-bed-thermo-benchmark.pdf'}")
    return 0


def _derive_accepted_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attempted = pd.read_csv(ATTEMPTED_RESULTS)
    required_columns = {
        "case_id",
        "campaign_year",
        "thermo_label",
        "thermo_model",
        "success",
        "capture_pct",
        "capture_error_pct",
        "runtime_s",
        "boundary_residual_norm",
        "invalid_state_count",
        "guard_penalty_count",
        "message",
    }
    missing = sorted(required_columns - set(attempted.columns))
    if missing:
        raise ValueError(
            f"Attempted-row artifact is missing required columns: {missing}"
        )

    attempted = attempted.copy()
    attempted["case_id"] = attempted["case_id"].astype(str)
    actual_cases = set(attempted["case_id"])
    actual_labels = set(attempted["thermo_label"].astype(str))
    row_pairs = attempted[["case_id", "thermo_label"]]
    if (
        actual_cases != EXPECTED_CASES
        or actual_labels != EXPECTED_THERMO_LABELS
        or len(attempted) != len(EXPECTED_CASES) * len(EXPECTED_THERMO_LABELS)
        or row_pairs.duplicated().any()
    ):
        raise ValueError(
            "Attempted-row artifact does not contain one row per expected case and thermodynamic lane: "
            f"cases={sorted(actual_cases)}, labels={sorted(actual_labels)}, rows={len(attempted)}"
        )

    success = attempted["success"].astype(str).str.lower().eq("true")
    attempted["accepted_gate"] = (
        success
        & attempted["boundary_residual_norm"].le(BOUNDARY_RESIDUAL_LIMIT)
        & attempted["capture_pct"].between(0.0, 100.0, inclusive="both")
        & attempted["runtime_s"].le(RUNTIME_LIMIT_S)
    )
    attempted["abs_capture_error_pct"] = attempted["capture_error_pct"].abs()
    accepted = attempted.loc[attempted["accepted_gate"]].copy()
    accepted = accepted.sort_values(
        ["campaign_year", "case_id", "thermo_label"],
        key=lambda values: (
            values.map(_case_sort_key) if values.name == "case_id" else values
        ),
    )
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
    return attempted, accepted, summary


def _build_case_table() -> pd.DataFrame:
    source_2014 = pd.read_csv(SOURCE_2014)
    source_2017 = pd.read_csv(SOURCE_2017)
    model_2017 = pd.read_csv(MODEL_2017).set_index("case_no")

    rows: list[dict[str, object]] = []
    for _, row in source_2014[
        source_2014["case_no"].isin(["K18", "K19", "K20"])
    ].iterrows():
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
    c_rows = source_2017[
        source_2017["case_no"].isin([f"{idx}C" for idx in range(1, 8)])
    ]
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
        r"    \caption{NCCC one-bed no-intercooler MEA physical-comparison scope}",
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
    (LATEX_TABLES / "nccc_one_bed_case_scope.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_latex_attempted_status_table(attempted: pd.DataFrame) -> None:
    by_case = {
        case_id: group.set_index("thermo_label")
        for case_id, group in attempted.groupby("case_id")
    }

    def accepted(case_id: str) -> bool:
        return bool(by_case[case_id]["accepted_gate"].all())

    if not all(accepted(case_id) for case_id in EXPECTED_CASES - {"K20"}) or accepted(
        "K20"
    ):
        raise ValueError(
            "The generated status-table grouping no longer matches the row-level acceptance gate."
        )

    k19 = by_case["K19"]
    c7 = by_case["7C"]
    rows = [
        (
            "2014",
            "K18",
            "Included",
            "Henry and ePC-SAFT",
            "Both rows satisfied all four criteria and recorded no domain-guard events.",
        ),
        (
            "2014",
            "K19",
            "Included",
            "Henry and ePC-SAFT",
            "Both rows satisfied all four criteria; the ePC-SAFT and Henry calculations recorded "
            f"{int(k19.loc['ePC-SAFT', 'guard_penalty_count'])} and "
            f"{int(k19.loc['Henry', 'guard_penalty_count'])} guard events, respectively.",
        ),
        (
            "2014",
            "K20",
            "Excluded",
            "Henry and ePC-SAFT",
            "Both solvers exceeded the maximum mesh-node limit and encountered hydraulics or "
            "pressure-drop domain violations.",
        ),
        (
            "2017",
            "1C--6C",
            "Included",
            "Henry and ePC-SAFT",
            "All twelve rows satisfied all four criteria and recorded no domain-guard events.",
        ),
        (
            "2017",
            "7C",
            "Included",
            "Henry and ePC-SAFT",
            f"Both rows satisfied all four criteria: ePC-SAFT ran in "
            f"{c7.loc['ePC-SAFT', 'runtime_s']:.2f} s with "
            f"{int(c7.loc['ePC-SAFT', 'guard_penalty_count'])} guard events; Henry ran in "
            f"{c7.loc['Henry', 'runtime_s']:.2f} s with "
            f"{int(c7.loc['Henry', 'guard_penalty_count'])} guard events.",
        ),
    ]
    lines = [
        r"\begin{center}",
        r"    \centering",
        r"    \small",
        rf"    \captionof{{table}}{{Attempted one-bed NCCC cases. Included rows require solver success, boundary-residual norm $\leq {BOUNDARY_RESIDUAL_LIMIT:.1f}$, predicted capture between 0 and 100\%, and runtime $\leq\SI{{{RUNTIME_LIMIT_S:.0f}}}{{s}}$.}}",
        r"    \label{tab:nccc-attempted-status}",
        r"    \renewcommand{\arraystretch}{1.15}",
        r"    \begin{tabularx}{\textwidth}{@{}ll>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}X@{}}",
        r"        \toprule",
        r"        \textbf{Year} & \textbf{Case} & \textbf{Aggregate status} & \textbf{Thermodynamic calculations} & \textbf{Evidence} \\",
        r"        \midrule",
    ]
    lines.extend("        " + " & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabularx}",
            r"\end{center}",
            "",
        ]
    )
    (LATEX_TABLES / "nccc_one_bed_attempted_status.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _plot_accepted_results(accepted: pd.DataFrame, summary: pd.DataFrame) -> None:
    colors = {"Henry": "#2f5d8c", "ePC-SAFT": "#8a4b2b"}
    markers = {2014: "s", 2017: "o"}
    accepted = accepted.copy()
    accepted["case_label"] = (
        accepted["campaign_year"].astype(str) + " " + accepted["case_id"].astype(str)
    )
    order = accepted.drop_duplicates("case_label")["case_label"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
    for (label, year), group in accepted.groupby(
        ["thermo_label", "campaign_year"], sort=False
    ):
        axes[0].plot(
            group["case_label"],
            group["capture_error_pct"],
            marker=markers.get(int(year), "o"),
            linestyle="None",
            markersize=5.5,
            color=colors.get(label),
            label=f"{label}, {year}",
        )
    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].set_xticks(range(len(order)), order, rotation=45, ha="right")
    axes[0].set_ylabel("Capture error, predicted - measured (p.p.)")
    axes[0].set_xlabel("Included one-bed NCCC case")
    axes[0].set_title("Case-wise capture error", pad=8)
    axes[0].grid(False)

    axes[1].bar(
        summary["thermo_label"],
        summary["median_runtime_s"],
        color=[colors.get(label) for label in summary["thermo_label"]],
        width=0.55,
    )
    axes[1].set_ylabel("Median runtime (s)")
    axes[1].set_title("Median runtime for included rows", pad=8)
    axes[1].grid(False)

    axes[0].legend(
        loc="upper center",
        ncol=2,
        fontsize=7.0,
        frameon=False,
    )
    fig.savefig(FIGURES / "nccc_one_bed_thermo_benchmark.pdf", bbox_inches="tight")
    fig.savefig(
        FIGURES / "nccc_one_bed_thermo_benchmark.png", bbox_inches="tight", dpi=220
    )
    fig.savefig(
        LATEX_FIGURES / "nccc-one-bed-thermo-benchmark.pdf", bbox_inches="tight"
    )
    plt.close(fig)


def _case_sort_key(case_id: str) -> tuple[str, int]:
    if case_id.startswith("K"):
        return ("K", int(case_id[1:]))
    return (case_id[-1], int(case_id[:-1]))


if __name__ == "__main__":
    raise SystemExit(main())
