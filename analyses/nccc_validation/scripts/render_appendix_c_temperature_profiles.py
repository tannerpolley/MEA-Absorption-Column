from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses" / "nccc_validation"
INPUT_DIR = ANALYSIS / "data" / "input"
TABLE_DIR = ANALYSIS / "results" / "final" / "tables"
FIGURE_DIR = ANALYSIS / "results" / "final" / "figures"
PAGE_DIR = FIGURE_DIR / "appendix_c_temperature_profile_pages"
CASE_FIGURE_DIR = FIGURE_DIR / "appendix_c_temperature_profile_cases"

C_PROFILE_RUN_DIR = ANALYSIS / "results" / "runs" / "clean_profile_csvs_c_cases"
C_PROFILE_RESULTS = C_PROFILE_RUN_DIR / "benchmark_results.csv"
K_PROFILE_RUN_DIR = ANALYSIS / "results" / "runs" / "clean_profile_csvs_k_cases_fast"
K_PROFILE_RESULTS = K_PROFILE_RUN_DIR / "benchmark_results.csv"


@dataclass(frozen=True)
class PageSpec:
    title: str
    cases: tuple[str, ...]


THREE_BED_X = (0.07, 0.13, 0.20, 0.27, 0.47, 0.53, 0.60, 0.73, 0.80, 0.87, 0.93, 1.00)
ONE_BED_X = (0.20, 0.40, 0.60, 0.80, 1.00)
TWO_BED_X = (0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90, 1.00)


APPENDIX_C_TEMPERATURE_C: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "1A": (THREE_BED_X, (43.9, 46.8, 49.1, 49.2, 54.4, 60.0, 54.5, 58.2, 58.2, 54.6, 51.5, 51.6)),
    "2A": (THREE_BED_X, (60.9, 60.5, 54.7, 63.3, 48.8, 55.3, 48.4, 45.2, 48.0, 47.9, 48.3, 50.5)),
    "3A": (THREE_BED_X, (68.8, 63.1, 51.8, 53.4, 46.9, 46.5, 46.3, 45.2, 45.4, 45.1, 45.5, 46.5)),
    "4A": (THREE_BED_X, (61.3, 56.2, 44.8, 49.9, 44.1, 43.0, 42.5, 45.0, 45.3, 45.2, 45.6, 46.5)),
    "5A": (THREE_BED_X, (59.9, 53.0, 43.6, 47.0, 45.5, 44.9, 44.6, 47.2, 47.5, 47.4, 47.9, 48.9)),
    "6A": (THREE_BED_X, (57.4, 53.2, 39.4, 47.8, 36.2, 39.0, 38.9, 40.0, 42.6, 42.6, 43.5, 47.4)),
    "7A": (THREE_BED_X, (66.2, 60.9, 49.4, 53.7, 48.5, 46.1, 45.9, 44.4, 44.9, 45.2, 45.5, 46.4)),
    "8A": (THREE_BED_X, (62.0, 54.5, 44.5, 48.4, 44.5, 43.4, 42.6, 44.4, 44.8, 44.5, 45.0, 45.9)),
    "9A": (THREE_BED_X, (66.1, 58.2, 43.8, 49.4, 42.7, 42.4, 42.2, 44.3, 44.9, 44.6, 45.1, 46.3)),
    "10A": (THREE_BED_X, (63.5, 60.4, 56.6, 62.9, 53.0, 51.9, 49.8, 45.1, 45.8, 45.9, 46.0, 46.4)),
    "11A": (THREE_BED_X, (60.7, 53.2, 42.2, 46.2, 43.4, 42.7, 42.4, 44.7, 45.0, 44.9, 45.2, 46.0)),
    "12A": (THREE_BED_X, (67.1, 60.5, 47.1, 53.1, 43.0, 45.0, 43.7, 45.1, 46.3, 46.8, 47.2, 49.9)),
    "13A": (THREE_BED_X, (66.8, 65.2, 57.2, 63.3, 51.5, 51.5, 48.2, 44.2, 45.8, 46.3, 46.5, 48.5)),
    "14A": (THREE_BED_X, (61.4, 59.5, 51.1, 58.7, 48.7, 47.3, 46.2, 44.2, 45.0, 45.0, 45.2, 46.0)),
    "15A": (THREE_BED_X, (63.2, 60.3, 55.9, 62.5, 51.9, 50.9, 49.1, 45.1, 45.7, 45.7, 45.7, 46.2)),
    "1B": (THREE_BED_X, (49.0, 50.9, 53.2, 53.2, 61.4, 59.4, 53.6, 49.7, 52.2, 51.7, 49.7, 48.9)),
    "2B": (THREE_BED_X, (46.3, 48.0, 52.4, 50.9, 58.2, 61.4, 58.3, 51.5, 53.7, 53.5, 52.5, 53.1)),
    "3B": (THREE_BED_X, (46.8, 54.7, 57.5, 60.4, 60.5, 60.2, 56.3, 49.7, 50.5, 50.7, 50.4, 50.4)),
    "1C": (ONE_BED_X, (73.5, 73.6, 69.4, 60.3, 55.2)),
    "2C": (ONE_BED_X, (73.9, 72.7, 67.7, 58.6, 54.8)),
    "3C": (ONE_BED_X, (74.9, 73.7, 68.7, 60.8, 57.0)),
    "4C": (ONE_BED_X, (70.5, 71.0, 68.7, 58.7, 59.1)),
    "5C": (ONE_BED_X, (64.9, 66.4, 64.3, 55.5, 57.6)),
    "6C": (ONE_BED_X, (62.6, 63.4, 58.0, 52.8, 52.4)),
    "7C": (ONE_BED_X, (51.2, 58.2, 54.7, 51.4, 54.9)),
    "1D": (TWO_BED_X, (70.3, 69.0, 68.8, 63.5, 58.3, 57.6, 55.8, 51.8)),
    "2D": (TWO_BED_X, (70.5, 65.1, 61.3, 57.2, 52.9, 50.5, 49.8, 48.4)),
    "3D": (TWO_BED_X, (76.5, 75.9, 74.6, 73.0, 69.8, 64.2, 59.8, 55.6)),
    "4D": (TWO_BED_X, (72.6, 72.4, 71.8, 70.2, 65.3, 59.0, 57.0, 51.2)),
}


CASE_META: dict[str, tuple[int, int, str]] = {
    **{case: (3, 2, "three-bed/two-intercooler Appendix C profile") for case in [f"{i}A" for i in range(1, 16)]},
    **{case: (3, 0, "three-bed/no-intercooler model group") for case in ["1B", "2B", "3B"]},
    **{case: (1, 0, "one-bed/no-intercooler Appendix C profile") for case in [f"{i}C" for i in range(1, 8)]},
    "1D": (2, 0, "two-bed/no-intercooler model group"),
    "2D": (2, 0, "two-bed/no-intercooler model group"),
    "3D": (2, 1, "two-bed/one-intercooler Appendix C profile"),
    "4D": (2, 1, "two-bed/one-intercooler Appendix C profile"),
}


A_K_ROWS = [f"K{i}" for i in range(1, 13)] + ["K14", "K15", "K16"]
K_TO_APPENDIX_STYLE = {
    **{legacy: f"{idx}A" for idx, legacy in enumerate(A_K_ROWS, start=1)},
    "K13": "1B",
    "K17": "1D",
    "K21": "2D",
    "K22": "3D",
    "K23": "4D",
}


PAGES = (
    PageSpec("True model profiles: 3 beds, 2 intercoolers, cases 1A-6A", ("1A", "2A", "3A", "4A", "5A", "6A")),
    PageSpec("True model profiles: 3 beds, 2 intercoolers, cases 7A-12A", ("7A", "8A", "9A", "10A", "11A", "12A")),
    PageSpec("True model profiles: 3 beds, 2 intercoolers, cases 13A-15A", ("13A", "14A", "15A")),
    PageSpec("True model profiles: 3 beds, 0 intercoolers, B cases", ("1B", "2B", "3B")),
    PageSpec("True model profiles: 1 bed, 0 intercoolers, cases 1C-6C", ("1C", "2C", "3C", "4C", "5C", "6C")),
    PageSpec("True model profiles: 1 bed, 0 intercoolers, case 7C", ("7C",)),
    PageSpec("True model profiles: 2 beds, 0 intercoolers, D cases", ("1D", "2D")),
    PageSpec("True model profiles: 2 beds, 1 intercooler, D cases", ("3D", "4D")),
)


def measured_profile_table() -> pd.DataFrame:
    rows = []
    for case_id, (positions, temperatures_c) in APPENDIX_C_TEMPERATURE_C.items():
        beds, intercoolers, group = CASE_META[case_id]
        for position, temperature_c in zip(positions, temperatures_c):
            rows.append(
                {
                    "case_id": case_id,
                    "relative_position_top_to_bottom": position,
                    "temperature_C": temperature_c,
                    "temperature_K": temperature_c + 273.15,
                    "beds": beds,
                    "intercoolers": intercoolers,
                    "group": group,
                    "source": "Morgan et al. 2020 Appendix C absorber temperature tables",
                }
            )
    return pd.DataFrame(rows)


def _load_best_c_case_model_profiles() -> dict[str, dict[str, object]]:
    if not C_PROFILE_RESULTS.exists():
        return {}
    results = pd.read_csv(C_PROFILE_RESULTS)
    results["success_bool"] = results["success"].astype(str).str.lower().eq("true")
    for column in ["temperature_rmse_K", "runtime_s", "capture_error_pct"]:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    candidates = results[
        results["case_id"].astype(str).str.endswith("C")
        & results["success_bool"]
        & results["temperature_rmse_K"].notna()
        & results["runtime_s"].notna()
        & (results["runtime_s"] <= 60.0)
    ].copy()

    profiles: dict[str, dict[str, object]] = {}
    for case_id, group in candidates.groupby("case_id"):
        row = group.sort_values(["temperature_rmse_K", "runtime_s"]).iloc[0]
        profile = _read_temperature_profile(ROOT / str(row["profile_csv_dir"]) / "T.csv")
        if profile is None:
            continue
        profiles[str(case_id)] = {
            **profile,
            "model_source": "C_cases_data",
            "source_row": str(case_id),
            "thermo_model": row["thermo_model"],
            "success": bool(row["success_bool"]),
            "runtime_s": float(row["runtime_s"]),
            "temperature_rmse_K": float(row["temperature_rmse_K"]),
            "capture_error_pct": float(row["capture_error_pct"]),
            "message": row.get("message", ""),
            "profile_csv": str((ROOT / str(row["profile_csv_dir"]) / "T.csv").relative_to(ROOT)).replace("\\", "/"),
        }
    return profiles


def _load_k_model_profiles() -> dict[str, dict[str, object]]:
    if not K_PROFILE_RESULTS.exists():
        return {}
    results = pd.read_csv(K_PROFILE_RESULTS)
    results["success_bool"] = results["success"].astype(str).str.lower().eq("true")
    for column in ["runtime_s", "capture_error_pct", "capture_pct"]:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    profiles: dict[str, dict[str, object]] = {}
    for _, row in results.iterrows():
        legacy_id = str(row["case_id"])
        case_id = K_TO_APPENDIX_STYLE.get(legacy_id)
        if not case_id:
            continue
        profile = _read_temperature_profile(ROOT / str(row["profile_csv_dir"]) / "T.csv")
        if profile is None:
            profiles[case_id] = {
                "model_status": "no_profile_csv",
                "model_source": "NCCC_Data_mole_based",
                "source_row": legacy_id,
                "success": bool(row["success_bool"]),
                "runtime_s": float(row["runtime_s"]) if pd.notna(row["runtime_s"]) else np.nan,
                "capture_error_pct": float(row["capture_error_pct"]) if pd.notna(row["capture_error_pct"]) else np.nan,
                "message": row.get("message", ""),
            }
            continue
        profiles[case_id] = {
            **profile,
            "model_status": "profile_exported",
            "model_source": "NCCC_Data_mole_based",
            "source_row": legacy_id,
            "thermo_model": row.get("thermo_model", "ideal_henry"),
            "success": bool(row["success_bool"]),
            "runtime_s": float(row["runtime_s"]) if pd.notna(row["runtime_s"]) else np.nan,
            "temperature_rmse_K": np.nan,
            "capture_error_pct": float(row["capture_error_pct"]) if pd.notna(row["capture_error_pct"]) else np.nan,
            "message": row.get("message", ""),
            "profile_csv": str((ROOT / str(row["profile_csv_dir"]) / "T.csv").relative_to(ROOT)).replace("\\", "/"),
        }
    return profiles


def _read_temperature_profile(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    profile = pd.read_csv(path)
    if not {"Position", "Tl"}.issubset(profile.columns):
        return None
    x = pd.to_numeric(profile["Position"], errors="coerce").to_numpy(dtype=float)
    liquid = pd.to_numeric(profile["Tl"], errors="coerce").to_numpy(dtype=float) - 273.15
    vapor = pd.to_numeric(profile["Tv"], errors="coerce").to_numpy(dtype=float) - 273.15 if "Tv" in profile else np.full_like(liquid, np.nan)
    ok = np.isfinite(x) & np.isfinite(liquid)
    if ok.sum() < 2:
        return None
    order = np.argsort(x[ok])
    vapor_ok = vapor[ok][order] if np.isfinite(vapor[ok]).any() else np.full(ok.sum(), np.nan)
    return {
        "x": x[ok][order],
        "liquid_temperature_C": liquid[ok][order],
        "vapor_temperature_C": vapor_ok,
        "model_status": "profile_exported",
    }


def _plot_case(ax: plt.Axes, case_id: str, model_profiles: dict[str, dict[str, object]]) -> dict[str, object]:
    positions, temperatures_c = APPENDIX_C_TEMPERATURE_C[case_id]
    measured_x = np.asarray(positions, dtype=float)
    measured_y = np.asarray(temperatures_c, dtype=float)
    beds, intercoolers, group = CASE_META[case_id]
    model = model_profiles.get(case_id)

    if model and model.get("model_status") == "profile_exported":
        is_success = bool(model.get("success"))
        line_prefix = "model" if is_success else "broken model"
        ax.plot(
            model["x"],
            model["liquid_temperature_C"],
            color="#2563EB",
            linewidth=1.7,
            label=f"{line_prefix} liquid",
        )
        vapor = np.asarray(model["vapor_temperature_C"], dtype=float)
        if np.isfinite(vapor).any():
            ax.plot(model["x"], vapor, color="#D97706", linewidth=1.2, linestyle="--", label=f"{line_prefix} vapor")
        success = "accepted" if is_success else "BROKEN model output"
        runtime = model.get("runtime_s", np.nan)
        ax.text(
            0.02,
            0.03,
            f"{success}; {runtime:.1f} s" if np.isfinite(runtime) else success,
            transform=ax.transAxes,
            fontsize=6.8,
            color="#374151" if is_success else "#9A3412",
            va="bottom",
        )
    else:
        ax.text(
            0.5,
            0.52,
            "BROKEN/MISSING\nmodel output",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#9A3412",
        )

    ax.scatter(
        measured_x,
        measured_y,
        s=24,
        marker="o",
        facecolor="white",
        edgecolor="#111827",
        linewidth=0.9,
        label="measured NCCC",
    )
    if beds > 1:
        for boundary in np.linspace(0, 1, beds + 1)[1:-1]:
            ax.axvline(boundary, color="#9CA3AF", linewidth=0.7, linestyle="--", alpha=0.8)

    y_values = list(measured_y)
    if model and model.get("model_status") == "profile_exported":
        y_values += list(np.asarray(model["liquid_temperature_C"], dtype=float))
        vapor = np.asarray(model["vapor_temperature_C"], dtype=float)
        y_values += list(vapor[np.isfinite(vapor)])
    ymin = max(30.0, float(np.nanmin(y_values)) - 3.0)
    ymax = min(95.0, float(np.nanmax(y_values)) + 3.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Case {case_id}: {beds} bed{'s' if beds != 1 else ''}, {intercoolers} intercooler{'s' if intercoolers != 1 else ''}", fontsize=9)
    ax.set_xlabel("Relative absorber position, top to bottom")
    ax.set_ylabel("Temperature (C)")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="best", fontsize=6.8, frameon=False)

    return {
        "case_id": case_id,
        "beds": beds,
        "intercoolers": intercoolers,
        "group": group,
        "measured_points": len(measured_x),
        "model_status": model.get("model_status", "no_profile_csv") if model else "no_profile_csv",
        "model_quality": (
            "accepted"
            if model and model.get("model_status") == "profile_exported" and bool(model.get("success"))
            else "broken_diagnostic"
            if model and model.get("model_status") == "profile_exported"
            else "missing_model_output"
        ),
        "model_source": model.get("model_source", "") if model else "",
        "source_row": model.get("source_row", "") if model else "",
        "thermo_model": model.get("thermo_model", "") if model else "",
        "success": model.get("success", "") if model else "",
        "runtime_s": model.get("runtime_s", np.nan) if model else np.nan,
        "temperature_rmse_K": model.get("temperature_rmse_K", np.nan) if model else np.nan,
        "capture_error_pct": model.get("capture_error_pct", np.nan) if model else np.nan,
        "profile_csv": model.get("profile_csv", "") if model else "",
        "message": model.get("message", "") if model else "No exported true model output profile found for this case.",
    }


def _write_case_name_crosswalk() -> None:
    nccc_path = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_Data.csv"
    if not nccc_path.exists():
        return
    k_cases = pd.read_csv(nccc_path)
    rows = []
    for _, row in k_cases.iterrows():
        legacy_id = str(row["Runs"])
        mapped = K_TO_APPENDIX_STYLE.get(legacy_id, "")
        rows.append(
            {
                "legacy_case_id": legacy_id,
                "appendix_style_case_id": mapped,
                "beds": row["Beds"],
                "intercoolers": row["Intercoolers"],
                "capture_pct": row["CO2  %"],
                "mapping_basis": (
                    "assigned by bed/intercooler group and legacy row order for plotting"
                    if mapped
                    else "not assigned; authoritative C-case profiles use C_cases_data.csv"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "nccc_legacy_k_case_name_crosswalk.csv", index=False)


def render() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    CASE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    measured = measured_profile_table()
    measured.to_csv(INPUT_DIR / "morgan2020_appendix_c_absorber_temperature_profiles.csv", index=False)

    model_profiles = {}
    model_profiles.update(_load_k_model_profiles())
    model_profiles.update(_load_best_c_case_model_profiles())

    index_rows: list[dict[str, object]] = []
    for case_id in APPENDIX_C_TEMPERATURE_C:
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        row = _plot_case(ax, case_id, model_profiles)
        fig.suptitle(
            f"Case {case_id}: true model output and measured NCCC data",
            fontsize=11,
            y=0.985,
        )
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
        case_png = CASE_FIGURE_DIR / f"{case_id}_temperature_profile.png"
        fig.savefig(case_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        row["case_plot_png"] = str(case_png.relative_to(ROOT)).replace("\\", "/")
        index_rows.append(row)

    pdf_path = FIGURE_DIR / "appendix_c_temperature_profiles.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_number, page in enumerate(PAGES, start=1):
            n_cases = len(page.cases)
            ncols = min(3, n_cases)
            nrows = int(np.ceil(n_cases / ncols))
            fig_height = 3.9 * nrows + 1.0
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, fig_height), sharex=False, sharey=False)
            axes_flat = np.atleast_1d(axes).ravel()
            for ax, case_id in zip(axes_flat, page.cases):
                _plot_case(ax, case_id, model_profiles)
            for ax in axes_flat[len(page.cases):]:
                ax.axis("off")
            fig.suptitle(page.title, fontsize=13, y=0.985)
            fig.text(
                0.5,
                0.012,
                "Measured points are unconnected. Model lines are exported true model profiles only; broken or missing model outputs are labeled.",
                ha="center",
                fontsize=8.4,
            )
            fig.tight_layout(rect=(0.035, 0.045, 0.985, 0.955), h_pad=1.2, w_pad=1.0)
            pdf.savefig(fig)
            fig.savefig(PAGE_DIR / f"page_{page_number:02d}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    pd.DataFrame(index_rows).to_csv(TABLE_DIR / "appendix_c_temperature_profile_index.csv", index=False)
    _write_case_name_crosswalk()

    print(f"Wrote {pdf_path}")
    print(f"Wrote {TABLE_DIR / 'appendix_c_temperature_profile_index.csv'}")
    print(f"Wrote {PAGE_DIR}")


if __name__ == "__main__":
    render()
