from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses" / "nccc_validation"
INPUT = ANALYSIS / "data" / "input"
FINAL = ANALYSIS / "results" / "final"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"
RUNS = ANALYSIS / "results" / "runs"

HARD_RESET_RUN = RUNS / "intercooler_hard_reset_k3"
DISTRIBUTED_RUN = RUNS / "intercooler_distributed_k3_s025"
PUMPAROUND_RUN = RUNS / "intercooler_pumparound_temperature_k3"
HARD_RESET_T = HARD_RESET_RUN / "profiles" / "NCCC_Data" / "K3" / "scipy-bvp" / "ideal_henry" / "T.csv"
DISTRIBUTED_T = DISTRIBUTED_RUN / "profiles" / "NCCC_Data" / "K3" / "scipy-bvp" / "ideal_henry" / "T.csv"
PUMPAROUND_T = PUMPAROUND_RUN / "profiles" / "NCCC_Data" / "K3" / "scipy-bvp" / "ideal_henry" / "T.csv"
GUIDELINE = INPUT / "case3a_supplied_image_model_guideline.csv"
MORGAN_POINTS = INPUT / "morgan2020_appendix_c_absorber_temperature_profiles.csv"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    hard = _read_model_profile(HARD_RESET_T, "hard liquid-temperature reset", HARD_RESET_RUN)
    distributed = _read_model_profile(DISTRIBUTED_T, "distributed liquid-cooling relaxation", DISTRIBUTED_RUN)
    pumparound = _read_model_profile(PUMPAROUND_T, "pumparound temperature-approach", PUMPAROUND_RUN)
    guideline = pd.read_csv(GUIDELINE)
    measured = pd.read_csv(MORGAN_POINTS)
    measured_3a = measured[measured["case_id"].astype(str).eq("3A")].copy()

    metrics = pd.DataFrame([_smoothness_metrics(hard), _smoothness_metrics(distributed), _smoothness_metrics(pumparound)])
    metrics.to_csv(TABLES / "intercooler_model_smoothness_case3a.csv", index=False)

    plotted = pd.concat(
        [
            hard.assign(series_type="model"),
            distributed.assign(series_type="model"),
            pumparound.assign(series_type="model"),
            guideline.rename(columns={"relative_position": "Position", "temperature_C": "Tl_C"}).assign(
                model="digitized supplied-image guideline",
                series_type="guideline",
            )[["Position", "Tl_C", "model", "series_type"]],
            measured_3a.rename(columns={"relative_position_top_to_bottom": "Position", "temperature_C": "Tl_C"}).assign(
                model="Morgan Appendix C measured points",
                series_type="measured",
            )[["Position", "Tl_C", "model", "series_type"]],
        ],
        ignore_index=True,
    )
    plotted.to_csv(TABLES / "intercooler_model_comparison_case3a_plotted_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(
        guideline["relative_position"],
        guideline["temperature_C"],
        color="#111827",
        linewidth=1.4,
        label="digitized 3A model guideline",
    )
    ax.plot(hard["Position"], hard["Tl_C"], color="#DC2626", linewidth=1.6, label="current hard reset")
    ax.plot(
        distributed["Position"],
        distributed["Tl_C"],
        color="#2563EB",
        linewidth=1.3,
        linestyle=":",
        label="distributed cooling diagnostic",
    )
    ax.plot(pumparound["Position"], pumparound["Tl_C"], color="#059669", linewidth=1.9, label="pumparound temperature approach")
    ax.scatter(
        measured_3a["relative_position_top_to_bottom"],
        measured_3a["temperature_C"],
        s=28,
        facecolors="white",
        edgecolors="#111827",
        linewidth=0.9,
        label="NCCC measured points",
        zorder=5,
    )
    for boundary in (1.0 / 3.0, 2.0 / 3.0):
        ax.axvline(boundary, color="#9CA3AF", linewidth=0.8, linestyle="--", alpha=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(38.0, 74.0)
    ax.set_xlabel("Relative column position")
    ax.set_ylabel("Liquid temperature (degC)")
    ax.set_title("Case 3A/K3 intercooler profile model comparison", pad=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "intercooler_model_comparison_case3a.png", dpi=220)
    fig.savefig(FIGURES / "intercooler_model_comparison_case3a.pdf")
    plt.close(fig)

    print(f"Wrote {FIGURES / 'intercooler_model_comparison_case3a.png'}")
    print(f"Wrote {TABLES / 'intercooler_model_smoothness_case3a.csv'}")
    return 0


def _read_model_profile(path: Path, model: str, run_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if not {"Position", "Tl"}.issubset(data.columns):
        raise ValueError(f"{path} must contain Position and Tl")
    out = pd.DataFrame(
        {
            "Position": pd.to_numeric(data["Position"], errors="coerce"),
            "Tl_C": pd.to_numeric(data["Tl"], errors="coerce") - 273.15,
            "model": model,
            "success": _run_success(run_dir),
        }
    )
    out = out[np.isfinite(out["Position"]) & np.isfinite(out["Tl_C"])].sort_values("Position")
    return out.reset_index(drop=True)


def _smoothness_metrics(profile: pd.DataFrame) -> dict[str, float | str]:
    x = profile["Position"].to_numpy(dtype=float)
    y = profile["Tl_C"].to_numpy(dtype=float)
    rows: dict[str, float | str] = {
        "model": str(profile["model"].iloc[0]),
        "success": bool(profile["success"].iloc[0]) if "success" in profile else "",
    }
    spike_depths = []
    for boundary in (1.0 / 3.0, 2.0 / 3.0):
        left = _interp(x, y, boundary - 0.025)
        center = _interp(x, y, boundary)
        right = _interp(x, y, boundary + 0.025)
        local_mean = 0.5 * (left + right)
        spike_depths.append(max(0.0, local_mean - center))
    rows["max_intercooler_dip_C"] = float(max(spike_depths))
    rows["mean_intercooler_dip_C"] = float(np.mean(spike_depths))
    rows["max_adjacent_slope_jump_C_per_position"] = float(_max_slope_jump(x, y))
    section_scores = [_quadratic_r2(x, y, left, right) for left, right in [(0.02, 0.31), (0.36, 0.64), (0.70, 0.98)]]
    rows["min_section_quadratic_r2"] = float(np.nanmin(section_scores))
    rows["mean_section_quadratic_r2"] = float(np.nanmean(section_scores))
    return rows


def _run_success(run_dir: Path) -> bool:
    path = run_dir / "benchmark_results.csv"
    if not path.exists():
        return False
    data = pd.read_csv(path)
    if data.empty or "success" not in data.columns:
        return False
    return str(data.iloc[0]["success"]).lower() == "true"


def _interp(x: np.ndarray, y: np.ndarray, value: float) -> float:
    return float(np.interp(value, x, y))


def _max_slope_jump(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4:
        return float("nan")
    slopes = np.diff(y) / np.maximum(np.diff(x), 1.0e-12)
    return float(np.nanmax(np.abs(np.diff(slopes))))


def _quadratic_r2(x: np.ndarray, y: np.ndarray, left: float, right: float) -> float:
    mask = (x >= left) & (x <= right)
    if mask.sum() < 4:
        return float("nan")
    xs = x[mask]
    ys = y[mask]
    coeffs = np.polyfit(xs, ys, deg=2)
    pred = np.polyval(coeffs, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


if __name__ == "__main__":
    raise SystemExit(main())
