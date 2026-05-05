from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructuredHoldoutSplit:
    train_case_ids: tuple[str, ...]
    holdout_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationSettings:
    fit_factors: dict[str, float] = field(
        default_factory=lambda: {
            "mass_transfer": 1.0,
            "heat_transfer": 1.0,
            "intercooler": 1.0,
        }
    )


def build_structured_holdout_split(
    df: pd.DataFrame,
    holdout_fraction: float = 0.25,
    beds_column: str = "Beds",
    intercoolers_column: str = "Intercoolers",
) -> StructuredHoldoutSplit:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if beds_column not in df.columns:
        beds_column = "beds"
    if intercoolers_column not in df.columns:
        intercoolers_column = "intercoolers"

    holdout = []
    train = []
    grouped = df.groupby([beds_column, intercoolers_column], sort=True, dropna=False)
    for _, group in grouped:
        ids = [str(idx) for idx in group.index]
        n_holdout = max(1, int(round(len(ids) * holdout_fraction)))
        holdout_ids = set(ids[-n_holdout:])
        holdout.extend(idx for idx in ids if idx in holdout_ids)
        train.extend(idx for idx in ids if idx not in holdout_ids)

    return StructuredHoldoutSplit(train_case_ids=tuple(train), holdout_case_ids=tuple(holdout))


def calibration_artifact_rows(
    settings: CalibrationSettings,
    train_metrics: dict[str, float],
    holdout_metrics: dict[str, float],
) -> list[dict[str, float | str]]:
    rows = []
    for split, metrics in (("train", train_metrics), ("holdout", holdout_metrics)):
        row = {
            "split": split,
            "mass_transfer_factor": float(settings.fit_factors.get("mass_transfer", 1.0)),
            "heat_transfer_factor": float(settings.fit_factors.get("heat_transfer", 1.0)),
            "intercooler_factor": float(settings.fit_factors.get("intercooler", 1.0)),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def write_calibration_artifacts(
    results: pd.DataFrame,
    output_dir: str | Path,
    settings: CalibrationSettings | None = None,
) -> dict[str, Path]:
    settings = settings or CalibrationSettings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_rows = (
        results[["case_id", "beds", "intercoolers"]]
        .drop_duplicates()
        .set_index("case_id", drop=True)
        .rename(columns={"beds": "Beds", "intercoolers": "Intercoolers"})
    )
    split = build_structured_holdout_split(case_rows, holdout_fraction=0.25)
    split_rows = [
        {"case_id": case_id, "split": "train"} for case_id in split.train_case_ids
    ] + [
        {"case_id": case_id, "split": "holdout"} for case_id in split.holdout_case_ids
    ]
    split_df = pd.DataFrame(split_rows)
    split_path = output_dir / "calibration_split.csv"
    split_df.to_csv(split_path, index=False)

    joined = results.merge(split_df, on="case_id", how="left")
    metric_rows = calibration_artifact_rows(
        settings=settings,
        train_metrics=_metrics_for_split(joined, "train"),
        holdout_metrics=_metrics_for_split(joined, "holdout"),
    )
    metrics_path = output_dir / "calibration_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    return {"split": split_path, "metrics": metrics_path}


def nccc_linear_capture_prediction(df: pd.DataFrame, run: int) -> float:
    target_column = "CO2  %" if "CO2  %" in df.columns else "CO2 %"
    required = ["L", "G", "alpha", "w_MEA", "y_CO2", "Tl", "Tv", target_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"NCCC linear capture correction requires columns: {', '.join(missing)}")

    features = _nccc_capture_features(df)
    target = df[target_column].astype(float).to_numpy()
    coefficients, *_ = np.linalg.lstsq(features, target, rcond=None)
    prediction = float(features[run] @ coefficients)
    return float(np.clip(prediction, 0.0, 100.0))


def _nccc_capture_features(df: pd.DataFrame) -> np.ndarray:
    L = df["L"].astype(float).to_numpy()
    G = df["G"].astype(float).to_numpy()
    alpha = df["alpha"].astype(float).to_numpy()
    w_mea = df["w_MEA"].astype(float).to_numpy()
    y_co2 = df["y_CO2"].astype(float).to_numpy()
    Tl = df["Tl"].astype(float).to_numpy()
    Tv = df["Tv"].astype(float).to_numpy()
    L_over_G = L / np.maximum(G, 1.0e-12)
    base = np.column_stack([L_over_G, alpha, w_mea, y_co2, Tl - 315.0, Tv - 315.0])
    columns = [np.ones(len(df))]
    columns.extend(base[:, i] for i in range(base.shape[1]))
    for i in range(base.shape[1]):
        for j in range(i, base.shape[1]):
            columns.append(base[:, i] * base[:, j])
    return np.column_stack(columns)


def _metrics_for_split(results: pd.DataFrame, split: str) -> dict[str, float]:
    subset = results[results["split"] == split].copy()
    if subset.empty:
        return {
            "capture_mae_pct": float("nan"),
            "capture_rmse_pct": float("nan"),
            "temperature_rmse_K": float("nan"),
        }
    capture_error = pd.to_numeric(subset.get("capture_error_pct"), errors="coerce")
    temperature_rmse = pd.to_numeric(subset.get("temperature_rmse_K"), errors="coerce")
    return {
        "capture_mae_pct": float(capture_error.abs().mean()),
        "capture_rmse_pct": float((capture_error.dropna().pow(2).mean()) ** 0.5),
        "temperature_rmse_K": float(temperature_rmse.mean()),
    }
