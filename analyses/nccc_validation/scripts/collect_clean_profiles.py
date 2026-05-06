from __future__ import annotations

import argparse
import re
from importlib import resources
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mea_absorption_column.Run_Model import run_model

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
FINAL = ANALYSIS / "results" / "final"
PROFILE_ROOT = FINAL / "profiles"
TABLES = FINAL / "tables"


def _data_path(filename: str):
    return resources.files("mea_absorption_column").joinpath(f"data/{filename}")


def load_cases(source: str) -> pd.DataFrame:
    if source == "c-cases":
        return pd.read_csv(_data_path("C_cases_data.csv"), index_col=0)
    if source == "nccc":
        return pd.read_csv(_data_path("NCCC_Data_mole_based.csv"), index_col=0)
    raise ValueError("source must be c-cases or nccc")


def generate_temperature_profiles(
    source: str,
    methods: tuple[str, ...],
    thermo_models: tuple[str, ...],
    output_dir: Path,
    limit: int | None = None,
    start: int = 0,
    staged_beds: str | bool = "auto",
    data_type: str = "mole",
) -> pd.DataFrame:
    df = load_cases(source)
    stop = len(df) if limit is None else min(len(df), start + limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in range(start, stop):
        case_id = str(df.index[run])
        for method in methods:
            for thermo_model in thermo_models:
                result = run_model(
                    df,
                    method=method,
                    data_type=data_type,
                    run=run,
                    thermo_model=thermo_model,
                    staged_beds=staged_beds,
                    save_run_results=False,
                    plot_temperature=False,
                    return_details=True,
                    solver_settings={"return_profiles": True},
                )
                profile_path = None
                profiles = result.get("_profiles") or {}
                if "T" in profiles:
                    profile_path = output_dir / case_id / thermo_model / _profile_filename(case_id, method, thermo_model)
                    profile_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_temperature_plot(
                        profiles["T"],
                        df=df,
                        run=run,
                        case_id=case_id,
                        method=method,
                        thermo_model=thermo_model,
                        result=result,
                        output_path=profile_path,
                    )
                rows.append(
                    {
                        "case_id": case_id,
                        "method": method,
                        "thermo_model": thermo_model,
                        "success": result.get("success"),
                        "message": result.get("message"),
                        "runtime_s": result.get("runtime_s"),
                        "capture_pct": result.get("capture_pct"),
                        "capture_error_pct": result.get("capture_error_pct"),
                        "temperature_rmse_K": result.get("temperature_rmse_K"),
                        "invalid_state_count": result.get("invalid_state_count"),
                        "guard_penalty_count": result.get("guard_penalty_count"),
                        "profile_png": "" if profile_path is None else profile_path.relative_to(ROOT).as_posix(),
                    }
                )
    summary = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "generated_temperature_profile_index.csv", index=False)
    return summary


def collect_existing_profiles(profile_root: Path = PROFILE_ROOT) -> pd.DataFrame:
    rows = []
    for path in sorted(profile_root.glob("*/*/*.png")):
        case_id = path.parents[1].name
        thermo_model = path.parent.name
        caveat = _profile_caveat(case_id, thermo_model)
        rows.append(
            {
                "case_id": case_id,
                "thermo_model": thermo_model,
                "profile_png": path.relative_to(ROOT).as_posix(),
                "clean_profile": True,
                "caveat": caveat,
            }
        )
    index = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    index.to_csv(TABLES / "clean_temperature_profile_index.csv", index=False)
    return index


def _profile_caveat(case_id: str, thermo_model: str) -> str:
    if case_id == "7C" and thermo_model == "epcsaft_neutral":
        return "converged profile retained as a difficult one-bed ePC-SAFT diagnostic with substantial validation error"
    return "accepted clean validation profile"


def _write_temperature_plot(profile, df, run, case_id, method, thermo_model, result, output_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    if "Tl" in profile:
        ax.plot(profile.index, profile["Tl"], label="Liquid model", linewidth=2.0)
    if "Tv" in profile:
        ax.plot(profile.index, profile["Tv"], label="Vapor model", linewidth=2.0)

    tap_columns = [column for column in df.columns if _is_float(column)]
    if tap_columns:
        taps = df.iloc[run][tap_columns].astype(float)
        ax.scatter([float(column) for column in tap_columns], taps.to_numpy(), label="NCCC liquid taps", s=28, c="black")

    ax.set_xlabel("Normalized column position")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(
        f"{case_id} | {method} | {thermo_model} | "
        f"success={result.get('success')} | capture={result.get('capture_pct'):.2f}%"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _is_float(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _profile_filename(case_id: str, method: str, thermo_model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{case_id}_{method}_{thermo_model}")
    return f"{safe}.png"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate temperature-profile PNGs from solved MEA model cases.")
    parser.add_argument("--collect-existing", action="store_true", help="Index existing final profile PNGs without rerunning simulations.")
    parser.add_argument("--source", choices=["c-cases", "nccc"], default="c-cases")
    parser.add_argument("--methods", nargs="+", default=["scipy-bvp"])
    parser.add_argument("--thermo-models", nargs="+", default=["ideal_henry", "epcsaft_neutral"])
    parser.add_argument("--output-dir", default=str(PROFILE_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--staged-beds", choices=["auto", "true", "false"], default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.collect_existing:
        summary = collect_existing_profiles(Path(args.output_dir))
        print(summary.to_string(index=False))
        return
    staged_beds = args.staged_beds
    if staged_beds == "true":
        staged_beds = True
    elif staged_beds == "false":
        staged_beds = False
    summary = generate_temperature_profiles(
        source=args.source,
        methods=tuple(args.methods),
        thermo_models=tuple(args.thermo_models),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        start=args.start,
        staged_beds=staged_beds,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
