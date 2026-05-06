from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable
import argparse
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ContinuationStep:
    name: str
    thermo_model: str
    staged_beds: str | bool
    solver_settings: dict = field(default_factory=dict)
    intercooler_settings: dict | None = None
    required: bool = True


@dataclass(frozen=True)
class ContinuationResult:
    success: bool
    failed_stage: str
    rows: tuple[dict, ...]

    @property
    def last(self) -> dict:
        return self.rows[-1] if self.rows else {}


def run_continuation_ladder(
    steps: Iterable[ContinuationStep],
    runner: Callable[[ContinuationStep], dict],
) -> ContinuationResult:
    rows = []
    for step in steps:
        row = dict(runner(step))
        row.setdefault("continuation_stage", step.name)
        rows.append(row)
        if not row.get("success", False) and step.required:
            return ContinuationResult(success=False, failed_stage=step.name, rows=tuple(rows))
    return ContinuationResult(success=True, failed_stage="", rows=tuple(rows))


def default_absorber_continuation_steps(include_epcsaft: bool = True) -> tuple[ContinuationStep, ...]:
    common = {
        "mesh_points": 7,
        "tol": 2.0,
        "bc_tol": 0.05,
        "max_nodes": 80,
        "success_capture_error_max_pct": 8.0,
        "co2_flux_mode": "absorption_only",
    }
    steps = [
        ContinuationStep(
            name="one_bed_henry",
            thermo_model="ideal_henry",
            staged_beds=False,
            required=False,
            solver_settings={
                **common,
                "transform_mode": "positive_flow_pressure",
                "continuation_stage": "one_bed_henry",
                "continuation_path": "one_bed_henry",
            },
        ),
        ContinuationStep(
            name="staged_henry_no_intercooler_reset",
            thermo_model="ideal_henry",
            staged_beds="auto",
            required=False,
            solver_settings={
                **common,
                "transform_mode": "case_bounded_flow_pressure",
                "intercooler_strength": 0.0,
                "continuation_stage": "staged_henry_no_intercooler_reset",
                "continuation_path": "one_bed_henry->staged_henry_no_intercooler_reset",
            },
        ),
        ContinuationStep(
            name="staged_henry_weak_intercooler_reset",
            thermo_model="ideal_henry",
            staged_beds="auto",
            required=False,
            solver_settings={
                **common,
                "transform_mode": "case_bounded_flow_pressure",
                "intercooler_strength": 0.25,
                "continuation_stage": "staged_henry_weak_intercooler_reset",
                "continuation_path": "one_bed_henry->staged_henry_no_intercooler_reset->staged_henry_weak_intercooler_reset",
            },
        ),
        ContinuationStep(
            name="staged_henry_full_intercooler_reset",
            thermo_model="ideal_henry",
            staged_beds="auto",
            solver_settings={
                **common,
                "transform_mode": "case_bounded_flow_pressure",
                "intercooler_strength": 1.0,
                "continuation_stage": "staged_henry_full_intercooler_reset",
                "continuation_path": (
                    "one_bed_henry->staged_henry_no_intercooler_reset->"
                    "staged_henry_weak_intercooler_reset->staged_henry_full_intercooler_reset"
                ),
            },
        ),
    ]
    if include_epcsaft:
        epcsaft_path = (
            "one_bed_henry->staged_henry_no_intercooler_reset->"
            "staged_henry_weak_intercooler_reset->"
            "staged_henry_full_intercooler_reset"
        )
        for blend in (0.25, 0.5, 0.75):
            steps.append(
                ContinuationStep(
                    name=f"epcsaft_fugacity_blend_{blend:g}",
                    thermo_model="epcsaft_neutral",
                    staged_beds="auto",
                    required=False,
                    solver_settings={
                        **common,
                        "transform_mode": "case_bounded_flow_pressure",
                        "intercooler_strength": 1.0,
                        "epcsaft_fugacity_blend": blend,
                        "continuation_stage": f"epcsaft_fugacity_blend_{blend:g}",
                        "continuation_path": f"{epcsaft_path}->epcsaft_blend_{blend:g}",
                    },
                )
            )
        steps.append(
            ContinuationStep(
                name="henry_seeded_epcsaft",
                thermo_model="epcsaft_neutral",
                staged_beds="auto",
                solver_settings={
                    **common,
                    "transform_mode": "case_bounded_flow_pressure",
                    "intercooler_strength": 1.0,
                    "epcsaft_fugacity_blend": 1.0,
                    "continuation_stage": "henry_seeded_epcsaft",
                    "continuation_path": f"{epcsaft_path}->epcsaft_blend_0.25->epcsaft_blend_0.5->epcsaft_blend_0.75->epcsaft_neutral",
                },
            )
        )
    return tuple(steps)


def run_absorber_continuation(
    df,
    run: int = 0,
    method: str = "scipy-bvp",
    steps: Iterable[ContinuationStep] | None = None,
    data_type: str = "mole",
    run_model_func=None,
) -> ContinuationResult:
    if run_model_func is None:
        from mea_absorption_column.Run_Model import run_model as run_model_func

    steps = tuple(steps or default_absorber_continuation_steps())

    previous_profile = None

    def runner(step: ContinuationStep):
        nonlocal previous_profile
        solver_settings = dict(step.solver_settings or {})
        solver_settings.setdefault("continuation_stage", step.name)
        solver_settings.setdefault("continuation_path", step.name)
        solver_settings.setdefault("return_internal_profile", True)
        if previous_profile is not None and "initial_guess_scaled" not in solver_settings:
            solver_settings["initial_guess_scaled"] = previous_profile
        row = run_model_func(
            df,
            method=method,
            data_type=data_type,
            run=run,
            show_info=False,
            save_run_results=False,
            plot_temperature=False,
            thermo_model=step.thermo_model,
            solver_settings=solver_settings,
            return_details=True,
            staged_beds=step.staged_beds,
            intercooler_settings=step.intercooler_settings,
        )
        if _row_can_seed_next_stage(row, solver_settings):
            previous_profile = row["_raw_solution_scaled"]
        public_row = dict(row)
        public_row.pop("_raw_solution_scaled", None)
        return public_row

    return run_continuation_ladder(steps, runner=runner)


def _row_can_seed_next_stage(row: dict, solver_settings: dict) -> bool:
    profile = row.get("_raw_solution_scaled")
    if profile is None:
        return False
    if row.get("success"):
        return True
    if not solver_settings.get("seed_failed_capture_close", True):
        return False
    capture_error = row.get("capture_error_pct")
    if capture_error is None:
        return False
    try:
        return abs(float(capture_error)) <= float(solver_settings.get("seed_capture_error_max_pct", 8.0))
    except (TypeError, ValueError):
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the MEA absorber continuation ladder.")
    parser.add_argument("--data", default="src/mea_absorption_column/data/C_cases_data.csv")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--method", default="scipy-bvp")
    parser.add_argument("--no-epcsaft", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = pd.read_csv(Path(args.data), index_col=0)
    result = run_absorber_continuation(
        df=df,
        run=args.run,
        method=args.method,
        steps=default_absorber_continuation_steps(include_epcsaft=not args.no_epcsaft),
    )
    print(pd.DataFrame(result.rows).to_string(index=False))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
