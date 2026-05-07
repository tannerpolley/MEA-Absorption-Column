from __future__ import annotations

import argparse
import json
from pathlib import Path

from mea_absorption_column.benchmark import BenchmarkSettings, run_benchmark


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = _load_spec(args)
    case_source = spec["case_source"]
    case_id = str(spec["case_id"])
    settings = BenchmarkSettings(
        methods=(spec.get("method", "scipy-bvp"),),
        thermo_models=(spec.get("thermo_model", "ideal_henry"),),
        output_dir=Path(spec.get("output_dir", "analyses/nccc_validation/results/runs/manual_case_profiles")),
        c_case_limit=0 if case_source != "C_cases_data" else None,
        nccc_case_limit=0 if case_source != "NCCC_Data" else None,
        srp_case_limit=0 if case_source != "SRP_method_cases" else None,
        c_case_ids=(case_id,) if case_source == "C_cases_data" else None,
        nccc_case_ids=(case_id,) if case_source == "NCCC_Data" else None,
        srp_case_ids=(case_id,) if case_source == "SRP_method_cases" else None,
        staged_beds=spec.get("staged_beds", "auto"),
        solver_settings=spec.get("solver_settings") or None,
        profile_csvs=True,
        profile_pngs=bool(spec.get("profile_pngs", True)),
    )
    results = run_benchmark(settings)
    print(results.to_string(index=False))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one NCCC validation case and export dense profile CSVs."
    )
    parser.add_argument("--spec", type=Path, help="JSON run spec written beside a profile export.")
    parser.add_argument("--case-source", choices=["C_cases_data", "NCCC_Data", "SRP_method_cases"], default="C_cases_data")
    parser.add_argument("--case-id", default="3C")
    parser.add_argument("--method", default="scipy-bvp")
    parser.add_argument("--thermo-model", default="ideal_henry")
    parser.add_argument("--output-dir", default="analyses/nccc_validation/results/runs/manual_case_profiles")
    parser.add_argument("--staged-beds", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--no-profile-png", action="store_true")
    return parser.parse_args(argv)


def _load_spec(args):
    if args.spec:
        return json.loads(args.spec.read_text(encoding="utf-8"))
    staged_beds = args.staged_beds
    if staged_beds == "true":
        staged_beds = True
    elif staged_beds == "false":
        staged_beds = False
    return {
        "case_source": args.case_source,
        "case_id": args.case_id,
        "method": args.method,
        "thermo_model": args.thermo_model,
        "output_dir": args.output_dir,
        "staged_beds": staged_beds,
        "profile_pngs": not args.no_profile_png,
    }


if __name__ == "__main__":
    raise SystemExit(main())
