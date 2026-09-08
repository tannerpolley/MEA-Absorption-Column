"""Bounded source-backed one-bed runs; never overwrite manuscript results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

from mea_absorption_column.benchmark import BenchmarkSettings, run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="epcsaft_reactive_nine")
    parser.add_argument("--methods", nargs="+", choices=("scipy-bvp", "single", "finite"), default=["scipy-bvp"])
    parser.add_argument("--cases", nargs="+", default=["3C"])
    parser.add_argument("--mesh", type=int, default=21)
    parser.add_argument("--tol", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = BenchmarkSettings(
        methods=tuple(args.methods), thermo_models=(args.model,), output_dir=args.output,
        c_case_limit=0, nccc_case_ids=tuple(args.cases), srp_case_limit=0,
        nccc_dataset="2017", data_type="mass", staged_beds=False,
        write_artifacts=False, profile_csvs=True, subprocess_timeout_s=args.timeout,
        solver_settings={"mesh_points": args.mesh, "tol": args.tol, "bc_tol": 0.001,
                         "max_nodes": 1000, "thermal_state_mode": "temperature",
                         "vapor_composition_mode": "dry_saturated",
                         "gas_flow_basis": "reported_dry_mass"},
    )
    inputs = [Path("src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv"),
              Path("src/mea_absorption_column/data/NCCC_2017_absorber_temperature_profiles.csv"),
              Path("uv.lock"), Path("pyproject.toml"), Path(__file__),
              Path("integration/epcsaft_contract.json")]
    inputs += sorted(Path("src/mea_absorption_column").rglob("*.py"))
    inputs += sorted(Path("src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle").glob("*.json"))
    metadata = {"command": sys.argv, "platform": platform.platform(),
                "cpu": next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")), "unknown"),
                "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "settings": settings.solver_settings,
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (args.output / "run_identity.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = run_benchmark(settings)
    results.to_csv(args.output / "benchmark_results.csv", index=False)
    print(results[["case_id", "thermo_model", "success", "message", "runtime_s",
                   "solver_cpu_time_s", "capture_pct", "boundary_residual_norm",
                   "solver_iterations", "final_mesh_nodes", "max_rms_residual"]].to_string(index=False))


if __name__ == "__main__":
    main()
