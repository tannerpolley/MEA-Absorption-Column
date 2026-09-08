"""Retain the small analytic adapter check; no absorber physics or cost ranking."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from run_case import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    case_path = root/"tests/test_conserved_methods.py"
    case = runpy.run_path(str(case_path))
    rows, results = [], []
    for method in ("trapezoidal", "central", "shooting", "collocation"):
        for count in (9, 17):
            result = case["solve_analytic_case"](method, count)
            row = dict(method=method, initial_nodes=count, accepted=result["accepted"],
                       final_nodes=len(result.get("solver_grid", result["grid"])),
                       profile_samples=len(result["grid"]), state_error_inf=None,
                       boundary_residual_inf=None, algebraic_residual_inf=None, failure=result.get("failure"))
            if result["profile"] is not None:
                row.update(state_error_inf=float(np.max(abs(result["profile"]-case["analytic_profile"](result["grid"])))),
                           boundary_residual_inf=float(np.max(abs(result["boundary_residual"]))),
                           algebraic_residual_inf=float(np.max(abs(result["algebraic_residual"]))))
            rows.append(row)
            results.append(dict(method=method, initial_nodes=count, result=result))
    source_paths = [case_path, Path(__file__),
                   root/"src/mea_absorption_column/BVP/Methods/Conserved_Reduction.py",
                   root/"src/mea_absorption_column/BVP/Methods/Casadi_Collocation.py"]
    report = dict(scope="Independent analytic DAE verification; no absorber or matched-accuracy cost claim",
                  equations="u=(c,T,j), B=(c,T^2), R=(-j,-2j), a=j-c; c(0)=2,T(1)=3; positive T",
                  exact="c=j=2exp(-z), T=sqrt(9+4(exp(-z)-exp(-1)))",
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  rows=rows, results=results)
    (args.output/"analytic_methods.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False)+"\n")
    with (args.output/"analytic_methods.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    if not all(row["accepted"] for row in rows):
        raise SystemExit("Analytic verification contains failures; retained without omission")


if __name__ == "__main__":
    main()
