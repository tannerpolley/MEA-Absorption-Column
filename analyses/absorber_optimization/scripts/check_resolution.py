"""Run a mesh, film quadrature, or tolerance check from a retained design optimum."""
import json
import sys
from pathlib import Path

from mea_absorption_column.column import run_column

request_path, source, output = map(Path, sys.argv[1:4])
nodes, points, tolerance, limit = sys.argv[4:]
attempt = json.loads((source / "attempt.json").read_text())
if attempt["numerical_acceptance"] != attempt["physical_acceptance"] or attempt["physical_acceptance"] != "accepted":
    raise ValueError(f"Seed must be numerically and physically accepted: {source}")
request = json.loads(request_path.read_text())
settings = attempt["config"]["numerics"]["solver_settings"]
request.pop("initialization", None)
if (int(points), float(tolerance)) == (settings["quadrature_points"], settings["tolerance"]):
    request["initialization"] = {"values": {"retained_profile": str(source / "attempt.json")}}
request["design"]["initial_scale"] = attempt["result"]["design"]["decision"][0]
request["numerics"]["solver_settings"].update(nodes=int(nodes), quadrature_points=int(points), tolerance=float(tolerance))
request["execution"].update(output_dir=str(output), wall_limit_s=int(limit), threads=1)
result = run_column(request)
print(result["execution"], result["solver"], flush=True)
if result["numerical_acceptance"] != "accepted" or result["physical_acceptance"] != "accepted":
    print(result["failure"] or result.get("diagnostics", {}).get("physical_acceptance_reason"), flush=True)
    sys.exit(1)
