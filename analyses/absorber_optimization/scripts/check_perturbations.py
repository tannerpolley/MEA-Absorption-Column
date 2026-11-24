"""Run Stage 3: check_perturbations.py RETAINED_RUN LABEL."""
import json
import resource
import sys
from pathlib import Path
from mea_absorption_column.column import run_column
base, label = Path(sys.argv[1]).resolve(), sys.argv[2]
request = json.loads((base / "request.json").read_text())
design = json.loads((base / "attempt.json").read_text())["result"]["design"]
s, p = design["decision"][0], design["parameters"]
prefix = base.parent / f"stage3_{label}"
def run(tag, settings):
    output = prefix.with_name(f"{prefix.name}_{tag}")
    if output.exists():
        result = json.loads((output / "attempt.json").read_text())
    else:
        before = sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2])
        execution = {**request["execution"], "output_dir": str(output), "wall_limit_s": 1800}
        result = run_column({**settings, "execution": execution})
        cpu_s = sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2]) - before
        log = dict(execution=result["execution"], worker_cpu_s=cpu_s, solver=result["solver"], failure=result["failure"])
        (output / "run.log").write_text(json.dumps(log, indent=2) + "\n")
    if result["numerical_acceptance"] != "accepted" or result["physical_acceptance"] != "accepted":
        raise RuntimeError(f"{output}: {result['solver']} {result['failure']}")
run("second_start", {**request, "design": {**request["design"], "initial_scale": 1.15}})
plain = {k: v for k, v in request.items() if k not in ("design", "initialization")}
for tag, factor in (("center", 1.), ("low", .99), ("high", 1.01)):
    case = json.loads(Path(request["case"]["physical_input_file"]).read_text())
    scale = s * factor
    case["physical_inputs"]["liquid_feed_mol_s"] = [scale * feed for feed in p[:3]]
    case["initial_bulk_state"][:2] = [scale * p[0], scale * p[2]]
    path = prefix.with_name(f"{prefix.name}_o4_{tag}.case.json")
    path.write_text(json.dumps(case, indent=2) + "\n")
    run(f"o4_{tag}", {**plain, "case": {"physical_input_file": str(path)}})
request["initialization"] = {"values": {"retained_profile": str(base / "attempt.json")}}
for name, index, delta in (("capture", 10, .005), ("loading", 0, .005 * p[1]), ("liquid_temperature", 7, 1.)):
    for step in (-2, -1, 1, 2):
        moved = p.copy()
        moved[index] += step * delta
        run(f"o5_{name}_{step:+d}", {**request, "design": {**request["design"], "initial_scale": s, "capture_target": moved[-1], "inlet_parameters": moved[:-1]}})
