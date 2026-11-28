"""Case 3C minimum solvent at 90 % capture (problem P, equilibrium-manifold film) with the Chapter 3 MEA set (fit A), 17 -> 33 -> 65 nodes.

Same requests as stage2_3c_n17_problem_p and stage5_3c_n33/n65 except the thermodynamics and its transport input; the 17-node start is the
fit-A ladder's manifold_3C_n17_fresh solve. Run from the repository root.
"""
import json, subprocess, sys, time
from pathlib import Path

OUT = Path("analyses/absorber_optimization/results/runs/fitA_3c")
INPUTS = Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128/inputs")
CASE = "analyses/bvp_solution_methods/input/case_3c.json"
seed = Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128/twelve_state/manifold_3C_n17_fresh/attempt.json")
def finished(path):  # the ladder rewrites the record in stages; a failed seed ends the wait and the solve refuses it
    try:
        return json.loads(path.read_text()).get("execution", {}).get("status") in ("completed", "failed")
    except (FileNotFoundError, json.JSONDecodeError):
        return False


while not finished(seed):
    time.sleep(30)
scale = 1.0
for nodes, wall in ((17, 1800), (33, 3600), (65, 7200)):
    name = f"n{nodes}"
    request = {
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": CASE},
        "dependencies": {"thermodynamics_input": str(INPUTS / "thermodynamics.json"), "transport_input": str(INPUTS / "transport_3c.json")},
        "model": {"film_model": "equilibrium_manifold"},
        "numerics": {"method": "upwind", "solver_settings": {"nodes": nodes, "end_clustering": 1.0, "quadrature_points": 9, "tolerance": 1e-7,
                                                             "max_iterations": 60, "liquid_density_source": "solved_state"}},
        "initialization": {"values": {"retained_profile": str(seed)}},
        "design": {"capture_target": 0.90, "scale_bounds": [0.5, 2.0], "initial_scale": scale},
        "execution": {"wall_limit_s": wall, "threads": 1},
    }
    (OUT / f"{name}_request.json").write_text(json.dumps(request, indent=2) + "\n")
    with open(OUT / f"{name}.log", "w") as log:
        subprocess.run(["nice", "-n", "10", sys.executable, "analyses/absorber_optimization/scripts/solve_design.py",
                        str(OUT / f"{name}_request.json"), f"fitA_3c/{name}"], stdout=log, stderr=subprocess.STDOUT)
    attempt = OUT / name / "attempt.json"
    result = json.loads(attempt.read_text()).get("result") or {} if attempt.exists() else {}
    print(name, result.get("status"), (result.get("design") or {}).get("decision"), flush=True)
    if result.get("status") != "Solve_Succeeded":
        break
    subprocess.run([sys.executable, "analyses/absorber_optimization/scripts/summarize_optimum.py", str(attempt), CASE], check=True)
    seed, scale = attempt, result["design"]["decision"][0]
