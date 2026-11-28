"""Case 3C minimum solvent at 90 % capture at 129 nodes, after n17/n33/n65 gave scale 0.926/0.957/0.981 (not mesh-independent).

Two starts run side by side, since the solve is serial:
  n129_from_sim  -- the converged 129-node manifold simulation (scale 1.0), the start that rescued n33;
  n129_from_n65  -- the 65-node optimum interpolated to 129 nodes, as n65 was started from n33.
Same request as n65 otherwise. Run from the repository root: python n129.py <from_sim|from_n65>

Run on commit 40215db (the n17-n65 code): later commits change Thermophysical_Properties.py, whose hash the fit-A inputs pin.
n129_from_sim converged (6 iterations, 500 s in IPOPT, s* = 0.99651); n129_from_n65 was stopped unfinished once it had, so no record is kept.
"""
import json, subprocess, sys
from pathlib import Path

OUT = Path("analyses/absorber_optimization/results/runs/fitA_3c")
INPUTS = Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128/inputs")
CASE = "analyses/bvp_solution_methods/input/case_3c.json"
STARTS = {
    "from_sim": (Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128/twelve_state/manifold_3C_n129/attempt.json"), 1.0),
    "from_n65": (OUT / "n65/attempt.json", 0.9806754832113022),
}
start = sys.argv[1]
seed, scale = STARTS[start]
name = f"n129_{start}"
request = json.loads((OUT / "n65_request.json").read_text())
request["numerics"]["solver_settings"]["nodes"] = 129
request["initialization"]["values"]["retained_profile"] = str(seed)
request["design"]["initial_scale"] = scale
request["execution"]["wall_limit_s"] = 3600
(OUT / f"{name}_request.json").write_text(json.dumps(request, indent=2) + "\n")
with open(OUT / f"{name}.log", "w") as log:
    subprocess.run([sys.executable, "analyses/absorber_optimization/scripts/solve_design.py",
                    str(OUT / f"{name}_request.json"), f"fitA_3c/{name}"], stdout=log, stderr=subprocess.STDOUT)
attempt = OUT / name / "attempt.json"
result = (json.loads(attempt.read_text()).get("result") or {}) if attempt.exists() else {}
print(name, result.get("status"), (result.get("design") or {}).get("decision"), flush=True)
if result.get("status") == "Solve_Succeeded":
    subprocess.run([sys.executable, "analyses/absorber_optimization/scripts/summarize_optimum.py", str(attempt), CASE], check=True)
