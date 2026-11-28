"""Finite-rate 6C/7C 33 -> 65 -> 129 nodes, warm-started from the converged [0, 0.5]-bracket 17-node solves, same case and transport copies.
Run from the repository root."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
OUT = Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128/twelve_state/bracket_probe")
RUN = "import json,sys; from mea_absorption_column.column import run_column; run_column(json.load(open(sys.argv[1])))"
def chain(case):
    seed = OUT / f"finite_rate_{case}_n17_b0.5/attempt.json"
    for nodes in (33, 65, 129):
        name = f"finite_rate_{case}_n{nodes}_b0.5"
        req = json.loads((OUT / f"finite_rate_{case}_n17_b0.5_request.json").read_text())
        req["numerics"]["solver_settings"]["nodes"] = nodes
        req["initialization"] = {"values": {"retained_profile": str(seed)}}
        req["execution"].update(output_dir=str(OUT / name), wall_limit_s=7200 if nodes == 129 else 3600)
        (OUT / f"{name}_request.json").write_text(json.dumps(req, indent=2) + "\n")
        with open(OUT / f"{name}.log", "w") as log:
            subprocess.run(["nice", "-n", "10", sys.executable, "-c", RUN, str(OUT / f"{name}_request.json")], stdout=log, stderr=subprocess.STDOUT)
        done = OUT / name / "attempt.json"
        status = (json.loads(done.read_text()).get("result") or {}).get("status") if done.exists() else None
        print(name, status, flush=True)
        if status != "Solve_Succeeded":
            return
        seed = done
with ThreadPoolExecutor(2) as pool:
    list(pool.map(chain, ("6C", "7C")))
