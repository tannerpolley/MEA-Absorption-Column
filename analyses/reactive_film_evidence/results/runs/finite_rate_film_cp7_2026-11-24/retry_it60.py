"""Retry the capped fresh-start enhancement runs with a 60-iteration cap (hypothesis: 20 iterations are too few from the declared initialization)."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("analyses/reactive_film_evidence/results/runs/finite_rate_film_cp7_2026-11-24")
RUN = "import json,sys; from mea_absorption_column.column import run_column; run_column(json.load(open(sys.argv[1])))"


def run(case, nodes):
    base = json.loads((OUT / f"enhancement_{case}_n33_fresh_request.json").read_text())
    name = f"enhancement_{case}_n{nodes}_fresh_it60"
    base["numerics"]["solver_settings"].update(nodes=nodes, max_iterations=60)
    base["execution"] = {"output_dir": str(OUT / name), "wall_limit_s": 7200 if nodes == 129 else 3600}
    (OUT / f"{name}_request.json").write_text(json.dumps(base, indent=2))
    with open(OUT / f"{name}.log", "w") as log:
        subprocess.run(["nice", "-n", "10", sys.executable, "-c", RUN, str(OUT / f"{name}_request.json")], stdout=log, stderr=subprocess.STDOUT)
    attempt = OUT / name / "attempt.json"
    status = (json.loads(attempt.read_text()).get("result") or {}).get("status") if attempt.exists() else None
    print(name, status, flush=True)


with ThreadPoolExecutor(2) as pool:
    list(pool.map(lambda job: run(*job), [("1C", 65), ("2C", 33), ("1C", 129), ("2C", 65), ("2C", 129)]))
