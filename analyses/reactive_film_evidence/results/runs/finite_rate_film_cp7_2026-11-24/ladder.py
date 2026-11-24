"""Checkpoint 7 mesh ladder: manifold (warm-started) and enhancement factor (fresh start), 17 -> 33 -> 65 -> 129 nodes."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("analyses/reactive_film_evidence/results/runs/finite_rate_film_cp7_2026-11-24")
SOURCES = {
    "manifold": "analyses/nccc_validation/results/runs/joint_fit_f2_20261123/bridge/{c}/17/cosine/attempt.json",
    "enhancement": "analyses/reactive_film_evidence/results/runs/finite_rate_film_cp3_2026-11-23/{c}/enhancement/try1/attempt.json",
}
RUN = "import json,sys; from mea_absorption_column.column import run_column; run_column(json.load(open(sys.argv[1])))"


def rel(path):  # source records hold absolute paths from other checkouts
    return path[path.index("analyses/"):]


def chain(film, case):
    src = SOURCES[film].format(c=case).replace("1C/17/cosine/", "1C/17/cosine/retry/")
    config = json.loads(Path(src).read_text())["config"]
    seed = src
    for nodes in (33, 65, 129):
        name = f"{film}_{case}_n{nodes}" + ("_fresh" if film == "enhancement" else "")  # enhancement_1C_n33 is the retained warm-start refusal
        done = OUT / name / "attempt.json"
        if film == "manifold" and case == "1C" and nodes == 33:
            done = OUT / "manifold_1C_n33_root_anchored" / "attempt.json"
        if not done.exists():
            request = {
                "preset": "twelve_state_conserved",
                "case": {"physical_input_file": rel(config["case"]["physical_input_file"])},
                "dependencies": {k: rel(config["dependencies"][k]) for k in ("thermodynamics_input", "transport_input")},
                "numerics": {"method": "upwind", "solver_settings": dict(config["numerics"]["solver_settings"], nodes=nodes)},
                "initialization": {"values": {"retained_profile": seed} if film == "manifold" else {}},
                "execution": {"output_dir": str(OUT / name), "wall_limit_s": 7200 if nodes == 129 else 3600},
                "model": {"film_model": config["model"]["film_model"]},
            }
            (OUT / f"{name}_request.json").write_text(json.dumps(request, indent=2))
            with open(OUT / f"{name}.log", "w") as log:
                subprocess.run(["nice", "-n", "10", sys.executable, "-c", RUN, str(OUT / f"{name}_request.json")], stdout=log, stderr=subprocess.STDOUT)
        attempt = json.loads(done.read_text()) if done.exists() else {}
        status = (attempt.get("result") or {}).get("status")
        print(name, status, flush=True)
        if status != "Solve_Succeeded":
            return  # later meshes depend on this one; report and stop this chain
        seed = str(done)


cases = [f"{i}C" for i in range(1, 8)]
with ThreadPoolExecutor(2) as pool:  # one manifold and one enhancement chain at a time
    list(pool.map(lambda job: chain(*job), [job for c in cases for job in (("manifold", c), ("enhancement", c))]))
