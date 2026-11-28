"""Twelve-state film ladder with the Chapter 3 MEA set (fit A): 7 cases x 3 films x 17 -> 33 -> 65 -> 129 nodes.

Same case and transport basis as the article film runs (finite_rate_film_cp6/cp7_2026-11-24), except the transport temperature domain [298, 393.15] K, which the validator derives from the Chapter 3 caloric intervals.
Run from the repository root.
"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUNS = Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128")
OUT = RUNS / "twelve_state"
FILMS = {
    "finite_rate": {"film_model": "finite_rate_film", "henry_basis": "n2o_analogy"},
    "manifold": {"film_model": "equilibrium_manifold"},
    "enhancement": {"film_model": "enhancement_reference"},
}
RUN = "import json,sys; from mea_absorption_column.column import run_column; run_column(json.load(open(sys.argv[1])))"


def case_file(case):
    return "analyses/bvp_solution_methods/input/case_3c.json" if case == "3C" else f"analyses/nccc_validation/inputs/joint_fit_f2/case_{case.lower()}.json"


def solve(film, case, nodes, seed):
    name = f"{film}_{case}_n{nodes}" + ("" if seed else "_fresh")
    done = OUT / name / "attempt.json"
    if not done.exists():
        request = {
            "preset": "twelve_state_conserved",
            "case": {"physical_input_file": case_file(case)},
            "dependencies": {"thermodynamics_input": str(RUNS / "inputs/thermodynamics.json"),
                             "transport_input": str(RUNS / f"inputs/transport_{case.lower()}.json")},
            "numerics": {"method": "upwind", "solver_settings": {
                "nodes": nodes, "quadrature_points": 9, "tolerance": 1e-7, "max_iterations": 60,
                "liquid_density_source": "solved_state", "end_clustering": 1.0}},
            "initialization": {"values": {"retained_profile": seed} if seed else {}},
            "execution": {"output_dir": str(OUT / name), "wall_limit_s": 7200 if nodes == 129 else 3600},
            "model": FILMS[film],
        }
        (OUT / f"{name}_request.json").write_text(json.dumps(request, indent=2) + "\n")
        with open(OUT / f"{name}.log", "w") as log:
            subprocess.run(["nice", "-n", "10", sys.executable, "-c", RUN, str(OUT / f"{name}_request.json")],
                           stdout=log, stderr=subprocess.STDOUT)
    attempt = json.loads(done.read_text()) if done.exists() else {}
    status = (attempt.get("result") or {}).get("status")
    print(name, status, flush=True)
    return str(done) if status == "Solve_Succeeded" else None


def chain(film, case):
    seed = None
    for nodes in (17, 33, 65, 129):
        if film == "enhancement":  # fresh on every mesh, as in the article
            solve(film, case, nodes, None)
            continue
        result = solve(film, case, nodes, seed)
        if result is None and seed:  # one fresh retry after a failed warm start
            result = solve(film, case, nodes, None)
        if result is None:
            return  # later meshes need this one as their start
        seed = result


OUT.mkdir(exist_ok=True)
jobs = [(film, f"{i}C") for i in range(1, 8) for film in FILMS]
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
    list(pool.map(lambda job: chain(*job), jobs))
