"""Finite-rate film with the ePC-SAFT free-CO2 Henry basis (henry_basis epcsaft_free_co2), Chapter 3 MEA set (fit A): 7 cases x 17 -> 33 -> 65 -> 129 nodes.
Same requests as ladder.py's finite_rate chains except henry_basis. A fresh 17-node start whose interface-loading root search fails on the [0, 1] bracket
retries once with the [0, 0.5] case and transport copies made as in bracket_probe/probe.py (only the bracket and the references between them differ);
later meshes reuse the inputs of the 17-node solve. Run from the repository root."""
import hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128")
OUT = R / "twelve_state/epcsaft_henry"
RUN = "import json,sys; from mea_absorption_column.column import run_column; run_column(json.load(open(sys.argv[1])))"


def case_file(case):
    return "analyses/bvp_solution_methods/input/case_3c.json" if case == "3C" else f"analyses/nccc_validation/inputs/joint_fit_f2/case_{case.lower()}.json"


def narrowed(case, upper=0.5):
    data = json.loads(Path(case_file(case)).read_text())
    data["initial_interface_bracket"] = [0.0, upper]
    case_path = OUT / f"case_{case.lower()}_b{upper}.json"
    case_path.write_text(json.dumps(data, indent=2) + "\n")
    transport = json.loads((R / f"inputs/transport_{case.lower()}.json").read_text())
    transport["case"] = {"repository_path": str(case_path), "sha256": hashlib.sha256(case_path.read_bytes()).hexdigest()}
    transport["execution_policy"]["initial_interface_bracket"] = [0.0, upper]
    transport["thermodynamic_input"] = "../../inputs/thermodynamics.json"  # resolved from this copy's folder
    transport_path = OUT / f"transport_{case.lower()}_b{upper}.json"
    transport_path.write_text(json.dumps(transport, indent=2) + "\n")
    return str(case_path), str(transport_path)


def solve(name, case_path, transport_path, nodes, seed):
    request = {
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": case_path},
        "dependencies": {"thermodynamics_input": str(R / "inputs/thermodynamics.json"), "transport_input": transport_path},
        "numerics": {"method": "upwind", "solver_settings": {"nodes": nodes, "quadrature_points": 9, "tolerance": 1e-7, "max_iterations": 60,
                                                             "liquid_density_source": "solved_state", "end_clustering": 1.0}},
        "initialization": {"values": {"retained_profile": seed} if seed else {}},
        "execution": {"output_dir": str(OUT / name), "wall_limit_s": 7200 if nodes == 129 else 3600},
        "model": {"film_model": "finite_rate_film", "henry_basis": "epcsaft_free_co2"},
    }
    (OUT / f"{name}_request.json").write_text(json.dumps(request, indent=2) + "\n")
    with open(OUT / f"{name}.log", "w") as log:
        subprocess.run(["nice", "-n", "10", sys.executable, "-c", RUN, str(OUT / f"{name}_request.json")], stdout=log, stderr=subprocess.STDOUT)
    done = OUT / name / "attempt.json"
    attempt = json.loads(done.read_text()) if done.exists() else {}
    status = (attempt.get("result") or {}).get("status")
    message = (attempt.get("failure") or {}).get("message", "")
    print(name, status, message[-80:] if status != "Solve_Succeeded" else "", flush=True)
    return (str(done) if status == "Solve_Succeeded" else None), message


def chain(case):
    inputs = (case_file(case), str(R / f"inputs/transport_{case.lower()}.json"))
    seed, message = solve(f"{case}_n17_fresh", *inputs, 17, None)
    if seed is None and "G - D m must be positive" in message:
        inputs = narrowed(case)
        seed, _ = solve(f"{case}_n17_fresh_b0.5", *inputs, 17, None)
    for nodes in (33, 65, 129):
        if seed is None:
            return
        seed, _ = solve(f"{case}_n{nodes}", *inputs, nodes, seed)


if __name__ == "__main__":
  OUT.mkdir(exist_ok=True)
  with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
    list(pool.map(chain, [f"{i}C" for i in range(1, 8)]))
