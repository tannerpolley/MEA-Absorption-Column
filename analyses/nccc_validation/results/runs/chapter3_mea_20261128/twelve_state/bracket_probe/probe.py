"""Finite-rate 6C/7C fresh 17-node starts failed in the interface-loading root search: its upper end 1.0 gives G - D m <= 0 with fit A.
Probe narrower upper ends (case and transport copies differ only in initial_interface_bracket and the references between them). Run from the repository root."""
import hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R = Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128"); OUT = R / "twelve_state/bracket_probe"
RUN = "import json,sys; from mea_absorption_column.column import run_column; run_column(json.load(open(sys.argv[1])))"
def probe(job):
    case, upper = job
    name = f"finite_rate_{case}_n17_b{upper}"
    case_data = json.loads(Path(f"analyses/nccc_validation/inputs/joint_fit_f2/case_{case.lower()}.json").read_text())
    case_data["initial_interface_bracket"] = [0.0, upper]
    case_path = OUT / f"case_{case.lower()}_b{upper}.json"
    case_path.write_text(json.dumps(case_data, indent=2) + "\n")
    data = json.loads((R / f"inputs/transport_{case.lower()}.json").read_text())
    data["case"] = {"repository_path": str(case_path), "sha256": hashlib.sha256(case_path.read_bytes()).hexdigest()}
    data["execution_policy"]["initial_interface_bracket"] = [0.0, upper]
    data["thermodynamic_input"] = "../../inputs/thermodynamics.json"  # resolved from this copy's folder
    (OUT / f"transport_{case.lower()}_b{upper}.json").write_text(json.dumps(data, indent=2) + "\n")
    req = json.loads((R / f"twelve_state/finite_rate_{case}_n17_fresh_request.json").read_text())
    req["case"]["physical_input_file"] = str(case_path)
    req["dependencies"]["transport_input"] = str(OUT / f"transport_{case.lower()}_b{upper}.json")
    req["execution"]["output_dir"] = str(OUT / name)
    (OUT / f"{name}_request.json").write_text(json.dumps(req, indent=2) + "\n")
    with open(OUT / f"{name}.log", "w") as log:
        subprocess.run(["nice", "-n", "10", sys.executable, "-c", RUN, str(OUT / f"{name}_request.json")], stdout=log, stderr=subprocess.STDOUT)
    a = json.loads((OUT / name / "attempt.json").read_text())
    f = a.get("failure") or {}
    print(name, (a.get("result") or {}).get("status"), f.get("message", "")[-90:], (f.get("last_checkpoint") or {}).get("initialization_evaluations"), flush=True)
with ThreadPoolExecutor(6) as pool:
    list(pool.map(probe, [(c, b) for c in ("6C", "7C") for b in (0.5, 0.3, 0.15)]))
