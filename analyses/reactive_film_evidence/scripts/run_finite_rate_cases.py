"""Run one #16 closure using a #91 bridge request."""
import json
import os
import sys
import time
from pathlib import Path
from mea_absorption_column.column import run_column
from mea_absorption_column.Thermodynamics.solvent_inputs import read_json as read
BRIDGE = Path("analyses/nccc_validation/results/runs/joint_fit_f2_20261123/bridge")
FILMS = {"enhancement": {"film_model": "enhancement_reference"},
         "n2o": {"film_model": "finite_rate_film", "henry_basis": "n2o_analogy"},
         "epcsaft": {"film_model": "finite_rate_film", "henry_basis": "epcsaft_free_co2"}}
FILMS["anchors"] = FILMS["n2o"] | {"mea_diffusivity_anchors": True}

def bridge_request(case):
    return BRIDGE / case / "17/cosine" / ("retry/request.json" if case == "1C" else "request.json")

def machine():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {"time_utc": time.strftime("%FT%TZ", time.gmtime()), "load": os.getloadavg(),
            "cpu_ticks": Path("/proc/stat").read_text().splitlines()[0],
            **{key: int(memory[key].split()[0]) for key in ("MemAvailable", "SwapFree")}}

def run(output, case, film, index="1"):
    directory = output / case / film
    log_path = directory / f"try{index}.log"
    if log_path.is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    reference, snapshot = bridge_request(case), machine()
    request = read(reference) | {"model": FILMS[film]}
    request["initialization"]["values"].pop("retained_profile", None)
    request["execution"].update(output_dir=str(directory / f"try{index}"), wall_limit_s=600)
    if index == "2":
        assert read(directory / "try1/attempt.json")["execution"]["status"] == "timed_out"
        assert snapshot["load"][0] < read(directory / "launch1.json")["machine"]["load"][0]
    with log_path.open("x") as log:
        (directory / f"launch{index}.json").write_text(json.dumps(
            dict(request=request, manifold_reference=str(reference), machine=snapshot), indent=2) + "\n")
        print(case, film, snapshot, file=log, flush=True)
        attempt = run_column(request)
        print(attempt["execution"], attempt["failure"], file=log, flush=True)
        if attempt["execution"]["status"] == "timed_out":
            print("Machine at wall limit:", machine(), file=log, flush=True)

if __name__ == "__main__":
    run(Path(sys.argv[1]), *sys.argv[2:])
