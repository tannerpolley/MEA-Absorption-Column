"""Regenerate optimum.json with: summarize_optimum.py ATTEMPT_JSON CASE_JSON."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


if __name__ == "__main__":
    attempt_path, case_path = map(Path, sys.argv[1:])
    attempt = json.loads(attempt_path.read_text())
    case = case_path.read_bytes()
    if hashlib.sha256(case).hexdigest() != attempt["resolved_inputs"]["source_lineage"]["physical_input_file_sha256"]:
        raise ValueError("Case file differs from the retained attempt")
    design = attempt["result"]["design"]
    p, s = np.asarray(design["parameters"]), design["decision"][0]
    ds = -np.asarray(design["lam_p"])
    masses = np.asarray(json.loads(case)["liquid_molar_masses_kg_mol"])
    mol_flow, mass_flow = p[:3].sum(), p[:3] @ masses
    molar_derivative = mol_flow * ds + s * np.r_[np.ones(3), np.zeros(8)]
    mass_derivative = mass_flow * ds + s * np.r_[masses, np.zeros(8)]
    names = ("liquid_CO2", "liquid_MEA", "liquid_H2O", "vapor_CO2", "vapor_H2O", "vapor_N2", "vapor_O2",
             "liquid_temperature", "vapor_temperature", "bottom_pressure", "capture_target")
    units = ["mol/s"] * 7 + ["K", "K", "Pa", "fraction"]
    summary = {
        "attempt_id": attempt["attempt_id"],
        "flow_definition": "L*(p) = s*(p) times the unscaled liquid feed total at p",
        "baseline_liquid_flow_mol_s": float(mol_flow),
        "baseline_liquid_flow_kg_s": float(mass_flow),
        "minimum_liquid_flow_mol_s": float(mol_flow * s),
        "minimum_liquid_flow_kg_s": float(mass_flow * s),
        "capture_fraction": 1 - attempt["result"]["profile"][2][-1] / p[3],
        "physical_capture_multiplier_per_mol_s": design["lam_g"][-1] / design["row_scale"][0],
        "sensitivities": [dict(parameter=name, parameter_unit=unit, ds_dp=float(ds[i]),
                               dL_mol_s_dp=float(molar_derivative[i]), dL_kg_s_dp=float(mass_derivative[i]))
                          for i, (name, unit) in enumerate(zip(names, units, strict=True))],
    }
    attempt_path.with_name("optimum.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
