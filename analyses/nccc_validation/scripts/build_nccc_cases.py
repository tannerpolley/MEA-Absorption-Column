"""Build the one-bed Morgan Table A1 cases after reproducing the retained 3C input."""
import csv
from hashlib import sha256
import json
from math import isfinite, pi
from os.path import relpath
from pathlib import Path

from mea_absorption_column.config.Constants import column_params, packing_params
from mea_absorption_column.Properties.Thermophysical_Properties import vapor_pressure
from mea_absorption_column.Thermodynamics.solvent_inputs import ROOT, SolventInputs, formula_masses, read_json

CASE_DIR = ROOT / "analyses/nccc_validation/inputs/joint_fit_f2"
TABLE = ROOT / "src/mea_absorption_column/data/NCCC_2017_cases.csv"
RETAINED = ROOT / "analyses/bvp_solution_methods/input"
THERMO, TRANSPORT = (RETAINED / "issue85" / f"{name}.json" for name in ("thermodynamics", "transport"))


def identity(path):
    return {"repository_path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()).hexdigest()}


def build_case(row, bridge, owner):
    def value(key):
        return float(row[f"absorber_{key}"])

    case = row["case_no"]
    liquid_mass, gas_mass = (value(f"{stream}_flow_kg_h") / 3600 for stream in ("lean_solvent", "flue_gas"))
    alpha, w = value("lean_loading_mol_co2_per_mol_mea"), value("lean_solvent_mea_weight_fraction")
    imputed = not row["absorber_lean_solvent_temp_C"]
    tl = 273.15 + (45.0 if imputed else value("lean_solvent_temp_C"))
    tv, pressure, top = 273.15 + value("inlet_gas_temp_C"), 1000 * value("inlet_gas_pressure_kPa"), 1000 * value("top_pressure_kPa")
    co2, o2 = (value(f"inlet_gas_{species}_mole_fraction") for species in ("co2", "o2"))
    ml = formula_masses(bridge.thermo, bridge.feeds, bridge.mapping)
    mv = [c["fixed"]["molar_mass"]["value"]["magnitude"] for c in read_json(bridge.paths["gas_parameters"])["components"]]
    unloaded = liquid_mass / (1 + alpha * w * ml[0] / ml[1])
    mea = unloaded * w / ml[1]
    fl = [alpha * mea, mea, unloaded * (1 - w) / ml[2]]
    water = float(vapor_pressure(tv)) / pressure
    wet = [co2 * (1 - water), water, (1 - co2 - o2) * (1 - water), o2 * (1 - water)]
    fv = [y * gas_mass / sum(x * mass for x, mass in zip(wet, mv, strict=True)) for y in wet]
    geometry, packing = column_params["NCCC"], packing_params["MellapakPlus252Y"]
    beds = int(row["absorber_beds"])
    policy = bridge.transport["execution_policy"]["branch_policy"] | {"loading_anchor": alpha}
    diffusion = bridge.transport["species_diffusivities"]
    carbon = diffusion[bridge.roles["co2"]]
    mobility = dict(co2_prefactor_m2_s=carbon["prefactor_m2_s"], co2_activation_j_mol=carbon["activation_j_mol"],
                    gas_constant_j_mol_k=carbon["gas_constant_j_mol_k"],
                    other_species_m2_s=[diffusion[s]["constant_m2_s"] for s in bridge.species[1:]],
                    pair_closure="harmonic mean, common Reactive_Film.binary_diffusivities_from_species")
    humidity = "Water saturated at inlet Tv/P using existing repository vapor_pressure correlation"
    physical = dict(case_id=f"corrected published {row['campaign_year']} {case}", height_m=geometry["H"] * beds,
                    area_m2=pi * .25 * geometry["D"] ** 2, packing=[packing[k] for k in ("a_p","eps","Cl","Cv","Cs","Cp_0","Ch")],
                    liquid_feed_mol_s=fl, vapor_feed_mol_s=fv, liquid_temperature_k=tl, vapor_temperature_k=tv,
                    bottom_pressure_pa=pressure, gas_mass_flow_basis="reported_total_wet; paper moisture basis unspecified",
                    liquid_mass_flow_basis=f"loaded liquid mass; unloaded-solvent MEA mass fraction {w:.2f}",
                    humidity_assumption=humidity, reported_top_pressure_pa=top, liquid_branch_policy=policy, species_diffusivity_model=mobility)
    assumptions = [f"Tl{tl:g}K is user-approved imputation, not measurement" if imputed else "Measured lean solvent inlet temperature",
                   humidity, "Reported gasmass interpreted as totalwet mass; source moisture basis unspecified",
                   "Loaded liquid mass and unloaded wMEA basis; molar conversion uses current native molar masses",
                   f"Pressure boundary is bottom/inlet{pressure:g}Pa; measured top{top:g}Pa reserved for comparison"]
    inputs = dict(L=liquid_mass, G=gas_mass, alpha=alpha, w_MEA=w, y_CO2=co2, Tl=tl, Tv=tv, P=pressure, Beds=beds,
                  Intercoolers=int(row["intercoolers"]), lean_solvent_temp_imputed=imputed,
                  lean_solvent_temp_imputed_C=45.0 if imputed else None, y_O2=o2)
    inputs["CO2  %"] = value("capture_pct_avg")
    return dict(physical_inputs=physical, source_record=owner["source_record"] if case == "3C" else identity(TABLE),
                model_input_row=inputs, assumptions=assumptions, initial_bulk_state=[fl[0], fl[2], *fv[:2], tl, tv, pressure, owner["holdup_seed"]],
                liquid_molar_masses_kg_mol=ml, vapor_molar_masses_kg_mol=mv,
                mobility_basis="Same nine species diffusivity estimates and harmonic-mean pair closure as the retained owner runner",
                central_readiness_point=owner["central_readiness_point"] if case == "3C" else None,
                **{key: owner[key] for key in ("initial_interface_bracket","physical_residual_tolerance","source_runner_sha256","claim_limits")})


def max_deviation(built, retained):
    if isinstance(retained, dict) and isinstance(built, dict) and built.keys() == retained.keys():
        return max((max_deviation(built[k], v) for k, v in retained.items()), default=0.)
    if isinstance(retained, list) and isinstance(built, list) and len(built) == len(retained):
        return max((max_deviation(a, b) for a, b in zip(built, retained, strict=True)), default=0.)
    if type(retained) in (int, float) and type(built) in (int, float) and isfinite(built):
        return abs(built - retained)
    return 0. if type(built) is type(retained) and built == retained else float("inf")


def main():
    owner = read_json(CASE_DIR / "case_3c_provenance.json")
    if identity(TABLE)["sha256"] != owner["table_a1_csv_sha256"]:
        raise SystemExit("NCCC CSV differs from the Table A1 comparison")
    bridge = SolventInputs(THERMO, TRANSPORT)
    with TABLE.open(newline="") as file:
        rows = {r["case_no"]: r for r in csv.DictReader(file)}
    rebuilt, retained = build_case(rows["3C"], bridge, owner), read_json(RETAINED / "case_3c.json")
    deviations = {key: max_deviation(rebuilt.get(key), expected) for key, expected in retained.items()}
    print(json.dumps(deviations, indent=2))
    if rebuilt.keys() != retained.keys() or max(deviations.values()) > 1e-12:
        raise SystemExit("3C reconstruction exceeds the 1e-12 absolute gate")
    for case in ("1C","2C","4C","5C","6C","7C"):
        case_file = CASE_DIR / f"case_{case.lower()}.json"
        payload = build_case(rows[case], bridge, owner)
        case_file.write_text(json.dumps(payload, indent=2) + "\n")
        transport = read_json(TRANSPORT)
        transport.update(case=identity(case_file), thermodynamic_input=relpath(THERMO, CASE_DIR))
        transport["execution_policy"].update(id=f"NCCC_{case}", branch_policy=payload["physical_inputs"]["liquid_branch_policy"],
                                             **{k: payload[k] for k in ("initial_bulk_state","initial_interface_bracket","physical_residual_tolerance")})
        transport["domain"]["unloaded_amine_mass_fraction"][bridge.roles["amines"][0]]["value"] = payload["model_input_row"]["w_MEA"]
        (CASE_DIR / f"transport_{case.lower()}.json").write_text(json.dumps(transport, indent=2) + "\n")


if __name__ == "__main__":
    main()
