"""Physical acceptance of the twelve-state absorber on the adopted MEA record (Engine #149).

Criteria are frozen by Engine #91 (docs/coordination/greenfield-migration.md): C3 vapor Cp,
C5 liquid Cp, C6 heat of absorption, 3C capture, liquid temperature taps and density at
matched Amundsen states. Column quantities come from one accepted cosine 65-node attempt.

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    uv run --frozen python analyses/physical_acceptance_149/scripts/physical_checks.py \
    analyses/bvp_solution_methods/results/replay_149/upwind_cos_n65/attempt.json
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from mea_absorption_column.Thermodynamics.casadi_reactive import EnginePhase, build_caloric_flow_function
from mea_absorption_column.Thermodynamics.reactive_bundle import engine_liquid, engine_vapor, ideal_gas_record

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analyses/physical_acceptance_149/results"
DATA = ROOT / "src/mea_absorption_column/data"
LIQUID_DATA = DATA / "epcsaft_datasets/MEA_greenfield_exploratory"
IDEAL = LIQUID_DATA / "ideal-gas-thermochemistry.json"
VAPOR_PARAMETERS = DATA / "epcsaft_datasets/MEA_neutral_vapor/parameters.json"
TAPS = DATA / "NCCC_2017_absorber_temperature_profiles.csv"
CASE = ROOT / "analyses/bvp_solution_methods/input/case_3c.json"
F_MEA, HEIGHT, P_REF = 9.7292821, 6.0, 101325.0
VAPORS = {"V1": (316.75, 110900., (1.6540435, 1.5624169, 14.5306834, 1.6006873)),
          "V2": (329.571, 110155., (0.038893, 2.44739, 14.5306834, 1.6006873))}
# NIST-JANAF ideal-gas Cp [J/(mol K)] (janaf.nist.gov tables C-095, H-064, N-023, O-029), retrieved 2026-09-28;
# the 300 K rows are omitted so the spline nodes are well spaced.
JANAF_T = (200.0, 298.15, 400.0, 500.0)
JANAF_CP = {"carbon-dioxide": (32.359, 37.129, 41.325, 44.627), "water": (33.349, 33.590, 34.262, 35.226),
            "nitrogen": (29.107, 29.124, 29.249, 29.580), "oxygen": (29.126, 29.376, 30.106, 31.091)}
# Hilliard 2008 Appendix G.2 (Zotero DNQUPRMT, PDF p. 994): 7 mol MEA / kg water, kJ/(kg loaded solution K).
HILLIARD = ((318.15, 0.0, 3.7195), (318.15, 0.358, 3.3675), (353.15, 0.358, 3.4707))
# IAPWS-95 liquid water Cp at 0.101325 MPa via the NIST WebBook (fluid.cgi, C7732185), retrieved 2026-09-28.
IAPWS_WATER_CP = {318.15: 75.3064, 353.15: 75.6056}
# Kim and Svendsen 2007 via #82: 353.15 K, 30 mass % MEA, loading 0.047 -> 0.090, released 90.904 kJ/mol CO2.
HEAT_PAIR, HEAT_TARGET, HEAT_LIMIT, MEA_116_HEAT = (353.15, 0.047, 0.090), 90.904, 10.0, 88.34924122546846
# Amundsen 2009 Table 3, 30 mass % MEA (CO2-free), g/cm3; Amine-Thermodynamics data/reference/MEA/observations/
# density_viscosity/Amundsen_2009_density_viscosity.csv at d46727f.
AMUNDSEN = {(313.15, 0.1): 1.0210, (313.15, 0.2): 1.0410, (313.15, 0.3): 1.0629, (313.15, 0.4): 1.0885,
            (313.15, 0.5): 1.1140, (323.15, 0.1): 1.0160, (323.15, 0.2): 1.0355, (323.15, 0.3): 1.0580,
            (323.15, 0.4): 1.0830, (323.15, 0.5): 1.1080, (343.15, 0.1): 1.0040, (343.15, 0.2): 1.0240,
            (343.15, 0.3): 1.0464, (343.15, 0.4): 1.0719, (353.15, 0.1): 0.9970, (353.15, 0.2): 1.0176,
            (353.15, 0.3): 1.0402, (353.15, 0.4): 1.0660}
rows: list[dict] = []


def record(check, state, quantity, model, reference, error, limit, status, note=""):
    rows.append(dict(check=check, state=state, quantity=quantity, model=model, reference=reference,
                     error=error, limit=limit, status=status, note=note))


def relative(check, state, quantity, model, reference, limit, note=""):
    error = (model - reference) / reference
    record(check, state, quantity, model, reference, error, limit, "pass" if abs(error) <= limit else "fail", note)
    return error


def liquid_feed(liquid, loading, temperature, water_kg=None, mea_mass_fraction=0.30):
    """Apparent CO2/MEA/water amounts for 1 kg water at 7 mol MEA (Hilliard) or 1 kg CO2-free solvent."""
    m_mea, m_water = liquid.molar_masses[1], liquid.molar_masses[2]
    mea, water = (7.0, 1.0 / m_water) if water_kg else (mea_mass_fraction / m_mea, (1 - mea_mass_fraction) / m_water)
    return (temperature, P_REF, loading * mea, mea, water)


def mass(liquid, u):
    return float(np.dot(liquid.molar_masses[:3], u[2:]))


def mass_density(liquid, u):
    values = np.asarray(liquid(u)).ravel()
    return values[11] * mass(liquid, u) / values[:9].sum() / 1000.0


def volume_matched_liquid(liquid):
    """Diagnostic only: ion segment counts giving MEAH+ the MEA hard-core volume m sigma^3 and MEACOO- that of
    MEA + CO2 (packing diameters kept). Speciation re-solves; the record is not refit, so VLE is not claimed."""
    mapping = json.loads((LIQUID_DATA / "parameters.json").read_text())
    reactions = json.loads((LIQUID_DATA / "engine-reactions.json").read_text())
    values = {}

    def walk(node, update=None):
        if isinstance(node, dict):
            identity = node.get("identity")
            if identity and isinstance(node.get("value"), dict) and "magnitude" in node["value"]:
                values[identity] = node["value"]["magnitude"]
                if update and identity in update:
                    node["value"]["magnitude"] = update[identity]
            for child in node.values():
                walk(child, update)
        elif isinstance(node, list):
            for child in node:
                walk(child, update)

    walk(mapping["components"])
    size = lambda c: values[f"component/{c}/segment_count"] * values[f"component/{c}/segment_diameter"] ** 3
    update = {"component/protonated-monoethanolamine/segment_count":
              size("monoethanolamine") / values["component/protonated-monoethanolamine/packing_diameter"] ** 3,
              "component/carbamate-anion/segment_count":
              (size("monoethanolamine") + size("carbon-dioxide")) / values["component/carbamate-anion/packing_diameter"] ** 3}
    walk(mapping["components"], update)
    for family in mapping["model_families"]:
        if family.get("kind") == "electrolyte" and family.get("choice") == "born":
            for key, value in reactions["born_runtime_defaults"].items():
                family.setdefault(key, value)
    probe = EnginePhase("volume_matched_liquid", mapping, ideal_gas_record(IDEAL, liquid.component_ids), kind="liquid",
                        feed_ids=liquid.feed_ids, molar_masses=liquid.molar_masses, reactions=liquid.reactions,
                        neutral_reference=liquid.neutral_reference)
    return probe, update


def vapor_cp(vapor):
    caloric = build_caloric_flow_function(vapor)
    for name, (temperature, pressure, feed) in VAPORS.items():
        y = np.asarray(feed) / sum(feed)
        model = float(np.asarray(caloric(np.r_[temperature, pressure, feed])[1]).ravel()[0])
        state = vapor.model.state(temperature, P=pressure, x=list(y), phase="vapor")
        janaf = sum(yi * CubicSpline(JANAF_T, JANAF_CP[c])(temperature) for yi, c in zip(y, vapor.component_ids))
        residual = state.residual_isobaric_heat_capacity
        # The action-based Cp equals the state's ideal + residual Cp on the same root.
        split = state.ideal_isobaric_heat_capacity + residual
        record("C3", name, "action Cp_V vs state ideal + residual Cp", model, split, (model - split) / split, 1e-8,
               "pass" if abs(model - split) <= 1e-8 * split else "fail", "consistency, not physical")
        relative("C3", name, "Cp_V J/(mol K) vs sum y_i Cp_JANAF + EOS residual", model, janaf + residual, 0.02,
                 f"EOS residual {residual:.4f} J/(mol K)")


def liquid_cp(liquid, vapor):
    for temperature, loading, observed in HILLIARD:
        # The callback needs a positive CO2 feed; loading 1e-5 stands for the unloaded row (Cp change
        # about 1e-5 relative). On this wheel 1e-6 stalls in the trace-inventory regime fixed by Engine #172.
        u = liquid_feed(liquid, max(loading, 1e-5), temperature, water_kg=1.0)
        model = liquid.enthalpy_temperature_derivative(u) / mass(liquid, u) / 1000.0
        amounts = liquid.actions(u, liquid.state_observables[:9])
        state = liquid.model.state(temperature, P=P_REF, x=list(amounts / amounts.sum()), phase="liquid")
        residual = state.residual_isobaric_heat_capacity * amounts.sum() / mass(liquid, u) / 1000.0
        relative("C5", f"{temperature:g} K, loading {loading:g}", "Cp kJ/(kg K) vs Hilliard 2008 G.2",
                 model, observed, 0.03, f"EOS residual Cp at the equilibrium speciation {residual:.4f}")
    water = [float(c == "water") for c in liquid.component_ids]
    for temperature, reference in IAPWS_WATER_CP.items():
        # The nine-species mixture completes ion records only inside reacting problems, so the pure-water
        # ideal part comes from the same water record in the vapor mixture.
        residual = liquid.model.state(temperature, P=P_REF, x=water, phase="liquid").residual_isobaric_heat_capacity
        ideal = vapor.model.state(temperature, P=1.0, x=[0., 1., 0., 0.], phase="vapor").ideal_isobaric_heat_capacity
        janaf = float(CubicSpline(JANAF_T, JANAF_CP["water"])(temperature))
        model = ideal + residual
        record("C5 diagnosis", f"pure water {temperature:g} K", "liquid Cp J/(mol K) vs IAPWS-95", model, reference,
               (model - reference) / reference, None, "diagnostic",
               f"ideal record {ideal:.3f} (JANAF {janaf:.3f}); EOS residual {residual:.3f} (IAPWS-95 minus JANAF "
               f"{reference - janaf:.3f})")


def absorption_heat(liquid, vapor):
    temperature, a, b = HEAT_PAIR
    co2 = vapor.model.state(temperature, P=1.0, x=[1., 0., 0., 0.], phase="vapor").ideal_enthalpy
    enthalpy = []
    for loading in (a, b):
        u = liquid_feed(liquid, loading, temperature)
        values = np.asarray(liquid(u)).ravel()
        enthalpy.append((values[-1], u[2]))
    (h_a, n_a), (h_b, n_b) = enthalpy
    q = (co2 * (n_b - n_a) - (h_b - h_a)) / (n_b - n_a) / 1000.0
    error = q - HEAT_TARGET
    record("C6", f"{temperature:g} K, loading {a:g} -> {b:g}", "released heat kJ/mol CO2 vs Kim-Svendsen 2007",
           q, HEAT_TARGET, error, HEAT_LIMIT, "pass" if abs(error) <= HEAT_LIMIT else "fail",
           f"absolute kJ/mol; MEA #116 evaluator gives {MEA_116_HEAT:.4f} on the same record")
    record("C6 cross-check", "same interval", "absorber chain minus MEA #116 evaluator kJ/mol", q, MEA_116_HEAT,
           q - MEA_116_HEAT, None, "diagnostic", "MEA wheel b66c7b96, bubble-point pressure; here 101325 Pa")


def column(liquid, attempt_path):
    attempt = json.loads(attempt_path.read_text())
    grid = np.asarray(attempt["native_profile"]["grid"])
    state = np.asarray(attempt["native_profile"]["state_matrix"])
    inlet, outlet = state[2, 0], state[2, -1]
    capture = 100.0 * (1.0 - outlet / inlet)
    record("capture", "3C", "CO2 capture % vs 89.5 % observed", capture, 89.5, capture - 89.5, 5.0,
           "pass" if abs(capture - 89.5) <= 5.0 else "fail", "absolute percentage points")
    # Morgan 2020 Table C2 taps; Appendix C p. 22 measures x from the top, so z = (1 - x) H.
    with TAPS.open() as stream:
        tap_row = next(r for r in csv.DictReader(stream) if r["case_no"] == "3C")
    taps = sorted(((1.0 - float(k.split("_")[1])) * HEIGHT, float(v) + 273.15) for k, v in tap_row.items()
                  if k.startswith("position_"))
    heights, observed = (np.array(v) for v in zip(*taps))
    model = np.interp(heights, grid, state[4])
    for z, t_obs, t_model in zip(heights, observed, model):
        record("taps", f"z = {z:.1f} m", "liquid T K", t_model, t_obs, t_model - t_obs, None, "residual")
    rmse = float(np.sqrt(np.mean((model - observed) ** 2)))
    record("taps", "3C", "RMSE K", rmse, 0.0, rmse, 6.0, "pass" if rmse <= 6.0 else "fail")
    peak = int(np.argmax(state[4]))
    error = state[4, peak] - observed.max()
    record("taps", "3C", "profile peak K vs peak tap", state[4, peak], observed.max(), error, 5.0,
           "pass" if abs(error) <= 5.0 else "fail",
           f"model peak z = {grid[peak]:.2f} m; peak tap z = {heights[np.argmax(observed)]:.1f} m")
    loading = state[0] / F_MEA
    bounds = (state[4].min(), state[4].max(), loading.min(), loading.max())
    matched = [(t, a) for t, a in AMUNDSEN if bounds[0] <= t <= bounds[1] and bounds[2] <= a <= bounds[3]]
    probe, update = volume_matched_liquid(liquid)
    counts = ", ".join(f"{k.split('/')[1]} m = {v:.3f}" for k, v in update.items())
    for t, a in [(323.15, 1e-4), *matched]:
        u = liquid_feed(liquid, a, t)
        if a > 1e-3:
            relative("density", f"{t:g} K, loading {a:g}", "mass density g/cm3 vs Amundsen 2009", mass_density(liquid, u),
                     AMUNDSEN[(t, a)], 0.016, "matched to the 65-node liquid T and loading range")
        reference = AMUNDSEN.get((t, a), 0.9981)  # Amundsen Table 1, 50 C unloaded
        record("density diagnosis", f"{t:g} K, loading {a:g}", "volume-matched ion segment counts g/cm3",
               mass_density(probe, u), reference, (mass_density(probe, u) - reference) / reference, None, "diagnostic",
               f"{counts}; record m = 1; not refit; record density {mass_density(liquid, u):.4f}")
    for index in (0, peak, len(grid) - 1):
        u = (state[4, index], state[6, index], state[0, index], F_MEA, state[1, index])
        density = mass_density(liquid, u)
        record("density", f"node z = {grid[index]:.2f} m", "column-state mass density g/cm3", density, None, None,
               None, "diagnostic", f"T {u[0]:.2f} K, loading {loading[index]:.4f}; unmatched")
    return dict(attempt_id=attempt["attempt_id"], capture_pct=capture, liquid_temperature_range_k=bounds[:2],
                loading_range=bounds[2:], matched_amundsen_states=matched)


def main():
    attempt_path = Path(sys.argv[1]).resolve()
    case = json.loads(CASE.read_text())
    liquid = engine_liquid(LIQUID_DATA, IDEAL, "physical_liquid", loading_policy=case["physical_inputs"]["liquid_branch_policy"])
    vapor = engine_vapor(VAPOR_PARAMETERS, IDEAL, "physical_vapor")
    vapor_cp(vapor)
    liquid_cp(liquid, vapor)
    absorption_heat(liquid, vapor)
    profile = column(liquid, attempt_path)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "physical-checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contract = json.loads((ROOT / "integration/epcsaft_contract.json").read_text())["final_identity"]
    summary = {
        "engine_commit": contract["engine_commit"], "engine_wheel_sha256": contract["wheel_sha256"],
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (LIQUID_DATA / "parameters.json", LIQUID_DATA / "engine-reactions.json", IDEAL,
                           VAPOR_PARAMETERS, TAPS, attempt_path)},
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256((OUT / "physical-checks.csv").read_bytes()).hexdigest(),
        "column": profile,
        "outcomes": {r["check"] + " " + r["state"] + ": " + r["quantity"]: r["status"] for r in rows
                     if r["status"] in ("pass", "fail")},
        "native_solves": liquid.stats,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    for r in rows:
        print(f"{r['check']:14s} {r['state']:28s} {r['quantity'][:52]:52s} model={r['model']!s:.10s} "
              f"ref={r['reference']!s:.10s} err={r['error']!s:.9s} {r['status']} {r['note']}")


if __name__ == "__main__":
    main()
