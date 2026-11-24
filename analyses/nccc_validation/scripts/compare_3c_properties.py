"""Evaluate matched 3C properties and decompose the retained upwind energy balance."""
import json
from pathlib import Path

import casadi as ca
import numpy as np
import pandas as pd

from mea_absorption_column.column import _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config
from mea_absorption_column.Thermodynamics.thermo_models import ensure_epcsaft_importable

RUN = Path(__file__).resolve().parents[1] / "results/runs/joint_fit_f2_20261123"
bridge = json.loads((RUN / "bridge/3C/33/attempt.json").read_bytes())
matched = np.asarray(bridge["result"]["profile"]).T
grid = np.asarray(bridge["result"]["grid"])
engine = ensure_epcsaft_importable()
properties, energy = [], []
for arm in ("bridge", "f2"):
    attempt = json.loads((RUN / arm / "3C/33/attempt.json").read_bytes())
    config = attempt["config"]
    declarations = {k: config["dependencies"][k] for k in ("thermodynamics_input", "transport_input")}
    request = dict(preset=attempt["preset"], dependencies=declarations, numerics=config["numerics"])
    resolved = resolve_column_config(request)
    prepared = _prepare_conserved_column_in_process(resolved)
    assert prepared["engine"]["actual_sha256"] == attempt["engine"]["actual_sha256"]
    assert prepared["resolved_inputs"]["parameters"] == attempt["resolved_inputs"]["parameters"]
    assert prepared["resolved_inputs"]["physical_input_sha256"] == bridge["resolved_inputs"]["physical_input_sha256"]
    assembly = prepared["assembly"]
    liquid, vapor = assembly["liquid"], assembly["vapor"]
    fl, fv = prepared["resolved_inputs"]["parameters"][:2]
    equilibrium = liquid._q
    water = engine.Pure(engine.Parameters.from_json(resolved.solvent.paths["parameters"], components=["water"]))
    u = ca.MX.sym("u", 2 + len(liquid.feed_ids))
    gradient = ca.Function("liquid_enthalpy_gradient", [u], [ca.jacobian(liquid(u)[-1], u)])
    co2, carbamate, bicarbonate = (liquid.component_ids.index(name) for name in ("carbon-dioxide", "carbamate-anion", "bicarbonate-anion"))
    for z, state in zip(grid, matched, strict=True):
        tl, tv, p = state[[4, 5, 6]]
        ul = np.r_[tl, p, liquid.flow(*state[:2], fl).full().ravel()]
        uv = np.r_[tv, p, vapor.flow(*state[2:4], fv).full().ravel()]
        amounts, fugacity, molar_density, _ = np.split(liquid(ul).full().ravel(), [9, 11, 12])
        x = amounts / amounts.sum()
        guess = dict(zip(("carbon-dioxide", "water", "monoethanolamine"), np.r_[fugacity, 1.] / (fugacity.sum() + 1.), strict=True))
        phases = [equilibrium.Phase("L", kind="liquid", support=liquid.support, composition_guess=x.tolist()),
                  equilibrium.Phase("V", kind="vapor", support=liquid.feed_ids, amount=equilibrium.Pinned(0.), composition_guess=guess)]
        bubble = equilibrium.Problem(phases=phases, T=tl, P=equilibrium.Free(guess=fugacity.sum() + 1.),
                                      feed=equilibrium.Amounts(dict(zip(liquid.feed_ids, ul[2:], strict=True))),
                                      reactions=liquid.reactions, neutral_reference=liquid.neutral_reference,
                                      topology_declared=True, row_tolerance=1e-10)
        bubble, saturation = equilibrium.solve_equilibrium(liquid.model, bubble), equilibrium.solve_equilibrium(water, water.saturation(tl))
        assert bubble.success and bubble.requested_tolerance_met, (arm, z, bubble.message)
        assert saturation.success and saturation.requested_tolerance_met, (arm, z, saturation.message)
        h_l, h_v = [water.state(tl, rho=rho).residual_enthalpy for rho in saturation.molar_densities]
        cp, _, h_co2, *_ = gradient(ul).full().ravel()
        gas_h = vapor.model.state(tl, P=1., x={"carbon-dioxide": 1.}, phase="vapor").ideal_enthalpy
        mass = amounts @ liquid.molar_masses
        properties.append([arm, z, tl, tv, p, ul[2] / ul[3], molar_density.item() * mass / amounts.sum(), cp / mass / 1000,
                           (gas_h - h_co2) / 1000, bubble.pressure * bubble.phase_mole_fractions(1)[co2], x[carbamate], x[bicarbonate],
                           (h_v - h_l) / 1000, vapor.enthalpy_temperature_derivative(uv) / uv[2:].sum(),
                           max(map(abs, bubble.classified_residuals)), max(map(abs, saturation.classified_residuals))])
    result = attempt["result"]
    assert attempt["physical_certification"]["accepted"] and np.array_equal(result["grid"], grid)
    states, cells = np.asarray(result["profile"]), np.asarray(result["cell_profile"])
    assert result["cell_sources"] == ["lower", "lower", "upper", "upper", "lower", "upper", "upper"] + ["cell"] * 5
    for k, dz in enumerate(np.diff(grid)):
        lower, upper = states[:, k:k + 2].T
        state = np.r_[lower[[0, 1]], upper[[2, 3]], lower[4], upper[[5, 6]], cells[:, k]]
        flow = liquid.flow(*state[:2], fl).full().ravel()
        cp, dh_p, h_co2, _, h_water = gradient(np.r_[state[4], state[6], flow]).full().ravel()
        heat, _, h_v = assembly["diagnostics"](state)[4:7]
        area, pressure_gradient = assembly["balance"](state[:8], state[8:11])[-1].full().ravel()[5:]
        exchanges = dz * area * state[8:11]
        direct = dz * area * float(heat) * (state[5] - state[4])
        uptake, water_transfer = exchanges[:2] * (h_v.full().ravel() - [h_co2, h_water])
        energy.append([arm, grid[k], grid[k + 1], direct, uptake, water_transfer, dz * pressure_gradient * dh_p,
                       cp * (lower[4] - upper[4]), exchanges[0], exchanges[1], exchanges[2],
                       exchanges[2] - direct - exchanges[:2] @ h_v.full().ravel()])
pd.DataFrame(properties, columns="arm z_m liquid_K vapor_K liquid_pressure_Pa loading_mol_mol density_kg_m3 liquid_cp_kJ_kg_K heat_of_absorption_kJ_mol co2_equilibrium_pressure_Pa carbamate_mole_fraction bicarbonate_mole_fraction water_vaporization_kJ_mol vapor_cp_J_mol_K bubble_residual_inf saturation_residual_inf".split()).to_csv(RUN / "comparison/3c_properties.csv", index=False)
pd.DataFrame(energy, columns="arm z_lower_m z_upper_m direct_sensible_W co2_uptake_heat_W water_transfer_heat_W pressure_W sensible_liquid_W co2_transfer_mol_s water_transfer_mol_s enthalpy_transfer_W interface_energy_defect_W".split()).to_csv(RUN / "comparison/3c_energy_cells.csv", index=False)
