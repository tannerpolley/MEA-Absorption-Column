"""Numerical node qualification of the absorber's Engine quantities (Engine #148; contract N1-N6, C1, C2, C4).

Numerical verification only (not physical validation) on the MEA exploratory record
868a5018, the physical ideal-gas records and one pinned wheel. Criteria are frozen by
Engine #91 (absorber docs/coordination/greenfield-migration.md at b67fb11).

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    uv run --frozen python analyses/greenfield_node_qualification/scripts/qualify_nodes.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import casadi as ca
import numpy as np
from epcsaft import equilibrium as q

from mea_absorption_column.column import _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config
from mea_absorption_column.Thermodynamics.casadi_reactive import (
    ActionUnavailable, build_caloric_flow_function, build_loading_path_function)
from mea_absorption_column.Thermodynamics.reactive_bundle import _FORMULAS, engine_liquid, engine_vapor
from mea_absorption_column.Transport.Reactive_Film import binary_diffusivities_from_species, onsager_mobility_expression

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analyses/greenfield_node_qualification/results"
DATA = ROOT / "src/mea_absorption_column/data/epcsaft_datasets"
LIQUID_DATA, IDEAL = DATA / "MEA_greenfield_exploratory", DATA / "MEA_greenfield_exploratory/ideal-gas-thermochemistry.json"
VAPOR_PARAMETERS = DATA / "MEA_neutral_vapor/parameters.json"
CASE = ROOT / "analyses/bvp_solution_methods/input/case_3c.json"
FIRST, SECOND, ZERO, ROWS, FUGACITY, BUDGET_S = 5e-6, 5e-5, 1e-12, 1e-7, 1e-10, 600.0
STATES = {"S1": (318.15, 110900., (2.4323205, 9.7292821, 76.9751925)),
          "S2": (329.39, 110900., (4.04747, 9.72928, 76.0902))}
LOADINGS = {"S1": (-0.2, 0.0, 0.05, 0.4, 0.7746), "S2": (0.0, 0.168)}
VAPORS = {"V1": (316.75, 110900., (1.6540435, 1.5624169, 14.5306834, 1.6006873)),
          "V2": (329.571, 110155., (0.038893, 2.44739, 14.5306834, 1.6006873))}
S3 = (360.0, 106400., (0.45 * 9.7292821, 9.7292821, 76.9751925))
S2_NODE = [4.04747099164368, 76.09022399713623, 1.6540435152488, 1.5624169373574066, 329.3895574786238,
           316.7500000049194, 110899.99998043675, 0.052016874592319975, 0.004152821929475733,
           -0.19396500939796837, 44024.633074237165, 0.16817814616992194]
# IAPWS-95 saturation states via the NIST WebBook (fluid.cgi, ID=C7732185, SatP), retrieved 2026-09-24.
SATURATION = ((313.15, 7384.94, 46.3625 - 3.01815), (329.4, 16730.5, 46.8809 - 4.24235),
              (348.5, 39164.0, 47.4737 - 5.68378))
FEED_STEPS, T_STEPS, P_STEPS = (1e-2, 5e-3, 2.5e-3), (0.4, 0.2, 0.1), (400., 200., 100.)
rows: list[dict] = []


def record(check, state, item, exact, reference, defect, criterion, status):
    rows.append(dict(check=check, state=state, item=item, exact=exact, reference=reference,
                     defect=defect, criterion=criterion, status=status))


def inputs(state, loading=0.0):
    temperature, pressure, feed = STATES[state] if isinstance(state, str) else state
    return (temperature, pressure, feed[0] * math.exp(loading), *feed[1:])


def ladder(function, point, direction, steps):
    """Richardson estimate from three centred differences; resolved when the two finest agree to 10 %."""
    point, direction = np.asarray(point, float), np.asarray(direction, float)
    d = [(function(point + h * direction) - function(point - h * direction)) / (2 * h) for h in steps]
    coarse, fine = (4 * d[1] - d[0]) / 3, (4 * d[2] - d[1]) / 3
    return fine, np.abs(coarse - fine) <= 0.1 * np.abs(fine)


def compare(check, state, labels, exact, reference, resolved, criterion):
    exact, reference = np.atleast_1d(exact), np.atleast_1d(reference)
    for label, e, r, ok in zip(labels, exact, reference, np.atleast_1d(resolved)):
        defect = abs(e - r) / max(abs(e), abs(r), 1e-300)
        status = ("pass" if defect <= criterion else "fail") if ok else "unresolved"
        record(check, state, label, e, r, defect, criterion, status)
    # Negative control: a resolved nonzero component set to zero must fail.
    ok = np.atleast_1d(resolved) & (np.abs(reference) > 0)
    if ok.any():
        index = int(np.argmax(np.where(ok, np.abs(reference), 0)))
        record(check, state, f"control: {labels[index]} set to zero", 0.0, reference[index], 1.0, criterion,
               "control-fails" if 1.0 > criterion else "control-passed")


def refused(check, state, item, call):
    try:
        call()
    except ActionUnavailable as error:
        record(check, state, item, None, None, None, None, f"refused: {error.status}")
    else:
        record(check, state, item, None, None, None, None, "fail: expected typed refusal is now available")


def direction(phase, index):
    values = np.zeros(2 + len(phase.feed_ids))
    values[index] = 1.
    feed = [0.] * len(phase.component_ids)
    for name, value in zip(phase.feed_ids, values[2:]):
        feed[phase.component_ids.index(name)] = float(value)
    return q.SolvedStateActionDirection(float(values[0]), float(values[1]), feed, [])


def node_state_checks(liquid, name):
    u = inputs(name)
    compiled, central = liquid.solve(u)
    amounts = liquid.actions(u, liquid.state_observables[:9])
    feed = np.asarray(u[2:])
    atoms = np.array([[_FORMULAS[c].get(e, 0) for c in liquid.component_ids] for e in "CHNO"], float)
    masses, charges = np.asarray(liquid.molar_masses), np.asarray(liquid.charges)
    for label, true, apparent in [*((f"element {e}", atoms[i] * amounts, atoms[i, :3] * feed) for i, e in enumerate("CHNO")),
                                  ("mass", masses * amounts, masses[:3] * feed), ("charge", charges * amounts, np.zeros(1))]:
        scale = max(1.0, np.abs(true).sum(), np.abs(apparent).sum())
        defect = abs(true.sum() - apparent.sum())
        record("N1", name, label, true.sum(), apparent.sum(), defect / scale, ZERO, "pass" if defect <= ZERO * scale else "fail")
    residual = float(np.max(np.abs(central.classified_residuals)))
    record("N1", name, "max classified row residual", residual, 0.0, residual, ROWS, "pass" if residual <= ROWS else "fail")
    x, rho = amounts / amounts.sum(), central.molar_densities[0]
    root = liquid.model.state(u[0], P=u[1], x=list(x), phase="liquid")
    defect = abs(root.molar_density - rho) / rho
    record("N1", name, f"liquid pressure root ({root.density_branch.name})", rho, root.molar_density, defect, 1e-8,
           "pass" if defect <= 1e-8 and root.density_branch.name == "Liquid" else "fail")
    values = np.asarray(liquid(u)).ravel()
    at_root = liquid.model.state(u[0], rho=rho, x=list(x))
    for k, species in ((9, 0), (10, 2)):
        reference = x[species] * math.exp(at_root.log_fugacity_coefficient[species]) * u[1]
        defect = abs(values[k] - reference) / reference
        record("N1", name, f"f=aRT rho0 vs x phi P: {liquid.component_ids[species]}", values[k], reference, defect,
               FUGACITY, "pass" if defect <= FUGACITY else "fail")


def loading_checks(liquid, path, name, loading):
    label = f"{name}@{loading:g}"
    bulk = inputs(name)
    u = inputs(name, loading)
    ln_a = lambda point: liquid.actions(point, liquid.log_activities)
    amounts = liquid.actions(u, liquid.state_observables[:9])
    x = amounts / amounts.sum()
    ions = np.asarray(liquid.charges) != 0
    compiled, central = liquid.solve(u)
    for index, tag in ((2, "v"), (3, "e_MEA"), (4, "e_H2O")):
        batch = q.solved_state_actions(compiled, central, q.SolvedStateActionRequest(
            list(liquid.log_activities), [direction(liquid, index)]))
        slope = np.array([item.action for item in batch.results]) * (u[2] if tag == "v" else 1.0)
        terms = x * slope
        scale = max(1.0, np.abs(terms).sum())
        ratio = abs(terms.sum()) / (ZERO * scale)
        diagnostic = batch.diagnostics
        record("N2", label, f"sum x_i D ln a_i[{tag}] / (1e-12 scale); central_residual "
               f"{diagnostic.central_residual:.2e} floor {diagnostic.central_floor:.2e}",
               terms.sum(), 0.0, ratio * ZERO, ZERO, "pass" if ratio <= 1 else "fail")
        control = abs(terms[~ions].sum()) / (ZERO * scale)
        record("N2", label, f"control: ionic terms dropped [{tag}]", terms[~ions].sum(), 0.0, control * ZERO, ZERO,
               "control-fails" if control > 1 else "control-passed")
    # N3: tangent t = D ln a[v] (CasADi loading path) against centred differences along lambda.
    tangent = np.asarray(path(bulk, loading)[1]).ravel()
    reference, resolved = ladder(lambda lam: ln_a(inputs(name, float(lam[0]))), [loading], [1.0], (2e-2, 1e-2, 5e-3))
    compare("N3", label, [f"t:{c}" for c in liquid.component_ids], tangent, reference, resolved, FIRST)
    # N4: outer actions dt/du_e = u_CO2 D2 ln a[e_CO2, e] + delta_e,CO2 D ln a[e_CO2] against differences of t.
    first = liquid.actions(u, liquid.log_activities, (2,))
    t = lambda point: point[2] * liquid.actions(point, liquid.log_activities, (2,))
    refused("N4", label, "D2 ln a[v, e_T]", lambda: liquid.actions(u, liquid.log_activities, (2, 0)))
    for index, tag, steps in ((1, "e_P", P_STEPS), (2, "e_CO2", None), (3, "e_MEA", None), (4, "e_H2O", None)):
        exact = u[2] * liquid.actions(u, liquid.log_activities, (2, index)) + (first if index == 2 else 0.0)
        e = np.eye(5)[index]
        reference, resolved = ladder(t, u, e, steps or tuple(s * u[index] for s in FEED_STEPS))
        compare("N4", label, [f"dt/d{tag}:{c}" for c in liquid.component_ids], exact, reference, resolved, SECOND)
        if index == 2:  # chain control: omitting D ln a[e_CO2] must fail
            chain = exact - first
            ok = resolved & (np.abs(reference) > 0)
            defect = float(np.max(np.abs(chain - reference)[ok] / np.maximum(np.abs(chain), np.abs(reference))[ok]))
            record("N4", label, "control: chain term D ln a[e_CO2] omitted", None, None, defect, SECOND,
                   "control-fails" if defect > SECOND else "control-passed")


def conductance_checks(path, name, loading, diffusivity, constraints):
    label = f"{name}@{loading:g}"
    bulk, temperature = inputs(name), inputs(name)[0]
    pairs = ca.DM(binary_diffusivities_from_species(diffusivity(temperature)))
    co2 = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0], float)
    for fraction in (np.linspace(0, 1, 9) if loading else (0.0,)):
        values, tangent = (np.asarray(v).ravel() for v in path(bulk, fraction * loading))
        x = values[:9] / values[:9].sum()
        mobility = np.asarray(onsager_mobility_expression(ca.DM(x), values[11], pairs,
                                                          additional_flux_constraints=constraints))
        bound = ZERO * max(1.0, np.abs(mobility).sum(axis=1).max())
        item = f"{label} q={fraction:.3f}"
        for tag, mode in (("charge", constraints[0]), ("MEA", constraints[1]), ("water", constraints[2]), ("total", np.ones(9))):
            defect = float(np.abs(mobility @ mode).max())
            record("N5", item, f"M {tag} = 0", defect, 0.0, defect / bound * ZERO, ZERO, "pass" if defect <= bound else "fail")
        quadratic, conductance = float(tangent @ mobility @ tangent), float(co2 @ mobility @ tangent)
        record("N5", item, "t^T M t >= 0", quadratic, 0.0, None, None, "pass" if quadratic >= 0 else "fail")
        record("N5", item, "g > 0", conductance, 0.0, None, None, "pass" if conductance > 0 else "fail")


def caloric_checks(liquid, vapor):
    observables = liquid.state_observables[:9] + (liquid.enthalpy,)

    def liquid_h(point, index=None):
        y = liquid.actions(point, observables, () if index is None else (index,))
        if index is None:
            return np.array([y[:9].sum() * y[-1]])
        values = liquid.actions(point, observables)
        return np.array([y[:9].sum() * values[-1] + values[:9].sum() * y[-1]])

    for name in STATES:
        u = inputs(name)
        total_h = liquid_h(u)[0]
        partials = np.array([liquid_h(u, j)[0] for j in (2, 3, 4)])
        terms = np.asarray(u[2:]) * partials
        scale = max(abs(total_h), np.abs(terms).sum())
        defect = abs(terms.sum() - total_h) / scale
        record("C1", name, "sum_j F_j dH_L/dF_j = H_L", terms.sum(), total_h, defect, FIRST, "pass" if defect <= FIRST else "fail")
        control = abs(terms[:2].sum() - total_h) / scale
        record("C1", name, "control: water term omitted", terms[:2].sum(), total_h, control, FIRST,
               "control-fails" if control > FIRST else "control-passed")
        for index, tag, steps in ((0, "T", T_STEPS), (2, "F_CO2", None), (3, "F_MEA", None), (4, "F_H2O", None)):
            reference, resolved = ladder(liquid_h, u, np.eye(5)[index], steps or tuple(s * u[index] for s in FEED_STEPS))
            compare("C2", name, [f"dH_L/d{tag}"], liquid_h(u, index), reference, resolved, FIRST)
        refused("C2", name, "dH_L/dP", lambda: liquid_h(u, 1))
    caloric = build_caloric_flow_function(vapor)
    symbol = ca.MX.sym("u", 6)
    outputs = ca.vertcat(*caloric(symbol))
    jacobian = ca.Function("caloric_jacobian", [symbol], [ca.jacobian(outputs, symbol)])
    evaluate = lambda point: np.asarray(ca.vertcat(*caloric(point))).ravel()
    labels = ("H_V", "Cp_V", "hbar_CO2", "hbar_H2O")
    for name, (temperature, pressure, feed) in VAPORS.items():
        u = np.r_[temperature, pressure, feed]
        values, exact = evaluate(u), np.asarray(jacobian(u))
        terms = u[2:] * exact[0, 2:]
        scale = max(abs(values[0]), np.abs(terms).sum())
        defect = abs(terms.sum() - values[0]) / scale
        record("C1", name, "sum_i F_i dH_V/dF_i = H_V", terms.sum(), values[0], defect, FIRST, "pass" if defect <= FIRST else "fail")
        control = abs(terms[:-1].sum() - values[0]) / scale
        record("C1", name, "control: oxygen term omitted", terms[:-1].sum(), values[0], control, FIRST,
               "control-fails" if control > FIRST else "control-passed")
        for index, tag in enumerate(("T", "P", "F_CO2", "F_H2O", "F_N2", "F_O2")):
            steps = T_STEPS if index == 0 else P_STEPS if index == 1 else tuple(s * u[index] for s in FEED_STEPS)
            reference, resolved = ladder(evaluate, u, np.eye(6)[index], steps)
            if index != 1:  # first order: dH_V along T and feeds
                compare("C2", name, [f"dH_V/d{tag}"], exact[0, index], reference[0], resolved[0], FIRST)
            compare("C2", name, [f"d{label}/d{tag}" for label in labels[1:]], exact[1:, index], reference[1:], resolved[1:], SECOND)
    return caloric


def latent_heat_checks(liquid, vapor, caloric):
    water = [float(c == "water") for c in liquid.component_ids]
    for temperature, pressure, reference in SATURATION:
        phases = [liquid.model.state(temperature, P=pressure, x=water, phase=p) for p in ("liquid", "vapor")]
        model = (phases[1].residual_enthalpy - phases[0].residual_enthalpy) / 1000.0
        defect = abs(model - reference) / reference
        record("C4", f"water {temperature:g} K", "dh_vap kJ/mol vs IAPWS-95", model, reference, defect, 0.02,
               "pass" if defect <= 0.02 else "fail")
    observables = liquid.state_observables[:9] + (liquid.enthalpy,)
    for liquid_name, vapor_name in (("S1", "V1"), ("S2", "V2")):
        u = inputs(liquid_name)
        y, dy = liquid.actions(u, observables), liquid.actions(u, observables, (4,))
        hbar_liquid = dy[:9].sum() * y[-1] + y[:9].sum() * dy[-1]
        temperature, pressure, feed = VAPORS[vapor_name]
        hbar_vapor = float(np.asarray(caloric(np.r_[temperature, pressure, feed])[2]).ravel()[1])
        record("C4", f"{vapor_name}/{liquid_name}", "diagnostic: hbar_H2O^V - hbar_H2O^L kJ/mol",
               (hbar_vapor - hbar_liquid) / 1000.0, None, None, None, "diagnostic")


def main():
    started = time.perf_counter()
    case = json.loads(CASE.read_text())
    policy = case["physical_inputs"]["liquid_branch_policy"]
    liquid = engine_liquid(LIQUID_DATA, IDEAL, "qualified_liquid", loading_policy=policy)
    vapor = engine_vapor(VAPOR_PARAMETERS, IDEAL, "qualified_vapor")
    path = build_loading_path_function(liquid)
    model = case["physical_inputs"]["species_diffusivity_model"]
    diffusivity = lambda t: np.r_[model["co2_prefactor_m2_s"] * math.exp(
        -model["co2_activation_j_mol"] / (model["gas_constant_j_mol_k"] * t)), model["other_species_m2_s"]]
    constraints = np.array([[0, 0, 0, 1, -1, -1, -2, 1, -1], [0, 1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1, 1]], float)
    for name in STATES:
        node_state_checks(liquid, name)
        for loading in LOADINGS[name]:
            loading_checks(liquid, path, name, loading)
            conductance_checks(path, name, loading, diffusivity, constraints)
    caloric = caloric_checks(liquid, vapor)
    latent_heat_checks(liquid, vapor, caloric)
    # Start policy: native continuation from the accepted S1 state reaches the cold-start S2 state.
    cold = np.asarray(liquid.solve(inputs("S2"))[1].amounts)
    liquid._accepted = (inputs("S1"), liquid.solve(inputs("S1"))[1])
    continued = liquid.continue_to(inputs("S2"))
    defect = float(np.max(np.abs(np.asarray(continued.amounts) - cold) / cold)) if continued.success else math.inf
    record("start", "S1->S2", "continuation vs cold solve, max relative amount difference", defect, 0.0, defect, 1e-8,
           ("pass" if defect <= 1e-8 else "fail") if continued.success else f"fail: {continued.message}")
    s3 = np.asarray(liquid(inputs(S3))).ravel()
    record("S3", "360 K, 106.4 kPa, loading 0.45", "value-only: liquid density mol/m3", s3[11], None, None, None, "evaluated")
    # N6: the node Jacobian consumes dH_L/dP and D2 ln a[v, e_T]; until Engine #147 it must refuse, never estimate.
    request = {"preset": "twelve_state_conserved", "case": {"physical_input_file": str(CASE.relative_to(ROOT))},
               "numerics": {"method": "trapezoidal", "nodes": 11}}
    prepared = _prepare_conserved_column_in_process(resolve_column_config(request))
    node, balance = prepared["assembly"]["node"], prepared["assembly"]["balance"]
    point = np.asarray(case["initial_bulk_state"], float)
    point[7] -= float(balance(point, [0.0, 0.0, 0.0])[2])
    symbol = ca.MX.sym("node_state", 12)
    graph = ca.Function("node_jacobian", [symbol], [ca.jacobian(ca.vertcat(*node.call([ca.MX(0.0), symbol], True, False)), symbol)])
    for label, state in (("3C", np.r_[point, 1e-4, 0.0, 0.0, 0.05]), ("S2 node", np.asarray(S2_NODE))):
        try:
            graph(state)
            record("N6", label, "19x12 node Jacobian", None, None, None, 1e-5, "fail: available; run the centred comparison")
        except RuntimeError as error:
            cause = next((s for s in ("ReferenceUnavailable",) if s in str(error)), "other")
            record("N6", label, "19x12 node Jacobian", None, None, None, 1e-5, f"refused: {cause}")
    elapsed = time.perf_counter() - started
    record("budget", "all", "node-check wall time s", elapsed, None, elapsed, BUDGET_S, "pass" if elapsed <= BUDGET_S else "fail")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "node-checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        status = row["status"].split(":")[0]
        counts.setdefault(row["check"], {}).setdefault(status, 0)
        counts[row["check"]][status] += 1
    contract = json.loads((ROOT / "integration/epcsaft_contract.json").read_text())["final_identity"]
    worst = {check: max((r["defect"] for r in rows if r["check"] == check and r["status"] == "pass"
                         and r["defect"] is not None), default=None) for check in counts}
    summary = {
        "engine_commit": contract["engine_commit"], "engine_wheel_sha256": contract["wheel_sha256"],
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (LIQUID_DATA / "parameters.json", LIQUID_DATA / "engine-reactions.json", IDEAL, VAPOR_PARAMETERS)},
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256((OUT / "node-checks.csv").read_bytes()).hexdigest(),
        "status_counts": counts, "largest_passing_defect": worst, "elapsed_s": elapsed, "native_solves": liquid.stats,
        "claim_limits": "Numerical verification of consumed node quantities on the exploratory MEA record 868a5018 "
                        "(not adopted) at S1/S2/V1/V2; C4 is the pure-water EOS residual only. Refusals (N4 e_T, "
                        "dH_L/dP, N6) wait for Engine #147; K1-K2 and physical checks (C3, C5, C6) are not run here.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
