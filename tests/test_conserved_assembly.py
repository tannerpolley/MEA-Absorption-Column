from __future__ import annotations

import json
from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from mea_absorption_column.column import _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config
from mea_absorption_column.Transport.Hydraulic_Variables_Correlations import interfacial_area_expression
from mea_absorption_column.Transport.Pressure_Drop import pressure_drop_expression


REQUEST = {
    "preset": "twelve_state_conserved",
    "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
    "numerics": {"method": "trapezoidal", "nodes": 11},
}


@pytest.fixture(scope="module")
def case():
    return json.loads(Path(REQUEST["case"]["physical_input_file"]).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def assembly():
    return _prepare_conserved_column_in_process(resolve_column_config(REQUEST))["assembly"]


@pytest.fixture(scope="module")
def point(assembly, case):
    value = np.asarray(case["initial_bulk_state"], dtype=float)
    value[7] -= float(assembly["balance"](value, [0.0, 0.0, 0.0])[2])
    return value


def test_conserved_layout_sources_bases_and_pressure_boundary(assembly, point, case):
    balance = assembly["balance"]
    liquid_feed = np.asarray(case["physical_inputs"]["liquid_feed_mol_s"], dtype=float)
    vapor_feed = np.asarray(case["physical_inputs"]["vapor_feed_mol_s"], dtype=float)
    transfer = np.array([1e-4, -5e-5, 10.0])
    conserved, sources, algebraic, liquid, vapor, hydraulics = [
        np.asarray(value).ravel() for value in balance(point, transfer)
    ]
    np.testing.assert_allclose(sources[:6], -hydraulics[5] * transfer[[0, 1, 0, 1, 2, 2]], rtol=1e-13)
    assert sources[6] == -hydraulics[6] < 0.0
    np.testing.assert_allclose([sources[2] - sources[0], sources[3] - sources[1], sources[5] - sources[4]], 0.0, atol=1e-12)
    assert abs(algebraic[0]) < 1e-12
    np.testing.assert_allclose(hydraulics[:2], [liquid[:9].sum() / liquid[11], vapor_feed.sum() / vapor[2]], rtol=1e-13)
    np.testing.assert_allclose(hydraulics[0] * hydraulics[2], liquid_feed @ np.asarray(assembly["liquid"].molar_masses[:3]), rtol=1e-10)
    np.testing.assert_allclose(hydraulics[1] * hydraulics[3], vapor_feed @ np.asarray(assembly["vapor"].molar_masses), rtol=1e-13)

    bottom, top = ca.MX.sym("bottom", 12), ca.MX.sym("top", 12)
    bc = assembly["boundary"](bottom, top)
    jac = ca.Function("conserved_boundary_derivatives", [bottom, top], [ca.jacobian(bc, bottom), ca.jacobian(bc, top)])
    left, right = (np.asarray(value) for value in jac(np.r_[point, [0.0] * 4], np.r_[point, [0.0] * 4]))
    assert np.linalg.matrix_rank(np.hstack((left, right))) == 7
    np.testing.assert_array_equal(left[6], np.eye(12)[6])
    np.testing.assert_array_equal(right[6], np.zeros(12))

    state, transfer_symbol = ca.MX.sym("bulk", 8), ca.MX.sym("transfer", 3)
    source = balance.call([state, transfer_symbol], True, False)[1]  # inline: only source paths are differentiated
    graph = ca.Function("conserved_source_derivatives", [state, transfer_symbol], [ca.jacobian(source, state)])
    jacobian = np.asarray(graph(point, transfer))
    assert jacobian.shape == (7, 8) and np.all(np.isfinite(jacobian))


# S2: bottom node of the accepted two-node 3C profile (contract N6).
S2_NODE = [4.04747099164368, 76.09022399713623, 1.6540435152488, 1.5624169373574066, 329.3895574786238,
           316.7500000049194, 110899.99998043675, 0.052016874592319975, 0.004152821929475733,
           -0.19396500939796837, 44024.633074237165, 0.16817814616992194]


@pytest.mark.parametrize("node_state", ["3C", "S2"])
def test_full_native_node_jacobian_matches_centred_difference(assembly, point, node_state):
    node = assembly["node"]
    state = ca.MX.sym("full_node_inputs", 12)
    expression = ca.vertcat(*node.call([ca.MX(0.0), state], True, False))
    graph = ca.Function("full_node_outer", [state], [ca.jacobian(expression, state)], {"cse": True})
    value = np.r_[point, 1e-4, 0.0, 0.0, 0.05] if node_state == "3C" else np.asarray(S2_NODE)
    jacobian = np.asarray(graph(value))
    assert jacobian.shape == (19, 12)
    assert np.all(np.isfinite(jacobian))

    direction = np.array([.1, 1., .2, .1, 1., 1., 1e4, .01, 1e-4, 0., 0., .05])
    step = 1e-3
    nearby = [
        np.concatenate([np.asarray(item).ravel() for item in node(0.0, value + sign * step * direction)])
        for sign in (1, -1)
    ]
    difference = (nearby[0] - nearby[1]) / (2 * step)
    exact = jacobian @ direction
    error = abs(exact - difference) / np.maximum(1e-12, np.maximum(abs(exact), abs(difference)))
    assert max(error) < 1e-5


def test_callbacks_are_repeatable_in_reverse_order(assembly, point):
    balance, liquid = assembly["balance"], assembly["liquid"]
    first = point.copy()
    second = point.copy()
    second[4] += 0.5
    initial = np.concatenate([np.asarray(item).ravel() for item in balance(first, [0.0, 0.0, 0.0])])
    balance(second, [0.0, 0.0, 0.0])
    before = liquid.stats["native_solves"]
    repeated = np.concatenate([np.asarray(item).ravel() for item in balance(first, [0.0, 0.0, 0.0])])
    np.testing.assert_array_equal(initial, repeated)
    assert liquid.stats["native_solves"] == before


def test_enhancement_reference_builds_the_selected_casadi_closure(point):
    request = {
        **REQUEST,
        "model": {"film_model": "enhancement_reference"},
    }
    prepared = _prepare_conserved_column_in_process(resolve_column_config(request))
    assembly = prepared["assembly"]
    diagnostics = assembly["diagnostics"]
    assert diagnostics.name_out(diagnostics.n_out() - 1) == "enhancement_factor"
    assert prepared["capabilities"]["a2_equilibrium_actions"] == "not_required_by_selected_film"
    state = np.r_[point, 1.0e-4, 0.0, 0.0, 0.05]
    values = np.concatenate([np.asarray(value).ravel() for value in assembly["node"](0.0, state)])
    assert np.all(np.isfinite(values))
    assert float(diagnostics(state)[-1]) >= 1.0


def test_shared_hydraulic_and_pressure_expressions_match_numeric_and_symbolic():
    packing = [250.0, .97, .203, .35, .017, .292, .119]
    inputs = ca.MX.sym("transport_inputs", 8)
    area = .32
    area_expr = interfacial_area_expression(inputs[0], inputs[1], inputs[2], area, packing)[0]
    pressure_expr = pressure_drop_expression(
        inputs[3], inputs[0], inputs[4], inputs[5], inputs[6], area, inputs[2], inputs[7], packing,
    )
    graph = ca.Function("shared_transport_values", [inputs], [area_expr, pressure_expr])
    point = np.array([950.0, .03, .1, .08, 1.2, 1.1e-3, 1.8e-5, .5])
    symbolic = np.asarray(graph(point)).ravel()
    numeric = np.array([
        interfacial_area_expression(point[0], point[1], point[2], area, packing)[0],
        pressure_drop_expression(point[3], point[0], point[4], point[5], point[6], area, point[2], point[7], packing),
    ])
    np.testing.assert_allclose(symbolic, numeric, rtol=1e-12, atol=1e-12)


def _exchanger(lam, nodes, scheme="upwind"):
    """Linear countercurrent exchanger L dx/dz = G dy/dz = -K(y - m x) on 0 <= z <= 6 m, liquid x
    entering at the top, gas y at the bottom. Its one nontrivial mode d = y - m x has rate
    lambda = -K(1/G - m/L). Returns the discrete x, y, d and the exact nodal x, y."""
    from mea_absorption_column.BVP.Methods.Casadi_Collocation import solve_conservative_collocation

    big_l, big_g, height, x_in, y_in = 2.0, 1.0, 6.0, 0.1, 1.0
    m = (0.5 if lam < 0 else 2.0) * big_l / big_g  # gas-side (lam < 0) or liquid-side controlled
    k = -lam / (1 / big_g - m / big_l)
    z, u = ca.MX.sym("z"), ca.MX.sym("u", 3)
    node = ca.Function("node", [z, u], [ca.vertcat(big_l * u[0], big_g * u[1]), -ca.vertcat(u[2], u[2]),
                                        u[2] - k * (u[1] - m * u[0])])
    bottom, top = ca.MX.sym("bottom", 3), ca.MX.sym("top", 3)
    boundary = ca.Function("inlets", [bottom, top], [ca.vertcat(top[0] - x_in, bottom[1] - y_in)])
    grid = np.linspace(0.0, height, nodes)
    result = solve_conservative_collocation(
        node, boundary, grid, np.tile([[x_in], [y_in], [0.0]], nodes), [-10, -10, -1e3], [10, 10, 1e3],
        state_scale=[1, 1, 1], balance_scale=[1, 1], algebraic_scale=[1], boundary_scale=[1, 1],
        tolerance=1e-12, scheme=scheme, cell_sources=("lower", "upper", "cell") if scheme == "upwind" else None)
    assert result["accepted"], result["failure"]
    x, y = result["profile"][:2]
    # Exact: d = D exp(lam (z - z_ref)), z_ref at the end the mode decays from; G y' = L x' = -K d.
    reference = 0.0 if lam < 0 else height
    integral = lambda s: (np.exp(lam * (s - reference)) - np.exp(-lam * reference)) / lam
    x0, amplitude = np.linalg.solve([[m, np.exp(-lam * reference)], [1, -k / big_l * integral(height)]], [y_in, x_in])
    return x, y, y - m * x, x0 - k / big_l * amplitude * integral(grid), y_in - k / big_g * amplitude * integral(grid)


@pytest.mark.parametrize("lam", [-15.0, -10.0, 10.0, 15.0])
def test_upwind_cells_damp_both_stiff_directions_without_alternation(lam):
    # Measured interface-mode rates |lambda| = 10-15 1/m (#148); coarsest ladder spacing 1.5 m.
    _, _, mode, _, _ = _exchanger(lam, 5)
    assert np.all(mode[1:] / mode[:-1] > 0), mode
    _, _, trapezoidal_mode, _, _ = _exchanger(lam, 5, "trapezoidal")
    assert np.any(trapezoidal_mode[1:] / trapezoidal_mode[:-1] < 0)  # the failure this scheme removes


@pytest.mark.parametrize("lam", [-15.0, 15.0])
def test_upwind_cells_are_first_order_and_conserve_on_the_grid(lam):
    errors = []
    for nodes in (385, 769, 1537):
        x, y, _, exact_x, exact_y = _exchanger(lam, nodes)
        errors.append(max(np.max(abs(x - exact_x)), np.max(abs(y - exact_y))))
        invariant = 1.0 * y - 2.0 * x  # G y - L x
        assert np.max(abs(invariant - invariant[0])) < 1e-10
    order = np.log2(np.asarray(errors[:-1]) / errors[1:])  # observed 0.89 then 0.94: approaching 1
    assert order[0] < order[1] and abs(order[1] - 1) < 0.1, (errors, order)
