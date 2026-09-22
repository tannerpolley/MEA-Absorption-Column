from __future__ import annotations

import json
import importlib
from pathlib import Path
from types import SimpleNamespace

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
    np.testing.assert_allclose(hydraulics[:2], [liquid_feed.sum() * liquid[:9].sum() / liquid[27], vapor_feed.sum() / vapor[4]], rtol=1e-13)
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
    source = balance(state, transfer_symbol)[1]
    graph = ca.Function("conserved_source_derivatives", [state, transfer_symbol], [ca.jacobian(source, state)])
    jacobian = np.asarray(graph(point, transfer))
    assert jacobian.shape == (7, 8) and np.all(np.isfinite(jacobian))


def test_native_a2_and_caloric_actions_are_available(assembly, point, case):
    from mea_absorption_column.Thermodynamics.thermo_models import ensure_epcsaft_importable

    ensure_epcsaft_importable()
    equilibrium = importlib.import_module("epcsaft.equilibrium")

    reactive = assembly["reactive_liquid"]
    liquid = assembly["liquid"]
    liquid_feed = np.asarray(case["physical_inputs"]["liquid_feed_mol_s"], dtype=float)
    vapor_feed = np.asarray(case["physical_inputs"]["vapor_feed_mol_s"], dtype=float)
    state = reactive.solve(point[4], point[6], [point[0], liquid_feed[1], point[1]], state_input_derivatives=True)
    block = state["state_input_derivatives"]
    n = len(block.input_identities)
    directions = tuple(tuple(float(i == row) for i in range(n)) for row in (0, 1))
    action = equilibrium.EquilibriumStateInputAction(
        "assembly-mu-A2", 2, block.input_identities, block.input_units, directions,
    )
    result = reactive.solve_actions(
        point[4], point[6], [point[0], liquid_feed[1], point[1]], [action], liquid.output_ids[9:18],
    )
    returned = result.state_input_actions[0]
    assert returned.failure is None
    assert returned.output_identities == liquid.output_ids[9:18]
    assert np.all(np.isfinite(returned.values))
    native_solves = reactive.stats["native_solves"]
    cache_hits = reactive.stats["cache_hits"]
    repeated = reactive.solve_actions(
        point[4], point[6], [point[0], liquid_feed[1], point[1]], [action], liquid.output_ids[9:18],
    ).state_input_actions[0]
    np.testing.assert_array_equal(repeated.values, returned.values)
    assert reactive.stats["native_solves"] == native_solves
    assert reactive.stats["cache_hits"] == cache_hits + 1

    vapor = assembly["vapor"]
    vapor_inputs = np.r_[point[5], point[6], vapor_feed]
    _, _, vapor_block, _, _ = vapor._state(vapor_inputs)
    vn = len(vapor_block.input_identities)
    vd = tuple(tuple(float(i == row) for i in range(vn)) for row in (0, 1))
    vapor_action = equilibrium.EquilibriumStateInputAction(
        "assembly-H-A2", 2, vapor_block.input_identities, vapor_block.input_units, vd, total_enthalpy=True,
    )
    vapor_result = vapor._state(vapor_inputs, actions=(vapor_action,))[4].state_input_actions[0]
    assert vapor_result.failure is None
    assert vapor_result.caloric_failure is None
    assert np.isfinite(vapor_result.total_enthalpy_action_j)


def test_failed_native_a2_action_is_not_cached(assembly, point, case, monkeypatch):
    import mea_absorption_column.Thermodynamics.reactive_bundle as bundle

    calls = []
    failed = SimpleNamespace(failure=object(), caloric_failure=None, values=(None,))
    monkeypatch.setattr(bundle, "_solve_homogeneous_reactive_result",
                        lambda *args, **kwargs: (calls.append(True) or SimpleNamespace(
                            state_input_actions=(failed,)), None))
    reactive = assembly["reactive_liquid"]
    inputs = (point[4], point[6], [point[0], case["physical_inputs"]["liquid_feed_mol_s"][1], point[1]])
    action = object()
    for _ in range(2):
        reactive.solve_actions(*inputs, [action], ["failed-output"])
    assert len(calls) == 2


def test_full_native_node_jacobian_is_19_by_12_and_finite(assembly, point):
    node = assembly["node"]
    state = ca.MX.sym("full_node_inputs", 12)
    expression = ca.vertcat(*node.call([ca.MX(0.0), state], True, False))
    graph = ca.Function("full_node_outer", [state], [ca.jacobian(expression, state)], {"cse": True})
    value = np.r_[point, 1e-4, 0.0, 0.0, 0.05]
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
    balance = assembly["balance"]
    reactive = assembly["reactive_liquid"]
    assert reactive.reuse_states
    assert not reactive.warm_starts
    assert reactive._accepted is None
    first = point.copy()
    second = point.copy()
    second[4] += 0.5
    initial = np.concatenate([np.asarray(item).ravel() for item in balance(first, [0.0, 0.0, 0.0])])
    balance(second, [0.0, 0.0, 0.0])
    before = reactive.stats["native_solves"]
    repeated = np.concatenate([np.asarray(item).ravel() for item in balance(first, [0.0, 0.0, 0.0])])
    np.testing.assert_allclose(initial, repeated, rtol=1e-12, atol=1e-12)
    assert reactive.stats["native_solves"] == before
    assert reactive._accepted is None


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
