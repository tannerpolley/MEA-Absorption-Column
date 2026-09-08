"""Analytic DAE checks, independent of absorber chemistry or native EOS cost."""
import casadi as ca
import numpy as np
import pytest

from mea_absorption_column.BVP.Methods.Casadi_Collocation import solve_conservative_collocation
from mea_absorption_column.BVP.Methods.Conserved_Reduction import ConservedReduction, solve_reduced_bvp, dop853_dense_derivative


def analytic_case():
    z, u = ca.MX.sym("z"), ca.MX.sym("u", 3)
    node = ca.Function("analytic_node", [z, u],
                       [ca.vertcat(u[0], u[1]**2), ca.vertcat(-u[2], -2*u[2]), u[2]-u[0]])
    left, right = ca.MX.sym("left", 3), ca.MX.sym("right", 3)
    boundary = ca.Function("analytic_boundary", [left, right], [ca.vertcat(left[0]-2, right[1]-3)])
    return node, boundary


def analytic_profile(z):
    c = 2*np.exp(-z)
    return np.array([c, np.sqrt(9+4*(np.exp(-z)-np.exp(-1))), c])


def reduction(node=None, **overrides):
    options = dict(initial_state=[2., 3.2, 2.], lower=[0., .1, 0.], upper=[10., 10., 10.],
                   state_scale=[2., 3., 2.], conserved_scale=[2., 10.], algebraic_scale=[2.],
                   boundary_scale=[2., 3.], tolerance=1e-11, solver_tolerance=1e-13,
                   max_evaluations=30, max_condition=1e8)
    options.update(overrides)
    default_node, boundary = analytic_case()
    return ConservedReduction(default_node if node is None else node, boundary, **options)


def test_implicit_derivative_and_mixed_end_boundary_chain_rule():
    model = reduction()
    point = analytic_profile(.4)
    w = np.array([point[0], point[1]**2])/model.conserved_scale
    value = model.evaluate(.4, w)
    assert model.counts["node_jacobians"] == model.last_local["njev"]
    assert model.counts["node_values"] == model.last_local["nfev"]
    count = model.counts.copy()
    np.testing.assert_array_equal(model.evaluate(.6, w)["state"], value["state"])
    assert model.counts == count  # The actual graph has no explicit height dependence.
    np.testing.assert_allclose(value["state"], point, atol=1e-9)
    # d(R/Sq)/d(q/Sq), with Sq=(2,10).
    np.testing.assert_allclose(value["rhs_jacobian"], [[-1., 0.], [-.4, 0.]], atol=1e-12)
    np.testing.assert_allclose(value["state_jacobian"], [[2., 0.], [0., 5/point[1]], [2., 0.]], atol=1e-12)
    bottom, top = analytic_profile(0.), analytic_profile(1.)
    wa, wb = (np.array([p[0], p[1]**2])/model.conserved_scale for p in (bottom, top))
    residual, ja, jb = model.boundary_values(0., 1., wa, wb)
    np.testing.assert_allclose(residual, 0., atol=1e-10)
    np.testing.assert_allclose(ja, [[1., 0.], [0., 0.]], atol=1e-12)
    np.testing.assert_allclose(jb, [[0., 0.], [0., 5/9]], atol=1e-12)
    z, u = ca.MX.sym("z"), ca.MX.sym("u", 3)
    varying = reduction(ca.Function("height_dependent", [z, u],
        [ca.vertcat(u[0]+z, u[1]**2), ca.vertcat(-u[2], -2*u[2]), u[2]-u[0]]))
    first = varying.evaluate(0., [1., 1.])
    second = varying.evaluate(.4, [1., 1.])
    assert not varying.height_independent
    np.testing.assert_allclose(first["state"][0]-second["state"][0], .4, atol=1e-10)


@pytest.mark.parametrize("method", ["trapezoidal", "central", "shooting", "collocation"])
def test_four_schemes_recover_independent_analytic_dae(method):
    errors = []
    for count in (9, 17):
        result = solve_analytic_case(method, count)
        assert result["accepted"], result.get("failure")
        exact = analytic_profile(result["grid"])
        error = np.max(abs(result["profile"]-exact))
        errors.append(error)
        assert np.max(abs(result["algebraic_residual"])) < 1e-8
        assert np.max(abs(result["boundary_residual"])) < 1e-7
        if method in ("shooting", "collocation"):
            assert result["differential_residual"] is not None
            assert result["scaled_differential_residual_inf"] <= 1e-7
            assert set(result["solver_grid"]).issubset(result["grid"])
            assert set(result["differential_residual_height_m"]).issubset(result["grid"])
            assert len(result["grid"]) > len(result["solver_grid"])
    assert errors[-1] < (0.01 if method == "central" else 0.001)
    if method in ("trapezoidal", "central"):
        assert errors[-1] < errors[0]/2


def solve_analytic_case(method, count):
    """Shared reproduction of this analytic verification, not a physical case factory."""
    grid = np.linspace(0., 1., count)**1.2
    initial = np.tile([1.5, 3.2, 1.5], (count, 1)).T
    if method in ("trapezoidal", "central"):
        return solve_conservative_collocation(*analytic_case(), grid, initial,
            [0., .1, 0.], [10., 10., 10.], state_scale=[2., 3., 2.],
            balance_scale=[2., 10.], algebraic_scale=[2.], boundary_scale=[2., 3.],
            tolerance=1e-9, max_iterations=50, scheme=method,
            boundary_slots=[(0, 0), (1, -1)] if method == "central" else None)
    return solve_reduced_bvp(reduction(), grid, initial, method=method,
        tolerance=1e-7, boundary_tolerance=1e-8, max_nodes=100,
        max_evaluations=40, ivp_rtol=1e-9, ivp_atol=1e-11)


def test_local_singularity_refusal_and_exhausted_budget_remain_failures():
    singular = reduction(initial_state=[2., 0., 2.], lower=[0., -1., 0.])
    with pytest.raises(RuntimeError):
        singular.evaluate(0., [1., 0.])
    limited = reduction(max_evaluations=1)
    with pytest.raises(RuntimeError):
        limited.evaluate(0., [1.5, 1.6])
    assert limited.last_local["accepted"] is False
    assert limited.last_local["state"] is not None
    node, boundary = analytic_case()
    z, u = ca.MX.sym("z"), ca.MX.sym("u", 3)
    b, r, a = node(z, u)
    refused = ca.Function("refusing_node", [z, u],
        [b.attachAssert(u[0] < 1., "manufactured native refusal"), r, a])
    model = reduction(node=refused)
    with pytest.raises(RuntimeError, match="manufactured native refusal"):
        model.evaluate(0., [1., 1.])
    failed = solve_reduced_bvp(model, [0., 1.], np.tile([2., 3.2, 2.], (2, 1)).T,
        method="collocation", tolerance=1e-7, boundary_tolerance=1e-8, max_nodes=10,
        max_evaluations=10, ivp_rtol=1e-9, ivp_atol=1e-11)
    assert not failed["accepted"] and failed["profile"] is None
    assert "manufactured native refusal" in failed["failure"]


def test_collocation_rejects_large_original_differential_defect(monkeypatch):
    import importlib
    module = importlib.import_module("mea_absorption_column.BVP.Methods.Conserved_Reduction")
    original = module.solve_bvp

    def corrupted_derivative(*args, **kwargs):
        solved = original(*args, **kwargs)
        dense = solved.sol
        solved.sol = lambda z, nu=0: dense(z, nu) + (1. if nu == 1 else 0.)
        return solved

    monkeypatch.setattr(module, "solve_bvp", corrupted_derivative)
    result = solve_analytic_case("collocation", 9)
    assert not result["accepted"]
    assert result["profile"] is not None
    assert result["scaled_differential_residual_inf"] > .9
    assert "differential" in result["failure"]


def test_dense_derivative_recovers_polynomial_and_rejects_wrong_degree():
    dense = lambda z: np.array([2+3*z+z**7, 5-z**3])
    z, values, derivative, error = dop853_dense_derivative(dense, np.array([0., .4, 1.]), 2)
    np.testing.assert_allclose(values, dense(z), atol=1e-13)
    np.testing.assert_allclose(derivative, [3+7*z**6, -3*z**2], atol=1e-12)
    assert error < 1e-14
    with pytest.raises(RuntimeError, match="dense reconstruction failed"):
        dop853_dense_derivative(lambda z: np.array([z**8]), np.array([0., 1.]), 1)
