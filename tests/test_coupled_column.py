import gc
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from mea_absorption_column.BVP.Coupled_Column import build_column_balance_functions, build_coupled_column_functions
from mea_absorption_column.BVP.Methods.Casadi_Collocation import solve_conservative_collocation
from mea_absorption_column.Thermodynamics.casadi_reactive import (
    ReactiveLiquidCallback, FixedCompositionVaporCallback, build_loading_path_function, build_caloric_flow_function,
)
from mea_absorption_column.Thermodynamics.reactive_bundle import (
    ReactiveLiquid, load_reference_thermochemistry, equilibrium_loading_direction,
)
from mea_absorption_column.Thermodynamics.thermo_models import MEA_THERMODYNAMICS_EPCSAFT_DATASET, ensure_epcsaft_importable


def _column():
    epcsaft = ensure_epcsaft_importable()
    path = MEA_THERMODYNAMICS_EPCSAFT_DATASET
    reference = load_reference_thermochemistry(path / "anchored-reference-thermochemistry.json")
    liquid = ReactiveLiquidCallback("balance_liquid", ReactiveLiquid(path, thermochemistry=reference))
    vapor_path = path.parent / "MEA_neutral_vapor"
    vapor = FixedCompositionVaporCallback("balance_vapor", epcsaft.Parameters.from_json(vapor_path / "parameters.json"),
        load_reference_thermochemistry(vapor_path / "reference-thermochemistry.json", liquid_reference=reference),
        packing_interval=(1e-6, .1))
    fl, fv = [.1529, 1., 7.911], [2.4, 1.2, 15.6, .8]
    packing = [250., .97, .203, .35, .017, .292, .119]
    balance, boundary = build_column_balance_functions(liquid, vapor,
        liquid_feed_mol_s=fl, vapor_feed_mol_s=fv, liquid_temperature_k=313.15,
        vapor_temperature_k=353.15, bottom_pressure_pa=101325., area_m2=.32, packing=packing)
    state = np.array([fl[0], fl[2], fv[0], fv[1], 313.15, 353.15, 101325., .02])
    state[7] -= float(balance(state, [0., 0., 0.])[2])
    return liquid, vapor, balance, boundary, state, np.array(fl), np.array(fv)


def test_native_column_sources_bases_and_pressure_boundary():
    liquid, vapor, balance, boundary, point, fl, fv = _column()
    liquid_masses, vapor_masses = liquid.molar_masses, vapor.molar_masses
    del liquid, vapor
    gc.collect()  # The returned balance must own both native callback lifetimes.
    # Prescribed exchange tests balance assembly, not an interface model.
    transfer = np.array([1e-4, -5e-5, 10.])
    b, r, algebraic, l, v, h = [np.asarray(out).ravel() for out in balance(point, transfer)]
    np.testing.assert_allclose(r[:6], -h[5] * transfer[[0, 1, 0, 1, 2, 2]], rtol=1e-13)
    assert r[6] == -h[6] < 0.
    np.testing.assert_allclose([r[2] - r[0], r[3] - r[1], r[5] - r[4]], 0., atol=1e-12)
    assert abs(algebraic[0]) < 1e-12
    np.testing.assert_allclose(h[:2], [fl.sum() * l[:9].sum() / l[27], fv.sum() / v[4]], rtol=1e-13)
    np.testing.assert_allclose(h[0] * h[2], fl @ np.asarray(liquid_masses[:3]), rtol=1e-10)
    np.testing.assert_allclose(h[1] * h[3], fv @ np.asarray(vapor_masses), rtol=1e-13)
    np.testing.assert_allclose(b[4:6], [fl.sum() * l[-1], fv.sum() * v[-1]], rtol=1e-13)
    # Apparent-flow/true-density mixing would fail this basis check.
    assert abs(l[:9].sum() - 1.) > 1e-4

    bottom, top = ca.MX.sym("bottom", 8), ca.MX.sym("top", 8)
    bc = boundary(bottom, top)
    jac = ca.Function("boundary_derivatives", [bottom, top], [ca.jacobian(bc, bottom), ca.jacobian(bc, top)])
    left, right = (np.asarray(x) for x in jac(point, point))
    assert np.linalg.matrix_rank(np.hstack((left, right))) == 7
    np.testing.assert_array_equal(left[6], [0., 0., 0., 0., 0., 0., 1., 0.])
    np.testing.assert_array_equal(right[6], np.zeros(8))

    state = ca.MX.sym("bulk", 8)
    source = balance(state, transfer)[1]
    graph = ca.Function("source_derivatives", [state], [ca.jacobian(source, state)])
    jacobian = np.asarray(graph(point))
    assert np.all(np.isfinite(jacobian))
    direction = np.array([.1, 1., .2, .1, 1., 1., 1e4, .01])
    step = 1e-3
    difference = (np.asarray(balance(point + step * direction, transfer)[1]).ravel()
                  - np.asarray(balance(point - step * direction, transfer)[1]).ravel()) / (2 * step)
    exact = jacobian @ direction
    scale = np.maximum(abs(r), np.maximum(abs(exact), abs(difference)))
    assert np.max(abs(exact - difference) / scale) < 1e-5


def _species_diffusivities(temperature):
    # Retained central species estimates, with their existing temperature law.
    co2 = .5 * 6.28e-7 * ca.exp(-15230. / (8.314462618 * temperature))
    return ca.vertcat(co2, 8.8e-10, 8.8e-10, 8.4e-10, 6.8e-10, 6.8e-10,
                      6.8e-10, np.sqrt(3.4e-18), np.sqrt(3.4e-18))


def test_native_loading_calorics_and_signed_coupled_interface_values():
    liquid, vapor, balance, _, bulk, fl, fv = _column()
    loading_path = build_loading_path_function(liquid)
    liquid_point = np.r_[bulk[4], bulk[6], fl]
    _, tangent = loading_path(liquid_point, 0.)
    independent = equilibrium_loading_direction(liquid.liquid.solve(
        *liquid_point[:2], fl, state_input_derivatives=True))
    np.testing.assert_allclose(np.asarray(tangent).ravel(), independent, atol=1e-12, rtol=1e-12)
    for callback, point in ((liquid, liquid_point), (vapor, np.r_[bulk[5], bulk[6], fv])):
        caloric = build_caloric_flow_function(callback)
        h, cp, partial_h = caloric(point)
        # Euler identity detects the missing total-flow product/normalization.
        np.testing.assert_allclose(float(ca.dot(partial_h, point[2:])), float(h), rtol=1e-12)
        assert float(cp) > 0.
        if callback is vapor:
            epcsaft = ensure_epcsaft_importable()
            units = epcsaft.unit_registry
            direct = vapor.model.state(T=point[0] * units.kelvin, P=point[1] * units.pascal,
                                       x=fv / fv.sum(), phase="vapor")
            np.testing.assert_allclose(float(cp), direct.cp(vapor.thermochemistry).to("J/mol/K").magnitude, rtol=1e-10)

    node, boundary, diagnostics = build_coupled_column_functions(liquid, vapor,
        species_diffusivities=_species_diffusivities, quadrature_points=3,
        liquid_feed_mol_s=fl, vapor_feed_mol_s=fv, liquid_temperature_k=313.15,
        vapor_temperature_k=353.15, bottom_pressure_pa=101325., area_m2=.32,
        packing=[250., .97, .203, .35, .017, .292, .119])
    signed = []
    for loading in (-.1, 0., .1):
        point = np.r_[bulk, 0., 0., 0., loading]
        d = diagnostics(state=point)
        assert np.all(np.asarray(d["conductances"]) > 0.)
        integral = float(d["integral"])
        assert np.sign(integral) == np.sign(loading)
        signed.append(integral)
        point[8] = integral / float(d["thickness"])
        native = balance(bulk, point[8:11])
        kg = np.asarray(d["gas_coefficients"]).ravel()
        point[9] = kg[1] * (float(native[4][1]) - float(native[3][20]))
        point[10] = float(d["heat_coefficient"]) * (bulk[5] - bulk[4]) + float(ca.dot(point[8:10], d["vapor_partial_enthalpies"][:2]))
        b, r, algebraic = node(0., point)
        np.testing.assert_allclose(np.asarray(algebraic)[[0, 1, 3, 4]], 0., atol=1e-10)
        expected_gas = point[8] - kg[0] * (float(native[4][0]) - float(d["interface_fugacity"]))
        np.testing.assert_allclose(float(algebraic[2]), expected_gas, atol=1e-14)
        np.testing.assert_allclose(float(r[5] - r[4]), 0., atol=1e-12)
        assert boundary(point, point).shape == (7, 1)
    print({"signed_integrals_mol_m_s": signed})
    invalid = point.copy()
    invalid[7] = 0.
    with pytest.raises(RuntimeError, match="Column holdup must satisfy"):
        node(0., invalid)
    reordered = list(liquid.component_ids)
    reordered[3], reordered[4] = reordered[4], reordered[3]
    with pytest.raises(ValueError, match="Column requires"):
        build_column_balance_functions(SimpleNamespace(component_ids=reordered), vapor,
            liquid_feed_mol_s=fl, vapor_feed_mol_s=fv, liquid_temperature_k=313.15,
            vapor_temperature_k=353.15, bottom_pressure_pa=101325., area_m2=.32,
            packing=[250., .97, .203, .35, .017, .292, .119])



def test_full_native_column_node_jacobian():
    liquid, vapor, _, _, bulk, fl, fv = _column()
    node, _, _ = build_coupled_column_functions(liquid, vapor,
        species_diffusivities=_species_diffusivities, quadrature_points=3,
        liquid_feed_mol_s=fl, vapor_feed_mol_s=fv, liquid_temperature_k=313.15,
        vapor_temperature_k=353.15, bottom_pressure_pa=101325., area_m2=.32,
        packing=[250., .97, .203, .35, .017, .292, .119])
    state = ca.MX.sym("full_node_inputs", 12)
    expression = ca.vertcat(*node.call([ca.MX(0.), state], True, False))
    graph = ca.Function("full_node_outer", [state], [ca.jacobian(expression, state)], {"cse": True})
    point = np.r_[bulk, 1e-4, 0., 0., .05]
    jacobian = np.asarray(graph(point))
    assert jacobian.shape == (19, 12) and np.all(np.isfinite(jacobian))
    direction = np.array([.1, 1., .2, .1, 1., 1., 1e4, .01, 1e-4, 0., 0., .05])
    step = 1e-3
    nearby = [np.concatenate([np.asarray(v).ravel() for v in node(0., point + sign * step * direction)])
              for sign in (1, -1)]
    difference = (nearby[0] - nearby[1]) / (2 * step)
    exact = jacobian @ direction
    # Per-row relative comparison retains sensitivity to the tiny film residual.
    error = abs(exact - difference) / np.maximum(1e-12, np.maximum(abs(exact), abs(difference)))
    assert max(error) < 1e-5
    np.testing.assert_array_equal(jacobian[15, [2, 3, 5, 7, 9, 10]], 0.)
    print({"full_node_direction_error": error.tolist()})


@pytest.mark.parametrize("transfer", ([0., 0., 0.], [2e-6, -1e-6, .5]))
def test_prescribed_exchange_column_solves_material_energy_momentum_and_raw_holdup(transfer):
    liquid, vapor, balance, boundary, point, _, _ = _column()
    del liquid, vapor
    gc.collect()
    # Zero/nonzero prescribed-transfer checks use the same native balances
    # with active momentum, not a surrogate thermodynamics or film closure.
    z, state = ca.MX.sym("z"), ca.MX.sym("bulk", 8)
    b, r, algebraic, *_ = balance(state, ca.DM(transfer))
    node = ca.Function("prescribed_exchange_column", [z, state], [b, r, algebraic])
    # Exclude singular void endpoints by a relative machine-precision margin;
    # this is not an empirical holdup limit or a saturation of the raw law.
    holdup_margin = .97 * np.finfo(float).eps
    result = solve_conservative_collocation(node, boundary, [0., 1.], np.tile(point, (2, 1)).T,
        [0., 0., 0., 0., 293.15, 293.15, 1., holdup_margin],
        [np.inf, np.inf, np.inf, np.inf, 393.15, 393.15, 1e7, .97 - holdup_margin],
        state_scale=[.2, 8., 3., 2., 300., 300., 1e5, .1],
        balance_scale=[.2, 8., 3., 2., 100., 100., 100.],
        algebraic_scale=[.1], boundary_scale=[.2, 8., 3., 2., 40., 40., 100.],
        tolerance=1e-7, max_iterations=30)
    assert result["accepted"], (result["status"], result["failure"])
    profile = result["profile"]
    evaluated = [balance(profile[:, k], transfer) for k in range(2)]
    conserved = np.column_stack([np.asarray(v[0]).ravel() for v in evaluated])
    hydraulics = np.column_stack([np.asarray(v[5]).ravel() for v in evaluated])
    integrated_exchange = hydraulics[5].mean() * np.asarray(transfer)
    np.testing.assert_allclose(np.diff(conserved[:4], axis=1).ravel(),
                               -integrated_exchange[[0, 1, 0, 1]], atol=1e-7, rtol=0.)
    np.testing.assert_allclose(np.diff(conserved[4:6], axis=1).ravel(), -integrated_exchange[2], atol=1e-5, rtol=0.)
    energy_drift = float(abs(np.diff(conserved[5] - conserved[4])[0]))
    assert energy_drift < 2e-5
    np.testing.assert_allclose(np.diff(conserved[2:4] - conserved[:2], axis=1), 0., atol=1e-7, rtol=0.)
    if transfer[0] > 0.:
        assert profile[0, 0] > profile[0, 1] and profile[2, 0] > profile[2, 1]
        assert profile[1, 0] < profile[1, 1] and profile[3, 0] < profile[3, 1]
    pressure_loss = profile[6, 0] - profile[6, 1]
    assert pressure_loss > 0.
    np.testing.assert_allclose(pressure_loss, hydraulics[6].mean(), atol=1e-5, rtol=0.)
    assert abs(profile[6, 0] - 101325.) < 1e-5
    assert np.all((hydraulics[4] > 0.) & (hydraulics[4] < .97))
    assert np.all((profile[7] > 0.) & (profile[7] < .97))
    np.testing.assert_allclose(profile[7], hydraulics[4], atol=1e-8, rtol=0.)
    print({"prescribed_transfer": transfer, "pressure_loss_pa": float(pressure_loss), "net_energy_drift_w": energy_drift,
           "integrated_exchange_mol_s_and_w": integrated_exchange.tolist(),
           "raw_holdup": hydraulics[4].tolist(), "temperatures_k": profile[4:6].tolist(),
           "scaled_residual_inf": result["scaled_residual_inf"]})
