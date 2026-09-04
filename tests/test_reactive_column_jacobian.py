"""Independent directional check of the nonisothermal conventional-film RHS."""
import numpy as np
import pandas as pd
import pytest

from mea_absorption_column.BVP.ABS_Column import abs_column
from mea_absorption_column.BVP.reactive_jacobian import ReactiveColumnJacobian
from mea_absorption_column.Thermodynamics.reactive_bundle import (
    DATASET, ReactiveLiquid, parameter_set, reaction_system, homogeneous_reactive_request,
)
from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.Transport.Enhancement_Factor import film_resistance_expression
from mea_absorption_column.Transport.Flux import molar_flux


@pytest.mark.parametrize('rich,kij_scale,reaction_scale', [
    (False, None, None), (True, None, None),
    (True, ('pair/monoethanolamine/water/k_ij', .95), None),
    (False, ('pair/carbon-dioxide/water/k_ij', 1.05), None),
    (False, None, ('R4', .95)), (True, None, ('R5', 1.05)),
])
def test_native_column_direction_and_exact_reuse(rich, kij_scale, reaction_scale):
    data = pd.read_csv('src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv', index_col=0)
    inputs, _, _ = convert_data(data, run=data.index.get_loc('3C'), type='mass', return_metadata=True,
                               vapor_composition_mode='dry_saturated', gas_flow_basis='reported_dry_mass')
    liquid, vapor, tl, tv, _, height, area, pressure, packing = inputs
    if rich:
        liquid = [4.087096190146402, 9.729875260332735, 75.2853477794609]
        tl, pressure = 325., 109500.
    runtime = ReactiveLiquid(DATASET, loading_anchor=.25, reuse_states=True,
                             kij_scale=kij_scale, reaction_scale=reaction_scale)
    options = dict(thermo_model='epcsaft_reactive_nine', chemical_equilibrium_model='epcsaft_reactive_nine',
                   thermal_state_mode='temperature', reactive_liquid=runtime)
    # Nonunit scales exercise the physical-to-column Jacobian mapping.
    scales = np.array([5., 100., 2., 1., 320., 320., 1e5])
    parameters = (scales, scales, (liquid[1], vapor[2], vapor[3]), height, area, packing, options)
    state = np.array([liquid[0], liquid[2], vapor[0], vapor[1], tl, tv, pressure])/scales
    jacobian = ReactiveColumnJacobian(parameters, 'raw')
    jac = jacobian(0., state)
    before = runtime.stats['native_solves']
    first = abs_column(0., state, parameters)
    second = abs_column(0., state.copy(), parameters)
    np.testing.assert_array_equal(first, second)
    assert runtime.stats['native_solves'] == before
    direction = np.array([.03, .02, -.01, .002, .2, -.1, 20.])/scales
    step = 1e-3
    centered = (abs_column(0, state+step*direction, parameters)
                - abs_column(0, state-step*direction, parameters))/(2*step)
    # Resolves the independent centered difference above native root noise,
    # with an absolute scale for the nearly zero pressure/material entries.
    np.testing.assert_allclose(jac@direction, centered, rtol=2e-5, atol=2e-8)
    accepted = runtime._accepted['amounts_mol'].copy()
    with pytest.raises(ValueError):
        runtime.solve(tl, pressure, [-1., liquid[1], liquid[2]])
    with pytest.raises(ValueError, match='outside'):
        runtime.solve(400., pressure, liquid, state_input_derivatives=True)
    np.testing.assert_array_equal(runtime._accepted['amounts_mol'], accepted)


def test_large_loading_jump_uses_declared_path_and_retains_liquid_root():
    runtime = ReactiveLiquid(DATASET, loading_anchor=.25, reuse_states=True,
                             water_per_mea_anchor=7.909,
                             kij_scale=('pair/carbon-dioxide/water/k_ij', 1.05))
    runtime.solve(316.75, 109500., [.25, 1., 7.909], state_input_derivatives=True)
    inputs = (326.2714274736848, 109500.,
              [3.979265359040124, 9.729875260332735, 75.6233916590782])
    before = runtime.stats['warm_starts']
    result = runtime.solve(*inputs, state_input_derivatives=True)
    assert runtime.stats['warm_starts'] == before
    assert result['density_mol_m3'] == pytest.approx(50951.5005892302, rel=1e-8)
    assert np.all(result['amounts_mol'] > 0.)
    # Nearby queries still reuse accepted liquid states.
    runtime.solve(inputs[0]+.01, *inputs[1:], state_input_derivatives=True)
    assert runtime.stats['warm_starts'] == before+1


def test_interaction_screen_changes_one_coefficient_without_changing_chemistry():
    baseline = parameter_set(str(DATASET))
    original = baseline.to_mapping()
    values = lambda doc: {c['identity']: c['value']['magnitude']
                          for p in doc['pairs'] for c in p['coefficients']}
    for identity in ('pair/monoethanolamine/water/k_ij', 'pair/carbon-dioxide/water/k_ij'):
        for factor in (.95, 1.05):
            changed = parameter_set(str(DATASET), (identity, factor))
            expected = values(original)
            expected[identity] *= factor
            assert values(changed.to_mapping()) == expected
            for key in original.keys()-{'sources','pairs'}:
                assert changed.to_mapping()[key] == original[key]
            assert changed.fingerprint != baseline.fingerprint
    assert parameter_set(str(DATASET)).to_mapping() == original
    with pytest.raises(ValueError):
        parameter_set(str(DATASET), ('absent', .95))
    with pytest.raises(ValueError):
        parameter_set(str(DATASET), ('pair/monoethanolamine/water/k_ij', float('nan')))


def test_reaction_multipliers_preserve_temperature_slope_and_other_equilibria():
    baseline = reaction_system(str(DATASET))
    def log_constants(reactions, temperature):
        request = homogeneous_reactive_request(str(DATASET), temperature, 109500., [.3, 1., 8.], reactions=reactions)
        return np.array([row[0] for row in request['reaction_system']['equilibrium_constants']])
    for identity, index in [('R4', 3), ('R5', 4)]:
        for factor in (.95, 1.05):
            changed = reaction_system(str(DATASET), (identity, factor))
            for i, row in enumerate(changed['reactions']):
                if i != index:
                    assert row == baseline['reactions'][i]
            expected = np.zeros(5)
            expected[index] = np.log(factor)
            for temperature in (298.15, 313.15, 348., 393.15):
                # Roundoff-scale tolerance for algebraic log-correlation arithmetic.
                np.testing.assert_allclose(log_constants(changed, temperature)-log_constants(baseline, temperature),
                                           expected, rtol=0., atol=1e-12)
            slope = lambda r: (log_constants(r, 325.01)-log_constants(r, 324.99))/.02
            np.testing.assert_allclose(slope(changed), slope(baseline), rtol=0., atol=1e-11)
    assert reaction_system(str(DATASET)) == baseline
    for invalid in [('R1', 1.05), ('R4', 0.), ('R5', float('nan'))]:
        with pytest.raises(ValueError):
            reaction_system(str(DATASET), invalid)


def test_frozen_bulk_conversion_matches_two_film_interface_balance():
    # Independent two-film algebra: N = kg(fg-fi) = E*kl(Ci-Cb), fi=H*Ci.
    # H is the local secant f_bulk/C_free, in Pa m^3/mol, not df/dC_total.
    free_co2, bulk_fugacity, kl, kg = 2., 2000., 2e-4, 1e-6
    h_bulk = bulk_fugacity/free_co2
    for enhancement in (1., 30., 1e4):
        for gas_fugacity in (1000., 2000., 5000.):
            _, factor = film_resistance_expression(enhancement, kl, kg, h_bulk, 1.)
            vapor, _, liquid, _ = molar_flux(bulk_fugacity, gas_fugacity, 100., 100., kg, kg, 1., factor)
            interface_co2 = (kg*gas_fugacity+enhancement*kl*free_co2)/(enhancement*kl+kg*h_bulk)
            expected = enhancement*kl*(interface_co2-free_co2)
            assert liquid == pytest.approx(expected, rel=1e-11, abs=1e-13)
            assert liquid == -vapor
            assert np.sign(liquid) == np.sign(gas_fugacity-bulk_fugacity)
