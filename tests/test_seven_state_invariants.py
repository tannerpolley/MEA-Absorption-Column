"""Empirical enthalpy, smooth absorption, and area effects with Engine fugacity."""
import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from mea_absorption_column.BVP.ABS_Column import abs_column
from mea_absorption_column.config.column import resolve_column_config
from mea_absorption_column.Properties.Thermophysical_Properties import enthalpy
from mea_absorption_column.misc.Convert_Data import convert_data


@pytest.fixture(params=["epcsaft_ionic", "epcsaft_reactive_nine"])
def column_state(request):
    data = pd.read_csv('src/mea_absorption_column/data/C_cases_campaign_inputs.csv', index_col=0)
    inputs, _, _ = convert_data(data, run=data.index.get_loc('3C'), return_metadata=True)
    liquid, vapor, tl, tv, _, height, area, pressure, packing = inputs
    config = resolve_column_config(dict(case=dict(source='C_cases_data', id='3C'),
                                        model=dict(thermo_model=request.param)))
    config = resolve_column_config(config.as_dict())
    assert config.engine.required
    options = dict(thermo_model=config.model.thermo_model, thermal_state_mode='enthalpy',
                   chemical_equilibrium_model='epcsaft_reactive_nine'
                   if request.param == 'epcsaft_reactive_nine' else 'legacy')
    # Nonunit scaling catches missing flow/enthalpy factors in the inversion chain.
    scales = np.array([5., 100., 2., 1., 1e6, 1e6, 1e5])
    parameters = (scales, scales, (liquid[1], vapor[2], vapor[3]), height, area, packing, options)
    state = np.array([liquid[0], liquid[2], vapor[0], vapor[1],
                      sum(liquid)*enthalpy(tl, liquid/sum(liquid))[1],
                      sum(vapor)*enthalpy(tv, vapor/sum(vapor), 'vapor')[1], pressure])/scales
    return state, parameters, (tl, tv)


def test_engine_enthalpy_inversion_chain(column_state):
    state, parameters, temperatures = column_state
    scales, _, fixed, *_, options = parameters
    enthalpy_rhs = abs_column(0., state, parameters)
    temperature_state = state.copy()
    temperature_state[4:6] = np.array(temperatures)/scales[4:6]
    options['thermal_state_mode'] = 'temperature'
    temperature_rhs = abs_column(0., temperature_state, parameters)
    # Equal material rates expose a temperature inversion error through fugacity.
    np.testing.assert_allclose(enthalpy_rhs[[0, 1, 2, 3, 6]],
                               temperature_rhs[[0, 1, 2, 3, 6]], rtol=1e-8, atol=1e-12)

    def energy(values):
        physical = values*scales
        liquid = np.array([physical[0], fixed[0], physical[1]])
        vapor = np.array([physical[2], physical[3], fixed[1], fixed[2]])
        return np.array([sum(liquid)*enthalpy(physical[4], liquid/sum(liquid))[1],
                         sum(vapor)*enthalpy(physical[5], vapor/sum(vapor), 'vapor')[1]])

    step = 1e-6
    derivative = (energy(temperature_state+step*temperature_rhs)
                  - energy(temperature_state-step*temperature_rhs))/(2*step)
    # Independent extensive-energy difference: 1 mW resolves cancellation;
    # relative tolerance is the existing empirical chain-rule check's 1e-7.
    np.testing.assert_allclose(derivative, enthalpy_rhs[4:6]*scales[4:6], rtol=1e-7, atol=1e-3)


@pytest.mark.parametrize('flux_over_smoothing', [-10., -1., 0., 1., 10.])
def test_absorption_cap_derivative_in_column(column_state, monkeypatch, flux_over_smoothing):
    state, parameters, _ = column_state
    scales, _, _, height, _, _, options = parameters
    options['co2_flux_mode'] = 'absorption_only'
    flux = flux_over_smoothing*1e-10
    module = importlib.import_module('mea_absorption_column.BVP.ABS_Column')
    # Prescribed pre-cap flux isolates the cap derivative while real Engine
    # fugacity and the column's material/energy assembly still execute.
    def evaluate(value):
        monkeypatch.setattr(module, 'molar_flux', lambda *args: (value, 0., -value, 0.))
        rhs = abs_column(0., state, parameters)*scales/height
        assert rhs[0] == pytest.approx(rhs[2], abs=1e-24)
        assert rhs[4] == pytest.approx(rhs[5], abs=1e-15)
        assert rhs[2]-1e-10 <= 0.
        return rhs[2]

    step = 1e-13
    derivative = (evaluate(flux+step)-evaluate(flux-step))/(2*step)
    # Logistic derivative of softplus; O((step/smoothing)^2) truncation
    # is <1e-7 absolute, including at the zero-driving-force transition.
    assert derivative == pytest.approx(expit(-flux_over_smoothing), abs=1e-7)


@pytest.mark.parametrize('reference_ratio,expected_factor', [(2., 2.), (10., 3.), (.01, .1)])
def test_gas_velocity_area_scales_transfer_conservatively(column_state, reference_ratio, expected_factor):
    state, parameters, _ = column_state
    scales, _, _, height, _, _, options = parameters
    baseline = abs_column(0., state, parameters)*scales/height
    output, _ = abs_column(0., state, parameters, run_type='saving')
    velocity = output['transport'][4]
    options.update(gas_velocity_area_exponent=1.,
                   gas_velocity_area_reference_m_s=velocity/reference_ratio,
                   gas_velocity_area_bounds=(.1, 3.))
    changed = abs_column(0., state, parameters)*scales/height
    # The existing 1e-10 mol/(s m) source offsets do not scale with area.
    offsets = np.array([1e-10]*4+[0., 0., 0.])
    np.testing.assert_allclose(changed-offsets, expected_factor*(baseline-offsets), rtol=1e-10, atol=1e-10)
    assert changed[0] == pytest.approx(changed[2], abs=1e-12)
    assert changed[1] == pytest.approx(changed[3], abs=1e-12)
    assert changed[4] == pytest.approx(changed[5], abs=1e-9)
