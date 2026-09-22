"""Conservation and chain-rule checks for the retained empirical caloric model."""

import numpy as np
import pandas as pd
import pytest

from mea_absorption_column.BVP.ABS_Column import abs_column
from mea_absorption_column.Properties.Thermophysical_Properties import enthalpy
from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.misc.special_functions import f_dHl_dT


@pytest.mark.parametrize("temperature", [298.15, 318.15, 353.15])
def test_liquid_enthalpy_temperature_derivative(temperature):
    x = np.array([0.03, 0.12, 0.85])
    step = 0.001  # Central difference resolves the polynomial without cancellation.
    expected = (enthalpy(temperature + step, x)[1]
                - enthalpy(temperature - step, x)[1]) / (2 * step)
    assert f_dHl_dT(temperature, x) == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("model", ["ideal_henry", "epcsaft_reactive_nine"])
def test_countercurrent_energy_and_temperature_state_equivalence(model):
    data = pd.read_csv("src/mea_absorption_column/data/C_cases_campaign_inputs.csv", index_col=0)
    inputs, _, _ = convert_data(data, run=data.index.get_loc("3C"), return_metadata=True)
    liquid, vapor, tl, tv, _, height, area, pressure, packing = inputs
    options = {"thermo_model": model, "thermal_state_mode": "temperature",
               "chemical_equilibrium_model": model if model == "epcsaft_reactive_nine" else "legacy"}
    parameters = (np.ones(7), np.ones(7), (liquid[1], vapor[2], vapor[3]),
                  height, area, packing, options)
    state = np.array([liquid[0], liquid[2], vapor[0], vapor[1], tl, tv, pressure])

    def enthalpy_flows(values):
        fl = np.array([values[0], liquid[1], values[1]])
        fv = np.array([values[2], values[3], vapor[2], vapor[3]])
        return np.array([sum(fl) * enthalpy(values[4], fl / sum(fl), "liquid")[1],
                         sum(fv) * enthalpy(values[5], fv / sum(fv), "vapor")[1]])

    rhs_temperature = abs_column(0, state, parameters)
    output, labels = abs_column(0, state, parameters, run_type='saving', column_names=True)
    hl = dict(zip(labels['Hl'], output['Hl'], strict=True))
    hv = dict(zip(labels['Hv'], output['Hv'], strict=True))
    assert hl['dHlf_dz'] == -hl['Hl_flux'] == hv['Hv_flux'] == hv['dHvf_dz']
    # Repeated chemistry evaluations can differ at roundoff level.
    assert hl['dTl_dz'] == pytest.approx(rhs_temperature[4], rel=1e-10)
    assert hv['dTv_dz'] == pytest.approx(rhs_temperature[5], rel=1e-10)
    step = 1e-6
    energy_derivative = (enthalpy_flows(state + step * rhs_temperature)
                         - enthalpy_flows(state - step * rhs_temperature)) / (2 * step)
    # Extensive enthalpy central differences lose several digits; 1e-3 W is
    # below 1e-8 relative to the local transfer scale, not a column-fit tolerance.
    assert energy_derivative[1] - energy_derivative[0] == pytest.approx(0, abs=1e-3)
    enthalpy_state = state.copy()
    enthalpy_state[4:6] = enthalpy_flows(state)
    options["thermal_state_mode"] = "enthalpy"
    rhs_enthalpy = abs_column(0, enthalpy_state, parameters)
    assert rhs_enthalpy[5] == pytest.approx(rhs_enthalpy[4], abs=1e-9)
    np.testing.assert_allclose(energy_derivative, rhs_enthalpy[4:6], rtol=1e-7, atol=1e-3)
