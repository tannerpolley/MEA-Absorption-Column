import numpy as np

from mea_absorption_column.Properties.Thermophysical_Properties import enthalpy
from mea_absorption_column.Properties.Thermophysical_Properties import heat_capacity
from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_liquid_temperature
from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_vapor_temperature


def test_liquid_enthalpy_accepts_root_solver_array_temperature():
    values, mixture_value = enthalpy(np.array([333.0]), [0.05, 0.25, 0.70], phase="liquid")

    assert values.shape == (3,)
    assert np.isfinite(mixture_value)


def test_vapor_properties_accept_integer_csv_temperature():
    y = [0.10, 0.05, 0.75, 0.10]

    heat_capacity_values, heat_capacity_mixture = heat_capacity(320, y, phase="vapor")
    enthalpy_values, enthalpy_mixture = enthalpy(320, y, phase="vapor")

    assert np.all(np.isfinite(heat_capacity_values))
    assert np.isfinite(heat_capacity_mixture)
    assert np.all(np.isfinite(enthalpy_values))
    assert np.isfinite(enthalpy_mixture)


def test_liquid_temperature_inversion_runs_with_numpy_root_inputs():
    target = enthalpy(333.0, [0.05, 0.25, 0.70], phase="liquid")[1]

    recovered = get_liquid_temperature([0.05, 0.25, 0.70], target)

    assert abs(recovered - 333.0) < 1e-6


def test_temperature_inversion_falls_back_to_physical_bounds_for_bad_enthalpy():
    liquid = get_liquid_temperature([0.05, 0.25, 0.70], -1.0e12)
    vapor = get_vapor_temperature([0.05, 0.05, 0.80, 0.10], -1.0e12)

    assert 250.0 <= liquid <= 500.0
    assert 250.0 <= vapor <= 500.0


def test_temperature_inversion_returns_bound_for_nonphysical_trial_composition():
    liquid = get_liquid_temperature([0.5, -0.1, 0.6], 1.0e30)

    assert 250.0 <= liquid <= 500.0
