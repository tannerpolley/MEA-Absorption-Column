import numpy as np

from mea_absorption_column.Properties.Thermophysical_Properties import enthalpy
from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_liquid_temperature
from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_vapor_temperature


def test_liquid_enthalpy_accepts_root_solver_array_temperature():
    values, mixture_value = enthalpy(np.array([333.0]), [0.05, 0.25, 0.70], phase="liquid")

    assert values.shape == (3,)
    assert np.isfinite(mixture_value)


def test_liquid_temperature_inversion_runs_with_numpy_root_inputs():
    target = enthalpy(333.0, [0.05, 0.25, 0.70], phase="liquid")[1]

    recovered = get_liquid_temperature([0.05, 0.25, 0.70], target)

    assert abs(recovered - 333.0) < 1e-6


def test_temperature_inversion_falls_back_to_physical_bounds_for_bad_enthalpy():
    liquid = get_liquid_temperature([0.05, 0.25, 0.70], -1.0e12)
    vapor = get_vapor_temperature([0.05, 0.05, 0.80, 0.10], -1.0e12)

    assert 250.0 <= liquid <= 500.0
    assert 250.0 <= vapor <= 500.0
