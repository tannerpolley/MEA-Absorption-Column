import numpy as np
import pytest

from mea_absorption_column.Properties.Thermophysical_Properties import CO2_HEAT_OF_ABSORPTION_J_PER_MOL
from mea_absorption_column.Properties.Thermophysical_Properties import CO2_MOLAR_VOLUME_COEFFS_ML_PER_MOL
from mea_absorption_column.Properties.Thermophysical_Properties import HENRY_LWM_COEFFS
from mea_absorption_column.Properties.Thermophysical_Properties import MEA_MOLAR_VOLUME_INTERACTION_COEFFS_ML_PER_MOL
from mea_absorption_column.Properties.Thermophysical_Properties import density
from mea_absorption_column.Properties.Thermophysical_Properties import enthalpy
from mea_absorption_column.Properties.Thermophysical_Properties import henrys_law
from mea_absorption_column.Properties.Thermophysical_Properties import heat_capacity
from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_liquid_temperature
from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_vapor_temperature


def test_liquid_enthalpy_accepts_root_solver_array_temperature():
    values, mixture_value = enthalpy(np.array([333.0]), [0.05, 0.25, 0.70], phase="liquid")

    assert values.shape == (3,)
    assert np.isfinite(mixture_value)


def test_liquid_co2_enthalpy_uses_idaes_heat_of_absorption_reference():
    values, _ = enthalpy(333.0, [0.05, 0.25, 0.70], phase="liquid")

    assert values[0] == CO2_HEAT_OF_ABSORPTION_J_PER_MOL
    assert values[0] == -84000.0


def test_henry_mixing_uses_idaes_mea_solvent_coefficients():
    assert HENRY_LWM_COEFFS == (1.70981, 0.03972, -4.3e-4, -2.20377)
    assert abs(henrys_law(333.15, [0.03, 0.10, 0.87]) - 6117.905925660748) < 1.0e-9


def test_liquid_density_uses_idaes_molar_volume_coefficients():
    assert CO2_MOLAR_VOLUME_COEFFS_ML_PER_MOL == (10.2074, 207.0, -563.3701)
    assert MEA_MOLAR_VOLUME_INTERACTION_COEFFS_ML_PER_MOL == (-2.2642, 3.0059)
    rho_mol_l, rho_mass_l, volume = density(333.15, [0.03, 0.10, 0.87], 101325.0, phase="liquid")

    assert abs(rho_mol_l - 43951.636206989046) < 1.0e-8
    assert abs(rho_mass_l - 1015.5333207078268) < 1.0e-8
    assert abs(volume[1] - 2.5273699e-5) < 1.0e-12


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


@pytest.mark.parametrize("invert,composition", [
    (get_liquid_temperature, [0.05, 0.25, 0.70]),
    (get_vapor_temperature, [0.05, 0.05, 0.80, 0.10]),
])
def test_temperature_inversion_rejects_unattainable_enthalpy(invert, composition):
    with pytest.raises((ValueError, RuntimeError)):
        invert(composition, -1.0e12)
