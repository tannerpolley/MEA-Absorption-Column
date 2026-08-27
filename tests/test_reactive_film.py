import numpy as np
import pytest

from mea_absorption_column.Transport.Reactive_Film import (
    ReactiveFilmDomainError,
    solve_reactive_film,
)


def _linear_fugacity(henry_pa_m3_mol):
    return lambda concentrations, _composition: henry_pa_m3_mol * concentrations[0]


def test_no_reaction_matches_linear_two_film_solution():
    bulk = np.array([1.0, 10.0])
    diffusivities = np.array([1.0e-9, 8.0e-10])
    delta = 1.0e-4
    k_g = 1.0e-7
    henry = 5.0e3
    vapor_fugacity = 1.0e4
    expected_flux = k_g * (vapor_fugacity - henry * bulk[0]) / (
        1.0 + k_g * henry * delta / diffusivities[0]
    )

    result = solve_reactive_film(
        bulk_concentrations_mol_m3=bulk,
        diffusivities_m2_s=diffusivities,
        stoichiometry=np.array([-1.0, -2.0]),
        liquid_co2_fugacity_pa=_linear_fugacity(henry),
        net_rate_mol_m3_s=lambda *_: 0.0,
        vapor_bulk_fugacity_pa=vapor_fugacity,
        gas_transfer_coefficient_mol_m2_s_pa=k_g,
        film_thickness_m=delta,
        co2_index=0,
        mesh_points=11,
    )

    assert result.fluxes_mol_m2_s[0, 0] == pytest.approx(expected_flux, rel=5.0e-4)
    assert result.maximum_interface_residual <= 1.0e-7
    assert result.maximum_conservation_residual <= 1.0e-7


def test_reactive_film_preserves_stoichiometry_and_direction_under_refinement():
    bulk = np.array([1.0, 1000.0, 5.0, 5.0])
    diffusivities = np.array([1.0e-9, 8.0e-10, 8.0e-10, 8.0e-10])
    stoichiometry = np.array([-1.0, -2.0, 1.0, 1.0])
    conservation_matrix = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )
    kwargs = dict(
        bulk_concentrations_mol_m3=bulk,
        diffusivities_m2_s=diffusivities,
        stoichiometry=stoichiometry,
        liquid_co2_fugacity_pa=_linear_fugacity(5.0e3),
        net_rate_mol_m3_s=lambda concentrations, *_: 0.1 * concentrations[0],
        vapor_bulk_fugacity_pa=1.0e4,
        gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
        film_thickness_m=1.0e-4,
        co2_index=0,
        conservation_matrix=conservation_matrix,
    )

    coarse = solve_reactive_film(**kwargs, mesh_points=11, initial_flux_factor=0.5)
    fine = solve_reactive_film(**kwargs, mesh_points=22, initial_flux_factor=2.0)
    flux_difference = abs(coarse.fluxes_mol_m2_s[0, 0] / fine.fluxes_mol_m2_s[0, 0] - 1.0)

    assert fine.fluxes_mol_m2_s[0, 0] > 0.0
    assert flux_difference <= 5.0e-3
    assert fine.maximum_conservation_residual <= 1.0e-7
    assert fine.maximum_invariant_source_residual <= 1.0e-14


def test_reactive_film_rejects_nonpositive_scientific_inputs():
    with pytest.raises(ReactiveFilmDomainError, match="diffusivities"):
        solve_reactive_film(
            bulk_concentrations_mol_m3=np.array([1.0, 10.0]),
            diffusivities_m2_s=np.array([1.0e-9, 0.0]),
            stoichiometry=np.array([-1.0, -2.0]),
            liquid_co2_fugacity_pa=_linear_fugacity(5.0e3),
            net_rate_mol_m3_s=lambda *_: 0.0,
            vapor_bulk_fugacity_pa=1.0e4,
            gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
            film_thickness_m=1.0e-4,
            co2_index=0,
        )


def test_high_hatta_state_converges_from_broad_initial_flux_guess():
    result = solve_reactive_film(
        bulk_concentrations_mol_m3=np.array([0.02244, 2491.0]),
        diffusivities_m2_s=np.array([2.15e-9, 1.18e-9]),
        stoichiometry=np.array([-1.0, -2.0]),
        liquid_co2_fugacity_pa=_linear_fugacity(1593.0),
        net_rate_mol_m3_s=lambda concentrations, *_: 64506.0 * concentrations[0],
        vapor_bulk_fugacity_pa=3077.0,
        gas_transfer_coefficient_mol_m2_s_pa=2.48e-5,
        film_thickness_m=7.787e-5,
        co2_index=0,
        mesh_points=21,
        initial_flux_factor=2.0,
        reaction_continuation_steps=8,
        solver_tolerance=1.0e-6,
    )

    assert result.fluxes_mol_m2_s[0, 0] > 0.0
    assert result.maximum_interface_residual <= 1.0e-7
    assert result.maximum_conservation_residual <= 1.0e-7
