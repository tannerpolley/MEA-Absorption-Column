import json
from pathlib import Path

import numpy as np
import pytest

import mea_absorption_column.Transport.Reactive_Film as reactive_film
from mea_absorption_column.Thermodynamics.thermo_models import (
    epcsaft_liquid_transport_state,
)
from mea_absorption_column.Transport.Reactive_Film import (
    EquilibriumManifoldState,
    FilmThermodynamicState,
    ReactiveFilmDomainError,
    binary_diffusivities_from_species,
    constrained_onsager_mobility,
    solve_equilibrium_manifold_film,
    solve_reactive_film,
)


ROOT = Path(__file__).parents[1]


def test_constrained_onsager_mobility_is_psd_and_recovers_binary_fick_limit():
    composition = np.array([0.3, 0.7])
    concentration = 1000.0
    diffusivity = 2.0e-9
    pairs = binary_diffusivities_from_species([diffusivity, diffusivity])
    mobility = constrained_onsager_mobility(composition, concentration, pairs)
    ideal_log_tangent = np.array([[1.0], [-composition[0] / composition[1]]])

    flux_per_log_gradient = -mobility @ ideal_log_tangent

    assert mobility == pytest.approx(mobility.T)
    assert np.linalg.eigvalsh(mobility).min() >= -1.0e-18
    assert mobility.sum(axis=0) == pytest.approx(0.0, abs=1.0e-18)
    assert flux_per_log_gradient[:, 0] == pytest.approx(
        [-concentration * diffusivity * composition[0],
         concentration * diffusivity * composition[0]]
    )


def test_constrained_onsager_mobility_enforces_zero_current_without_equal_ions():
    composition = np.array([0.8, 0.1, 0.1])
    charges = np.array([0.0, 1.0, -1.0])
    pairs = binary_diffusivities_from_species([2.0e-9, 8.4e-10, 6.8e-10])
    mobility = constrained_onsager_mobility(
        composition, 50000.0, pairs, charge_numbers=charges
    )
    force = np.array([0.2, -0.4, 0.1])
    flux = -mobility @ force

    assert np.linalg.eigvalsh(mobility).min() >= -1.0e-18
    assert flux.sum() == pytest.approx(0.0, abs=1.0e-18)
    assert charges @ flux == pytest.approx(0.0, abs=1.0e-18)
    assert -(flux @ force) >= 0.0


def _linear_thermodynamics(henry_pa_m3_mol, derivative=1.0):
    def evaluate(concentrations, _composition):
        fugacities = np.ones_like(concentrations)
        fugacities[0] = henry_pa_m3_mol * concentrations[0]
        return FilmThermodynamicState(fugacities, derivative)

    return evaluate


@pytest.mark.parametrize("vapor_fugacity", (1.0e4, 5.0e3, 2.5e3))
def test_equilibrium_manifold_matches_linear_two_film_solution(vapor_fugacity):
    bulk_fraction = 0.2
    total_concentration = 5.0
    diffusivity = 1.0e-9
    delta = 1.0e-4
    k_g = 1.0e-7
    henry = 5.0e3 / (total_concentration * bulk_fraction)
    expected_flux = (
        k_g
        * (vapor_fugacity - henry * total_concentration * bulk_fraction)
        / (1.0 + k_g * henry * delta / diffusivity)
    )

    def state_at_log_loading(log_loading):
        odds = bulk_fraction / (1.0 - bulk_fraction) * np.exp(log_loading)
        composition = np.array([odds / (1.0 + odds), 1.0 / (1.0 + odds)])
        tangent = np.array([[composition[1]], [-composition[0]]])
        return EquilibriumManifoldState(
            composition=composition,
            total_concentration_mol_m3=total_concentration,
            fugacities_pa=np.array(
                [henry * total_concentration * composition[0], composition[1]]
            ),
            chemical_potentials_over_rt=np.log(composition),
            log_composition_basis=tangent,
            chemical_potential_derivatives_over_rt=tangent,
        )

    result = solve_equilibrium_manifold_film(
        state_at_log_loading=state_at_log_loading,
        species_diffusivities_m2_s=np.full(2, diffusivity),
        co2_component_coefficients=np.array([1.0, 0.0]),
        vapor_bulk_fugacity_pa=vapor_fugacity,
        gas_transfer_coefficient_mol_m2_s_pa=k_g,
        film_thickness_m=delta,
        co2_index=0,
    )

    assert result.co2_component_flux_mol_m2_s == pytest.approx(
        expected_flux, rel=5.0e-4, abs=1.0e-14
    )
    assert result.minimum_composition > 0.0
    assert result.maximum_interface_residual <= 1.0e-10
    assert result.maximum_component_flux_residual <= 1.0e-12
    assert result.maximum_zero_total_flux_residual <= 1.0e-12
    assert result.minimum_entropy_production_over_r >= -1.0e-12
    assert result.maximum_tangent_directional_error <= 5.0e-4


def test_reactive_film_preserves_stoichiometry_and_direction_under_refinement(
    monkeypatch,
):
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
        liquid_thermodynamic_state=_linear_thermodynamics(5.0e3),
        net_rate_mol_m3_s=lambda concentrations, *_: 0.1 * concentrations[0],
        vapor_bulk_fugacity_pa=1.0e4,
        gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
        film_thickness_m=1.0e-4,
        co2_index=0,
        conservation_matrix=conservation_matrix,
    )

    initial_fluxes = []
    scipy_solve_bvp = reactive_film.solve_bvp

    def capture_initial_flux(*args, **kwargs):
        initial_fluxes.append(float(args[3][4, 0]))
        return scipy_solve_bvp(*args, **kwargs)

    monkeypatch.setattr(reactive_film, "solve_bvp", capture_initial_flux)
    coarse = solve_reactive_film(**kwargs, mesh_points=11, initial_flux_factor=0.5)
    fine_start = len(initial_fluxes)
    fine = solve_reactive_film(**kwargs, mesh_points=22, initial_flux_factor=2.0)
    flux_difference = abs(
        coarse.fluxes_mol_m2_s[0, 0] / fine.fluxes_mol_m2_s[0, 0] - 1.0
    )

    assert fine.fluxes_mol_m2_s[0, 0] > 0.0
    assert initial_fluxes[fine_start] / initial_fluxes[0] == pytest.approx(4.0)
    assert flux_difference <= 5.0e-3
    assert fine.maximum_conservation_residual <= 1.0e-7
    assert fine.maximum_invariant_source_residual <= 1.0e-14


def test_reactive_film_rejects_nonpositive_scientific_inputs():
    with pytest.raises(ReactiveFilmDomainError, match="diffusivities"):
        solve_reactive_film(
            bulk_concentrations_mol_m3=np.array([1.0, 10.0]),
            diffusivities_m2_s=np.array([1.0e-9, 0.0]),
            stoichiometry=np.array([-1.0, -2.0]),
            liquid_thermodynamic_state=_linear_thermodynamics(5.0e3),
            net_rate_mol_m3_s=lambda *_: 0.0,
            vapor_bulk_fugacity_pa=1.0e4,
            gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
            film_thickness_m=1.0e-4,
            co2_index=0,
        )


def test_high_hatta_state_rejects_negative_concentration_branch():
    with pytest.raises(ReactiveFilmDomainError, match="positive finite domain"):
        solve_reactive_film(
            bulk_concentrations_mol_m3=np.array([0.02244, 2491.0]),
            diffusivities_m2_s=np.array([2.15e-9, 1.18e-9]),
            stoichiometry=np.array([-1.0, -2.0]),
            liquid_thermodynamic_state=_linear_thermodynamics(1593.0),
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


@pytest.mark.parametrize(
    ("vapor_fugacity", "expected_direction"),
    ((5.0e3, 0), (1.0e4, 1), (2.5e3, -1)),
)
def test_nine_species_reversible_film_closes_for_both_directions(
    vapor_fugacity, expected_direction
):
    bulk = np.array([1.0, 100.0, 1000.0, 10.0, 7.0, 2.0, 0.5, 1.0, 1.0])
    charges = np.array([0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -2.0, 1.0, -1.0])
    stoichiometry = np.array(
        [
            [-1.0, -1.0, 0.0],
            [-2.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    def thermodynamics(concentrations, _composition):
        fugacities = concentrations.copy()
        fugacities[0] *= 5.0e3
        return FilmThermodynamicState(fugacities, 1.0)

    bulk_fugacities = thermodynamics(bulk, bulk / bulk.sum()).fugacities_pa

    def reversible_rates(_concentrations, _composition, fugacities):
        activity_ratio = fugacities / bulk_fugacities
        return 1.0e-4 * np.array(
            [
                activity_ratio[0] * activity_ratio[1] ** 2
                - activity_ratio[3] * activity_ratio[4],
                activity_ratio[0] * activity_ratio[1] * activity_ratio[2]
                - activity_ratio[4] * activity_ratio[7],
                activity_ratio[0] * activity_ratio[8] - activity_ratio[5],
            ]
        )

    assert reversible_rates(bulk, bulk / bulk.sum(), bulk_fugacities) == pytest.approx(
        0.0
    )
    result = solve_reactive_film(
        bulk_concentrations_mol_m3=bulk,
        diffusivities_m2_s=np.full(9, 1.0e-9),
        stoichiometry=stoichiometry,
        liquid_thermodynamic_state=thermodynamics,
        net_rate_mol_m3_s=reversible_rates,
        vapor_bulk_fugacity_pa=vapor_fugacity,
        gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
        film_thickness_m=1.0e-4,
        co2_index=0,
        charge_numbers=charges,
        mesh_points=11,
        reaction_continuation_steps=3,
    )

    assert np.sign(result.fluxes_mol_m2_s[0, 0]) == expected_direction
    if expected_direction == 0:
        assert abs(result.fluxes_mol_m2_s[0, 0]) <= 1.0e-12
    assert result.net_rate_mol_m3_s.shape[0] == 3
    assert result.maximum_interface_residual <= 1.0e-7
    assert result.maximum_conservation_residual <= 1.0e-7
    assert result.maximum_electroneutrality_residual <= 1.0e-12
    assert result.maximum_zero_current_residual <= 1.0e-12


def test_exact_epcsaft_tangent_closes_through_zero_drive_film():
    bulk = np.array([1.0, 20.0, 70.0, 3.0, 2.0, 0.5, 0.25, 0.5, 0.5])
    charges = np.array([0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -2.0, 1.0, -1.0])
    temperature = 318.15
    pressure = 109500.0
    calls = 0

    def thermodynamics(_concentrations, composition):
        nonlocal calls
        state = epcsaft_liquid_transport_state(temperature, pressure, composition)
        calls += 1
        return FilmThermodynamicState(
            state.fugacities_pa,
            state.fixed_other_concentrations_log_fugacity_derivative(0),
        )

    bulk_state = thermodynamics(bulk, bulk / bulk.sum())
    tangent = epcsaft_liquid_transport_state(
        temperature, pressure, bulk / bulk.sum()
    )
    constrained_hessian = (
        tangent.log_composition_basis.T
        @ np.diag(tangent.composition)
        @ tangent.chemical_potential_derivatives_over_rt
    )
    assert constrained_hessian == pytest.approx(
        constrained_hessian.T, abs=1.0e-12
    )
    assert np.linalg.eigvalsh(constrained_hessian).min() >= -1.0e-12
    result = solve_reactive_film(
        bulk_concentrations_mol_m3=bulk,
        diffusivities_m2_s=np.full(9, 1.0e-9),
        stoichiometry=np.zeros(9),
        liquid_thermodynamic_state=thermodynamics,
        net_rate_mol_m3_s=lambda *_: 0.0,
        vapor_bulk_fugacity_pa=float(bulk_state.fugacities_pa[0]),
        gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
        film_thickness_m=1.0e-4,
        co2_index=0,
        charge_numbers=charges,
        mesh_points=5,
    )

    assert calls > 1
    assert abs(result.fluxes_mol_m2_s[0, 0]) <= 1.0e-12
    assert result.maximum_interface_residual <= 1.0e-7
    assert result.maximum_electroneutrality_residual <= 1.0e-12
    assert result.maximum_zero_current_residual <= 1.0e-12


def test_issue16_retained_identity_remains_historical():
    identity = json.loads(
        (
            ROOT
            / "analyses/nccc_validation/inputs/issue16_reactive_film_identity.json"
        ).read_text(encoding="utf-8")
    )
    assert identity["claim_label"] == "provisional_concept_only"
    assert identity["retained_position_1"]["domain_admitted"] is False
