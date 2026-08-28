import json
from pathlib import Path

import numpy as np
import pytest

import mea_absorption_column.Transport.Reactive_Film as reactive_film
from mea_absorption_column.Thermodynamics.thermo_models import (
    epcsaft_liquid_transport_state,
)
from mea_absorption_column.Transport.Reactive_Film import (
    FilmThermodynamicState,
    ReactiveFilmDomainError,
    solve_reactive_film,
)


ROOT = Path(__file__).parents[1]


def _linear_thermodynamics(henry_pa_m3_mol, derivative=1.0):
    def evaluate(concentrations, _composition):
        fugacities = np.ones_like(concentrations)
        fugacities[0] = henry_pa_m3_mol * concentrations[0]
        return FilmThermodynamicState(fugacities, derivative)

    return evaluate


@pytest.mark.parametrize("vapor_fugacity", (1.0e4, 5.0e3, 2.5e3))
def test_no_reaction_matches_linear_two_film_solution(monkeypatch, vapor_fugacity):
    bulk = np.array([1.0, 10.0])
    diffusivities = np.array([1.0e-9, 8.0e-10])
    delta = 1.0e-4
    k_g = 1.0e-7
    henry = 5.0e3
    derivative = 1.7
    expected_flux = (
        k_g
        * (vapor_fugacity - henry * bulk[0])
        / (1.0 + k_g * henry * delta / diffusivities[0])
    )

    bvp_calls = 0
    scipy_solve_bvp = reactive_film.solve_bvp

    def counted_solve_bvp(*args, **kwargs):
        nonlocal bvp_calls
        bvp_calls += 1
        assert callable(kwargs.get("bc_jac"))
        interface_jacobian, _ = kwargs["bc_jac"](args[3][:, 0], args[3][:, -1])
        closure_scale = max(
            diffusivities[0] * bulk[0] / delta,
            k_g * abs(vapor_fugacity - henry * bulk[0]),
            1.0e-30,
        )
        assert interface_jacobian[2, 0] == pytest.approx(
            k_g * henry * bulk[0] * derivative / closure_scale
        )
        return scipy_solve_bvp(*args, **kwargs)

    monkeypatch.setattr(reactive_film, "solve_bvp", counted_solve_bvp)
    result = solve_reactive_film(
        bulk_concentrations_mol_m3=bulk,
        diffusivities_m2_s=diffusivities,
        stoichiometry=np.array([-1.0, -2.0]),
        liquid_thermodynamic_state=_linear_thermodynamics(henry, derivative),
        net_rate_mol_m3_s=lambda *_: 0.0,
        vapor_bulk_fugacity_pa=vapor_fugacity,
        gas_transfer_coefficient_mol_m2_s_pa=k_g,
        film_thickness_m=delta,
        co2_index=0,
        mesh_points=11,
    )

    assert result.fluxes_mol_m2_s[0, 0] == pytest.approx(expected_flux, rel=5.0e-4)
    assert bvp_calls == 1
    assert result.maximum_interface_residual <= 1.0e-7
    assert result.maximum_conservation_residual <= 1.0e-7


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


def test_issue16_retained_identity_matches_integration_contract():
    identity = json.loads(
        (
            ROOT
            / "analyses/nccc_validation/inputs/issue16_reactive_film_identity.json"
        ).read_text(encoding="utf-8")
    )
    contract = json.loads(
        (ROOT / "integration/epcsaft_contract.json").read_text(encoding="utf-8")
    )

    assert identity["engine"]["commit"] == contract["final_identity"]["engine_commit"]
    assert (
        identity["engine"]["wheel_sha256"] == contract["final_identity"]["wheel_sha256"]
    )
    assert (
        identity["engine"]["core_sha256"] == contract["final_identity"]["core_sha256"]
    )
    assert identity["claim_label"] == "provisional_concept_only"
    assert identity["retained_position_1"]["domain_admitted"] is False
