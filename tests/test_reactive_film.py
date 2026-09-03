import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import mea_absorption_column.Transport.Reactive_Film as reactive_film
from mea_absorption_column.Thermodynamics.thermo_models import (
    IONIC_CHARGE_BY_SPECIES,
    IONIC_LIQUID_SPECIES_9,
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    epcsaft_liquid_transport_state,
)
from mea_absorption_column.Thermodynamics.reactive_bundle import validate_reactive_bundle
from mea_absorption_column.Transport.Reactive_Film import (
    FilmThermodynamicState,
    ReactiveFilmDomainError,
    solve_reactive_film,
)


ROOT = Path(__file__).parents[1]


def test_onsager_bvp_uses_exact_tangent_and_explicit_fick_comparison():
    bulk = np.array([1.0, 10.0, 20.0])
    diffusivities = np.array([1.0e-9, 2.0e-9, 3.0e-9])

    def thermodynamics(concentrations, composition):
        basis = np.column_stack(
            (
                [1.0, 0.0, -composition[0] / composition[2]],
                [0.0, 1.0, -composition[1] / composition[2]],
            )
        )
        return FilmThermodynamicState(
            np.array([5.0e3 * concentrations[0], concentrations[1], concentrations[2]]),
            1.0,
            basis,
            basis.copy(),
        )

    kwargs = dict(
        bulk_concentrations_mol_m3=bulk,
        diffusivities_m2_s=diffusivities,
        stoichiometry=np.zeros(3),
        liquid_thermodynamic_state=thermodynamics,
        net_rate_mol_m3_s=lambda *_: 0.0,
        vapor_bulk_fugacity_pa=1.0e4,
        gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
        film_thickness_m=1.0e-4,
        co2_index=0,
        mesh_points=7,
    )
    onsager = solve_reactive_film(**kwargs, transport_model="onsager")
    effective_fick = solve_reactive_film(**kwargs, transport_model="effective_fick")

    assert onsager.fluxes_mol_m2_s[0, 0] > 0.0
    assert onsager.fluxes_mol_m2_s[:, 0].sum() == pytest.approx(0.0, abs=1.0e-18)
    assert onsager.transport_rank == 2
    assert onsager.transport_nullity == 1
    assert onsager.minimum_mobility_eigenvalue >= -1.0e-18
    assert effective_fick.fluxes_mol_m2_s[0, 0] > 0.0
    assert onsager.fluxes_mol_m2_s[0, 0] != pytest.approx(
        effective_fick.fluxes_mol_m2_s[0, 0]
    )


def test_onsager_requires_exact_tangent_in_zero_drive_limit():
    bulk = np.array([1.0, 2.0])

    def incomplete_thermodynamics(concentrations, _composition):
        return FilmThermodynamicState(np.array([1.0e3 * concentrations[0], concentrations[1]]), 1.0)

    with pytest.raises(ReactiveFilmDomainError, match="exact thermodynamic composition derivatives"):
        solve_reactive_film(
            bulk_concentrations_mol_m3=bulk,
            diffusivities_m2_s=np.array([1.0e-9, 2.0e-9]),
            stoichiometry=np.zeros(2),
            liquid_thermodynamic_state=incomplete_thermodynamics,
            net_rate_mol_m3_s=lambda *_: 0.0,
            vapor_bulk_fugacity_pa=1.0e3,
            gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
            film_thickness_m=1.0e-4,
            co2_index=0,
            mesh_points=5,
        )


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
        transport_model="effective_fick",
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
        transport_model="effective_fick",
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
            transport_model="effective_fick",
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
        transport_model="effective_fick",
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
            state.log_composition_basis,
            state.chemical_potential_derivatives_over_rt,
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
        transport_model="onsager",
    )

    assert calls > 1
    assert abs(result.fluxes_mol_m2_s[0, 0]) <= 1.0e-12
    assert result.maximum_interface_residual <= 1.0e-7
    assert result.maximum_electroneutrality_residual <= 1.0e-12
    assert result.maximum_zero_current_residual <= 1.0e-12
    assert result.transport_rank == 7
    assert result.transport_nullity == 2


def test_nine_species_onsager_nonzero_reversible_bounded_tangent_gate():
    temperature = 318.15
    pressure = 109500.0
    bulk = np.array([1.0, 100.0, 1000.0, 10.0, 7.0, 2.0, 0.5, 1.0, 1.0])
    charges = np.asarray(
        [IONIC_CHARGE_BY_SPECIES[name] for name in IONIC_LIQUID_SPECIES_9], dtype=float
    )
    base = bulk / bulk.sum()
    exact = epcsaft_liquid_transport_state(temperature, pressure, base)
    coordinate = np.asarray([0, 1, 4, 5, 6, 7, 8], dtype=int)
    dependent = np.asarray([2, 3], dtype=int)

    def local_basis(composition):
        result = np.zeros((9, 7), dtype=float)
        constraints = np.vstack(
            (composition[dependent], charges[dependent] * composition[dependent])
        )
        for column, index in enumerate(coordinate):
            result[index, column] = 1.0
            result[dependent, column] = np.linalg.solve(
                constraints,
                -np.asarray((composition[index], charges[index] * composition[index])),
            )
        return result

    def local_state(_concentrations, composition):
        composition = np.asarray(composition, dtype=float)
        fugacities = exact.fugacities_pa * np.exp(
            exact.chemical_potential_derivatives_over_rt
            @ np.log(composition[coordinate] / base[coordinate])
        )
        return FilmThermodynamicState(
            fugacities,
            exact.fixed_other_concentrations_log_fugacity_derivative(0),
            local_basis(composition),
            exact.chemical_potential_derivatives_over_rt,
        )

    bulk_fugacities = local_state(bulk, base).fugacities_pa

    exact_errors = []
    for factor in (0.995, 1.005):
        sample = bulk.copy()
        sample[0] *= factor
        sample_composition = sample / sample.sum()
        exact_sample = epcsaft_liquid_transport_state(
            temperature, pressure, sample_composition
        )
        local_sample = local_state(sample, sample_composition)
        exact_errors.append(
            float(np.max(np.abs(np.log(exact_sample.fugacities_pa / local_sample.fugacities_pa))))
        )
    assert max(exact_errors) <= 5.0e-3

    def reversible_rates(_concentrations, _composition, fugacities):
        activity = np.asarray(fugacities) / bulk_fugacities
        raw = np.asarray(
            (
                activity[0] * activity[1] ** 2 - activity[3] * activity[4],
                activity[0] * activity[1] * activity[2] - activity[4] * activity[7],
                activity[0] * activity[8] - activity[5],
            )
        )
        return 1.0e-8 * (raw - np.mean(raw))

    policy = json.loads(
        (ROOT / "analyses/reactive_film_evidence/inputs/onsager_diffusivity_closure_policy.json")
        .read_text(encoding="utf-8")
    )
    stoichiometry = np.asarray(
        (
            (-1, -1, -1), (-2, -1, 0), (0, -1, 0), (1, 0, 0),
            (1, 1, 0), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 0, -1),
        ),
        dtype=float,
    )
    conservation = np.asarray(
        (
            (1, 2, 0, 2, 3, 1, 1, 0, 0),
            (0, 7, 2, 8, 6, 1, 0, 3, 1),
            (0, 1, 0, 1, 1, 0, 0, 0, 0),
            (2, 1, 1, 1, 3, 3, 3, 1, 1),
            (0, 0, 0, 1, -1, -1, -2, 1, -1),
        ),
        dtype=float,
    )
    common = dict(
        bulk_concentrations_mol_m3=bulk,
        diffusivities_m2_s=np.asarray(policy["species_diffusivities_m2_s"], dtype=float),
        diffusivity_metadata=policy,
        stoichiometry=stoichiometry,
        conservation_matrix=conservation,
        charge_numbers=charges,
        liquid_thermodynamic_state=local_state,
        net_rate_mol_m3_s=reversible_rates,
        gas_transfer_coefficient_mol_m2_s_pa=1.0e-7,
        film_thickness_m=1.0e-4,
        co2_index=0,
        solver_tolerance=1.0e-6,
    )
    results = []
    for factor in (1.01, 0.99):
        direction_results = []
        for mesh, start, order in ((5, 0.5, 1), (7, 1.0, 1), (9, 2.0, 2)):
            result = solve_reactive_film(
                **common,
                vapor_bulk_fugacity_pa=bulk_fugacities[0] * factor,
                mesh_points=mesh,
                initial_flux_factor=start,
                reaction_continuation_steps=order,
            )
            assert np.sign(result.fluxes_mol_m2_s[0, 0]) == np.sign(factor - 1.0)
            assert result.maximum_interface_residual <= 1.0e-6
            assert result.maximum_conservation_residual <= 1.0e-7
            assert result.maximum_invariant_source_residual <= 1.0e-10
            assert result.maximum_electroneutrality_residual <= 1.0e-12
            assert result.maximum_zero_current_residual <= 1.0e-12
            assert result.minimum_mobility_eigenvalue >= -1.0e-18
            assert result.diffusivity_policy_id == policy["policy_id"]
            direction_results.append(result)
        currents = np.asarray([result.fluxes_mol_m2_s[0, 0] for result in direction_results])
        j_floor = 1.0e-12
        assert (currents.max() - currents.min()) / max(abs(currents.mean()), j_floor) <= 5.0e-3
        results.extend(direction_results)

    assert all(np.isfinite(result.net_rate_mol_m3_s).all() for result in results)


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
    bundle = validate_reactive_bundle(str(MEA_THERMODYNAMICS_EPCSAFT_DATASET))["bundle"]

    assert identity["engine"]["commit"] == contract["final_identity"]["engine_commit"]
    assert (
        identity["engine"]["wheel_sha256"] == contract["final_identity"]["wheel_sha256"]
    )
    assert (
        identity["engine"]["core_sha256"] == contract["final_identity"]["core_sha256"]
    )
    assert identity["claim_label"] == "provisional_concept_only"
    assert identity["retained_position_1"]["domain_admitted"] is False
    assert identity["bundle"]["bundle_id"] == bundle["bundle_id"]
    assert identity["bundle"]["parameter_document_sha256"] == bundle["parameter_document_sha256"]
    assert identity["bundle"]["state_packet_sha256"] == bundle["state_packet_sha256"]
    assert identity["bundle"]["reaction_system_sha256"] == hashlib.sha256(
        (MEA_THERMODYNAMICS_EPCSAFT_DATASET / "reaction-system.json").read_bytes()
    ).hexdigest()
