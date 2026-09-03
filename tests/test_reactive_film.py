import json
from pathlib import Path

import numpy as np
import pytest

from mea_absorption_column.Transport.Reactive_Film import (
    EquilibriumManifoldState,
    ReactiveFilmSolveError,
    binary_diffusivities_from_species,
    constrained_onsager_mobility,
    solve_equilibrium_manifold_film,
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
    composition = np.array([0.7, 0.1, 0.1, 0.1])
    charges = np.array([0.0, 1.0, -1.0, 0.0])
    stationary = np.array([0.0, 0.0, 0.0, 1.0])
    pairs = binary_diffusivities_from_species([2.0e-9, 8.4e-10, 6.8e-10, 8.8e-10])
    mobility = constrained_onsager_mobility(
        composition,
        50000.0,
        pairs,
        charge_numbers=charges,
        additional_flux_constraints=stationary,
    )
    force = np.array([0.2, -0.4, 0.1, 0.3])
    flux = -mobility @ force

    assert np.linalg.eigvalsh(mobility).min() >= -1.0e-18
    assert flux.sum() == pytest.approx(0.0, abs=1.0e-18)
    assert charges @ flux == pytest.approx(0.0, abs=1.0e-18)
    assert stationary @ flux == pytest.approx(0.0, abs=1.0e-18)
    assert -(flux @ force) >= 0.0


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

    if vapor_fugacity != 5.0e3:
        with pytest.raises(ReactiveFilmSolveError, match="quadrature did not converge"):
            solve_equilibrium_manifold_film(
                state_at_log_loading=state_at_log_loading,
                species_diffusivities_m2_s=np.full(2, diffusivity),
                co2_component_coefficients=np.array([1.0, 0.0]),
                vapor_bulk_fugacity_pa=vapor_fugacity,
                gas_transfer_coefficient_mol_m2_s_pa=k_g,
                film_thickness_m=delta,
                co2_index=0,
                maximum_quadrature_points=9,
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


def test_issue16_retained_identity_remains_historical():
    identity = json.loads(
        (
            ROOT
            / "analyses/nccc_validation/inputs/issue16_reactive_film_identity.json"
        ).read_text(encoding="utf-8")
    )
    assert identity["claim_label"] == "provisional_concept_only"
    assert identity["retained_position_1"]["domain_admitted"] is False


def test_column_film_campaign_does_not_forward_previous_solver_profile(monkeypatch):
    import importlib.util
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd

    script = ROOT / "analyses/reactive_film_evidence/scripts/run_column_film_comparison.py"
    spec = importlib.util.spec_from_file_location("column_film_comparison", script)
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    calls = []

    class ProbeComplete(Exception):
        pass

    def column(*args, **kwargs):
        calls.append(kwargs["solver_settings"])
        if len(calls) == 2:
            assert "initial_guess_scaled" not in calls[-1]
            assert "return_internal_profile" not in calls[-1]
            raise ProbeComplete
        return {"_profiles": {}, "_raw_solution_scaled": object(), "capture_pct": 20.}

    monkeypatch.setattr(campaign, "run_model", column)
    monkeypatch.setattr(campaign, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(campaign, "_sample_jobs", lambda *args: [None])
    monkeypatch.setattr(campaign, "_film_node", lambda job: {
        "film_conductance_mol_m2_s_Pa": 1., "reactive_bulk_fugacity_Pa": 1.,
    })
    with pytest.raises(ProbeComplete):
        campaign._run_case(
            "probe", pd.DataFrame(index=["probe"]), "mole", ROOT,
            np.array([0.]), 2, 1, .5, 5, .1, 100,
        )
    assert len(calls) == 2
