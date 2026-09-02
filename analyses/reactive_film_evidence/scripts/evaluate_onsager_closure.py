from __future__ import annotations

import json
import math

import numpy as np

from mea_absorption_column.Thermodynamics.reactive_bundle import (
    solve_homogeneous_reactive_state,
    validate_reactive_bundle,
)
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    epcsaft_liquid_transport_state,
)
from mea_absorption_column.Transport.Reactive_Film import (
    binary_diffusivities_from_species,
    constrained_onsager_mobility,
)


SPECIES = ("CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-")
CHARGES = np.asarray((0, 0, 0, 1, -1, -1, -2, 1, -1), dtype=float)
FILM_THICKNESS_M = 1.0e-4


def _diffusivity_bounds(temperature_k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    polat_30wt = 6.28e-7 * math.exp(-15230.0 / (8.314462618 * temperature_k))
    central = np.asarray(
        (
            0.5 * polat_30wt,
            8.8e-10,
            8.8e-10,
            8.4e-10,
            6.8e-10,
            6.8e-10,
            6.8e-10,
            math.sqrt(3.4e-10 * 1.0e-8),
            math.sqrt(3.4e-10 * 1.0e-8),
        )
    )
    low = np.asarray((0.4 * polat_30wt, 3.94e-10, 4.4e-10, 3.94e-10, 3.4e-10, 3.4e-10, 3.4e-10, 3.4e-10, 3.4e-10))
    high = np.asarray((0.6 * polat_30wt, 8.8e-10, 1.76e-9, 8.4e-10, 6.8e-10, 1.36e-9, 1.36e-9, 1.0e-8, 1.0e-8))
    labels = [
        "Polat2023 30 wt% self-diffusion correlation times Melnikov2019 loading-0.5 trend",
        "Jerng2022 measured DOSY; Lv2026 probe value supplies lower sensitivity bound",
        "estimated from retained molecular-anchor range",
        "Jerng2022 unresolved MEA/MEAH+ measured DOSY anchor",
        "Jerng2022 measured MEACOO- DOSY anchor",
        "estimated from MEACOO- anchor; no retained species-resolved source",
        "estimated from MEACOO- anchor; no retained species-resolved source",
        "broad estimate; retained source inventory has no H3O+ mobility",
        "broad estimate; retained source inventory has no OH- mobility",
    ]
    return low, central, high, labels


def _transport_case(
    state, total_concentration_mol_m3: float, species_diffusivities: np.ndarray
) -> dict[str, float]:
    pairs = binary_diffusivities_from_species(species_diffusivities)
    mobility = constrained_onsager_mobility(
        state.composition,
        total_concentration_mol_m3,
        pairs,
        charge_numbers=CHARGES,
    )
    coordinate_change = np.zeros(7)
    coordinate_change[0] = 0.01
    chemical_force_gradient = (
        state.chemical_potential_derivatives_over_rt @ coordinate_change
    ) / FILM_THICKNESS_M
    ideal_force_gradient = (
        state.log_composition_basis @ coordinate_change
    ) / FILM_THICKNESS_M
    flux = -mobility @ chemical_force_gradient
    ideal_flux = -mobility @ ideal_force_gradient
    return {
        "co2_flux_mol_m2_s": float(flux[0]),
        "ideal_force_co2_flux_mol_m2_s": float(ideal_flux[0]),
        "thermodynamic_force_delta_percent": float(100.0 * (flux[0] / ideal_flux[0] - 1.0)),
        "minimum_mobility_eigenvalue": float(np.linalg.eigvalsh(mobility).min()),
        "zero_total_flux_residual_mol_m2_s": float(abs(flux.sum())),
        "zero_current_residual_mol_m2_s_charge": float(abs(CHARGES @ flux)),
        "entropy_production_over_r": float(-(flux @ chemical_force_gradient)),
    }


def main() -> None:
    temperature = 313.15
    pressure = 101325.0
    equilibrium = solve_homogeneous_reactive_state(
        str(MEA_THERMODYNAMICS_EPCSAFT_DATASET),
        temperature,
        pressure,
        (0.1529, 1.0, 7.911),
    )
    state = epcsaft_liquid_transport_state(
        temperature, pressure, equilibrium["composition"]
    )
    hessian = (
        state.log_composition_basis.T
        @ np.diag(state.composition)
        @ state.chemical_potential_derivatives_over_rt
    )
    low, central, high, labels = _diffusivity_bounds(temperature)
    reactions = validate_reactive_bundle(str(MEA_THERMODYNAMICS_EPCSAFT_DATASET))[
        "reactions"
    ]["reactions"]
    reaction_matrix = np.asarray(
        [reaction["stoichiometry"] for reaction in reactions], dtype=float
    )
    f1_f2 = np.vstack(
        (reaction_matrix[1] - reaction_matrix[3] - reaction_matrix[4],
         reaction_matrix[1] - reaction_matrix[3])
    )
    coordinate_change = np.zeros(7)
    coordinate_change[0] = 0.01
    affinities = f1_f2 @ state.chemical_potential_derivatives_over_rt @ coordinate_change
    concentrations_kmol_m3 = equilibrium["density_mol_m3"] * equilibrium["composition"] / 1000.0
    k1 = 3.1732e9 * math.exp(-4936.6 / temperature)
    k2 = 1.0882e8 * math.exp(-3900.0 / temperature)
    forward_scales = np.asarray(
        (
            k1 * concentrations_kmol_m3[1] ** 2 * concentrations_kmol_m3[0],
            k2 * concentrations_kmol_m3[2] * concentrations_kmol_m3[1] * concentrations_kmol_m3[0],
        )
    ) * 1000.0
    rates = -forward_scales * np.expm1(affinities)
    diffusion_time_s = FILM_THICKNESS_M**2 / central[0]
    reaction_time_s = (
        equilibrium["density_mol_m3"] * equilibrium["composition"][0]
        / float(np.sum(rates))
    )
    payload = {
        "state": {
            "temperature_K": temperature,
            "pressure_Pa": pressure,
            "composition": equilibrium["composition"].tolist(),
            "density_mol_m3": equilibrium["density_mol_m3"],
            "reaction_affinity_inf_norm": float(
                np.max(np.abs(reaction_matrix @ equilibrium["chemical_potentials_over_rt"]))
            ),
            "thermodynamic_hessian_symmetry_residual": float(np.max(np.abs(hessian - hessian.T))),
            "thermodynamic_hessian_minimum_eigenvalue": float(np.linalg.eigvalsh(hessian).min()),
            "thermodynamic_hessian_condition_number": float(np.linalg.cond(hessian)),
        },
        "species_diffusivity_closure": [
            {
                "species": species,
                "low_m2_s": float(low[index]),
                "central_m2_s": float(central[index]),
                "high_m2_s": float(high[index]),
                "basis": labels[index],
            }
            for index, species in enumerate(SPECIES)
        ],
        "transport_sensitivity": {
            name: _transport_case(state, equilibrium["density_mol_m3"], values)
            for name, values in (("low", low), ("central", central), ("high", high))
        },
        "finite_rate_local_perturbation": {
            "coordinate_change": "1% increase in the provider CO2 log-composition coordinate",
            "F1_F2_affinities_over_rt": affinities.tolist(),
            "F1_F2_rates_mol_m3_s": rates.tolist(),
            "minimum_reaction_entropy_production_over_r": float(np.min(-rates * affinities)),
            "co2_diffusion_time_s": float(diffusion_time_s),
            "co2_reaction_time_s": float(reaction_time_s),
            "local_damkohler_estimate": float(diffusion_time_s / reaction_time_s),
            "source_rate_basis": "Putta2016 concentration forward scale; provider affinity supplies detailed-balance reverse factor",
            "standard_state_limit": "forward prefactor retains Putta concentration basis and is not independently validated",
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
