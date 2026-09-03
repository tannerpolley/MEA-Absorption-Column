from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad, solve_bvp
from scipy.optimize import brentq


class ReactiveFilmDomainError(ValueError):
    """The requested film state is outside the numerical or physical domain."""


class ReactiveFilmSolveError(RuntimeError):
    """The film boundary-value problem did not satisfy its numerical checks."""


DEFAULT_ONSAGER_DIFFUSIVITY_METADATA = {
    "policy_id": "estimated_compact_onsager_v1",
    "status": "estimated_compact_policy",
    "relative_uncertainty_factor": 2.0,
    "sensitivity_multipliers": (0.5, 1.0, 2.0),
    "unavailable_species": (
        "H2O",
        "HCO3-",
        "CO3^2-",
        "H3O+",
        "OH-",
    ),
}


@dataclass(frozen=True)
class FilmThermodynamicState:
    fugacities_pa: np.ndarray
    co2_log_fugacity_derivative: float
    log_composition_basis: np.ndarray | None = None
    chemical_potential_derivatives_over_rt: np.ndarray | None = None


@dataclass(frozen=True)
class ReactiveFilmResult:
    coordinate_m: np.ndarray
    concentrations_mol_m3: np.ndarray
    compositions: np.ndarray
    fluxes_mol_m2_s: np.ndarray
    liquid_species_fugacity_pa: np.ndarray
    net_rate_mol_m3_s: np.ndarray
    maximum_interface_residual: float
    maximum_conservation_residual: float
    maximum_invariant_source_residual: float
    maximum_electroneutrality_residual: float
    maximum_zero_current_residual: float
    solver_message: str
    transport_rank: int = 0
    transport_nullity: int = 0
    minimum_mobility_eigenvalue: float = float("nan")
    maximum_mobility_symmetry_residual: float = float("nan")
    diffusivity_policy_id: str = ""
    diffusivity_uncertainty_factor: float = float("nan")


def binary_diffusivities_from_species(
    species_diffusivities_m2_s,
    source_metadata: Mapping[str, object] | None = None,
):
    """Build a compact symmetric pair closure from species diffusivity anchors."""

    metadata = (
        DEFAULT_ONSAGER_DIFFUSIVITY_METADATA
        if source_metadata is None
        else source_metadata
    )
    if metadata.get("status") != "estimated_compact_policy":
        raise ReactiveFilmDomainError(
            "Onsager diffusivities require an explicit estimated compact policy"
        )
    if (
        not math.isfinite(float(metadata.get("relative_uncertainty_factor", float("nan"))))
        or float(metadata["relative_uncertainty_factor"]) <= 1.0
        or tuple(metadata.get("sensitivity_multipliers", ())) != (0.5, 1.0, 2.0)
    ):
        raise ReactiveFilmDomainError(
            "Onsager diffusivity policy must declare the bounded 0.5/1/2 sensitivity"
        )
    values = np.asarray(species_diffusivities_m2_s, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ReactiveFilmDomainError(
            "species diffusivity estimates must be a positive finite 1-D array"
        )
    pairs = 2.0 * values[:, None] * values[None, :] / (values[:, None] + values[None, :])
    np.fill_diagonal(pairs, 0.0)
    return pairs


def constrained_onsager_mobility(
    composition,
    total_concentration_mol_m3: float,
    binary_diffusivities_m2_s,
    charge_numbers=None,
):
    """Return the pair-friction mobility constrained to zero molar flux/current."""

    x = np.asarray(composition, dtype=float)
    pairs = np.asarray(binary_diffusivities_m2_s, dtype=float)
    if (
        x.ndim != 1 or x.size < 2 or np.any(~np.isfinite(x)) or np.any(x <= 0.0)
        or abs(float(x.sum()) - 1.0) > 1.0e-12
    ):
        raise ReactiveFilmDomainError("Onsager composition must be positive, finite, and normalized")
    if (
        pairs.shape != (x.size, x.size) or np.any(~np.isfinite(pairs))
        or not np.allclose(pairs, pairs.T, rtol=1.0e-12, atol=0.0)
        or np.any(pairs[np.triu_indices(x.size, 1)] <= 0.0)
    ):
        raise ReactiveFilmDomainError("binary diffusivities must be a finite symmetric matrix with positive pairs")
    if not np.isfinite(total_concentration_mol_m3) or total_concentration_mol_m3 <= 0.0:
        raise ReactiveFilmDomainError("total concentration must be positive and finite")
    weights = float(total_concentration_mol_m3) * x[:, None] * x[None, :] * pairs
    mobility = np.diag(weights.sum(axis=1)) - weights
    if charge_numbers is not None:
        charges = np.asarray(charge_numbers, dtype=float)
        if charges.shape != x.shape or np.any(~np.isfinite(charges)):
            raise ReactiveFilmDomainError("charge_numbers must have one finite value per species")
        electrical_direction = mobility @ charges
        denominator = float(charges @ electrical_direction)
        if denominator > np.finfo(float).eps * max(float(np.trace(mobility)), 1.0):
            mobility -= np.outer(electrical_direction, electrical_direction) / denominator
    return 0.5 * (mobility + mobility.T)


def solve_reactive_film(
    *,
    bulk_concentrations_mol_m3,
    diffusivities_m2_s,
    stoichiometry,
    liquid_thermodynamic_state: Callable[
        [np.ndarray, np.ndarray], FilmThermodynamicState
    ],
    net_rate_mol_m3_s: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray | float
    ],
    vapor_bulk_fugacity_pa: float,
    gas_transfer_coefficient_mol_m2_s_pa: float,
    film_thickness_m: float,
    co2_index: int,
    conservation_matrix=None,
    charge_numbers=None,
    mesh_points: int = 21,
    initial_flux_factor: float = 1.0,
    reaction_continuation_steps: int = 1,
    solver_tolerance: float = 1.0e-8,
    transport_model: str = "onsager",
    diffusivity_metadata: Mapping[str, object] | None = None,
) -> ReactiveFilmResult:
    """Solve one isothermal reactive film with an explicit transport law.

    Concentrations are scaled by their bulk values and fluxes by
    ``D_i C_i,bulk / delta``. In the Onsager molar-average frame the CO2
    carrier flux is balanced by the constrained counterflux; the explicit
    effective-Fick comparison retains zero non-CO2 interfacial flux. The
    supplied fugacity and rate callbacks own the thermodynamic and kinetic
    bases.
    """

    transport_model = str(transport_model).lower()
    if transport_model not in {"onsager", "effective_fick"}:
        raise ReactiveFilmDomainError(
            "transport_model must be 'onsager' or the explicit 'effective_fick' comparison"
        )
    use_onsager = transport_model == "onsager"
    transport_metadata = (
        DEFAULT_ONSAGER_DIFFUSIVITY_METADATA
        if diffusivity_metadata is None
        else diffusivity_metadata
    )

    bulk = np.asarray(bulk_concentrations_mol_m3, dtype=float)
    diffusivities = np.asarray(diffusivities_m2_s, dtype=float)
    nu = np.asarray(stoichiometry, dtype=float)
    if nu.ndim == 1:
        nu = nu[:, None]
    if (
        bulk.ndim != 1
        or bulk.size < 2
        or diffusivities.shape != bulk.shape
        or nu.ndim != 2
        or nu.shape[0] != bulk.size
        or nu.shape[1] < 1
    ):
        raise ReactiveFilmDomainError(
            "bulk concentrations and diffusivities must be equal 1-D arrays and "
            "stoichiometry must have one row per species"
        )
    if not np.all(np.isfinite(bulk)) or np.any(bulk <= 0.0):
        raise ReactiveFilmDomainError("bulk concentrations must be positive and finite")
    if not np.all(np.isfinite(diffusivities)) or np.any(diffusivities <= 0.0):
        raise ReactiveFilmDomainError("diffusivities must be positive and finite")
    if not np.all(np.isfinite(nu)):
        raise ReactiveFilmDomainError("stoichiometry must be finite")
    if not 0 <= int(co2_index) < bulk.size:
        raise ReactiveFilmDomainError("co2_index is outside the species array")
    if not np.isfinite(vapor_bulk_fugacity_pa) or vapor_bulk_fugacity_pa < 0.0:
        raise ReactiveFilmDomainError(
            "vapor bulk fugacity must be nonnegative and finite"
        )
    positive = {
        "gas transfer coefficient": gas_transfer_coefficient_mol_m2_s_pa,
        "film thickness": film_thickness_m,
        "solver tolerance": solver_tolerance,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in positive.values()):
        raise ReactiveFilmDomainError(
            f"{', '.join(positive)} must be positive and finite"
        )
    if mesh_points < 5:
        raise ReactiveFilmDomainError("mesh_points must be at least 5")
    if reaction_continuation_steps < 1:
        raise ReactiveFilmDomainError("reaction_continuation_steps must be at least 1")
    if not np.isfinite(initial_flux_factor) or initial_flux_factor <= 0.0:
        raise ReactiveFilmDomainError("initial_flux_factor must be positive and finite")

    invariants = np.empty((0, bulk.size), dtype=float)
    if conservation_matrix is not None:
        invariants = np.asarray(conservation_matrix, dtype=float)
        if (
            invariants.ndim != 2
            or invariants.shape[1] != bulk.size
            or not np.all(np.isfinite(invariants))
        ):
            raise ReactiveFilmDomainError(
                "conservation_matrix must have one column per species"
            )

    charges = np.zeros(bulk.size, dtype=float)
    dependent_index = None
    if charge_numbers is not None:
        charges = np.asarray(charge_numbers, dtype=float)
        if charges.shape != bulk.shape or not np.all(np.isfinite(charges)):
            raise ReactiveFilmDomainError(
                "charge_numbers must have one finite value per species"
            )
        charge_scale = max(float(np.dot(np.abs(charges), bulk)), 1.0)
        if abs(float(np.dot(charges, bulk))) / charge_scale > 1.0e-12:
            raise ReactiveFilmDomainError("bulk concentrations must be electroneutral")
        if np.max(np.abs(charges @ nu)) > 1.0e-12:
            raise ReactiveFilmDomainError("every finite reaction must conserve charge")
        charged = np.flatnonzero(charges)
        if charged.size < 2:
            raise ReactiveFilmDomainError(
                "charged films require at least two charged species"
            )
        if not use_onsager and not np.allclose(
            diffusivities[charged], diffusivities[charged[0]], rtol=1.0e-12, atol=0.0
        ):
            raise ReactiveFilmDomainError(
                "effective-Fick electroneutral closure requires equal charged-species diffusivities"
            )
        dependent_index = int(
            charged[np.argmax(np.abs(charges[charged] * bulk[charged]))]
        )

    n_species = bulk.size
    independent = np.asarray(
        [index for index in range(n_species) if index != dependent_index], dtype=int
    )
    n_independent = independent.size
    co2_variable = int(np.flatnonzero(independent == co2_index)[0])
    delta = float(film_thickness_m)
    flux_scale = diffusivities * bulk / delta

    constraint_rows = [np.ones(n_species)]
    if charge_numbers is not None and np.any(charges):
        constraint_rows.append(charges.copy())
    constraint_matrix = np.asarray(constraint_rows, dtype=float)
    constraint_gram = constraint_matrix @ constraint_matrix.T
    if np.linalg.matrix_rank(constraint_gram) != constraint_matrix.shape[0]:
        raise ReactiveFilmDomainError("molar-average and current constraints are rank deficient")

    def project_fluxes(physical_fluxes: np.ndarray) -> np.ndarray:
        multipliers = np.linalg.solve(
            constraint_gram, constraint_matrix @ physical_fluxes
        )
        return physical_fluxes - constraint_matrix.T @ multipliers

    flux_projection = np.eye(n_species) - constraint_matrix.T @ np.linalg.solve(
        constraint_gram, constraint_matrix
    )
    projected_flux_coordinate_matrix = (
        flux_projection[:, independent] * flux_scale[independent][None, :]
    )
    flux_gauge_variable = n_independent - 1
    flux_free_variables = np.delete(np.arange(n_independent), flux_gauge_variable)
    flux_physical_rows = independent[:-1]
    flux_derivative_matrix = projected_flux_coordinate_matrix[
        np.ix_(flux_physical_rows, flux_free_variables)
    ]
    if np.linalg.matrix_rank(flux_derivative_matrix) != flux_derivative_matrix.shape[0]:
        raise ReactiveFilmDomainError("Onsager flux coordinate system is rank deficient")

    def expand_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        columns = values.shape[1]
        ratios = np.ones((n_species, columns), dtype=float)
        ratios[independent] = values[:n_independent]
        physical_fluxes = np.zeros_like(ratios)
        physical_fluxes[independent] = (
            values[n_independent:] * flux_scale[independent, None]
        )
        if dependent_index is not None:
            ratios[dependent_index] = -np.sum(
                charges[independent, None]
                * bulk[independent, None]
                * ratios[independent],
                axis=0,
            ) / (charges[dependent_index] * bulk[dependent_index])
            physical_fluxes[dependent_index] = (
                -np.sum(
                    charges[independent, None] * physical_fluxes[independent], axis=0
                )
                / charges[dependent_index]
            )
        if use_onsager:
            physical_fluxes = flux_projection @ physical_fluxes
        return ratios, physical_fluxes / flux_scale[:, None]

    def evaluate(concentration_ratios: np.ndarray):
        if np.any(~np.isfinite(concentration_ratios)) or np.any(
            concentration_ratios <= 0.0
        ):
            raise ReactiveFilmDomainError(
                "film concentrations left the positive finite domain"
            )
        concentrations = bulk[:, None] * concentration_ratios
        compositions = concentrations / np.sum(concentrations, axis=0)
        states = [
            liquid_thermodynamic_state(
                concentrations[:, column], compositions[:, column]
            )
            for column in range(concentrations.shape[1])
        ]
        fugacities = np.column_stack(
            [np.asarray(state.fugacities_pa, dtype=float) for state in states]
        )
        if (
            fugacities.shape != concentrations.shape
            or np.any(~np.isfinite(fugacities))
            or np.any(fugacities <= 0.0)
        ):
            raise ReactiveFilmDomainError(
                "liquid species fugacities must remain positive and finite"
            )
        rate_columns = [
            np.atleast_1d(
                np.asarray(
                    net_rate_mol_m3_s(
                        concentrations[:, column],
                        compositions[:, column],
                        fugacities[:, column],
                    ),
                    dtype=float,
                )
            )
            for column in range(concentrations.shape[1])
        ]
        rates = np.column_stack(rate_columns)
        if rates.shape != (nu.shape[1], concentrations.shape[1]):
            raise ReactiveFilmDomainError(
                "net reaction rate must return one value per stoichiometric column"
            )
        if np.any(~np.isfinite(rates)):
            raise ReactiveFilmDomainError("net reaction rate must remain finite")
        if use_onsager:
            for state in states:
                basis = getattr(state, "log_composition_basis", None)
                derivatives = getattr(state, "chemical_potential_derivatives_over_rt", None)
                if basis is None or derivatives is None:
                    raise ReactiveFilmDomainError(
                        "Onsager transport requires exact thermodynamic composition derivatives"
                    )
        return concentrations, compositions, fugacities, rates, states

    def co2_trial_concentrations(log_ratio: float) -> np.ndarray:
        trial = bulk.copy()
        trial[co2_index] *= np.exp(log_ratio)
        if dependent_index is not None:
            trial[dependent_index] = -np.sum(
                charges[np.arange(n_species) != dependent_index]
                * trial[np.arange(n_species) != dependent_index]
            ) / charges[dependent_index]
        if np.any(~np.isfinite(trial)) or np.any(trial <= 0.0):
            raise ReactiveFilmDomainError(
                "CO2 fugacity bracket left the positive electroneutral domain"
            )
        return trial

    reaction_scale = 1.0
    recovery_used = False

    transport_rank = n_independent
    transport_nullity = 0
    minimum_mobility_eigenvalue = float("inf")
    maximum_mobility_symmetry_residual = 0.0

    def onsager_log_derivative(
        concentration_ratios: np.ndarray,
        scaled_fluxes: np.ndarray,
        state: FilmThermodynamicState,
    ) -> np.ndarray:
        nonlocal transport_rank, transport_nullity
        nonlocal minimum_mobility_eigenvalue, maximum_mobility_symmetry_residual
        ratios = concentration_ratios[:, 0]
        concentrations = bulk * ratios
        composition = concentrations / float(np.sum(concentrations))
        binary = binary_diffusivities_from_species(diffusivities, transport_metadata)
        mobility = constrained_onsager_mobility(
            composition,
            float(np.sum(concentrations)),
            binary,
            charge_numbers=charges if charge_numbers is not None else None,
        )
        maximum_mobility_symmetry_residual = max(
            maximum_mobility_symmetry_residual,
            float(np.max(np.abs(mobility - mobility.T))),
        )
        minimum_mobility_eigenvalue = min(
            minimum_mobility_eigenvalue,
            float(np.min(np.linalg.eigvalsh(mobility))),
        )
        basis = np.asarray(state.log_composition_basis, dtype=float)
        derivatives = np.asarray(state.chemical_potential_derivatives_over_rt, dtype=float)
        if basis.shape[0] != n_species or derivatives.shape[0] != n_species:
            raise ReactiveFilmDomainError("exact thermodynamic tangent has the wrong species dimension")
        tangent_columns = []
        for index in independent:
            dlogc = np.zeros(n_species, dtype=float)
            dlogc[index] = 1.0
            if dependent_index is not None:
                dconcentration = -charges[index] * concentrations[index] / charges[dependent_index]
                dlogc[dependent_index] = dconcentration / concentrations[dependent_index]
            dlogx = dlogc - float(composition @ dlogc)
            q, r = np.linalg.qr(basis, mode="reduced")
            if np.linalg.matrix_rank(r) != r.shape[0]:
                raise ReactiveFilmDomainError("exact thermodynamic composition basis is rank deficient")
            coordinates = np.linalg.solve(r, q.T @ dlogx)
            if np.max(np.abs(basis @ coordinates - dlogx)) > 1.0e-9:
                raise ReactiveFilmDomainError("exact thermodynamic derivative basis is inconsistent")
            tangent_columns.append(-mobility @ derivatives @ coordinates / delta)
        tangent = np.asarray(tangent_columns, dtype=float).T
        tangent_independent = tangent[independent]
        transport_rank = int(np.linalg.matrix_rank(tangent_independent))
        transport_nullity = int(n_independent - transport_rank)
        if transport_nullity != 1:
            raise ReactiveFilmDomainError(
                f"Onsager transport operator expected one gauge nullspace, got rank={transport_rank}"
            )
        gram = tangent_independent.T @ tangent_independent
        _, singular_values, right_vectors = np.linalg.svd(
            tangent_independent, full_matrices=True
        )
        if singular_values[-1] > 1.0e-10 * singular_values[0]:
            raise ReactiveFilmDomainError("Onsager transport nullspace is unresolved")
        gauge = right_vectors[-1]
        kkt = np.block([[gram, gauge[:, None]], [gauge[None, :], np.zeros((1, 1))]])
        rhs = np.r_[
            tangent_independent.T
            @ (scaled_fluxes[independent, 0] * flux_scale[independent]),
            0.0,
        ]
        if np.linalg.matrix_rank(kkt) != kkt.shape[0]:
            raise ReactiveFilmDomainError("Onsager KKT system is rank deficient")
        solution = np.linalg.solve(kkt, rhs)[:-1]
        requested_flux = scaled_fluxes[independent, 0] * flux_scale[independent]
        residual = tangent_independent @ solution - requested_flux
        residual_scale = max(
            float(np.max(np.abs(requested_flux))),
            1.0e-8 * float(np.max(flux_scale)),
        )
        if np.max(np.abs(residual)) > 1.0e-7 * residual_scale:
            raise ReactiveFilmDomainError("Onsager KKT solve is incompatible with the supplied flux")
        dlogr = np.zeros(n_species, dtype=float)
        dlogr[independent] = solution
        if dependent_index is not None:
            dlogr[dependent_index] = (
                -np.sum(charges[independent] * concentrations[independent] * solution)
                / (charges[dependent_index] * concentrations[dependent_index])
            )
        return ratios * dlogr

    def onsager_derivative(
        concentration_ratios: np.ndarray,
        scaled_fluxes: np.ndarray,
        states: list[FilmThermodynamicState],
    ) -> np.ndarray:
        derivatives = []
        for column, state in enumerate(states):
            derivative = onsager_log_derivative(
                concentration_ratios[:, column:column + 1],
                scaled_fluxes[:, column:column + 1],
                state,
            )
            derivatives.append(derivative)
        return np.column_stack(derivatives)

    def equations(_coordinate: np.ndarray, values: np.ndarray) -> np.ndarray:
        concentration_ratios, scaled_fluxes = expand_values(values)
        _, _, _, rates, states = evaluate(concentration_ratios)
        sources = nu @ rates
        if use_onsager:
            source_constraint_residual = constraint_matrix @ sources
            source_scale = max(float(np.max(np.abs(sources))), 1.0e-30)
            if np.max(np.abs(source_constraint_residual)) > 1.0e-10 * source_scale:
                raise ReactiveFilmDomainError(
                    "Onsager reaction source violates the flux constraints"
                )
            flux_derivative = np.zeros(
                (n_independent, values.shape[1]), dtype=float
            )
            for column in range(values.shape[1]):
                flux_derivative[flux_free_variables, column] = np.linalg.solve(
                    flux_derivative_matrix,
                    reaction_scale * delta * sources[flux_physical_rows, column],
                )
        else:
            flux_derivative = (
                reaction_scale
                * delta
                * sources[independent]
                / flux_scale[independent, None]
            )
        concentration_derivative = (
            onsager_derivative(concentration_ratios, scaled_fluxes, states)
            if use_onsager
            else -scaled_fluxes[independent]
        )
        return np.vstack(
            (
                concentration_derivative[independent] if use_onsager else concentration_derivative,
                flux_derivative,
            )
        )

    bulk_composition = bulk / np.sum(bulk)
    bulk_fugacity = float(
        liquid_thermodynamic_state(bulk, bulk_composition).fugacities_pa[co2_index]
    )

    def phase_residual(log_ratio: float) -> float:
        interface_concentrations = co2_trial_concentrations(log_ratio)
        interface_composition = interface_concentrations / np.sum(
            interface_concentrations
        )
        return float(
            liquid_thermodynamic_state(
                interface_concentrations, interface_composition
            ).fugacities_pa[co2_index]
            - float(vapor_bulk_fugacity_pa)
        )

    direction = np.sign(float(vapor_bulk_fugacity_pa) - bulk_fugacity)
    equilibrium_log_ratio = 0.0
    if direction:
        at_bulk = phase_residual(0.0)
        previous_log, previous_residual = 0.0, at_bulk
        bracket = None
        bracket_error = None
        for magnitude in np.linspace(0.25, 12.0, 48):
            candidate_log = float(direction * magnitude)
            try:
                candidate_residual = phase_residual(candidate_log)
            except Exception as error:
                bracket_error = error
                break
            if previous_residual * candidate_residual <= 0.0:
                bracket = (
                    min(previous_log, candidate_log),
                    max(previous_log, candidate_log),
                )
                break
            previous_log, previous_residual = candidate_log, candidate_residual
        if bracket is None:
            if bracket_error is not None:
                raise bracket_error
            raise ReactiveFilmSolveError(
                "could not bracket the interfacial fugacity-equilibrium state"
            )
        equilibrium_log_ratio = float(brentq(phase_residual, *bracket, xtol=1.0e-12))

    continuation = np.linspace(0.0, 1.0, int(reaction_continuation_steps) + 1) ** 2

    def physical_film_residual(log_ratio: float) -> float:
        trial_concentrations = co2_trial_concentrations(log_ratio)
        trial_composition = trial_concentrations / np.sum(trial_concentrations)
        trial_fugacity = float(
            liquid_thermodynamic_state(
                trial_concentrations, trial_composition
            ).fugacities_pa[co2_index]
        )
        liquid_flux = (np.exp(log_ratio) - 1.0) * flux_scale[co2_index]
        gas_flux = float(gas_transfer_coefficient_mol_m2_s_pa) * (
            float(vapor_bulk_fugacity_pa) - trial_fugacity
        )
        return float(liquid_flux - gas_flux)

    physical_log_ratio = 0.0
    if direction:
        physical_log_ratio = float(
            brentq(
                physical_film_residual,
                min(0.0, equilibrium_log_ratio),
                max(0.0, equilibrium_log_ratio),
                xtol=1.0e-12,
            )
        )

    coordinate = np.linspace(0.0, 1.0, int(mesh_points)) ** 3
    concentration_ratio = np.exp(physical_log_ratio)
    interface_concentrations = co2_trial_concentrations(physical_log_ratio)
    interface_composition = interface_concentrations / np.sum(interface_concentrations)
    interface_state = liquid_thermodynamic_state(
        interface_concentrations, interface_composition
    )
    interface_rates = np.atleast_1d(
        np.asarray(
            net_rate_mol_m3_s(
                interface_concentrations,
                interface_composition,
                np.asarray(interface_state.fugacities_pa, dtype=float),
            ),
            dtype=float,
        )
    )
    interface_source = nu @ interface_rates
    rate_coefficient = (
        abs(float(interface_source[co2_index])) / interface_concentrations[co2_index]
    )
    if rate_coefficient == 0.0:
        continuation = np.asarray((1.0,))
    closure_scale = max(
        flux_scale[co2_index],
        float(gas_transfer_coefficient_mol_m2_s_pa)
        * abs(float(vapor_bulk_fugacity_pa) - bulk_fugacity),
        1.0e-30,
    )

    if use_onsager and not direction and rate_coefficient == 0.0:
        check_coordinate = np.linspace(0.0, 1.0, max(201, 10 * int(mesh_points)))
        concentrations = np.repeat(bulk[:, None], check_coordinate.size, axis=1)
        compositions = np.repeat(bulk_composition[:, None], check_coordinate.size, axis=1)
        state = liquid_thermodynamic_state(bulk, bulk_composition)
        if (
            getattr(state, "log_composition_basis", None) is None
            or getattr(state, "chemical_potential_derivatives_over_rt", None) is None
        ):
            raise ReactiveFilmDomainError(
                "Onsager transport requires exact thermodynamic composition derivatives"
            )
        fugacities = np.repeat(
            np.asarray(state.fugacities_pa, dtype=float)[:, None],
            check_coordinate.size,
            axis=1,
        )
        mobility = constrained_onsager_mobility(
            bulk_composition,
            float(np.sum(bulk)),
            binary_diffusivities_from_species(diffusivities, transport_metadata),
            charge_numbers=charges if charge_numbers is not None else None,
        )
        rank = int(np.linalg.matrix_rank(mobility))
        return ReactiveFilmResult(
            coordinate_m=check_coordinate * delta,
            concentrations_mol_m3=concentrations,
            compositions=compositions,
            fluxes_mol_m2_s=np.zeros_like(concentrations),
            liquid_species_fugacity_pa=fugacities,
            net_rate_mol_m3_s=np.zeros((nu.shape[1], check_coordinate.size)),
            maximum_interface_residual=0.0,
            maximum_conservation_residual=0.0,
            maximum_invariant_source_residual=0.0,
            maximum_electroneutrality_residual=0.0,
            maximum_zero_current_residual=0.0,
            solver_message="zero-drive equilibrium state (analytical Onsager limit)",
            transport_rank=rank,
            transport_nullity=n_species - rank,
            minimum_mobility_eigenvalue=float(np.min(np.linalg.eigvalsh(mobility))),
            maximum_mobility_symmetry_residual=float(np.max(np.abs(mobility - mobility.T))),
            diffusivity_policy_id=str(transport_metadata["policy_id"]),
            diffusivity_uncertainty_factor=float(
                transport_metadata["relative_uncertainty_factor"]
            ),
        )

    def boundary(interface: np.ndarray, bulk_edge: np.ndarray) -> np.ndarray:
        interface_ratios, interface_fluxes = expand_values(interface[:, None])
        if np.any(~np.isfinite(interface_ratios)) or np.any(interface_ratios <= 0.0):
            raise ReactiveFilmDomainError(
                "film interface left the positive finite domain"
            )
        interface_concentrations = bulk * interface_ratios[:, 0]
        interface_composition = interface_concentrations / np.sum(
            interface_concentrations
        )
        interface_fugacity = float(
            liquid_thermodynamic_state(
                interface_concentrations, interface_composition
            ).fugacities_pa[co2_index]
        )
        liquid_flux = interface_fluxes[co2_index, 0] * flux_scale[co2_index]
        gas_flux = float(gas_transfer_coefficient_mol_m2_s_pa) * (
            float(vapor_bulk_fugacity_pa) - interface_fugacity
        )
        residual = np.empty(2 * n_independent, dtype=float)
        residual[:n_independent] = bulk_edge[:n_independent] - 1.0
        residual[n_independent] = (liquid_flux - gas_flux) / closure_scale
        other_fluxes = np.delete(
            interface_fluxes[independent, 0] * flux_scale[independent], co2_variable
        )
        if use_onsager:
            unit_co2 = np.zeros(n_species, dtype=float)
            unit_co2[co2_index] = 1.0
            projected = project_fluxes(unit_co2)
            if abs(projected[co2_index]) <= np.finfo(float).eps:
                raise ReactiveFilmDomainError("Onsager constraints cannot carry CO2 flux")
            target = projected * (liquid_flux / projected[co2_index])
            residual[n_independent + 1 :] = np.asarray(
                [
                    (other_fluxes[row] - target[index]) / flux_scale[index]
                    for row, index in enumerate(np.delete(independent, co2_variable))
                ]
            )
        else:
            residual[n_independent + 1 :] = np.delete(
                interface[n_independent:], co2_variable
            )
        return residual

    def boundary_jacobian(interface: np.ndarray, _bulk_edge: np.ndarray):
        interface_jacobian = np.zeros(
            (2 * n_independent, 2 * n_independent), dtype=float
        )
        bulk_jacobian = np.zeros_like(interface_jacobian)
        bulk_jacobian[:n_independent, :n_independent] = np.eye(n_independent)

        ratios, _ = expand_values(interface[:, None])
        if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0.0):
            raise ReactiveFilmDomainError(
                "film interface left the positive finite domain"
            )
        concentrations = bulk * ratios[:, 0]
        composition = concentrations / np.sum(concentrations)
        state = liquid_thermodynamic_state(concentrations, composition)
        fugacity = float(state.fugacities_pa[co2_index])
        log_derivative = float(state.co2_log_fugacity_derivative)
        if not math.isfinite(log_derivative):
            raise ReactiveFilmDomainError("CO2 log-fugacity derivative must be finite")
        interface_jacobian[n_independent, co2_variable] = (
            float(gas_transfer_coefficient_mol_m2_s_pa)
            * fugacity
            * log_derivative
            / interface[co2_variable]
            / closure_scale
        )
        interface_jacobian[n_independent, n_independent + co2_variable] = (
            flux_scale[co2_index] / closure_scale
        )
        other_variables = [
            index for index in range(n_independent) if index != co2_variable
        ]
        for row, index in enumerate(other_variables, start=n_independent + 1):
            interface_jacobian[row, n_independent + index] = 1.0
        if use_onsager:
            unit_co2 = np.zeros(n_species, dtype=float)
            unit_co2[co2_index] = 1.0
            projected = project_fluxes(unit_co2)
            if abs(projected[co2_index]) <= np.finfo(float).eps:
                raise ReactiveFilmDomainError("Onsager constraints cannot carry CO2 flux")
            for row, index in enumerate(np.delete(independent, co2_variable), start=n_independent + 1):
                ratio = projected[index] / projected[co2_index]
                interface_jacobian[row, n_independent + co2_variable] = (
                    -ratio * flux_scale[co2_index] / flux_scale[index]
                )
        return interface_jacobian, bulk_jacobian

    def initial_guess(first_scale: float, flux_factor: float) -> np.ndarray:
        guess = np.zeros((2 * n_species, coordinate.size), dtype=float)
        if first_scale == 0.0:
            guess[:n_species] = 1.0
            guess[co2_index] = (
                concentration_ratio + (1.0 - concentration_ratio) * coordinate
            )
            guess[n_species + co2_index] = (
                flux_factor * (concentration_ratio - 1.0) * flux_scale[co2_index]
            )
            if use_onsager:
                guess[n_species:] = np.column_stack(
                    [
                        project_fluxes(guess[n_species:, column])
                        for column in range(coordinate.size)
                    ]
                ) / flux_scale[:, None]
            else:
                guess[n_species:] /= flux_scale[:, None]
            return np.vstack((guess[independent], guess[n_species + independent]))
        hatta = delta * math.sqrt(
            rate_coefficient * first_scale / diffusivities[co2_index]
        )
        reaction_shape = np.exp(-hatta * coordinate)
        concentration_ratios = np.ones((n_species, coordinate.size), dtype=float)
        concentration_ratios[co2_index] += (concentration_ratio - 1.0) * reaction_shape
        co2_flux = (
            flux_factor
            * flux_scale[co2_index]
            * hatta
            * (concentration_ratio - 1.0)
            * reaction_shape
        )
        physical_fluxes = np.zeros((n_species, coordinate.size), dtype=float)
        physical_fluxes[co2_index] = co2_flux
        for index in range(n_species):
            if index == co2_index:
                continue
            if abs(float(interface_source[co2_index])) > 1.0e-30:
                source_ratio = -float(interface_source[index]) / float(
                    interface_source[co2_index]
                )
                physical_fluxes[index] = source_ratio * (co2_flux[0] - co2_flux)
                reverse_integral = -cumulative_trapezoid(
                    physical_fluxes[index, ::-1], coordinate[::-1], initial=0.0
                )[::-1]
                candidate = 1.0 + delta * reverse_integral / (
                    diffusivities[index] * bulk[index]
                )
                if np.all(candidate > 0.0):
                    concentration_ratios[index] = candidate
        guess[:n_species] = concentration_ratios
        guess[n_species:] = physical_fluxes / flux_scale[:, None]
        if use_onsager:
            guess[n_species:] = np.column_stack(
                [project_fluxes(physical_fluxes[:, column]) for column in range(coordinate.size)]
            ) / flux_scale[:, None]
        return np.vstack((guess[independent], guess[n_species + independent]))

    alternate = np.linspace(0.0, 1.0, max(6, int(reaction_continuation_steps) + 1)) ** 2
    schedules = (
        (continuation, float(initial_flux_factor), False),
        (alternate, float(initial_flux_factor), False),
        (alternate, 1.0, True),
    )
    solution = None
    failure = None
    for scales, flux_factor, recovered in schedules:
        solution = None
        guess = initial_guess(float(scales[0]), flux_factor)
        for scale in scales:
            reaction_scale = float(scale)
            solution = solve_bvp(
                equations,
                boundary,
                coordinate if solution is None else solution.x,
                guess if solution is None else solution.y,
                tol=float(solver_tolerance),
                max_nodes=20000,
                bc_jac=boundary_jacobian,
            )
            if not solution.success:
                failure = (
                    f"reactive film solve failed at reaction scale {reaction_scale:g}: "
                    f"{solution.message}"
                )
                break
        if solution.success:
            recovery_used = recovery_used or recovered
            break
    if solution is None or not solution.success:
        raise ReactiveFilmSolveError(str(failure))
    closure_residual = float(
        boundary(solution.y[:, 0], solution.y[:, -1])[n_independent]
    )

    check_coordinate = np.linspace(0.0, 1.0, max(201, 10 * int(mesh_points)))
    check_values = solution.sol(check_coordinate)
    check_ratios, check_scaled_fluxes = expand_values(check_values)
    if np.any(~np.isfinite(check_ratios)) or np.any(check_ratios <= 0.0):
        raise ReactiveFilmDomainError(
            "film concentrations left the positive finite domain"
        )
    concentrations, compositions, fugacities, rates, _ = evaluate(check_ratios)
    fluxes = check_scaled_fluxes * flux_scale[:, None]
    integrated_rates = np.asarray(
        [
            quad(
                lambda value, index=index: float(
                    evaluate(expand_values(solution.sol(value)[:, None])[0])[3][
                        index, 0
                    ]
                ),
                0.0,
                1.0,
                epsabs=1.0e-8,
                epsrel=1.0e-10,
                limit=500,
            )[0]
            for index in range(nu.shape[1])
        ]
    )
    integrated_source = delta * (nu @ integrated_rates)
    endpoint_values = solution.sol(np.asarray((0.0, 1.0)))
    _, endpoint_scaled_fluxes = expand_values(endpoint_values)
    conservation_fluxes = endpoint_scaled_fluxes * flux_scale[:, None]
    flux_change = conservation_fluxes[:, -1] - conservation_fluxes[:, 0]
    conservation_scale = np.maximum.reduce(
        (
            np.abs(flux_change),
            np.abs(integrated_source),
            np.full(n_species, float(np.max(flux_scale))),
        )
    )
    conservation_residual = np.max(
        np.abs(flux_change - integrated_source) / conservation_scale
    )
    invariant_residual = 0.0
    if invariants.size:
        source = nu @ rates
        invariant_source = invariants @ source
        source_scale = max(float(np.max(np.abs(source))), 1.0e-30)
        invariant_residual = float(np.max(np.abs(invariant_source)) / source_scale)

    electroneutrality_residual = float(
        np.max(np.abs(charges @ concentrations))
        / max(float(np.max(np.sum(concentrations, axis=0))), 1.0)
    )
    zero_current_residual = float(
        np.max(np.abs(charges @ fluxes))
        / max(float(np.max(np.sum(np.abs(fluxes), axis=0))), 1.0e-30)
    )
    if electroneutrality_residual > 1.0e-12 or zero_current_residual > 1.0e-12:
        raise ReactiveFilmSolveError(
            "film charge/current closure exceeded the 1e-12 acceptance tolerance"
        )

    return ReactiveFilmResult(
        coordinate_m=check_coordinate * delta,
        concentrations_mol_m3=concentrations,
        compositions=compositions,
        fluxes_mol_m2_s=fluxes,
        liquid_species_fugacity_pa=fugacities,
        net_rate_mol_m3_s=rates,
        maximum_interface_residual=float(
            max(
                np.max(np.abs(boundary(solution.y[:, 0], solution.y[:, -1]))),
                abs(closure_residual),
            )
        ),
        maximum_conservation_residual=float(conservation_residual),
        maximum_invariant_source_residual=invariant_residual,
        maximum_electroneutrality_residual=electroneutrality_residual,
        maximum_zero_current_residual=zero_current_residual,
        solver_message=str(solution.message)
        + ("; canonical initialization recovery used" if recovery_used else ""),
        transport_rank=transport_rank,
        transport_nullity=transport_nullity,
        minimum_mobility_eigenvalue=(
            minimum_mobility_eigenvalue
            if np.isfinite(minimum_mobility_eigenvalue)
            else float("nan")
        ),
        maximum_mobility_symmetry_residual=maximum_mobility_symmetry_residual,
        diffusivity_policy_id=str(transport_metadata["policy_id"]),
        diffusivity_uncertainty_factor=float(
            transport_metadata["relative_uncertainty_factor"]
        ),
    )
