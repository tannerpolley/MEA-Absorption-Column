from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import casadi as ca
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


class ReactiveFilmDomainError(ValueError):
    """The requested film state is outside the numerical or physical domain."""


class ReactiveFilmSolveError(RuntimeError):
    """The scalar film calculation did not satisfy its numerical checks."""


def interface_balance_residuals(
    transfer, *, liquid_conductance_integral, film_thickness_m,
    vapor_fugacities_pa, interface_co2_fugacity_pa, liquid_water_fugacity_pa,
    gas_coefficients_mol_m2_s_pa, heat_coefficient_w_m2_k,
    vapor_temperature_k, liquid_temperature_k, transfer_enthalpies_j_mol,
):
    """Four algebraic equations for explicit CO2 interface loading + 3 fluxes.

    Transfer is (j_CO2,j_water,total energy flux), positive vapor to liquid.
    The supplied integral is integral_0^lambda_I c_CO2^T L d(mu/RT)/dlambda
    in mol/(m s); it is not a fitted conductance or a finite-difference tangent.
    Caller supplies native interface fugacity, Cp-dependent heat coefficient,
    and CO2/water transfer partial enthalpies on the shared phase reference.
    Water retains the separate gas-film law against bulk liquid water fugacity.
    This implements residual arithmetic only, never missing native derivatives.

    Residual units: mol/(m s), mol/(m² s), mol/(m² s), W/m². The caller must
    retain positivity/domain constraints for conductance, thickness and transfer
    coefficients. No clipping or one-way absorption restriction is applied.
    """
    transfer = ca.reshape(transfer, 3, 1)
    return ca.vertcat(
        film_thickness_m * transfer[0] - liquid_conductance_integral,
        transfer[0] - gas_coefficients_mol_m2_s_pa[0] * (vapor_fugacities_pa[0] - interface_co2_fugacity_pa),
        transfer[1] - gas_coefficients_mol_m2_s_pa[1] * (vapor_fugacities_pa[1] - liquid_water_fugacity_pa),
        transfer[2] - heat_coefficient_w_m2_k * (vapor_temperature_k - liquid_temperature_k)
        - ca.dot(transfer[:2], transfer_enthalpies_j_mol),
    )


@dataclass(frozen=True)
class EquilibriumManifoldState:
    """One homogeneous state and its exact equilibrium loading derivative."""

    composition: np.ndarray
    total_concentration_mol_m3: float
    fugacities_pa: np.ndarray
    chemical_potential_loading_derivative_over_rt: np.ndarray


@dataclass(frozen=True)
class EquilibriumManifoldFilmResult:
    """Fast-equilibrium film result in one conserved loading coordinate."""

    coordinate_m: np.ndarray
    log_loading_coordinate: np.ndarray
    compositions: np.ndarray
    fluxes_mol_m2_s: np.ndarray
    co2_component_flux_mol_m2_s: float
    interface_fugacity_pa: float
    maximum_interface_residual: float
    maximum_component_flux_residual: float
    maximum_stationary_component_flux_residual: float
    maximum_zero_total_flux_residual: float
    maximum_zero_current_residual: float
    minimum_entropy_production_over_r: float
    minimum_mobility_eigenvalue: float
    minimum_composition: float
    quadrature_points: int
    quadrature_relative_change: float
    solver_message: str


def binary_diffusivities_from_species(species_diffusivities_m2_s):
    """Harmonic-mean pairs; symbolic callers enforce positive finite species D."""

    if isinstance(species_diffusivities_m2_s, (ca.MX, ca.SX)):
        values = ca.reshape(species_diffusivities_m2_s, -1, 1)
        if values.numel() < 2:
            raise ReactiveFilmDomainError("species diffusivity estimates require at least two species")
        pairs = 2 * (values @ values.T) / (
            ca.repmat(values, 1, values.numel()) + ca.repmat(values.T, values.numel(), 1))
        return pairs - ca.diag(ca.diag(pairs))
    values = np.asarray(species_diffusivities_m2_s, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ReactiveFilmDomainError(
            "species diffusivity estimates must be a positive finite 1-D array"
        )
    pairs = 2.0 * values[:, None] * values[None, :] / (
        values[:, None] + values[None, :]
    )
    np.fill_diagonal(pairs, 0.0)
    return pairs


def constrained_onsager_mobility(
    composition,
    total_concentration_mol_m3: float,
    binary_diffusivities_m2_s,
    charge_numbers=None,
    additional_flux_constraints=None,
):
    """Return a symmetric mobility satisfying the requested zero-flux modes.

    Pair weights ``c*x_i*x_j*D_ij`` recover ordinary binary Fick diffusion for
    ideal chemical potentials.  Eliminating the electric-potential force by a
    Schur complement imposes zero current and any stationary component fluxes
    without sacrificing symmetry or positive semidefiniteness.
    """

    x = np.asarray(composition, dtype=float)
    pairs = np.asarray(binary_diffusivities_m2_s, dtype=float)
    if (
        x.ndim != 1
        or x.size < 2
        or np.any(~np.isfinite(x))
        or np.any(x <= 0.0)
        or abs(float(x.sum()) - 1.0) > 1.0e-12
    ):
        raise ReactiveFilmDomainError(
            "Onsager composition must be positive, finite, and normalized"
        )
    if (
        pairs.shape != (x.size, x.size)
        or np.any(~np.isfinite(pairs))
        or not np.allclose(pairs, pairs.T, rtol=1.0e-12, atol=0.0)
        or np.any(pairs[np.triu_indices(x.size, 1)] <= 0.0)
    ):
        raise ReactiveFilmDomainError(
            "binary diffusivities must be a finite symmetric matrix with positive pairs"
        )
    if not np.isfinite(total_concentration_mol_m3) or total_concentration_mol_m3 <= 0.0:
        raise ReactiveFilmDomainError(
            "total concentration must be positive and finite"
        )

    mobility = np.asarray(onsager_mobility_expression(
        ca.DM(x), float(total_concentration_mol_m3), ca.DM(pairs),
        charge_numbers, additional_flux_constraints,
    ))
    if np.any(~np.isfinite(mobility)):
        raise ReactiveFilmDomainError("Onsager constraint projection is singular or non-finite")
    return mobility


def onsager_mobility_expression(
    composition, total_concentration_mol_m3, binary_diffusivities_m2_s,
    charge_numbers=None, additional_flux_constraints=None,
):
    """Shared CasADi DM/MX expression for numerical and differentiable transport.

    Symbolic callers must constrain composition to the positive simplex,
    concentration to positive values, and symmetric pair diffusivities to
    positive off-diagonal values. Constant zero-flux modes must be independent
    modulo the automatic zero-total-flux mode; none are silently discarded.
    """
    x = ca.reshape(composition, -1, 1)
    pairs = binary_diffusivities_m2_s
    if x.numel() < 2 or pairs.shape != (x.numel(), x.numel()):
        raise ReactiveFilmDomainError("Onsager composition and pair dimensions disagree")
    constraints = []
    if charge_numbers is not None:
        charges = np.asarray(charge_numbers, dtype=float)
        if charges.shape != (x.numel(),) or np.any(~np.isfinite(charges)):
            raise ReactiveFilmDomainError(
                "charge_numbers must have one finite value per species"
            )
        constraints.append(charges)
    if additional_flux_constraints is not None:
        additional = np.atleast_2d(np.asarray(additional_flux_constraints, dtype=float))
        if additional.shape[1] != x.numel() or np.any(~np.isfinite(additional)):
            raise ReactiveFilmDomainError(
                "additional_flux_constraints must have one finite column per species"
            )
        constraints.extend(additional)
    weights = total_concentration_mol_m3 * (x @ x.T) * pairs
    mobility = ca.diag(ca.sum2(weights)) - weights
    if constraints:
        constraint_matrix = np.asarray(constraints, dtype=float)
        norms = np.linalg.norm(constraint_matrix, axis=1)
        if np.any(norms <= np.finfo(float).eps):
            raise ReactiveFilmDomainError("zero-flux constraints must be nonzero")
        constraint_matrix /= norms[:, None]
        independent = constraint_matrix - constraint_matrix.mean(axis=1, keepdims=True)
        rank_tolerance = np.finfo(float).eps * max(constraint_matrix.shape) * np.linalg.norm(constraint_matrix, 2)
        if np.linalg.matrix_rank(independent, tol=rank_tolerance) != len(constraints):
            raise ReactiveFilmDomainError("zero-flux constraints must be independent modulo total flux")
        directions = mobility @ constraint_matrix.T
        gram = constraint_matrix @ directions
        # Scale both sides, not the model: physical diffusivities make the
        # unscaled Gram entries smaller than the symbolic solver's pivot test.
        scale = ca.trace(gram)
        # Explicit QR also avoids DM's small-matrix default losing accuracy
        # for trace-species constraints; use the same solver in DM and MX.
        mobility -= directions @ ca.solve(gram / scale, directions.T / scale, "qr")
    return 0.5 * (mobility + mobility.T)


def solve_equilibrium_manifold_film(
    *,
    state_at_log_loading: Callable[[float], EquilibriumManifoldState],
    species_diffusivities_m2_s,
    co2_component_coefficients,
    vapor_bulk_fugacity_pa: float,
    gas_transfer_coefficient_mol_m2_s_pa: float,
    film_thickness_m: float,
    co2_index: int,
    charge_numbers=None,
    stationary_component_coefficients=None,
    quadrature_points: int = 9,
    maximum_quadrature_points: int = 65,
    quadrature_tolerance: float = 1.0e-3,
    profile_points: int = 21,
) -> EquilibriumManifoldFilmResult:
    """Solve the fast-reaction film on a one-dimensional equilibrium manifold.

    The log-loading coordinate is zero in the bulk.  Reactions remain at
    homogeneous equilibrium, while the supplied component invariant is
    transported by a symmetric constrained Onsager mobility.  The resulting
    scalar resistance integral and gas-film closure require no spatial BVP.
    """

    diffusivities = np.asarray(species_diffusivities_m2_s, dtype=float)
    component = np.asarray(co2_component_coefficients, dtype=float)
    charges = None if charge_numbers is None else np.asarray(charge_numbers, dtype=float)
    stationary = (
        np.empty((0, diffusivities.size))
        if stationary_component_coefficients is None
        else np.atleast_2d(np.asarray(stationary_component_coefficients, dtype=float))
    )
    if (
        diffusivities.ndim != 1
        or diffusivities.size < 2
        or np.any(~np.isfinite(diffusivities))
        or np.any(diffusivities <= 0.0)
        or component.shape != diffusivities.shape
        or np.any(~np.isfinite(component))
        or not np.any(component)
    ):
        raise ReactiveFilmDomainError(
            "equilibrium-manifold diffusivities and component coefficients are invalid"
        )
    if charges is not None and (
        charges.shape != diffusivities.shape or np.any(~np.isfinite(charges))
    ):
        raise ReactiveFilmDomainError("charge_numbers must have one finite value per species")
    if stationary.shape[1] != diffusivities.size or np.any(~np.isfinite(stationary)):
        raise ReactiveFilmDomainError(
            "stationary_component_coefficients must have one finite column per species"
        )
    if not 0 <= int(co2_index) < diffusivities.size:
        raise ReactiveFilmDomainError("co2_index is outside the species array")
    positive = (
        gas_transfer_coefficient_mol_m2_s_pa,
        film_thickness_m,
        quadrature_tolerance,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive):
        raise ReactiveFilmDomainError("film, transfer, and quadrature scales must be positive")
    if quadrature_points < 5 or quadrature_points % 2 == 0:
        raise ReactiveFilmDomainError("quadrature_points must be odd and at least 5")
    if maximum_quadrature_points < quadrature_points or profile_points < 3:
        raise ReactiveFilmDomainError("quadrature and profile limits are inconsistent")

    pairs = binary_diffusivities_from_species(diffusivities)

    def state(log_loading: float) -> EquilibriumManifoldState:
        raw = state_at_log_loading(float(log_loading))
        composition = np.asarray(raw.composition, dtype=float)
        fugacities = np.asarray(raw.fugacities_pa, dtype=float)
        derivative = np.asarray(raw.chemical_potential_loading_derivative_over_rt, dtype=float)
        if (
            composition.shape != diffusivities.shape
            or fugacities.shape != diffusivities.shape
            or derivative.shape != diffusivities.shape
            or np.any(~np.isfinite(composition))
            or np.any(composition <= 0.0)
            or abs(float(composition.sum()) - 1.0) > 1.0e-10
            or np.any(~np.isfinite(fugacities))
            or np.any(fugacities <= 0.0)
            or np.any(~np.isfinite(derivative))
            or not np.isfinite(raw.total_concentration_mol_m3)
            or raw.total_concentration_mol_m3 <= 0.0
        ):
            raise ReactiveFilmDomainError("invalid equilibrium-manifold state")
        return EquilibriumManifoldState(
            composition=composition,
            total_concentration_mol_m3=float(raw.total_concentration_mol_m3),
            fugacities_pa=fugacities,
            chemical_potential_loading_derivative_over_rt=derivative,
        )

    bulk = state(0.0)
    bulk_fugacity = float(bulk.fugacities_pa[co2_index])
    drive = float(vapor_bulk_fugacity_pa) - bulk_fugacity
    drive_scale = max(abs(float(vapor_bulk_fugacity_pa)), bulk_fugacity, 1.0)
    if abs(drive) <= 1.0e-13 * drive_scale:
        coordinate = np.linspace(0.0, float(film_thickness_m), int(profile_points))
        mobility = constrained_onsager_mobility(
            bulk.composition,
            bulk.total_concentration_mol_m3,
            pairs,
            charge_numbers=charges,
            additional_flux_constraints=stationary,
        )
        return EquilibriumManifoldFilmResult(
            coordinate_m=coordinate,
            log_loading_coordinate=np.zeros_like(coordinate),
            compositions=np.repeat(bulk.composition[:, None], coordinate.size, axis=1),
            fluxes_mol_m2_s=np.zeros((diffusivities.size, coordinate.size)),
            co2_component_flux_mol_m2_s=0.0,
            interface_fugacity_pa=bulk_fugacity,
            maximum_interface_residual=0.0,
            maximum_component_flux_residual=0.0,
            maximum_stationary_component_flux_residual=0.0,
            maximum_zero_total_flux_residual=0.0,
            maximum_zero_current_residual=0.0,
            minimum_entropy_production_over_r=0.0,
            minimum_mobility_eigenvalue=float(np.linalg.eigvalsh(mobility).min()),
            minimum_composition=float(bulk.composition.min()),
            quadrature_points=int(quadrature_points),
            quadrature_relative_change=0.0,
            solver_message="zero-drive equilibrium-manifold limit",
        )

    direction = float(np.sign(drive))

    def transport(item):
        tangent = item.chemical_potential_loading_derivative_over_rt
        mobility = constrained_onsager_mobility(
            item.composition, item.total_concentration_mol_m3, pairs,
            charge_numbers=charges, additional_flux_constraints=stationary,
        )
        conductance = float(component @ mobility @ tangent)
        if not np.isfinite(conductance) or conductance <= 0.0:
            raise ReactiveFilmDomainError(
                "equilibrium-manifold component conductance is not positive"
            )
        return mobility, tangent, conductance

    def grid(count: int, endpoint: float):
        loading = np.linspace(min(0.0, endpoint), max(0.0, endpoint), count)
        states = [state(float(value)) for value in loading]
        compositions = np.column_stack([item.composition for item in states])
        mobilities = []
        tangents = []
        conductance = np.empty(count)
        for index, item in enumerate(states):
            mobility, tangent, value = transport(item)
            mobilities.append(mobility)
            tangents.append(tangent)
            conductance[index] = value
        antiderivative = cumulative_trapezoid(conductance, loading, initial=0.0)
        return (
            loading,
            states,
            compositions,
            mobilities,
            np.column_stack(tangents),
            conductance,
            antiderivative,
        )

    interface_bound = None
    # Advance the same trapezoidal bracketing integral once. Its running sum
    # and preceding ordinate are numerical quadrature state, not a lookup
    # cache of equilibrium results or a previous-solve warm start.
    previous_conductance = transport(bulk)[2]
    bracket_integral = 0.0
    previous_residual = -float(gas_transfer_coefficient_mol_m2_s_pa) * drive
    for step in range(1, 161):
        candidate = direction * 0.025 * step
        candidate_state = state(candidate)
        candidate_conductance = transport(candidate_state)[2]
        bracket_integral += 0.5 * (previous_conductance + candidate_conductance) * direction * 0.025
        previous_conductance = candidate_conductance
        liquid_flux = bracket_integral / float(film_thickness_m)
        gas_flux = float(gas_transfer_coefficient_mol_m2_s_pa) * (
            float(vapor_bulk_fugacity_pa) - float(candidate_state.fugacities_pa[co2_index])
        )
        residual = liquid_flux - gas_flux
        if step >= 4 and previous_residual * residual <= 0.0:
            interface_bound = candidate
            break
        if step >= 4:
            previous_residual = residual
    if interface_bound is None:
        raise ReactiveFilmSolveError("could not bracket the two-film interface flux root")

    previous_flux = None
    relative_change = float("inf")
    count = int(quadrature_points)
    while True:
        data = grid(count, interface_bound)
        loading, states, compositions, mobilities, tangents, conductance, antiderivative = data
        integral = PchipInterpolator(loading, antiderivative)
        fugacity = PchipInterpolator(
            loading, [item.fugacities_pa[co2_index] for item in states]
        )

        def interface_residual(value: float) -> float:
            liquid_flux = -float(integral(0.0) - integral(value)) / float(film_thickness_m)
            gas_flux = float(gas_transfer_coefficient_mol_m2_s_pa) * (
                float(vapor_bulk_fugacity_pa) - float(fugacity(value))
            )
            return liquid_flux - gas_flux

        # The bracketing scan and this quadrature can give different endpoint
        # signs. Refine within the same budget before asking Brent for a root.
        if interface_residual(0.0) * interface_residual(interface_bound) > 0.0:
            if count >= int(maximum_quadrature_points):
                raise ReactiveFilmSolveError(
                    f"Film quadrature could not retain the interface bracket at {count} points"
                )
            count = min(2 * count - 1, int(maximum_quadrature_points))
            continue
        interface_loading = float(
            brentq(interface_residual, min(0.0, interface_bound), max(0.0, interface_bound))
        )
        component_flux = -float(integral(0.0) - integral(interface_loading)) / float(
            film_thickness_m
        )
        if previous_flux is not None:
            relative_change = abs(component_flux - previous_flux) / max(abs(component_flux), 1.0e-30)
            if relative_change <= float(quadrature_tolerance):
                break
        if count >= int(maximum_quadrature_points):
            raise ReactiveFilmSolveError(
                f"Film quadrature did not converge at {count} points: "
                f"relative flux change={relative_change:.6g}"
            )
        previous_flux = component_flux
        count = min(2 * count - 1, int(maximum_quadrature_points))

    path_loading = np.asarray(
        sorted(
            {interface_loading, 0.0}
            | {
                float(value)
                for value in loading
                if min(interface_loading, 0.0) < value < max(interface_loading, 0.0)
            },
            reverse=interface_loading > 0.0,
        )
    )
    path_distance = -(
        np.asarray(integral(path_loading), dtype=float) - float(integral(interface_loading))
    ) / component_flux
    coordinate = np.linspace(0.0, float(film_thickness_m), int(profile_points))
    profile_loading = PchipInterpolator(path_distance, path_loading)(coordinate)
    profile_compositions = np.vstack(
        [PchipInterpolator(loading, row)(profile_loading) for row in compositions]
    )

    flux_grid = np.column_stack(
        [
            mobilities[index] @ tangents[:, index] * component_flux / conductance[index]
            for index in range(count)
        ]
    )
    # A common linear weight preserves every projected linear flux constraint.
    profile_fluxes = np.vstack(
        [np.interp(profile_loading, loading, row) for row in flux_grid]
    )
    component_residual = np.max(np.abs(component @ profile_fluxes - component_flux))
    stationary_residual = (
        0.0
        if stationary.size == 0
        else float(np.max(np.abs(stationary @ profile_fluxes)))
    )
    total_residual = np.max(np.abs(np.sum(profile_fluxes, axis=0)))
    current_residual = (
        0.0 if charges is None else float(np.max(np.abs(charges @ profile_fluxes)))
    )
    entropy = np.asarray(
        [
            component_flux**2
            * float(tangents[:, index] @ mobilities[index] @ tangents[:, index])
            / conductance[index] ** 2
            for index in range(count)
        ]
    )
    interface_fugacity = float(fugacity(interface_loading))
    interface_scale = max(abs(component_flux), 1.0e-30)
    return EquilibriumManifoldFilmResult(
        coordinate_m=coordinate,
        log_loading_coordinate=np.asarray(profile_loading, dtype=float),
        compositions=profile_compositions,
        fluxes_mol_m2_s=profile_fluxes,
        co2_component_flux_mol_m2_s=float(component_flux),
        interface_fugacity_pa=interface_fugacity,
        maximum_interface_residual=abs(interface_residual(interface_loading)) / interface_scale,
        maximum_component_flux_residual=float(component_residual),
        maximum_stationary_component_flux_residual=stationary_residual,
        maximum_zero_total_flux_residual=float(total_residual),
        maximum_zero_current_residual=float(current_residual),
        minimum_entropy_production_over_r=float(entropy.min()),
        minimum_mobility_eigenvalue=float(
            min(np.linalg.eigvalsh(mobility).min() for mobility in mobilities)
        ),
        minimum_composition=float(profile_compositions.min()),
        quadrature_points=int(count),
        quadrature_relative_change=float(relative_change),
        solver_message="fast-equilibrium manifold resistance solved",
    )
