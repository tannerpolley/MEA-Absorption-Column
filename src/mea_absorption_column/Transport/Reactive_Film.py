from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad, solve_bvp
from scipy.optimize import brentq


class ReactiveFilmDomainError(ValueError):
    """The requested film state is outside the numerical or physical domain."""


class ReactiveFilmSolveError(RuntimeError):
    """The film boundary-value problem did not satisfy its numerical checks."""


@dataclass(frozen=True)
class ReactiveFilmResult:
    coordinate_m: np.ndarray
    concentrations_mol_m3: np.ndarray
    compositions: np.ndarray
    fluxes_mol_m2_s: np.ndarray
    liquid_co2_fugacity_pa: np.ndarray
    net_rate_mol_m3_s: np.ndarray
    maximum_interface_residual: float
    maximum_conservation_residual: float
    maximum_invariant_source_residual: float
    solver_message: str


def solve_reactive_film(
    *,
    bulk_concentrations_mol_m3,
    diffusivities_m2_s,
    stoichiometry,
    liquid_co2_fugacity_pa: Callable[[np.ndarray, np.ndarray], float],
    net_rate_mol_m3_s: Callable[[np.ndarray, np.ndarray, float], float],
    vapor_bulk_fugacity_pa: float,
    gas_transfer_coefficient_mol_m2_s_pa: float,
    film_thickness_m: float,
    co2_index: int,
    conservation_matrix=None,
    mesh_points: int = 21,
    initial_flux_factor: float = 1.0,
    reaction_continuation_steps: int = 1,
    solver_tolerance: float = 1.0e-8,
) -> ReactiveFilmResult:
    """Solve one isothermal effective-Fick reactive film.

    Concentrations are scaled by their bulk values and fluxes by
    ``D_i C_i,bulk / delta``. Only CO2 crosses the mathematical interface;
    every other species has zero interfacial normal flux. The supplied
    fugacity and rate callbacks own the thermodynamic and kinetic bases.
    """

    bulk = np.asarray(bulk_concentrations_mol_m3, dtype=float)
    diffusivities = np.asarray(diffusivities_m2_s, dtype=float)
    nu = np.asarray(stoichiometry, dtype=float)
    if bulk.ndim != 1 or bulk.size < 2 or diffusivities.shape != bulk.shape or nu.shape != bulk.shape:
        raise ReactiveFilmDomainError("bulk concentrations, diffusivities, and stoichiometry must be equal 1-D arrays")
    if not np.all(np.isfinite(bulk)) or np.any(bulk <= 0.0):
        raise ReactiveFilmDomainError("bulk concentrations must be positive and finite")
    if not np.all(np.isfinite(diffusivities)) or np.any(diffusivities <= 0.0):
        raise ReactiveFilmDomainError("diffusivities must be positive and finite")
    if not np.all(np.isfinite(nu)):
        raise ReactiveFilmDomainError("stoichiometry must be finite")
    if not 0 <= int(co2_index) < bulk.size:
        raise ReactiveFilmDomainError("co2_index is outside the species array")
    if not np.isfinite(vapor_bulk_fugacity_pa) or vapor_bulk_fugacity_pa < 0.0:
        raise ReactiveFilmDomainError("vapor bulk fugacity must be nonnegative and finite")
    positive = {
        "gas transfer coefficient": gas_transfer_coefficient_mol_m2_s_pa,
        "film thickness": film_thickness_m,
        "solver tolerance": solver_tolerance,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in positive.values()):
        raise ReactiveFilmDomainError(f"{', '.join(positive)} must be positive and finite")
    if mesh_points < 5:
        raise ReactiveFilmDomainError("mesh_points must be at least 5")
    if reaction_continuation_steps < 1:
        raise ReactiveFilmDomainError("reaction_continuation_steps must be at least 1")
    if not np.isfinite(initial_flux_factor) or initial_flux_factor <= 0.0:
        raise ReactiveFilmDomainError("initial_flux_factor must be positive and finite")

    invariants = np.empty((0, bulk.size), dtype=float)
    if conservation_matrix is not None:
        invariants = np.asarray(conservation_matrix, dtype=float)
        if invariants.ndim != 2 or invariants.shape[1] != bulk.size or not np.all(np.isfinite(invariants)):
            raise ReactiveFilmDomainError("conservation_matrix must have one column per species")

    n_species = bulk.size
    delta = float(film_thickness_m)
    flux_scale = diffusivities * bulk / delta

    def evaluate(concentration_ratios: np.ndarray, *, include_fugacity: bool):
        concentrations = bulk[:, None] * np.maximum(concentration_ratios, 1.0e-30)
        compositions = concentrations / np.sum(concentrations, axis=0)
        if include_fugacity:
            fugacities = np.asarray(
                [
                    liquid_co2_fugacity_pa(concentrations[:, column], compositions[:, column])
                    for column in range(concentrations.shape[1])
                ],
                dtype=float,
            )
        else:
            fugacities = np.full(concentrations.shape[1], np.nan)
        rates = np.asarray(
            [
                net_rate_mol_m3_s(
                    concentrations[:, column], compositions[:, column], fugacities[column]
                )
                for column in range(concentrations.shape[1])
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(concentrations)) or np.any(concentrations <= 0.0):
            raise ReactiveFilmDomainError("film concentrations left the positive finite domain")
        if include_fugacity and (np.any(~np.isfinite(fugacities)) or np.any(fugacities <= 0.0)):
            raise ReactiveFilmDomainError("liquid CO2 fugacity must remain positive and finite")
        if np.any(~np.isfinite(rates)):
            raise ReactiveFilmDomainError("net reaction rate must remain finite")
        return concentrations, compositions, fugacities, rates

    reaction_scale = 1.0
    recovery_used = False

    def equations(_coordinate: np.ndarray, values: np.ndarray) -> np.ndarray:
        concentration_ratios = values[:n_species]
        scaled_fluxes = values[n_species:]
        _, _, _, rates = evaluate(concentration_ratios, include_fugacity=False)
        return np.vstack(
            (
                -scaled_fluxes,
                reaction_scale * delta * nu[:, None] * rates[None, :] / flux_scale[:, None],
            )
        )

    bulk_composition = bulk / np.sum(bulk)
    bulk_fugacity = float(liquid_co2_fugacity_pa(bulk, bulk_composition))

    def phase_residual(log_ratio: float) -> float:
        interface_concentrations = bulk.copy()
        interface_concentrations[co2_index] *= np.exp(log_ratio)
        interface_composition = interface_concentrations / np.sum(interface_concentrations)
        return float(
            liquid_co2_fugacity_pa(interface_concentrations, interface_composition)
            - float(vapor_bulk_fugacity_pa)
        )

    direction = np.sign(float(vapor_bulk_fugacity_pa) - bulk_fugacity)
    equilibrium_log_ratio = 0.0
    if direction:
        at_bulk = phase_residual(0.0)
        previous_log, previous_residual = 0.0, at_bulk
        bracket = None
        for magnitude in np.linspace(0.25, 12.0, 48):
            candidate_log = float(direction * magnitude)
            try:
                candidate_residual = phase_residual(candidate_log)
            except Exception:
                break
            if previous_residual * candidate_residual <= 0.0:
                bracket = (min(previous_log, candidate_log), max(previous_log, candidate_log))
                break
            previous_log, previous_residual = candidate_log, candidate_residual
        if bracket is None:
            raise ReactiveFilmSolveError("could not bracket the interfacial fugacity-equilibrium state")
        equilibrium_log_ratio = float(brentq(phase_residual, *bracket, xtol=1.0e-12))

    continuation = np.linspace(0.0, 1.0, int(reaction_continuation_steps) + 1) ** 2
    def physical_film_residual(log_ratio: float) -> float:
        trial_concentrations = bulk.copy()
        trial_concentrations[co2_index] *= np.exp(log_ratio)
        trial_composition = trial_concentrations / np.sum(trial_concentrations)
        trial_fugacity = float(
            liquid_co2_fugacity_pa(trial_concentrations, trial_composition)
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
    interface_concentrations = bulk.copy()
    interface_concentrations[co2_index] *= concentration_ratio
    interface_composition = interface_concentrations / np.sum(interface_concentrations)
    interface_fugacity = math.nan
    interface_rate = abs(
        net_rate_mol_m3_s(
            interface_concentrations,
            interface_composition,
            interface_fugacity,
        )
    )
    rate_coefficient = interface_rate / interface_concentrations[co2_index]
    if rate_coefficient == 0.0:
        continuation = np.asarray((1.0,))
    closure_scale = max(
        flux_scale[co2_index],
        float(gas_transfer_coefficient_mol_m2_s_pa)
        * abs(float(vapor_bulk_fugacity_pa) - bulk_fugacity),
        1.0e-30,
    )

    def boundary(interface: np.ndarray, bulk_edge: np.ndarray) -> np.ndarray:
        interface_concentrations = bulk * np.maximum(interface[:n_species], 1.0e-30)
        interface_composition = interface_concentrations / np.sum(interface_concentrations)
        interface_fugacity = float(
            liquid_co2_fugacity_pa(interface_concentrations, interface_composition)
        )
        liquid_flux = interface[n_species + co2_index] * flux_scale[co2_index]
        gas_flux = float(gas_transfer_coefficient_mol_m2_s_pa) * (
            float(vapor_bulk_fugacity_pa) - interface_fugacity
        )
        residual = np.empty(2 * n_species, dtype=float)
        residual[:n_species] = bulk_edge[:n_species] - 1.0
        residual[n_species] = (liquid_flux - gas_flux) / closure_scale
        residual[n_species + 1 :] = np.delete(interface[n_species:], co2_index)
        return residual

    def boundary_jacobian(interface: np.ndarray, _bulk_edge: np.ndarray):
        interface_jacobian = np.zeros((2 * n_species, 2 * n_species), dtype=float)
        bulk_jacobian = np.zeros_like(interface_jacobian)
        bulk_jacobian[:n_species, :n_species] = np.eye(n_species)

        ratio = max(float(interface[co2_index]), 1.0e-30)
        step = max(1.0e-6 * ratio, 1.0e-8)
        lower = max(ratio - step, 0.5 * ratio)
        upper = ratio + step

        def fugacity_at(co2_ratio: float) -> float:
            concentrations = bulk * np.maximum(interface[:n_species], 1.0e-30)
            concentrations[co2_index] = bulk[co2_index] * co2_ratio
            composition = concentrations / np.sum(concentrations)
            return float(liquid_co2_fugacity_pa(concentrations, composition))

        fugacity_derivative = (fugacity_at(upper) - fugacity_at(lower)) / (upper - lower)
        interface_jacobian[n_species, co2_index] = (
            float(gas_transfer_coefficient_mol_m2_s_pa)
            * fugacity_derivative
            / closure_scale
        )
        interface_jacobian[n_species, n_species + co2_index] = (
            flux_scale[co2_index] / closure_scale
        )
        other_species = [index for index in range(n_species) if index != co2_index]
        for row, index in enumerate(other_species, start=n_species + 1):
            interface_jacobian[row, n_species + index] = 1.0
        return interface_jacobian, bulk_jacobian

    def initial_guess(first_scale: float, flux_factor: float) -> np.ndarray:
        guess = np.zeros((2 * n_species, coordinate.size), dtype=float)
        if first_scale == 0.0:
            guess[:n_species] = 1.0
            guess[co2_index] = concentration_ratio + (1.0 - concentration_ratio) * coordinate
            guess[n_species + co2_index] = flux_factor * (concentration_ratio - 1.0)
            return guess
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
        physical_fluxes = np.zeros_like(concentration_ratios)
        physical_fluxes[co2_index] = co2_flux
        for index in range(n_species):
            if index == co2_index:
                continue
            physical_fluxes[index] = nu[index] * (co2_flux[0] - co2_flux)
            reverse_integral = -cumulative_trapezoid(
                physical_fluxes[index, ::-1], coordinate[::-1], initial=0.0
            )[::-1]
            concentration_ratios[index] += (
                delta * reverse_integral / (diffusivities[index] * bulk[index])
            )
        guess[:n_species] = concentration_ratios
        guess[n_species:] = physical_fluxes / flux_scale[:, None]
        return guess

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
    closure_residual = float(boundary(solution.y[:, 0], solution.y[:, -1])[n_species])

    check_coordinate = np.linspace(0.0, 1.0, max(201, 10 * int(mesh_points)))
    check_values = solution.sol(check_coordinate)
    if np.any(~np.isfinite(check_values[:n_species])) or np.any(check_values[:n_species] < -1.0e-9):
        minimum = np.unravel_index(np.nanargmin(check_values[:n_species]), (n_species, check_coordinate.size))
        raise ReactiveFilmDomainError(
            "film concentrations left the positive finite domain: "
            f"species {minimum[0]}, coordinate {check_coordinate[minimum[1]]:.6g}, "
            f"ratio {check_values[minimum]:.6g}"
        )
    concentrations, compositions, fugacities, rates = evaluate(
        check_values[:n_species], include_fugacity=True
    )
    fluxes = check_values[n_species:] * flux_scale[:, None]
    def rate_integrand(coordinate: float) -> float:
        values = solution.sol(coordinate)[:n_species, None]
        return float(evaluate(values, include_fugacity=False)[3][0])

    integrated_rate = quad(rate_integrand, 0.0, 1.0, epsabs=1.0e-8, epsrel=1.0e-10, limit=500)[0]
    integrated_source = delta * nu * integrated_rate
    endpoint_values = solution.sol(np.asarray((0.0, 1.0)))
    conservation_fluxes = endpoint_values[n_species:] * flux_scale[:, None]
    flux_change = conservation_fluxes[:, -1] - conservation_fluxes[:, 0]
    conservation_scale = np.maximum.reduce(
        (np.abs(flux_change), np.abs(integrated_source), np.full(n_species, 1.0e-30))
    )
    conservation_residual = np.max(np.abs(flux_change - integrated_source) / conservation_scale)
    invariant_residual = 0.0
    if invariants.size:
        source = nu[:, None] * rates[None, :]
        invariant_source = invariants @ source
        source_scale = max(float(np.max(np.abs(source))), 1.0e-30)
        invariant_residual = float(np.max(np.abs(invariant_source)) / source_scale)

    return ReactiveFilmResult(
        coordinate_m=check_coordinate * delta,
        concentrations_mol_m3=concentrations,
        compositions=compositions,
        fluxes_mol_m2_s=fluxes,
        liquid_co2_fugacity_pa=fugacities,
        net_rate_mol_m3_s=rates,
        maximum_interface_residual=float(
            max(
                np.max(np.abs(boundary(solution.y[:, 0], solution.y[:, -1]))),
                abs(closure_residual),
            )
        ),
        maximum_conservation_residual=float(conservation_residual),
        maximum_invariant_source_residual=invariant_residual,
        solver_message=str(solution.message) + ("; canonical initialization recovery used" if recovery_used else ""),
    )
