from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from mea_absorption_column.Thermodynamics.thermo_models import (
    IONIC_CHARGE_BY_SPECIES,
    IONIC_LIQUID_SPECIES_9,
    epcsaft_liquid_transport_state,
)


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "analyses/nccc_validation/results/final/tables/issue40_apparent_true_species.csv"
SPECIES = tuple(IONIC_LIQUID_SPECIES_9)
COMPONENT_ID_TO_SPECIES = {
    "carbon-dioxide": "CO2",
    "monoethanolamine": "MEA",
    "water": "H2O",
    "protonated-monoethanolamine": "MEAH+",
    "carbamate-anion": "MEACOO-",
    "bicarbonate-anion": "HCO3-",
    "carbonate-anion": "CO3^2-",
    "hydronium-cation": "H3O+",
    "hydroxide-anion": "OH-",
}
CHARGES = np.asarray([IONIC_CHARGE_BY_SPECIES[name] for name in SPECIES], dtype=float)
CO2_INDEX = SPECIES.index("CO2")
FILM_THICKNESS_M = 1.0e-4
PROVISIONAL_MOBILITY_MOL_PER_M_S = 1.0e-9
DRIVES = {"zero_drive": 0.0, "absorption": 1.0e-2, "desorption": -1.0e-2}


class ChemicalPotentialFilmError(RuntimeError):
    pass


@dataclass(frozen=True)
class BulkCase:
    temperature_K: float
    pressure_Pa: float
    composition: np.ndarray


@dataclass(frozen=True)
class FilmResult:
    calculation_status: str
    quadrature_points: int
    target_fugacity_ratio: float
    interface_fugacity_ratio_residual: float
    minimum_composition: float
    maximum_normalization_residual: float
    maximum_electroneutrality_residual: float
    maximum_zero_total_flux_residual: float
    maximum_zero_current_residual: float
    maximum_species_conservation_residual: float
    minimum_dissipation: float
    co2_flux_mol_m2_s: float
    constrained_ideal_log_force_reference_co2_flux_mol_m2_s: float
    chemical_potential_to_constrained_reference_ratio: float | None


def _load_case() -> BulkCase:
    with INPUT.open(newline="") as stream:
        rows = csv.DictReader(stream)
        row = next(
            row
            for row in rows
            if row.get("case_id") == "3C" and abs(float(row["position"]) - 1.0) < 1.0e-12
        )
    composition = np.asarray([float(row[f"true_x_{name}"]) for name in SPECIES], dtype=float)
    return BulkCase(float(row["temperature_K"]), float(row["pressure_Pa"]), composition)


def _rank_or_stop(matrix: np.ndarray, expected: int, label: str) -> None:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != expected:
        raise ChemicalPotentialFilmError(
            f"{label} rank deficient: rank={rank}, expected={expected}, "
            f"singular_values={singular_values.tolist()}"
        )


def _coordinate_indices(state) -> tuple[np.ndarray, np.ndarray]:
    independent = np.asarray(
        [SPECIES.index(COMPONENT_ID_TO_SPECIES[name]) for name in state.coordinate_component_ids], dtype=int
    )
    dependent = np.asarray(
        [SPECIES.index(COMPONENT_ID_TO_SPECIES[name]) for name in state.dependent_component_ids], dtype=int
    )
    if independent.size != 7 or dependent.size != 2 or np.intersect1d(independent, dependent).size:
        raise ChemicalPotentialFilmError(
            "under-determined composition coordinates: expected seven independent and two dependent species"
        )
    _rank_or_stop(np.vstack((np.ones(2), CHARGES[dependent])), 2, "dependent constraint block")
    return independent, dependent


def _composition_from_coordinates(
    bulk: np.ndarray, values: np.ndarray, independent: np.ndarray, dependent: np.ndarray
) -> np.ndarray:
    composition = bulk.copy()
    composition[independent] = bulk[independent] * np.exp(values)
    rhs = np.asarray(
        (
            1.0 - np.sum(composition[independent]),
            -np.dot(CHARGES[independent], composition[independent]),
        )
    )
    composition[dependent] = np.linalg.solve(
        np.vstack((np.ones(2), CHARGES[dependent])), rhs
    )
    if np.any(~np.isfinite(composition)) or np.any(composition <= 0.0):
        raise ChemicalPotentialFilmError("film composition left the positive domain")
    return composition


def _constrained_block(state) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basis = np.asarray(state.log_composition_basis, dtype=float)
    derivative = np.asarray(state.chemical_potential_derivatives_over_rt, dtype=float)
    if basis.shape != (9, 7) or derivative.shape != (9, 7):
        raise ChemicalPotentialFilmError(
            f"unexpected ePC-SAFT tangent shapes: basis={basis.shape}, derivative={derivative.shape}"
        )
    _rank_or_stop(basis, 7, "log-composition basis")
    _rank_or_stop(derivative, 7, "chemical-potential derivative block")
    constraints = np.vstack((np.ones(9), CHARGES))
    gram = constraints @ constraints.T
    _rank_or_stop(gram, 2, "total-flux/current constraint block")
    projection = np.eye(9) - constraints.T @ np.linalg.solve(gram, constraints)
    constrained = projection @ derivative
    _rank_or_stop(constrained, 7, "constrained chemical-potential block")
    return basis, derivative, constrained, projection


def _expand_flux(values: np.ndarray, independent: np.ndarray, dependent: np.ndarray) -> np.ndarray:
    flux = np.zeros(9, dtype=float)
    flux[independent] = values
    flux[dependent] = np.linalg.solve(
        np.vstack((np.ones(2), CHARGES[dependent])), -np.vstack((np.ones(independent.size), CHARGES[independent])) @ values
    )
    return flux


def _interface_coordinates(case: BulkCase, bulk_state, ratio: float, independent: np.ndarray, dependent: np.ndarray) -> np.ndarray:
    if ratio == 1.0:
        return np.zeros(7)

    def residual(log_ratio: float) -> float:
        values = np.zeros(7)
        co2_coordinate = int(np.flatnonzero(independent == CO2_INDEX)[0])
        values[co2_coordinate] = log_ratio
        composition = _composition_from_coordinates(case.composition, values, independent, dependent)
        state = epcsaft_liquid_transport_state(case.temperature_K, case.pressure_Pa, composition)
        return float(state.fugacities_pa[CO2_INDEX] / bulk_state.fugacities_pa[CO2_INDEX] - ratio)

    log_ratio = brentq(residual, -0.1, 0.1, xtol=1.0e-12)
    values = np.zeros(7)
    values[int(np.flatnonzero(independent == CO2_INDEX)[0])] = log_ratio
    return values


def _solve(case: BulkCase, interface_values: np.ndarray, independent: np.ndarray, dependent: np.ndarray, quadrature_points: int) -> FilmResult:
    # ponytail: fixed straight path; upgrade to a nonlinear BVP after admitted finite-rate and mobility inputs exist.
    coordinate = np.linspace(0.0, 1.0, quadrature_points)
    values = interface_values[:, None] * (1.0 - coordinate)
    compositions = np.column_stack(
        [_composition_from_coordinates(case.composition, values[:, i], independent, dependent) for i in range(quadrature_points)]
    )
    states = [
        epcsaft_liquid_transport_state(case.temperature_K, case.pressure_Pa, compositions[:, i])
        for i in range(quadrature_points)
    ]
    blocks = [_constrained_block(state) for state in states]
    flux_basis = np.column_stack([_expand_flux(np.eye(7)[:, i], independent, dependent) for i in range(7)])
    chemical_integral = np.zeros((7, 7))
    reference_integral = np.zeros((7, 7))
    chemical_maps = []
    reference_maps = []
    for i, (basis, _derivative, constrained, projection) in enumerate(blocks):
        chemical_map = np.linalg.lstsq(constrained, flux_basis, rcond=None)[0]
        reference_map = np.linalg.lstsq(projection @ basis, flux_basis, rcond=None)[0]
        if max(
            np.max(np.abs(constrained @ chemical_map - flux_basis)),
            np.max(np.abs(projection @ basis @ reference_map - flux_basis)),
        ) > 1.0e-10:
            raise ChemicalPotentialFilmError(
                "constrained solve residual exceeded 1e-10: "
                f"chemical={np.max(np.abs(constrained @ chemical_map - flux_basis)):.6e}, "
                f"constrained_reference={np.max(np.abs(projection @ basis @ reference_map - flux_basis)):.6e}"
            )
        chemical_maps.append(chemical_map)
        reference_maps.append(reference_map)
        weight = 0.5 / (quadrature_points - 1) if i in (0, quadrature_points - 1) else 1.0 / (quadrature_points - 1)
        chemical_integral += weight * chemical_map
        reference_integral += weight * reference_map
    _rank_or_stop(chemical_integral, 7, "integrated constrained chemical-potential block")
    _rank_or_stop(reference_integral, 7, "integrated constrained ideal-log-composition-force block")
    chemical_u = np.linalg.solve(chemical_integral, interface_values)
    reference_u = np.linalg.solve(reference_integral, interface_values)
    if max(
        np.max(np.abs(chemical_integral @ chemical_u - interface_values)),
        np.max(np.abs(reference_integral @ reference_u - interface_values)),
    ) > 1.0e-10:
        raise ChemicalPotentialFilmError("integrated constrained solve residual exceeded 1e-10")
    chemical_flux = flux_basis @ chemical_u
    reference_flux = flux_basis @ reference_u
    qprimes = np.column_stack([-chemical_maps[i] @ chemical_u for i in range(quadrature_points)])
    reference_qprimes = np.column_stack([-reference_maps[i] @ reference_u for i in range(quadrature_points)])
    fluxes = np.tile(chemical_flux[:, None], (1, quadrature_points))
    forces = np.column_stack([blocks[i][1] @ qprimes[:, i] for i in range(quadrature_points)])
    reference_forces = np.column_stack([blocks[i][0] @ reference_qprimes[:, i] for i in range(quadrature_points)])
    bulk_fugacity = float(states[-1].fugacities_pa[CO2_INDEX])
    interface_ratio = float(states[0].fugacities_pa[CO2_INDEX] / bulk_fugacity)
    return FilmResult(
        calculation_status="fixed_path_constrained_quadrature_completed",
        quadrature_points=quadrature_points,
        target_fugacity_ratio=interface_ratio,
        interface_fugacity_ratio_residual=0.0,
        minimum_composition=float(np.min(compositions)),
        maximum_normalization_residual=float(np.max(np.abs(np.sum(compositions, axis=0) - 1.0))),
        maximum_electroneutrality_residual=float(np.max(np.abs(CHARGES @ compositions))),
        maximum_zero_total_flux_residual=float(max(abs(np.sum(chemical_flux)), abs(np.sum(reference_flux)))),
        maximum_zero_current_residual=float(max(abs(CHARGES @ chemical_flux), abs(CHARGES @ reference_flux))),
        maximum_species_conservation_residual=float(np.max(np.abs(fluxes[:, -1] - fluxes[:, 0]))),
        minimum_dissipation=float(min(np.min(-np.sum(fluxes * forces, axis=0)), -np.max(np.sum(reference_flux[:, None] * reference_forces, axis=0)))),
        co2_flux_mol_m2_s=float(PROVISIONAL_MOBILITY_MOL_PER_M_S / FILM_THICKNESS_M * fluxes[CO2_INDEX, 0]),
        constrained_ideal_log_force_reference_co2_flux_mol_m2_s=float(
            PROVISIONAL_MOBILITY_MOL_PER_M_S / FILM_THICKNESS_M * reference_flux[CO2_INDEX]
        ),
        chemical_potential_to_constrained_reference_ratio=(
            float(chemical_flux[CO2_INDEX] / reference_flux[CO2_INDEX])
            if reference_flux[CO2_INDEX]
            else None
        ),
    )


def run(quadrature_points: int = 11) -> dict[str, object]:
    case = _load_case()
    bulk_state = epcsaft_liquid_transport_state(case.temperature_K, case.pressure_Pa, case.composition)
    independent, dependent = _coordinate_indices(bulk_state)
    result: dict[str, object] = {
        "case": asdict(case) | {"composition": case.composition.tolist()},
        "state_source": str(INPUT.relative_to(ROOT)),
        "state_disposition": "basis_unresolved_unadmitted_provisional_numerical_state",
        "reaction_status": "disabled_unresolved_finite_rate_inputs",
        "reaction_source_mol_m3_s": 0.0,
        "mobility": "common_unit_diagonal_provisional; dimensional flux arbitrary",
    }
    for name, drive in DRIVES.items():
        ratio = 1.0 + drive
        interface_values = _interface_coordinates(case, bulk_state, ratio, independent, dependent)
        chemical = _solve(case, interface_values, independent, dependent, quadrature_points)
        chemical = FilmResult(**{**asdict(chemical), "target_fugacity_ratio": ratio, "interface_fugacity_ratio_residual": chemical.target_fugacity_ratio - ratio})
        result[name] = {"chemical_potential": asdict(chemical)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory constrained chemical-potential film prototype")
    parser.add_argument("--quadrature-points", type=int, default=11)
    args = parser.parse_args()
    if args.quadrature_points < 5:
        parser.error("--quadrature-points must be at least 5")
    print(json.dumps(run(args.quadrature_points), indent=2, default=float))


if __name__ == "__main__":
    main()
