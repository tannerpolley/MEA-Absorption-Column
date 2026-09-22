from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable


LEGACY_TO_COMPONENT_ID = {
    "CO2": "carbon-dioxide",
    "MEA": "monoethanolamine",
    "H2O": "water",
    "MEAH+": "protonated-monoethanolamine",
    "MEACOO-": "carbamate-anion",
    "HCO3-": "bicarbonate-anion",
    "CO3^2-": "carbonate-anion",
    "H3O+": "hydronium-cation",
    "OH-": "hydroxide-anion",
}

_SOURCE_ID = "mea-absorber-selected-parameter-artifact"
_DOMAIN_ID = "mea-absorber-diagnostic-domain"
_SOURCE = {
    "source_id": _SOURCE_ID,
    "citation": "MEA Absorption Column selected ePC-SAFT parameter artifact.",
    "use_basis": (
        "Baygi and Pahlavanzadeh (2015) and Held-style neutral parameters, "
        "source-transferred carbonate parameters, and explicitly provisional "
        "MEAH+/MEACOO- parameters. This is a diagnostic benchmark input, not a "
        "proven predictive nine-species fit."
    ),
}
_PROVENANCE_STATUS = {
    "CO2": "retained-literature-lineage",
    "MEA": "retained-literature-lineage",
    "H2O": "retained-literature-lineage",
    "MEAH+": "provisional-historical-evaluation",
    "MEACOO-": "provisional-historical-evaluation",
    "HCO3-": "transferred-diagnostic",
    "CO3^2-": "placeholder-diagnostic",
    "H3O+": "placeholder-diagnostic",
    "OH-": "placeholder-diagnostic",
}


def _value(magnitude: float | int, unit: str) -> dict[str, object]:
    return {"magnitude": magnitude, "unit": unit}


def _provenance(dataset: Path, component: str, family: str) -> dict[str, str]:
    status = _PROVENANCE_STATUS[component]
    return {
        "source_id": _SOURCE_ID,
        "locator": (
            f"{dataset.as_posix()}/pure/any_solvent.csv:"
            f"component={component}:family={family}:status={status}"
        ),
        "domain_id": _DOMAIN_ID,
    }


def _coefficient(
    dataset: Path,
    component: str,
    family: str,
    magnitude: float,
    unit: str,
) -> dict[str, object]:
    component_id = LEGACY_TO_COMPONENT_ID[component]
    return {
        "identity": f"component/{component_id}/{family}",
        "family": family,
        "value": _value(magnitude, unit),
        "provenance": _provenance(dataset, component, family),
    }


def _load_pure_rows(dataset: Path) -> list[dict[str, str]]:
    path = dataset / "pure" / "any_solvent.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = set(LEGACY_TO_COMPONENT_ID)
    observed = {row["component"] for row in rows}
    missing = expected - observed
    if missing:
        raise ValueError(f"ePC-SAFT parameter artifact is missing species: {sorted(missing)}")
    return rows


def _component_record(dataset: Path, row: dict[str, str]) -> dict[str, object]:
    component = row["component"]
    component_id = LEGACY_TO_COMPONENT_ID[component]
    charge = int(float(row["z"]))
    fixed = {
        "molar_mass": {
            "value": _value(float(row["MW"]), "kilogram / mole"),
            "provenance": _provenance(dataset, component, "molar_mass"),
        },
        "charge_number": {
            "value": _value(charge, "elementary-charge"),
            "provenance": _provenance(dataset, component, "charge_number"),
        },
    }
    coefficients = [
        _coefficient(dataset, component, "segment_count", float(row["m"]), "dimensionless"),
        _coefficient(
            dataset,
            component,
            "dispersion_energy_over_k",
            float(row["e"]),
            "kelvin",
        ),
        _coefficient(
            dataset,
            component,
            "solvation_factor",
            float(row["f_solv"]),
            "dimensionless",
        ),
    ]
    if component != "H2O":
        segment_diameter = float(row["s"])
        coefficients.append(
            _coefficient(
                dataset,
                component,
                "segment_diameter",
                segment_diameter,
                "angstrom",
            )
        )
    if charge:
        segment_diameter = float(row["s"])
        coefficients.extend(
            (
                _coefficient(
                    dataset,
                    component,
                    "packing_diameter",
                    0.88 * segment_diameter,
                    "angstrom",
                ),
                _coefficient(
                    dataset,
                    component,
                    "debye_huckel_diameter",
                    0.88 * segment_diameter,
                    "angstrom",
                ),
                _coefficient(
                    dataset,
                    component,
                    "born_diameter",
                    float(row["d_born"]),
                    "angstrom",
                ),
            )
        )
    return {
        "component_id": component_id,
        "name": component,
        "formula": component,
        "aliases": [component],
        "fixed": fixed,
        "coefficients": sorted(coefficients, key=lambda item: str(item["family"])),
    }


def _model_provenance(locator: str) -> dict[str, str]:
    return {
        "source_id": _SOURCE_ID,
        "locator": locator,
        "domain_id": _DOMAIN_ID,
    }


def _model_families(dataset: Path) -> list[dict[str, object]]:
    base = dataset.as_posix()
    return [
        {
            "family_id": "model/association",
            "kind": "association",
            "choice": "general-site",
            "provenance": _model_provenance(f"{base}/pure/any_solvent.csv:2B sites"),
        },
        {
            "family_id": "model/base",
            "kind": "base",
            "choice": "pc-saft",
            "provenance": _model_provenance(f"{base}/pure/any_solvent.csv"),
        },
        {
            "family_id": "model/electrolyte",
            "kind": "electrolyte",
            "choice": "born",
            "c_shell": 1.0,
            "c_dielectric": 1.0,
            "provenance": _model_provenance(
                f"{base}/user_options.json:solvation-shell and dielectric-saturation enabled"
            ),
        },
        {
            "family_id": "model/relative_permittivity",
            "kind": "permittivity",
            "choice": "ion-fraction-suppression",
            "provenance": _model_provenance(
                f"{base}/user_options.json:empirical relative-permittivity rule"
            ),
        },
    ]


def _model_coefficient(family: str, magnitude: float, locator: str) -> dict[str, object]:
    return {
        "identity": f"model/relative_permittivity/{family}",
        "family": family,
        "value": _value(magnitude, "dimensionless"),
        "provenance": _model_provenance(locator),
    }


def _correlations(dataset: Path, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    correlations: list[dict[str, object]] = []
    for row in rows:
        component = row["component"]
        if int(float(row["z"])) != 0:
            continue
        component_id = LEGACY_TO_COMPONENT_ID[component]
        correlation_id = f"component/{component_id}/relative_permittivity/constant"
        correlations.append(
            {
                "correlation_id": correlation_id,
                "component_id": component_id,
                "family": "relative_permittivity",
                "form": "constant",
                "independent_variables": [],
                "constant": {
                    "identity": f"{correlation_id}/constant",
                    "value": _value(float(row["dielc"]), "dimensionless"),
                },
                "provenance": _provenance(dataset, component, "relative_permittivity"),
            }
        )
    diameter_id = "component/water/segment_diameter/constant-plus-sum-of-exponentials"
    correlations.append(
        {
            "correlation_id": diameter_id,
            "component_id": "water",
            "family": "segment_diameter",
            "form": "constant-plus-sum-of-exponentials",
            "independent_variables": ["temperature"],
            "constant": {
                "identity": f"{diameter_id}/constant",
                "value": _value(2.7927, "angstrom"),
            },
            "terms": [
                {
                    "amplitude": {
                        "identity": f"{diameter_id}/term-0/amplitude",
                        "value": _value(10.11, "angstrom"),
                    },
                    "exponent_coefficient": {
                        "identity": f"{diameter_id}/term-0/exponent_coefficient",
                        "value": _value(-0.01775, "1 / kelvin"),
                    },
                },
                {
                    "amplitude": {
                        "identity": f"{diameter_id}/term-1/amplitude",
                        "value": _value(-1.417, "angstrom"),
                    },
                    "exponent_coefficient": {
                        "identity": f"{diameter_id}/term-1/exponent_coefficient",
                        "value": _value(-0.01146, "1 / kelvin"),
                    },
                },
            ],
            "provenance": _provenance(dataset, "H2O", "segment_diameter"),
        }
    )
    return correlations


def _association_topology(dataset: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    associating = [row for row in rows if row["assoc_scheme"].strip().upper() == "2B"]
    sites: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    for row in associating:
        component = row["component"]
        component_id = LEGACY_TO_COMPONENT_ID[component]
        for site_id, site_role in (("a", "donor"), ("b", "acceptor")):
            sites.append(
                {
                    "component_id": component_id,
                    "site_id": site_id,
                    "site_role": site_role,
                    "multiplicity": 1,
                    "provenance": _provenance(dataset, component, f"association_site_{site_id}"),
                }
            )
        prefix = f"association/{component_id}/a/{component_id}/b"
        edges.append(
            {
                "endpoint_a": {"component_id": component_id, "site_id": "a"},
                "endpoint_b": {"component_id": component_id, "site_id": "b"},
                "energy_over_k": {
                    "identity": f"{prefix}/energy_over_k",
                    "value": _value(float(row["e_assoc"]), "kelvin"),
                },
                "volume": {
                    "identity": f"{prefix}/volume",
                    "value": _value(float(row["vol_a"]), "dimensionless"),
                },
                "source": {
                    "kind": "explicit",
                    "provenance": [
                        _provenance(dataset, component, "association_energy_over_k"),
                        _provenance(dataset, component, "association_volume"),
                    ],
                },
            }
        )
    for left in associating:
        for right in associating:
            if left is right:
                continue
            left_id = LEGACY_TO_COMPONENT_ID[left["component"]]
            right_id = LEGACY_TO_COMPONENT_ID[right["component"]]
            edges.append(
                {
                    "endpoint_a": {"component_id": left_id, "site_id": "a"},
                    "endpoint_b": {"component_id": right_id, "site_id": "b"},
                    "source": {
                        "kind": "combining-rule",
                        "rule_id": "arithmetic-energy-geometric-volume",
                        "inputs": {
                            "energy_over_k": [
                                f"association/{left_id}/a/{left_id}/b/energy_over_k",
                                f"association/{right_id}/a/{right_id}/b/energy_over_k",
                            ],
                            "volume": [
                                f"association/{left_id}/a/{left_id}/b/volume",
                                f"association/{right_id}/a/{right_id}/b/volume",
                            ],
                        },
                        "provenance": [
                            _model_provenance(
                                "Gross and Sadowski (2002), Eqs. 2-3 cross-association combining rule"
                            )
                        ],
                    },
                }
            )
    return {"presets": [], "sites": sites, "edges": edges}


def _pair_records(dataset: Path) -> list[dict[str, object]]:
    path = dataset / "mixed" / "binary_interaction" / "k_ij.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        left = row["component"]
        for right, raw_value in row.items():
            if right == "component" or not raw_value:
                continue
            endpoints = tuple(sorted((left, right)))
            if left == right or endpoints in seen:
                continue
            seen.add(endpoints)
            value = float(raw_value)
            left_id, right_id = sorted(LEGACY_TO_COMPONENT_ID[item] for item in endpoints)
            records.append(
                {
                    "component_id_a": left_id,
                    "component_id_b": right_id,
                    "coefficients": [
                        {
                            "identity": f"pair/{left_id}/{right_id}/k_ij",
                            "family": "k_ij",
                            "value": _value(value, "dimensionless"),
                            "provenance": _model_provenance(
                                f"{path.as_posix()}:row={left}:column={right}:"
                                "status=explicit-selected-dataset"
                            ),
                        }
                    ],
                }
            )
    return records


@lru_cache(maxsize=8)
def parameter_document(dataset_text: str) -> dict[str, object]:
    dataset = Path(dataset_text)
    rows = _load_pure_rows(dataset)
    return {
        "schema": "epcsaft.parameters",
        "schema_version": 1,
        "document_id": "mea-absorber-selected-epcsaft",
        "document_version": 1,
        "purpose": "user-provided",
        "sources": [_SOURCE],
        "domains": [{"domain_id": _DOMAIN_ID, "kind": "unknown"}],
        "components": [_component_record(dataset, row) for row in rows],
        "pairs": _pair_records(dataset),
        "model_families": _model_families(dataset),
        "model_coefficients": [
            _model_coefficient(
                "ion_fraction_suppression_coefficient",
                7.01,
                "Figiel, Yu, and Held (2025), Eq. 11 denominator coefficient",
            ),
            _model_coefficient(
                "ionic_region_relative_permittivity",
                8.0,
                "selected dataset ion dielc value and Figiel, Yu, and Held (2025), Table 1",
            ),
        ],
        "correlations": _correlations(dataset, rows),
        "topology": _association_topology(dataset, rows),
    }


def component_ids(species: Iterable[str]) -> tuple[str, ...]:
    try:
        return tuple(LEGACY_TO_COMPONENT_ID[item] for item in species)
    except KeyError as exc:
        raise ValueError(f"Unknown MEA ePC-SAFT species: {exc.args[0]}") from exc


@lru_cache(maxsize=32)
def parameters(dataset_text: str, species: tuple[str, ...]):
    import epcsaft

    return epcsaft.Parameters.from_mapping(
        parameter_document(dataset_text),
        components=component_ids(species),
    )


@lru_cache(maxsize=32)
def mixture(dataset_text: str, species: tuple[str, ...]):
    import epcsaft

    return epcsaft.Mixture(parameters(dataset_text, species))


def state(mixture_model, *, temperature_k: float, pressure_pa: float, composition, phase: str):
    import epcsaft

    normalized_phase = {"liq": "liquid", "liquid": "liquid", "vap": "vapor", "vapor": "vapor"}.get(
        str(phase).lower()
    )
    if normalized_phase is None:
        raise ValueError("phase must be liquid/liq or vapor/vap")
    return mixture_model.state(
        T=float(temperature_k) * epcsaft.unit_registry.kelvin,
        P=float(pressure_pa) * epcsaft.unit_registry.pascal,
        x=tuple(float(value) for value in composition),
        phase=normalized_phase,
    )


def state_at_density(mixture_model, *, temperature_k: float, density_mol_m3: float, composition):
    import epcsaft

    return mixture_model.state(
        T=float(temperature_k) * epcsaft.unit_registry.kelvin,
        rho=float(density_mol_m3) * epcsaft.unit_registry.mole / epcsaft.unit_registry.meter**3,
        x=tuple(float(value) for value in composition),
    )


def molar_density_value(state_value) -> float:
    return float(state_value.molar_density.to("mole / meter ** 3").magnitude)


def pressure_value(state_value) -> float:
    return float(state_value.pressure.to("pascal").magnitude)


def fugacity_coefficients(state_value) -> tuple[float, ...]:
    fugacity = state_value.fugacity
    if fugacity is None:
        raise RuntimeError("ePC-SAFT state did not expose fugacity coefficients")
    coefficients = tuple(float(value) for value in fugacity.coefficient)
    if any(not math.isfinite(value) or value <= 0.0 for value in coefficients):
        raise RuntimeError("ePC-SAFT returned a nonpositive or nonfinite fugacity coefficient")
    return coefficients
