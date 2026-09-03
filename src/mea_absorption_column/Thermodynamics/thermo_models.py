from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
import importlib
import importlib.metadata
from importlib import resources
from pathlib import Path

import numpy as np

from mea_absorption_column.BVP.robust_core import (
    record_invalid_state,
)
from mea_absorption_column.Thermodynamics.epcsaft_v02 import (
    fugacity_coefficients as _v02_fugacity_coefficients,
    mixture as _v02_mixture,
    molar_density_value as _v02_molar_density_value,
    state as _v02_state,
)


SPECIES = ["CO2", "MEA", "H2O"]
IONIC_LIQUID_SPECIES_6 = ["CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-"]
IONIC_LIQUID_SPECIES_9 = [
    "CO2",
    "MEA",
    "H2O",
    "MEAH+",
    "MEACOO-",
    "HCO3-",
    "CO3^2-",
    "H3O+",
    "OH-",
]
IONIC_LIQUID_SPECIES = IONIC_LIQUID_SPECIES_6
IONIC_CHARGE_BY_SPECIES = {
    "CO2": 0.0,
    "MEA": 0.0,
    "H2O": 0.0,
    "MEAH+": 1.0,
    "MEACOO-": -1.0,
    "HCO3-": -1.0,
    "CO3^2-": -2.0,
    "H3O+": 1.0,
    "OH-": -1.0,
}
CO2_INDEX = 0
MEA_INDEX = 1
H2O_INDEX = 2
COMPOSITION_FLOOR = 1e-12
TEMPERATURE_MIN_K = 250.0
TEMPERATURE_MAX_K = 500.0
PACKAGED_EPCSAFT_DATASETS = resources.files("mea_absorption_column").joinpath(
    "data/epcsaft_datasets"
)
DEFAULT_EPCSAFT_DATASET_NAME = os.environ.get(
    "MEA_EPCSAFT_DATASET_NAME", "MEA_reactive_epcsaft_bundle"
)


def _resolve_epcsaft_dataset_path() -> Path:
    packaged = Path(
        str(PACKAGED_EPCSAFT_DATASETS.joinpath(DEFAULT_EPCSAFT_DATASET_NAME))
    )
    override = os.environ.get("MEA_THERMODYNAMICS_EPCSAFT_DATASET")
    if override:
        override_path = Path(override)
        if override_path.exists():
            return override_path
    return packaged


MEA_THERMODYNAMICS_EPCSAFT_DATASET = _resolve_epcsaft_dataset_path()
class EpcsaftFixedPressureDerivativeError(RuntimeError):
    def __init__(self, code: str, diagnostic: str):
        self.code = str(code)
        self.diagnostic = str(diagnostic)
        super().__init__(f"{self.code}: {self.diagnostic}")


@dataclass(frozen=True)
class EpcsaftLiquidTransportState:
    composition: np.ndarray
    fugacities_pa: np.ndarray
    log_composition_basis: np.ndarray
    chemical_potential_derivatives_over_rt: np.ndarray
    coordinate_component_ids: tuple[str, ...]
    dependent_component_ids: tuple[str, ...]
    condition_measure: float | None
    artifact_fingerprint: str
    parameter_fingerprint: str

    def log_fugacity_derivative(
        self, component_index: int, log_composition_direction
    ) -> float:
        direction = np.asarray(log_composition_direction, dtype=float)
        if direction.shape != self.composition.shape or not np.all(
            np.isfinite(direction)
        ):
            raise ValueError(
                "log-composition direction must have one finite value per species"
            )
        coordinates, *_ = np.linalg.lstsq(
            self.log_composition_basis, direction, rcond=None
        )
        reconstructed = self.log_composition_basis @ coordinates
        scale = max(float(np.max(np.abs(direction))), 1.0)
        if float(np.max(np.abs(reconstructed - direction))) > 1.0e-10 * scale:
            raise EpcsaftFixedPressureDerivativeError(
                "invalid_composition_direction",
                "requested direction is outside the normalization/electroneutral tangent",
            )
        return float(
            self.chemical_potential_derivatives_over_rt[component_index] @ coordinates
        )

    def fixed_other_concentrations_log_fugacity_derivative(
        self, component_index: int
    ) -> float:
        direction = np.full(self.composition.size, -self.composition[component_index])
        direction[component_index] += 1.0
        return self.log_fugacity_derivative(component_index, direction)


def _parameter_resource():
    return resources.files("mea_absorption_column").joinpath(
        "data/epcsaft_neutral/parameters.json"
    )


def load_epcsaft_parameter_dataset() -> dict:
    path = _parameter_resource()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["metadata_path"] = str(path)
    return payload


def build_epcsaft_params() -> dict:
    payload = load_epcsaft_parameter_dataset()
    raw = payload["parameters"]
    params = {
        "species": list(payload["species"]),
        "metadata_path": payload["metadata_path"],
        "m": np.asarray(raw["m"], dtype=float),
        "s": np.asarray(raw["s"], dtype=float),
        "e": np.asarray(raw["e"], dtype=float),
        "vol_a": np.asarray(raw["vol_a"], dtype=float),
        "e_assoc": np.asarray(raw["e_assoc"], dtype=float),
        "assoc_scheme": list(raw["assoc_scheme"]),
        "MW": np.asarray(raw["MW"], dtype=float),
        "k_ij": np.asarray(raw["k_ij"], dtype=float),
    }
    return params


def _epcsaft_direct_url() -> dict | None:
    try:
        dist = importlib.metadata.distribution("epcsaft")
    except importlib.metadata.PackageNotFoundError:
        return None
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    return json.loads(text)


def _epcsaft_source_kind(direct_url: dict | None) -> str:
    if not direct_url:
        return "release"
    if "vcs_info" in direct_url:
        return "pinned_git"
    url = str(direct_url.get("url", ""))
    return "local_file" if url.startswith("file:") else "direct_url"


def epcsaft_source_fingerprint() -> dict:
    try:
        module = importlib.import_module("epcsaft")
    except Exception as exc:
        return {
            "package": "epcsaft",
            "installed": False,
            "import_error": f"{type(exc).__name__}: {exc}",
        }
    module_file = Path(module.__file__).resolve()
    direct_url = _epcsaft_direct_url()
    return {
        "package": "epcsaft",
        "installed": True,
        "version": importlib.metadata.version("epcsaft"),
        "module_path": str(module_file),
        "exists": module_file.exists(),
        "source_kind": _epcsaft_source_kind(direct_url),
        "source_detail": json.dumps(direct_url, sort_keys=True)
        if direct_url
        else str(module_file),
    }


def neutral_liquid_composition(x_true) -> np.ndarray:
    x_arr = np.asarray(x_true, dtype=float)
    neutral = np.asarray([x_arr[0], x_arr[1], x_arr[2]], dtype=float)
    neutral = np.maximum(neutral, COMPOSITION_FLOOR)
    total = float(neutral.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Neutral liquid composition must have a positive finite sum.")
    return neutral / total


def neutral_vapor_composition(y) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    neutral = np.asarray([y_arr[0], COMPOSITION_FLOOR, y_arr[1]], dtype=float)
    neutral = np.maximum(neutral, COMPOSITION_FLOOR)
    return neutral / float(neutral.sum())


def ionic_liquid_composition(x_true) -> np.ndarray:
    x_arr = np.asarray(x_true, dtype=float)
    species = _ionic_species_for_size(x_arr.size)
    if x_arr.size < len(species):
        raise ValueError(
            f"ePC-SAFT ionic liquid composition requires {len(species)} species, got {x_arr.size}."
        )
    ionic = np.asarray(x_arr[: len(species)], dtype=float)
    ionic = np.maximum(ionic, COMPOSITION_FLOOR)
    total = float(ionic.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Ionic liquid composition must have a positive finite sum.")
    ionic = ionic / total
    return _enforce_electroneutrality(ionic, species)


def _enforce_electroneutrality(
    composition: np.ndarray, species: list[str] | tuple[str, ...]
) -> np.ndarray:
    values = np.asarray(composition, dtype=float).copy()
    charges = np.asarray(
        [IONIC_CHARGE_BY_SPECIES[item] for item in species], dtype=float
    )
    residual = float(np.dot(values, charges))
    if abs(residual) > 1.0e-15:
        candidate_sign = -1.0 if residual > 0.0 else 1.0
        candidates = np.flatnonzero(charges * candidate_sign > 0.0)
        if candidates.size == 0:
            raise ValueError(
                "Ionic ePC-SAFT composition cannot be projected to electroneutrality."
            )
        index = int(candidates[np.argmax(values[candidates])])
        values[index] += abs(residual / charges[index])
        values /= float(np.sum(values))
    if abs(float(np.dot(values, charges))) > 1.0e-12:
        raise ValueError(
            "Ionic ePC-SAFT composition does not satisfy electroneutrality."
        )
    return values


def _ionic_species_for_size(size: int) -> list[str]:
    if int(size) >= len(IONIC_LIQUID_SPECIES_9):
        return IONIC_LIQUID_SPECIES_9
    return IONIC_LIQUID_SPECIES_6


def ensure_epcsaft_importable():
    try:
        import epcsaft

        for symbol in ("Parameters", "Mixture", "State", "unit_registry"):
            if not hasattr(epcsaft, symbol):
                raise AttributeError(f"missing public ePC-SAFT 0.2 symbol: {symbol}")
        return epcsaft
    except Exception as exc:
        raise RuntimeError(
            "Could not import the installed ePC-SAFT package from the active environment. "
            "Install the immutable ePC-SAFT 0.2 wheel; local checkout import fallbacks are disabled."
        ) from exc


def epcsaft_mixture():
    ensure_epcsaft_importable()
    return _v02_mixture(str(MEA_THERMODYNAMICS_EPCSAFT_DATASET), tuple(SPECIES))


def _canonical_user_options_json(user_options: dict | None) -> str:
    if not user_options:
        return "{}"
    return json.dumps(user_options, sort_keys=True, separators=(",", ":"))


def _user_options_from_json(user_options_json: str) -> dict:
    if not user_options_json or user_options_json == "{}":
        return {}
    return json.loads(user_options_json)


def epcsaft_runtime_user_options() -> dict:
    options_json = os.environ.get("MEA_EPCSAFT_USER_OPTIONS_JSON")
    if options_json and _user_options_from_json(options_json):
        raise RuntimeError(
            "MEA_EPCSAFT_USER_OPTIONS_JSON is not supported by the ePC-SAFT 0.2 API. "
            "Model-family and derivative choices are immutable parameter-document inputs; "
            "CppAD is the package's sole production derivative authority."
        )
    return {}


def epcsaft_dataset_mixture(
    species_key: tuple[str, ...], T_key: float, user_options_json: str = "{}"
):
    ensure_epcsaft_importable()
    if user_options_json and _user_options_from_json(user_options_json):
        raise ValueError(
            "Runtime ePC-SAFT user-option overrides were removed in API 0.2. "
            "Create a separately identified parameter document for a different model family."
        )
    if not MEA_THERMODYNAMICS_EPCSAFT_DATASET.exists():
        raise FileNotFoundError(
            "MEA ePC-SAFT dataset not found at "
            f"{MEA_THERMODYNAMICS_EPCSAFT_DATASET}. Expected a repo-vendored dataset under "
            "src/mea_absorption_column/data/epcsaft_datasets, or set MEA_THERMODYNAMICS_EPCSAFT_DATASET "
            "for an explicit external comparison."
        )
    return _v02_mixture(
        str(MEA_THERMODYNAMICS_EPCSAFT_DATASET),
        tuple(species_key),
        float(T_key),
    )


def epcsaft_liquid_transport_state(T, P, composition) -> EpcsaftLiquidTransportState:
    """Evaluate one charged liquid state and its exact fixed-T,P tangent block."""

    values = np.asarray(composition, dtype=float)
    species = tuple(_ionic_species_for_size(values.size))
    if (
        values.shape != (len(species),)
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise EpcsaftFixedPressureDerivativeError(
            "provider_domain_rejection",
            "composition must contain one positive finite mole fraction per ionic species",
        )
    if abs(float(np.sum(values)) - 1.0) > 1.0e-12:
        raise EpcsaftFixedPressureDerivativeError(
            "normalization_failure", "composition must sum to one without clipping"
        )
    charges = np.asarray([IONIC_CHARGE_BY_SPECIES[item] for item in species])
    if abs(float(charges @ values)) > 1.0e-12:
        raise EpcsaftFixedPressureDerivativeError(
            "electroneutrality_failure",
            "composition must be electroneutral without projection",
        )

    model = epcsaft_dataset_mixture(species, float(T))
    state = _v02_state(
        model,
        temperature_k=float(T),
        pressure_pa=float(P),
        composition=values,
        phase="liquid",
    )
    block = state.fixed_pressure_composition_derivatives
    if block is None or block.get("status") != "available":
        failure = None if block is None else block.get("failure")
        raise EpcsaftFixedPressureDerivativeError(
            getattr(failure, "code", "fixed_pressure_derivatives_unavailable"),
            getattr(failure, "diagnostic", str(failure)),
        )
    if state.fugacity is None:
        raise EpcsaftFixedPressureDerivativeError(
            "fugacity_unavailable", "liquid state did not return species fugacities"
        )
    fugacities = np.asarray(
        [float(value.to("pascal").magnitude) for value in state.fugacity.value],
        dtype=float,
    )
    return EpcsaftLiquidTransportState(
        composition=values.copy(),
        fugacities_pa=fugacities,
        log_composition_basis=np.asarray(block["log_composition_basis"], dtype=float),
        chemical_potential_derivatives_over_rt=np.asarray(
            block["chemical_potential_derivatives_over_rt"], dtype=float
        ),
        coordinate_component_ids=tuple(block["coordinate_component_ids"]),
        dependent_component_ids=tuple(block["dependent_component_ids"]),
        condition_measure=(
            None
            if block["condition_measure"] is None
            else float(block["condition_measure"])
        ),
        artifact_fingerprint=str(block["artifact_fingerprint"]),
        parameter_fingerprint=str(model.parameter_fingerprint),
    )


def epcsaft_dataset_user_options(dataset: Path | None = None) -> dict:
    dataset_path = (
        Path(dataset) if dataset is not None else MEA_THERMODYNAMICS_EPCSAFT_DATASET
    )
    options_path = dataset_path / "user_options.json"
    if not options_path.exists():
        return {}
    return json.loads(options_path.read_text(encoding="utf-8"))


def epcsaft_state_contribution_diagnostics(
    T,
    P,
    composition,
    *,
    phase="liq",
    mixture_kind="neutral",
    user_options=None,
) -> dict:
    if user_options:
        raise ValueError(
            "Runtime ePC-SAFT user-option overrides were removed in API 0.2. "
            "Diagnostic variants must use separately identified parameter documents."
        )
    species_key = tuple(IONIC_LIQUID_SPECIES if mixture_kind == "ionic" else SPECIES)
    composition_arr = np.asarray(composition, dtype=float)
    composition_arr = np.maximum(composition_arr, COMPOSITION_FLOOR)
    composition_arr = composition_arr / float(np.sum(composition_arr))
    if mixture_kind == "ionic":
        composition_arr = _enforce_electroneutrality(composition_arr, species_key)
    if mixture_kind == "neutral":
        mixture = epcsaft_mixture()
    elif mixture_kind in {"ionic", "external_neutral"}:
        mixture = epcsaft_dataset_mixture(species_key, float(T))
    else:
        raise ValueError(f"Unknown ePC-SAFT mixture kind: {mixture_kind}")

    state = _v02_state(
        mixture, temperature_k=float(T), pressure_pa=float(P),
        composition=composition_arr, phase=phase,
    )
    phi = np.asarray(_v02_fugacity_coefficients(state), dtype=float)
    ares_terms = {
        "hc": float(state.hard_chain),
        "disp": float(state.dispersion),
        "assoc": float(state.association),
        "ion": float(state.debye_huckel),
        "born": float(state.born),
    }
    return {
        "mixture_kind": mixture_kind,
        "phase": str(phase),
        "species": list(species_key),
        "temperature_K": float(T),
        "pressure_Pa": float(P),
        "composition": composition_arr.tolist(),
        "density_mol_m3": _v02_molar_density_value(state),
        "parameter_fingerprint": str(mixture.parameter_fingerprint),
        "phi_co2": float(phi[CO2_INDEX]),
        "lnfugcoef_co2_terms": {},
        "ares_terms": ares_terms,
    }


def epcsaft_phi_co2(
    T, P, composition, phase, mixture_kind="neutral"
) -> float:
    composition_arr = np.asarray(composition, dtype=float)
    if mixture_kind == "ionic":
        epcsaft_runtime_user_options()
        species_key = tuple(_ionic_species_for_size(composition_arr.size))
        composition_arr = _enforce_electroneutrality(composition_arr, species_key)
    elif mixture_kind == "external_neutral":
        epcsaft_runtime_user_options()
        species_key = tuple(SPECIES)
    else:
        species_key = tuple(SPECIES)
    if mixture_kind == "neutral":
        mixture = epcsaft_mixture()
    elif mixture_kind == "ionic":
        mixture = epcsaft_dataset_mixture(species_key, float(T))
    elif mixture_kind == "external_neutral":
        mixture = epcsaft_dataset_mixture(species_key, float(T))
    else:
        raise ValueError(f"Unknown ePC-SAFT mixture kind: {mixture_kind}")
    state = _v02_state(
        mixture, temperature_k=float(T), pressure_pa=float(P),
        composition=composition_arr, phase=phase,
    )
    phi = np.asarray(_v02_fugacity_coefficients(state), dtype=float)
    phi_co2 = float(phi[CO2_INDEX])
    if not math.isfinite(phi_co2) or phi_co2 <= 0.0:
        raise RuntimeError(f"Invalid ePC-SAFT CO2 fugacity coefficient: {phi_co2!r}")
    return phi_co2


def ideal_henry_fugacity(y, x_true, Cl_true, H_CO2_mix, P, P_sat_H2O):
    y_co2 = float(y[0])
    y_h2o = float(y[1])
    x_h2o_true = float(x_true[2])
    cl_co2_true = float(Cl_true[0])

    fl_co2 = cl_co2_true * float(H_CO2_mix)
    fv_co2 = y_co2 * float(P)
    fl_h2o = x_h2o_true * float(P_sat_H2O)
    fv_h2o = y_h2o * float(P)
    return fl_co2, fv_co2, fl_h2o, fv_h2o


def epcsaft_neutral_fugacity(y, x_true, Tl, Tv, P, P_sat_H2O):
    liquid_x = neutral_liquid_composition(x_true)
    vapor_x = neutral_vapor_composition(y)
    phi_l_co2 = epcsaft_phi_co2(Tl, P, liquid_x, phase="liq")
    phi_v_co2 = epcsaft_phi_co2(Tv, P, vapor_x, phase="vap")

    fl_co2 = liquid_x[CO2_INDEX] * phi_l_co2 * float(P)
    fv_co2 = float(y[0]) * phi_v_co2 * float(P)
    fl_h2o = liquid_x[H2O_INDEX] * float(P_sat_H2O)
    fv_h2o = float(y[1]) * float(P)
    return fl_co2, fv_co2, fl_h2o, fv_h2o


def epcsaft_ionic_fugacity(y, x_true, Tl, Tv, P, P_sat_H2O):
    liquid_x = ionic_liquid_composition(x_true)
    vapor_x = neutral_vapor_composition(y)
    phi_l_co2 = epcsaft_phi_co2(Tl, P, liquid_x, phase="liq", mixture_kind="ionic")
    phi_v_co2 = epcsaft_phi_co2(
        Tv, P, vapor_x, phase="vap", mixture_kind="external_neutral"
    )

    fl_co2 = liquid_x[CO2_INDEX] * phi_l_co2 * float(P)
    fv_co2 = float(y[0]) * phi_v_co2 * float(P)
    fl_h2o = liquid_x[H2O_INDEX] * float(P_sat_H2O)
    fv_h2o = float(y[1]) * float(P)
    return fl_co2, fv_co2, fl_h2o, fv_h2o


def compute_fugacity(
    model,
    y,
    x_true,
    Cl_true,
    Tl,
    Tv,
    H_CO2_mix,
    P,
    P_sat_H2O,
):
    normalized_model = (model or "ideal_henry").lower()
    if normalized_model in {"ideal", "ideal_henry", "henry"}:
        return ideal_henry_fugacity(y, x_true, Cl_true, H_CO2_mix, P, P_sat_H2O)
    if normalized_model in {"epcsaft", "epcsaft_neutral", "epc-saft"}:
        return epcsaft_neutral_fugacity(y, x_true, Tl, Tv, P, P_sat_H2O)
    if normalized_model in {
        "epcsaft_ionic",
        "epcsaft_electrolyte",
        "epcsaft_full_ionic",
        "epcsaft_reactive_six",
        "epcsaft_reactive_six_concentration",
        "epcsaft_reactive_six_activity",
        "epcsaft_reactive_six_activity_converted",
        "epcsaft_reactive_six_activity_rebased",
        "epcsaft_reactive_nine",
        "epcsaft_reactive_nine_activity",
        "epcsaft_reactive_nine_activity_converted",
        "epcsaft_reactive_nine_activity_rebased",
        "epcsaft_reactive_nine_tabulated",
        "epcsaft_full_species_activity",
        "epcsaft_full_species_activity_converted",
        "epcsaft_full_species_activity_rebased",
    }:
        return epcsaft_ionic_fugacity(y, x_true, Tl, Tv, P, P_sat_H2O)
    raise ValueError(
        "Choose ideal_henry, epcsaft_neutral, epcsaft_ionic, "
        "or an experimental epcsaft_reactive_* mode."
    )


def guarded_compute_fugacity(
    model,
    y,
    x_true,
    Cl_true,
    Tl,
    Tv,
    H_CO2_mix,
    P,
    P_sat_H2O,
    diagnostics=None,
):
    try:
        _validate_fugacity_state(y, x_true, Tl, Tv, P)
        values = compute_fugacity(
            model,
            y,
            x_true,
            Cl_true,
            Tl,
            Tv,
            H_CO2_mix,
            P,
            P_sat_H2O,
        )
        values = np.asarray(values, dtype=float)
        if values.shape != (4,) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Fugacities must be four finite nonnegative values")
        return tuple(values)
    except Exception as exc:
        record_invalid_state(diagnostics, f"fugacity guard: {exc}")
        raise


def _validate_fugacity_state(y, x_true, Tl, Tv, P):
    values = np.asarray([Tl, Tv, P], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("non-finite T/P state")
    if not (TEMPERATURE_MIN_K <= float(Tl) <= TEMPERATURE_MAX_K):
        raise ValueError(f"liquid temperature outside guarded range: {Tl!r}")
    if not (TEMPERATURE_MIN_K <= float(Tv) <= TEMPERATURE_MAX_K):
        raise ValueError(f"vapor temperature outside guarded range: {Tv!r}")
    if float(P) <= 0.0:
        raise ValueError(f"non-positive pressure: {P!r}")
    for name, composition in {"vapor": y, "liquid_true": x_true}.items():
        arr = np.asarray(composition, dtype=float)
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"non-finite {name} composition")
        if np.sum(arr[:3]) <= 0.0:
            raise ValueError(f"non-positive {name} composition sum")
