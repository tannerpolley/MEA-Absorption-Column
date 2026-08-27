from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
import importlib
import importlib.metadata
from importlib import resources
from pathlib import Path

import numpy as np

from mea_absorption_column.BVP.robust_core import record_guard_penalty, record_invalid_state
from mea_absorption_column.Thermodynamics.epcsaft_v02 import (
    fugacity_coefficients as _v02_fugacity_coefficients,
    mixture as _v02_mixture,
    molar_density_value as _v02_molar_density_value,
    pressure_value as _v02_pressure_value,
    state as _v02_state,
    state_at_density as _v02_state_at_density,
)


SPECIES = ["CO2", "MEA", "H2O"]
IONIC_LIQUID_SPECIES_6 = ["CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-"]
IONIC_LIQUID_SPECIES_9 = ["CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-"]
IONIC_LIQUID_SPECIES = IONIC_LIQUID_SPECIES_6
CO2_INDEX = 0
MEA_INDEX = 1
H2O_INDEX = 2
COMPOSITION_FLOOR = 1e-12
TEMPERATURE_MIN_K = 250.0
TEMPERATURE_MAX_K = 500.0
FUGACITY_FLOOR_PA = 1.0e-9
PACKAGED_EPCSAFT_DATASETS = resources.files("mea_absorption_column").joinpath("data/epcsaft_datasets")
DEFAULT_EPCSAFT_DATASET_NAME = os.environ.get("MEA_EPCSAFT_DATASET_NAME", "MEA_CO2_H2O_ionic_fit")


def _resolve_epcsaft_dataset_path() -> Path:
    packaged = Path(str(PACKAGED_EPCSAFT_DATASETS.joinpath(DEFAULT_EPCSAFT_DATASET_NAME)))
    override = os.environ.get("MEA_THERMODYNAMICS_EPCSAFT_DATASET")
    if override:
        override_path = Path(override)
        if override_path.exists():
            return override_path
    return packaged


MEA_THERMODYNAMICS_EPCSAFT_DATASET = _resolve_epcsaft_dataset_path()
EPCSAFT_CACHE_T_DIGITS = int(os.environ.get("MEA_EPCSAFT_CACHE_T_DIGITS", "2"))
EPCSAFT_CACHE_X_DIGITS = int(os.environ.get("MEA_EPCSAFT_CACHE_X_DIGITS", "5"))
EPCSAFT_CACHE_P_ROUND_PA = float(os.environ.get("MEA_EPCSAFT_CACHE_P_ROUND_PA", "10.0"))
EPCSAFT_DATASET_T_DIGITS = int(os.environ.get("MEA_EPCSAFT_DATASET_T_DIGITS", "1"))
_EPCSAFT_PHI_CACHE: dict[tuple, float] = {}
_EPCSAFT_RHO_GUESS_CACHE: dict[tuple, float] = {}
_EPCSAFT_CACHE_STATS = {
    "epcsaft_cache_hits": 0,
    "epcsaft_cache_misses": 0,
    "epcsaft_direct_density_solve_s": 0.0,
    "epcsaft_rho_guess_hits": 0,
    "epcsaft_rho_guess_misses": 0,
}


def _parameter_resource():
    return resources.files("mea_absorption_column").joinpath("data/epcsaft_neutral/parameters.json")


@lru_cache(maxsize=1)
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
        "source_detail": json.dumps(direct_url, sort_keys=True) if direct_url else str(module_file),
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


def _enforce_electroneutrality(composition: np.ndarray, species: list[str] | tuple[str, ...]) -> np.ndarray:
    charges_by_species = {
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
    values = np.asarray(composition, dtype=float).copy()
    charges = np.asarray([charges_by_species[item] for item in species], dtype=float)
    residual = float(np.dot(values, charges))
    if abs(residual) > 1.0e-15:
        candidate_sign = -1.0 if residual > 0.0 else 1.0
        candidates = np.flatnonzero(charges * candidate_sign > 0.0)
        if candidates.size == 0:
            raise ValueError("Ionic ePC-SAFT composition cannot be projected to electroneutrality.")
        index = int(candidates[np.argmax(values[candidates])])
        values[index] += abs(residual / charges[index])
        values /= float(np.sum(values))
    if abs(float(np.dot(values, charges))) > 1.0e-12:
        raise ValueError("Ionic ePC-SAFT composition does not satisfy electroneutrality.")
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


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=512)
def epcsaft_dataset_mixture(species_key: tuple[str, ...], T_key: float, user_options_json: str = "{}"):
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


def epcsaft_dataset_user_options(dataset: Path | None = None) -> dict:
    dataset_path = Path(dataset) if dataset is not None else MEA_THERMODYNAMICS_EPCSAFT_DATASET
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
        mixture = epcsaft_dataset_mixture(species_key, _epcsaft_dataset_T_key(T))
    else:
        raise ValueError(f"Unknown ePC-SAFT mixture kind: {mixture_kind}")

    state = _pressure_state_with_optional_rho_guess(
        mixture,
        T,
        P,
        composition_arr,
        phase,
        f"{mixture_kind}_diagnostic",
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


def clear_epcsaft_phi_cache():
    _EPCSAFT_PHI_CACHE.clear()
    _EPCSAFT_RHO_GUESS_CACHE.clear()
    _EPCSAFT_CACHE_STATS["epcsaft_cache_hits"] = 0
    _EPCSAFT_CACHE_STATS["epcsaft_cache_misses"] = 0
    _EPCSAFT_CACHE_STATS["epcsaft_direct_density_solve_s"] = 0.0
    _EPCSAFT_CACHE_STATS["epcsaft_rho_guess_hits"] = 0
    _EPCSAFT_CACHE_STATS["epcsaft_rho_guess_misses"] = 0


def epcsaft_cache_stats() -> dict:
    return dict(_EPCSAFT_CACHE_STATS)


def _round_pressure_for_cache(P):
    pressure = float(P)
    increment = max(EPCSAFT_CACHE_P_ROUND_PA, 1.0e-12)
    return float(np.round(pressure / increment) * increment)


def _epcsaft_cache_key(T, P, composition, phase):
    comp = np.asarray(composition, dtype=float)
    rounded = tuple(float(np.round(value, EPCSAFT_CACHE_X_DIGITS)) for value in comp)
    return (
        str(phase),
        float(np.round(float(T), EPCSAFT_CACHE_T_DIGITS)),
        _round_pressure_for_cache(P),
        rounded,
    )


def _epcsaft_dataset_T_key(T):
    return float(np.round(float(T), EPCSAFT_DATASET_T_DIGITS))


def _rho_guess_key(mixture_kind, phase):
    return (str(mixture_kind), str(phase))


def _rho_guess_from_cache(mixture_kind, phase):
    key = _rho_guess_key(mixture_kind, phase)
    value = _EPCSAFT_RHO_GUESS_CACHE.get(key)
    if value is not None and math.isfinite(float(value)) and float(value) > 0.0:
        _EPCSAFT_CACHE_STATS["epcsaft_rho_guess_hits"] += 1
        return float(value)
    _EPCSAFT_CACHE_STATS["epcsaft_rho_guess_misses"] += 1
    return None


def _store_rho_guess(mixture_kind, phase, state):
    try:
        rho = _v02_molar_density_value(state)
    except Exception:
        return
    if math.isfinite(rho) and rho > 0.0:
        _EPCSAFT_RHO_GUESS_CACHE[_rho_guess_key(mixture_kind, phase)] = rho


def _pressure_state_with_optional_rho_guess(mixture, T, P, composition, phase, mixture_kind):
    rho_guess = _rho_guess_from_cache(mixture_kind, phase)
    if rho_guess is not None:
        try:
            state = _state_from_density_newton(
                mixture,
                T=float(T),
                P=float(P),
                composition=np.asarray(composition, dtype=float),
                rho_guess=rho_guess,
            )
            _store_rho_guess(mixture_kind, phase, state)
            return state
        except Exception:
            pass
    state = _v02_state(
        mixture,
        temperature_k=float(T),
        pressure_pa=float(P),
        composition=np.asarray(composition, dtype=float),
        phase=phase,
    )
    _store_rho_guess(mixture_kind, phase, state)
    return state


def _state_from_density_newton(mixture, *, T, P, composition, rho_guess):
    rho = max(float(rho_guess), 1.0e-9)
    target = float(P)
    for _ in range(12):
        current = _v02_state_at_density(
            mixture,
            temperature_k=T,
            density_mol_m3=rho,
            composition=composition,
        )
        residual = _v02_pressure_value(current) - target
        if abs(residual) <= max(1.0e-5 * target, 1.0e-2):
            if current.fugacity is None:
                raise RuntimeError("Density-closed ePC-SAFT state has no stable fugacity value.")
            return current

        step = max(1.0e-5 * rho, 1.0e-4)
        plus = _v02_state_at_density(
            mixture,
            temperature_k=T,
            density_mol_m3=rho + step,
            composition=composition,
        )
        derivative = (_v02_pressure_value(plus) - _v02_pressure_value(current)) / step
        if not math.isfinite(derivative) or abs(derivative) < 1.0e-12:
            raise RuntimeError("Invalid ePC-SAFT pressure-density slope.")
        delta = residual / derivative
        max_delta = 0.2 * rho
        rho = max(rho - float(np.clip(delta, -max_delta, max_delta)), 1.0e-9)
    raise RuntimeError("Warm-started ePC-SAFT density closure did not converge.")


def epcsaft_phi_co2(T, P, composition, phase, cache=True, mixture_kind="neutral") -> float:
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
    key = (str(mixture_kind), *_epcsaft_cache_key(T, P, composition_arr, phase))
    if cache and key in _EPCSAFT_PHI_CACHE:
        _EPCSAFT_CACHE_STATS["epcsaft_cache_hits"] += 1
        return _EPCSAFT_PHI_CACHE[key]
    _EPCSAFT_CACHE_STATS["epcsaft_cache_misses"] += 1
    if mixture_kind == "neutral":
        mixture = epcsaft_mixture()
    elif mixture_kind == "ionic":
        mixture = epcsaft_dataset_mixture(species_key, _epcsaft_dataset_T_key(T))
    elif mixture_kind == "external_neutral":
        mixture = epcsaft_dataset_mixture(species_key, _epcsaft_dataset_T_key(T))
    else:
        raise ValueError(f"Unknown ePC-SAFT mixture kind: {mixture_kind}")
    start = time.perf_counter()
    state = _pressure_state_with_optional_rho_guess(mixture, T, P, composition_arr, phase, mixture_kind)
    _EPCSAFT_CACHE_STATS["epcsaft_direct_density_solve_s"] += time.perf_counter() - start
    phi = np.asarray(_v02_fugacity_coefficients(state), dtype=float)
    phi_co2 = float(phi[CO2_INDEX])
    if not math.isfinite(phi_co2) or phi_co2 <= 0.0:
        raise RuntimeError(f"Invalid ePC-SAFT CO2 fugacity coefficient: {phi_co2!r}")
    if cache:
        _EPCSAFT_PHI_CACHE[key] = phi_co2
    return phi_co2


def epcsaft_phi_co2_batch(records, cache=True) -> list[float]:
    """Evaluate CO2 fugacity coefficients for many states with cache-aware de-duplication.

    This is intentionally a MEA-side batching seam. The external ePC-SAFT package
    still performs each state evaluation; this helper prevents repeated Python
    calls for duplicate or near-duplicate BVP mesh states under the local cache
    quantization.
    """
    resolved: dict[tuple, float] = {}
    results: list[float] = []
    for record in records:
        mixture_kind = record.get("mixture_kind", "neutral")
        key = (
            str(mixture_kind),
            *_epcsaft_cache_key(record["T"], record["P"], record["composition"], record["phase"]),
        )
        if key not in resolved:
            resolved[key] = epcsaft_phi_co2(
                record["T"],
                record["P"],
                record["composition"],
                record["phase"],
                cache=cache,
                mixture_kind=mixture_kind,
            )
        results.append(resolved[key])
    return results


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
    phi_v_co2 = epcsaft_phi_co2(Tv, P, vapor_x, phase="vap", mixture_kind="external_neutral")

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
    epcsaft_fugacity_blend=1.0,
):
    normalized_model = (model or "ideal_henry").lower()
    if normalized_model in {"ideal", "ideal_henry", "henry"}:
        return ideal_henry_fugacity(y, x_true, Cl_true, H_CO2_mix, P, P_sat_H2O)
    if normalized_model in {"epcsaft", "epcsaft_neutral", "epc-saft"}:
        blend = float(np.clip(epcsaft_fugacity_blend, 0.0, 1.0))
        epcsaft_values = epcsaft_neutral_fugacity(y, x_true, Tl, Tv, P, P_sat_H2O)
        if blend >= 1.0:
            return epcsaft_values
        henry_values = ideal_henry_fugacity(y, x_true, Cl_true, H_CO2_mix, P, P_sat_H2O)
        if blend <= 0.0:
            return henry_values
        return tuple(
            (1.0 - blend) * float(henry_value) + blend * float(epcsaft_value)
            for henry_value, epcsaft_value in zip(henry_values, epcsaft_values)
        )
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
        blend = float(np.clip(epcsaft_fugacity_blend, 0.0, 1.0))
        epcsaft_values = epcsaft_ionic_fugacity(y, x_true, Tl, Tv, P, P_sat_H2O)
        if blend >= 1.0:
            return epcsaft_values
        henry_values = ideal_henry_fugacity(y, x_true, Cl_true, H_CO2_mix, P, P_sat_H2O)
        if blend <= 0.0:
            return henry_values
        return tuple(
            (1.0 - blend) * float(henry_value) + blend * float(epcsaft_value)
            for henry_value, epcsaft_value in zip(henry_values, epcsaft_values)
        )
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
    epcsaft_fugacity_blend=1.0,
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
            epcsaft_fugacity_blend=epcsaft_fugacity_blend,
        )
        return tuple(_positive_finite(value) for value in values)
    except Exception as exc:
        record_invalid_state(diagnostics, f"fugacity guard: {exc}")
        record_guard_penalty(diagnostics)
        return _fallback_fugacity(y, x_true, Cl_true, H_CO2_mix, P, P_sat_H2O)


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


def _positive_finite(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        return FUGACITY_FLOOR_PA
    return value


def _fallback_fugacity(y, x_true, Cl_true, H_CO2_mix, P, P_sat_H2O):
    y_arr = np.nan_to_num(np.asarray(y, dtype=float), nan=COMPOSITION_FLOOR)
    x_arr = np.nan_to_num(np.asarray(x_true, dtype=float), nan=COMPOSITION_FLOOR)
    cl_arr = np.nan_to_num(np.asarray(Cl_true, dtype=float), nan=COMPOSITION_FLOOR)
    pressure = max(float(np.nan_to_num(P, nan=101325.0)), 1.0)
    henry = max(float(np.nan_to_num(H_CO2_mix, nan=1.0)), 1.0)
    psat = max(float(np.nan_to_num(P_sat_H2O, nan=FUGACITY_FLOOR_PA)), FUGACITY_FLOOR_PA)
    return (
        max(float(cl_arr[0]) * henry, FUGACITY_FLOOR_PA),
        max(float(y_arr[0]) * pressure, FUGACITY_FLOOR_PA),
        max(float(x_arr[2]) * psat, FUGACITY_FLOOR_PA),
        max(float(y_arr[1]) * pressure, FUGACITY_FLOOR_PA),
    )
