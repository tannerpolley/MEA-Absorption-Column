from __future__ import annotations

import json
import math
import os
import sys
import time
from functools import lru_cache
import importlib.util
from importlib import resources
from pathlib import Path

import numpy as np

from mea_absorption_column.BVP.robust_core import record_guard_penalty, record_invalid_state


EPCSAFT_SOURCE_ROOT = Path(os.environ.get("MEA_EPCSAFT_ROOT", r"C:\Users\Tanner\Documents\git\ePC-SAFT"))
EPCSAFT_SOURCE_SRC = EPCSAFT_SOURCE_ROOT / "src"
EPCSAFT_BUILD_DIRS = (
    EPCSAFT_SOURCE_ROOT / "build" / "dev",
    EPCSAFT_SOURCE_ROOT / "build" / f"cp{sys.version_info.major}{sys.version_info.minor}-cp{sys.version_info.major}{sys.version_info.minor}-win_amd64",
)
SPECIES = ["CO2", "MEA", "H2O"]
IONIC_LIQUID_SPECIES = ["CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-"]
CO2_INDEX = 0
MEA_INDEX = 1
H2O_INDEX = 2
COMPOSITION_FLOOR = 1e-12
TEMPERATURE_MIN_K = 250.0
TEMPERATURE_MAX_K = 500.0
FUGACITY_FLOOR_PA = 1.0e-9
PACKAGED_EPCSAFT_DATASETS = resources.files("mea_absorption_column").joinpath("data/epcsaft_datasets")
DEFAULT_EPCSAFT_DATASET_NAME = os.environ.get("MEA_EPCSAFT_DATASET_NAME", "MEA_CO2_H2O_draft")


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


def epcsaft_source_fingerprint() -> dict:
    init_file = EPCSAFT_SOURCE_SRC / "epcsaft" / "__init__.py"
    stat = init_file.stat() if init_file.exists() else None
    return {
        "source_root": str(EPCSAFT_SOURCE_ROOT),
        "source_src": str(EPCSAFT_SOURCE_SRC),
        "exists": EPCSAFT_SOURCE_ROOT.exists(),
        "import_init": str(init_file),
        "modified_at_utc": None if stat is None else stat.st_mtime,
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
    if x_arr.size < len(IONIC_LIQUID_SPECIES):
        raise ValueError(
            f"ePC-SAFT ionic liquid composition requires {len(IONIC_LIQUID_SPECIES)} species, got {x_arr.size}."
        )
    ionic = np.asarray(x_arr[: len(IONIC_LIQUID_SPECIES)], dtype=float)
    ionic = np.maximum(ionic, COMPOSITION_FLOOR)
    total = float(ionic.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Ionic liquid composition must have a positive finite sum.")
    return ionic / total


def ensure_epcsaft_importable():
    try:
        from epcsaft import ePCSAFTMixture

        return ePCSAFTMixture
    except Exception as first_error:
        if EPCSAFT_SOURCE_SRC.exists() and str(EPCSAFT_SOURCE_SRC) not in sys.path:
            sys.path.insert(0, str(EPCSAFT_SOURCE_SRC))
        _preload_epcsaft_core()
        for module_name in list(sys.modules):
            if module_name == "epcsaft" or (module_name.startswith("epcsaft.") and module_name != "epcsaft._core"):
                del sys.modules[module_name]
        try:
            from epcsaft import ePCSAFTMixture

            return ePCSAFTMixture
        except Exception as second_error:
            raise RuntimeError(
                "Could not import the external ePC-SAFT package from the active environment "
                f"or from {EPCSAFT_SOURCE_SRC}. First error: {first_error}; second error: {second_error}"
            ) from second_error


def _preload_epcsaft_core():
    if "epcsaft._core" in sys.modules:
        return
    suffix = f"_core.cp{sys.version_info.major}{sys.version_info.minor}"
    candidates = []
    for build_dir in EPCSAFT_BUILD_DIRS:
        if build_dir.exists():
            candidates.extend(build_dir.glob(f"{suffix}*.pyd"))
    for candidate in candidates:
        spec = importlib.util.spec_from_file_location("epcsaft._core", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["epcsaft._core"] = module
        try:
            spec.loader.exec_module(module)
            return
        except Exception:
            sys.modules.pop("epcsaft._core", None)
            continue


@lru_cache(maxsize=1)
def epcsaft_mixture():
    ePCSAFTMixture = ensure_epcsaft_importable()
    params = build_epcsaft_params()
    native_params = {key: value for key, value in params.items() if key not in {"species", "metadata_path"}}
    return ePCSAFTMixture.from_params(native_params, species=params["species"])


@lru_cache(maxsize=256)
def epcsaft_dataset_mixture(species_key: tuple[str, ...], T_key: float):
    ePCSAFTMixture = ensure_epcsaft_importable()
    if not MEA_THERMODYNAMICS_EPCSAFT_DATASET.exists():
        raise FileNotFoundError(
            "MEA ePC-SAFT dataset not found at "
            f"{MEA_THERMODYNAMICS_EPCSAFT_DATASET}. Expected a repo-vendored dataset under "
            "src/mea_absorption_column/data/epcsaft_datasets, or set MEA_THERMODYNAMICS_EPCSAFT_DATASET "
            "for an explicit external comparison."
        )
    species = list(species_key)
    seed = np.full(len(species), 1.0 / len(species), dtype=float)
    return ePCSAFTMixture.from_dataset(
        str(MEA_THERMODYNAMICS_EPCSAFT_DATASET),
        species,
        seed,
        float(T_key),
    )


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
        rho = float(state.molar_density())
    except Exception:
        return
    if math.isfinite(rho) and rho > 0.0:
        _EPCSAFT_RHO_GUESS_CACHE[_rho_guess_key(mixture_kind, phase)] = rho


def _pressure_state_with_optional_rho_guess(mixture, T, P, composition, phase, mixture_kind):
    rho_guess = _rho_guess_from_cache(mixture_kind, phase)
    kwargs = {
        "T": float(T),
        "x": np.asarray(composition, dtype=float),
        "P": float(P),
        "phase": phase,
    }
    if rho_guess is not None:
        kwargs["rho_guess"] = rho_guess
    try:
        state = mixture.state(**kwargs)
    except TypeError:
        kwargs.pop("rho_guess", None)
        state = mixture.state(**kwargs)
    _store_rho_guess(mixture_kind, phase, state)
    return state


def epcsaft_phi_co2(T, P, composition, phase, cache=True, mixture_kind="neutral") -> float:
    species_key = tuple(IONIC_LIQUID_SPECIES if mixture_kind == "ionic" else SPECIES)
    key = (str(mixture_kind), *_epcsaft_cache_key(T, P, composition, phase))
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
    state = _pressure_state_with_optional_rho_guess(mixture, T, P, composition, phase, mixture_kind)
    _EPCSAFT_CACHE_STATS["epcsaft_direct_density_solve_s"] += time.perf_counter() - start
    phi = np.asarray(state.fugacity_coefficient(natural_log=False), dtype=float)
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
