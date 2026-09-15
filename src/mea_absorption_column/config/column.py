"""Immutable configuration records for selectable absorber runs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import sys
import tomllib
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse


SCHEMA_VERSION = 1
SEVEN_PRESET = "seven_state_legacy"
TWELVE_PRESET = "twelve_state_conserved"
_ROOT = Path(__file__).resolve().parents[3]
_REACTIVE_DATASET = "src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle"
_NEUTRAL_VAPOR_DATASET = "src/mea_absorption_column/data/epcsaft_datasets/MEA_neutral_vapor"
_CASE_3C_INPUT = "analyses/bvp_solution_methods/input/case_3c.json"

_METHOD_DEFAULTS = {
    "single": {
        "integrator": "euler", "ivp_method": "BDF", "ivp_rtol": 1e-5,
        "ivp_atol": 1e-8, "root_method": "Krylov", "fatol": 0.1,
        "maxiter": 50, "line_search": "armijo", "display": False,
    },
    "scipy-bvp": {
        "mesh_points": 51, "max_nodes": 1000, "tol": 0.5,
        "bc_tol": 0.001, "verbose": 0, "use_finite_jacobian": False,
    },
    "finite": {"maxfev": 500, "tol": 1e-4},
}
_COMMON_NUMERIC_KEYS = {
    "co2_mass_transfer_model", "enhancement_type", "chemical_equilibrium_model",
    "co2_capture_guess_pct", "h2o_capture_guess_pct", "mass_transfer_factor",
    "heat_transfer_factor", "eta_psi", "co2_flux_mode", "epcsaft_fugacity_blend",
    "guard_rhs", "strict_domain_guards", "transform_mode", "thermal_state_mode",
    "gas_velocity_area_exponent", "gas_velocity_area_reference_m_s", "gas_velocity_area_bounds",
    "jacobian_mode", "shooting_seed_jacobian_mode", "continuation_stage", "continuation_path",
    "return_profiles", "return_internal_profile", "profile_pngs", "profile_csvs", "profile_csv_dir",
    "reactive_dataset", "reactive_loading_anchor", "reactive_reuse_states", "reactive_kij_scale",
    "reactive_reaction_scale", "reactive_max_log_loading_step", "reactive_max_loading_steps",
    "success_boundary_residual_max", "success_capture_error_max_pct",
    "intercooler_strength", "intercooler_model", "max_runtime_s",
    "capture_correction_model", "seed_from_shooting", "seed_from_collapsed",
    "seed_from_henry", "use_finite_jacobian", "seed_jacobian_mode",
    "multistart_capture_guesses", "multistart_mass_transfer_factors",
    "multistart_intercooler_strengths", "multistart_co2_flux_modes",
    "initial_guess_scaled", "initial_guess_z", "case_source",
}


class ConfigurationError(ValueError):
    """A request cannot be resolved without changing its scientific meaning."""


class CapabilityRefusal(RuntimeError):
    """A resolved formulation cannot be prepared by the selected runtime."""


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError("Configuration values must be finite")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ConfigurationError(f"Unsupported configuration value: {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


def _sha256(value: Any) -> str:
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _section(request: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = request.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a table/mapping")
    return value


@dataclass(frozen=True)
class CaseRequest:
    source: str
    case_id: str
    data_type: str = "mole"
    vapor_composition_mode: str = "legacy_ratio"
    gas_flow_basis: str = "reported_total_wet"
    physical_input_file: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "id": self.case_id,
            "data_type": self.data_type,
            "vapor_composition_mode": self.vapor_composition_mode,
            "gas_flow_basis": self.gas_flow_basis,
            "physical_input_file": self.physical_input_file,
        }


@dataclass(frozen=True)
class ModelConfig:
    formulation: str
    thermo_model: str
    film_model: str
    energy_model: str
    pressure_model: str
    layout: str
    coordinate: str

    def as_dict(self) -> dict[str, str]:
        return {
            "formulation": self.formulation,
            "thermo_model": self.thermo_model,
            "film_model": self.film_model,
            "energy_model": self.energy_model,
            "pressure_model": self.pressure_model,
            "layout": self.layout,
            "coordinate": self.coordinate,
        }


@dataclass(frozen=True)
class NumericConfig:
    method: str
    settings: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "solver_settings": _thaw(self.settings)}


@dataclass(frozen=True)
class InitializationConfig:
    policy: str = "legacy_capture_temperature_guesses"
    values: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"policy": self.policy, "values": _thaw(self.values)}


@dataclass(frozen=True)
class AcceptanceConfig:
    numerical: str = "solver_termination_and_original_equations"
    physical: str = "not_evaluated"
    boundary_residual_max: float | None = None
    capture_error_max_pct: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "numerical": self.numerical,
            "physical": self.physical,
            "boundary_residual_max": self.boundary_residual_max,
            "capture_error_max_pct": self.capture_error_max_pct,
        }


@dataclass(frozen=True)
class ExecutionConfig:
    output_dir: str | None = None
    wall_limit_s: float | None = None
    threads: int = 1
    process_isolation: bool = False
    cache_policy: str = "legacy_explicit"
    resume: bool = False
    retry: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "wall_limit_s": self.wall_limit_s,
            "threads": self.threads,
            "process_isolation": self.process_isolation,
            "cache_policy": self.cache_policy,
            "resume": self.resume,
            "retry": self.retry,
        }


@dataclass(frozen=True)
class EngineConfig:
    wheel: str | None
    sha256: str | None
    commit: str | None
    python: str | None
    required: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "wheel": self.wheel,
            "sha256": self.sha256,
            "commit": self.commit,
            "python": self.python,
            "required": self.required,
        }


@dataclass(frozen=True)
class DependencyConfig:
    dataset: str | None = None
    mobility_law: str | None = None
    thermal_reference: str | None = None
    references: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "mobility_law": self.mobility_law,
            "thermal_reference": self.thermal_reference,
            "references": list(self.references),
        }


@dataclass(frozen=True)
class ColumnConfig:
    preset: str
    case: CaseRequest
    model: ModelConfig
    numerics: NumericConfig
    initialization: InitializationConfig
    acceptance: AcceptanceConfig
    execution: ExecutionConfig
    dependencies: DependencyConfig
    engine: EngineConfig
    schema_version: int = SCHEMA_VERSION
    resolved_config_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        identity = self.as_dict(include_fingerprint=False)
        identity["execution"] = dict(identity["execution"])
        identity["execution"].pop("resume", None)
        identity["execution"].pop("retry", None)
        object.__setattr__(self, "resolved_config_sha256", _sha256(identity))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "preset": self.preset,
            "case": self.case.as_dict(),
            "model": self.model.as_dict(),
            "numerics": self.numerics.as_dict(),
            "initialization": self.initialization.as_dict(),
            "acceptance": self.acceptance.as_dict(),
            "execution": self.execution.as_dict(),
            "dependencies": self.dependencies.as_dict(),
            "engine": self.engine.as_dict(),
        }
        if include_fingerprint:
            result["resolved_config_sha256"] = self.resolved_config_sha256
        return result

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _default_engine() -> EngineConfig:
    wheel: str | None = None
    pyproject = _ROOT / "pyproject.toml"
    if pyproject.exists():
        with pyproject.open("rb") as stream:
            for dependency in tomllib.load(stream)["project"]["dependencies"]:
                if dependency.startswith("epcsaft @ file:"):
                    wheel = unquote(urlparse(dependency.split("@", 1)[1].strip()).path)
                    break
    contract = _ROOT / "integration" / "epcsaft_contract.json"
    sha256 = commit = None
    if contract.exists():
        record = json.loads(contract.read_text(encoding="utf-8"))
        identity = record.get("final_identity", {})
        sha256 = identity.get("wheel_sha256")
        commit = identity.get("engine_commit")
    return EngineConfig(wheel, sha256, commit, sys.executable, False)


def _engine(request: Mapping[str, Any], *, required: bool) -> EngineConfig:
    values = dict(request)
    unknown = set(values) - {"wheel", "sha256", "commit", "python", "required"}
    if unknown:
        raise ConfigurationError(f"Unknown engine keys: {sorted(unknown)}")
    default = _default_engine()
    wheel = values.get("wheel", default.wheel)
    sha256 = values.get("sha256", default.sha256)
    commit = values.get("commit", default.commit)
    python = values.get("python", default.python)
    if wheel is not None and not isinstance(wheel, str):
        raise ConfigurationError("engine.wheel must be a path string")
    if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256.lower())):
        raise ConfigurationError("engine.sha256 must be a 64-character hexadecimal digest")
    if isinstance(sha256, str):
        sha256 = sha256.lower()
    if python is not None and not isinstance(python, str):
        raise ConfigurationError("engine.python must be a path string")
    supplied_required = values.get("required", required)
    if not isinstance(supplied_required, bool):
        raise ConfigurationError("engine.required must be boolean")
    return EngineConfig(wheel, sha256, commit, python, bool(required or supplied_required))


def _case(request: Mapping[str, Any]) -> CaseRequest:
    values = dict(request)
    unknown = set(values) - {
        "source", "dataset", "id", "case_id", "data_type", "vapor_composition_mode",
        "gas_flow_basis", "physical_input_file",
    }
    if unknown:
        raise ConfigurationError(f"Unknown case keys: {sorted(unknown)}")
    source = values.get("source", values.get("dataset"))
    case_id = values.get("id", values.get("case_id"))
    data_type = values.get("data_type", "mole")
    vapor_mode = values.get("vapor_composition_mode", "legacy_ratio")
    gas_basis = values.get("gas_flow_basis", "reported_total_wet")
    if data_type not in {"mole", "mass"}:
        raise ConfigurationError("case.data_type must be 'mole' or 'mass'")
    if vapor_mode not in {"legacy_ratio", "dry_saturated", "input_o2"}:
        raise ConfigurationError(f"Unsupported vapor_composition_mode: {vapor_mode!r}")
    if gas_basis not in {"reported_total_wet", "reported_dry_mass"}:
        raise ConfigurationError(f"Unsupported gas_flow_basis: {gas_basis!r}")
    physical = values.get("physical_input_file")
    if physical is not None and not isinstance(physical, str):
        raise ConfigurationError("case.physical_input_file must be a path string")
    if source is None and physical is not None:
        source = "physical_input_file"
    if not isinstance(source, str) or not source.strip():
        raise ConfigurationError("case.source is required")
    source = source.strip()
    if source not in {"C_cases_data", "C_cases_campaign_inputs", "NCCC_Data", "NCCC_2014_cases", "NCCC_2017_cases", "SRP_method_cases"}:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = _ROOT / source_path
        source = str(source_path.resolve())
    if physical is not None:
        physical_path = Path(physical)
        if not physical_path.is_absolute():
            physical_path = _ROOT / physical_path
        physical = str(physical_path.resolve())
        if source == str((_ROOT / "physical_input_file").resolve()):
            source = physical
        if case_id is None and physical_path.is_file():
            try:
                payload = json.loads(physical_path.read_text(encoding="utf-8"))
                case_id = (payload.get("physical_inputs") or {}).get("case_id")
            except (OSError, json.JSONDecodeError):
                case_id = None
    if not isinstance(case_id, str) or not case_id.strip():
        raise ConfigurationError("case.id is required")
    return CaseRequest(source, case_id.strip(), data_type, vapor_mode, gas_basis, physical)


def _numeric(request: Mapping[str, Any]) -> NumericConfig:
    values = dict(request)
    settings = values.pop("solver_settings", values.pop("settings", {}))
    method = values.pop("method", "single")
    if method == "finite-difference":
        method = "finite"
    if values:
        raise ConfigurationError(f"Unknown numerics keys: {sorted(values)}")
    if method not in _METHOD_DEFAULTS:
        raise ConfigurationError(f"Unsupported seven-state method: {method!r}")
    if not isinstance(settings, Mapping):
        raise ConfigurationError("numerics.solver_settings must be a table/mapping")
    settings = dict(settings)
    unknown = set(settings) - set(_METHOD_DEFAULTS[method]) - _COMMON_NUMERIC_KEYS
    if unknown:
        raise ConfigurationError(f"Unknown or irrelevant solver settings: {sorted(unknown)}")
    if settings.get("co2_mass_transfer_model", "enhancement_factor") != "enhancement_factor":
        raise ConfigurationError("seven_state_legacy requires co2_mass_transfer_model='enhancement_factor'")
    settings.setdefault("co2_mass_transfer_model", "enhancement_factor")
    if settings.get("jacobian_mode") == "native" and method != "scipy-bvp":
        raise ConfigurationError("native seven-state Jacobian is supported only by unstaged scipy-bvp")
    if settings.get("thermal_state_mode", "enthalpy") != "enthalpy":
        raise ConfigurationError("seven_state_legacy uses its enthalpy state layout")
    if settings.get("transform_mode") not in {None, "raw", "bounded_guarded_raw_state"}:
        raise ConfigurationError("Unsupported seven-state transform_mode")
    if settings.get("co2_flux_mode") not in {None, "bidirectional", "forward_only", "absorption_only"}:
        raise ConfigurationError("Unsupported co2_flux_mode")
    if settings.get("enhancement_type") not in {None, "explicit", "implicit"}:
        raise ConfigurationError("Unsupported enhancement_type")
    resolved = dict(_METHOD_DEFAULTS[method])
    resolved.update(settings)
    integer_keys = {"mesh_points", "max_nodes", "maxiter", "maxfev", "reactive_max_loading_steps"}
    positive_keys = {"mesh_points", "max_nodes", "maxiter", "maxfev", "tol", "bc_tol", "fatol", "ivp_rtol", "ivp_atol", "mass_transfer_factor", "heat_transfer_factor", "eta_psi"}
    for key, value in resolved.items():
        if key == "verbose":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError("numerics.solver_settings.verbose must be a nonnegative integer")
        elif key in integer_keys:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"numerics.solver_settings.{key} must be a positive integer")
        elif key in positive_keys:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"numerics.solver_settings.{key} must be finite and positive")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ConfigurationError(f"numerics.solver_settings.{key} must be finite")
    for key in (
        "guard_rhs", "strict_domain_guards", "return_profiles", "return_internal_profile",
        "profile_pngs", "profile_csvs", "reactive_reuse_states", "seed_from_shooting",
        "seed_from_collapsed", "seed_from_henry", "use_finite_jacobian",
    ):
        if key in resolved and not isinstance(resolved[key], bool):
            raise ConfigurationError(f"numerics.solver_settings.{key} must be boolean")
    if resolved.get("intercooler_model") not in {None, "liquid_temperature_reset", "pumparound_temperature_approach"}:
        raise ConfigurationError("Unsupported intercooler_model")
    if resolved.get("capture_correction_model") not in {None, "nccc_linear"}:
        raise ConfigurationError("Unsupported capture_correction_model")
    if resolved.get("max_runtime_s") is not None:
        value = resolved["max_runtime_s"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ConfigurationError("numerics.solver_settings.max_runtime_s must be finite and positive")
    bounds = resolved.get("gas_velocity_area_bounds")
    if bounds is not None:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ConfigurationError("gas_velocity_area_bounds must contain two values")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in bounds):
            raise ConfigurationError("gas_velocity_area_bounds must contain finite positive values")
    for key in (
        "multistart_capture_guesses", "multistart_mass_transfer_factors",
        "multistart_intercooler_strengths", "multistart_co2_flux_modes",
    ):
        if key in resolved:
            values = resolved[key]
            if not isinstance(values, (list, tuple)) or not values:
                raise ConfigurationError(f"numerics.solver_settings.{key} must be a nonempty sequence")
            if key == "multistart_co2_flux_modes":
                if any(value not in {"bidirectional", "forward_only", "absorption_only"} for value in values):
                    raise ConfigurationError(f"Unsupported values in {key}")
            elif any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                raise ConfigurationError(f"{key} must contain finite numeric values")
    return NumericConfig(method, _freeze(resolved))


def _conserved_numeric(request: Mapping[str, Any]) -> NumericConfig:
    """Resolve only the settings consumed by the native conserved methods."""
    values = dict(request)
    settings = values.pop("solver_settings", values.pop("settings", {}))
    method = values.pop("method", "trapezoidal")
    direct_nodes = values.pop("nodes", None)
    if values:
        raise ConfigurationError(f"Unknown numerics keys: {sorted(values)}")
    if method not in {"trapezoidal", "central", "shooting", "collocation"}:
        raise ConfigurationError(f"Unsupported twelve-state method: {method!r}")
    if not isinstance(settings, Mapping):
        raise ConfigurationError("numerics.solver_settings must be a table/mapping")
    defaults = {
        "trapezoidal": {"nodes": 11, "quadrature_points": 9, "tolerance": 1.0e-7, "max_iterations": 20},
        "central": {"nodes": 11, "quadrature_points": 9, "tolerance": 1.0e-7, "max_iterations": 20},
        "shooting": {"nodes": 11, "quadrature_points": 9, "tolerance": 1.0e-7, "boundary_tolerance": 1.0e-7, "max_nodes": 100},
        "collocation": {"nodes": 11, "quadrature_points": 9, "tolerance": 1.0e-7, "boundary_tolerance": 1.0e-7, "max_nodes": 100},
    }[method]
    settings = dict(settings)
    if direct_nodes is not None:
        if "nodes" in settings:
            raise ConfigurationError("nodes must be supplied either directly or in solver_settings")
        settings["nodes"] = direct_nodes
    override_keys = {"reactive_loading_anchor", "reactive_max_log_loading_step", "reactive_max_loading_steps"}
    unknown = set(settings) - set(defaults) - override_keys
    if unknown:
        raise ConfigurationError(f"Unknown or irrelevant conserved solver settings: {sorted(unknown)}")
    resolved = dict(defaults)
    resolved.update(settings)
    integer_keys = {"nodes", "quadrature_points", "max_iterations", "max_nodes", "reactive_max_loading_steps"}
    for key, value in resolved.items():
        if key in integer_keys:
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ConfigurationError(f"numerics.solver_settings.{key} must be an integer >= 2")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ConfigurationError(f"numerics.solver_settings.{key} must be finite and positive")
    if resolved["quadrature_points"] < 2:
        raise ConfigurationError("numerics.solver_settings.quadrature_points must be >= 2")
    if method == "central" and resolved["nodes"] < 3:
        raise ConfigurationError("central conserved differences require at least three nodes")
    return NumericConfig(method, _freeze(resolved))


def _execution(request: Mapping[str, Any]) -> ExecutionConfig:
    values = dict(request)
    unknown = set(values) - {"output_dir", "wall_limit_s", "threads", "process_isolation", "cache_policy", "resume", "retry"}
    if unknown:
        raise ConfigurationError(f"Unknown execution keys: {sorted(unknown)}")
    output_dir = values.get("output_dir")
    wall = values.get("wall_limit_s")
    threads = values.get("threads", 1)
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        raise ConfigurationError("execution.output_dir must be a nonempty path string")
    if output_dir is not None:
        output_dir = str(Path(output_dir).expanduser().resolve())
    if wall is not None and (not isinstance(wall, (int, float)) or isinstance(wall, bool) or not math.isfinite(wall) or wall <= 0):
        raise ConfigurationError("execution.wall_limit_s must be a finite positive number")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads != 1:
        raise ConfigurationError("execution.threads currently supports only serial execution (1)")
    process_isolation = values.get("process_isolation", False)
    if not isinstance(process_isolation, bool):
        raise ConfigurationError("execution.process_isolation must be boolean")
    cache_policy = values.get("cache_policy", "legacy_explicit")
    if cache_policy not in {"legacy_explicit", "disabled"}:
        raise ConfigurationError(f"Unsupported cache policy: {cache_policy!r}")
    if cache_policy == "disabled" and not process_isolation:
        raise ConfigurationError("cache_policy='disabled' requires process_isolation=true")
    resume = values.get("resume", False)
    retry = values.get("retry", False)
    if not isinstance(resume, bool) or not isinstance(retry, bool):
        raise ConfigurationError("execution.resume and execution.retry must be boolean")
    if retry and not resume:
        raise ConfigurationError("execution.retry requires execution.resume=true")
    return ExecutionConfig(output_dir, None if wall is None else float(wall), threads, process_isolation, cache_policy, resume, retry)


def _acceptance(request: Mapping[str, Any]) -> AcceptanceConfig:
    values = dict(request)
    unknown = set(values) - {"numerical", "physical", "boundary_residual_max", "capture_error_max_pct"}
    if unknown:
        raise ConfigurationError(f"Unknown acceptance keys: {sorted(unknown)}")
    numerical = values.get("numerical", "solver_termination_and_original_equations")
    physical = values.get("physical", "not_evaluated")
    if numerical != "solver_termination_and_original_equations":
        raise ConfigurationError("Only solver_termination_and_original_equations is implemented")
    if physical != "not_evaluated":
        raise ConfigurationError("Physical acceptance is not implemented for seven_state_legacy")
    for name in ("boundary_residual_max", "capture_error_max_pct"):
        value = values.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ConfigurationError(f"acceptance.{name} must be finite and nonnegative")
    return AcceptanceConfig(numerical, physical, values.get("boundary_residual_max"), values.get("capture_error_max_pct"))


def _conserved_acceptance(request: Mapping[str, Any]) -> AcceptanceConfig:
    values = dict(request)
    unknown = set(values) - {"numerical", "physical", "boundary_residual_max", "capture_error_max_pct"}
    if unknown:
        raise ConfigurationError(f"Unknown acceptance keys: {sorted(unknown)}")
    numerical = values.get("numerical", "conserved_original_equations")
    physical = values.get("physical", "not_evaluated")
    if numerical not in {"conserved_original_equations", "solver_termination_and_original_equations"}:
        raise ConfigurationError("Unsupported conserved numerical acceptance")
    if physical not in {"not_evaluated", "exploratory_not_evaluated"}:
        raise ConfigurationError("Conserved physical acceptance is exploratory and cannot be marked implemented")
    for name in ("boundary_residual_max", "capture_error_max_pct"):
        value = values.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ConfigurationError(f"acceptance.{name} must be finite and nonnegative")
    return AcceptanceConfig(numerical, physical, values.get("boundary_residual_max"), values.get("capture_error_max_pct"))


def _dependencies(request: Mapping[str, Any]) -> DependencyConfig:
    values = dict(request)
    unknown = set(values) - {"dataset", "mobility_law", "thermal_reference", "references"}
    if unknown:
        raise ConfigurationError(f"Unknown dependencies keys: {sorted(unknown)}")
    refs = values.get("references", ())
    if not isinstance(refs, (list, tuple)) or any(not isinstance(item, str) for item in refs):
        raise ConfigurationError("dependencies.references must be a sequence of strings")
    for name in ("dataset", "mobility_law", "thermal_reference"):
        value = values.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigurationError(f"dependencies.{name} must be a nonempty string")
    return DependencyConfig(values.get("dataset"), values.get("mobility_law"), values.get("thermal_reference"), tuple(refs))


def resolve_column_config(request: Mapping[str, Any]) -> ColumnConfig:
    """Resolve a new preset without importing thermodynamics or building a solver."""
    if not isinstance(request, Mapping):
        raise ConfigurationError("Column configuration must be a mapping")
    allowed = {
        "schema_version", "preset", "case", "model", "numerics", "initialization",
        "acceptance", "execution", "dependencies", "engine", "resolved_config_sha256",
    }
    unknown = set(request) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown column configuration keys: {sorted(unknown)}")
    if request.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ConfigurationError(f"Unsupported schema_version: {request.get('schema_version')!r}")
    preset = request.get("preset", SEVEN_PRESET)
    if preset not in {SEVEN_PRESET, TWELVE_PRESET}:
        raise ConfigurationError(f"Unknown absorber preset: {preset!r}")
    case = _case(_section(request, "case"))
    model_values = dict(_section(request, "model"))
    dependencies = _dependencies(_section(request, "dependencies"))
    if preset == TWELVE_PRESET:
        film_model = model_values.get("film_model", "equilibrium_manifold")
        if film_model not in {"equilibrium_manifold", "enhancement_reference"}:
            raise ConfigurationError(
                "twelve_state_conserved model.film_model must be "
                "'equilibrium_manifold' or 'enhancement_reference'"
            )
        if film_model == "enhancement_reference" and dependencies.mobility_law is not None:
            raise ConfigurationError(
                "enhancement_reference does not consume dependencies.mobility_law"
            )
        dependencies = replace(
            dependencies,
            dataset=dependencies.dataset or _REACTIVE_DATASET,
            mobility_law=(dependencies.mobility_law or "harmonic_mean_onsager_v1")
            if film_model == "equilibrium_manifold" else None,
            thermal_reference=dependencies.thermal_reference or f"{_REACTIVE_DATASET}/anchored-reference-thermochemistry.json",
            references=dependencies.references or (
                f"{_NEUTRAL_VAPOR_DATASET}/parameters.json",
                f"{_NEUTRAL_VAPOR_DATASET}/reference-thermochemistry.json",
            ),
        )
        if film_model == "equilibrium_manifold" and dependencies.mobility_law != "harmonic_mean_onsager_v1":
            raise ConfigurationError("twelve_state_conserved requires mobility_law='harmonic_mean_onsager_v1'")
        if len(dependencies.references) != 2:
            raise ConfigurationError("twelve_state_conserved requires neutral-vapor parameter and reference assets")
        expected = {
            "formulation": "twelve_state_conserved",
            "thermo_model": "reactive_epcsaft",
            "film_model": film_model,
            "energy_model": "native_total_enthalpy",
            "pressure_model": "hydraulic_pressure_drop",
            "layout": "twelve_conserved",
            "coordinate": "physical_height",
        }
    else:
        expected = {
            "formulation": "seven_state",
            "thermo_model": "ideal_henry",
            "film_model": "enhancement_factor",
            "energy_model": "empirical_enthalpy",
            "pressure_model": "constant",
            "layout": "seven_enthalpy",
            "coordinate": "normalized_height",
        }
    for name, value in model_values.items():
        if name not in expected:
            raise ConfigurationError(f"Unknown model key: {name}")
        if value != expected[name]:
            raise ConfigurationError(f"{preset} fixes model.{name}={expected[name]!r}; got {value!r}")
    model = ModelConfig(**expected)
    numerics_request = _section(request, "numerics")
    numerics = _conserved_numeric(numerics_request) if preset == TWELVE_PRESET else _numeric(numerics_request)
    init_values = _section(request, "initialization")
    if set(init_values) - {"policy", "values"}:
        raise ConfigurationError(f"Unknown initialization keys: {sorted(set(init_values) - {'policy', 'values'})}")
    init_payload = init_values.get("values", {})
    if not isinstance(init_payload, Mapping):
        raise ConfigurationError("initialization.values must be a table/mapping")
    default_policy = "case_declared_native_inputs" if preset == TWELVE_PRESET else "legacy_capture_temperature_guesses"
    policy = init_values.get("policy", default_policy)
    if preset == TWELVE_PRESET:
        if policy != "case_declared_native_inputs":
            raise ConfigurationError("Only case_declared_native_inputs is implemented for twelve_state_conserved")
        unknown_init = set(init_payload) - {"interface_bracket", "loading_anchor", "max_log_loading_step", "max_loading_steps"}
    else:
        if policy != "legacy_capture_temperature_guesses":
            raise ConfigurationError("Only legacy_capture_temperature_guesses is implemented")
        unknown_init = set(init_payload) - {"co2_capture_guess_pct", "h2o_capture_guess_pct"}
    if unknown_init:
        raise ConfigurationError(f"Unknown initialization values: {sorted(unknown_init)}")
    for key, value in init_payload.items():
        if key == "interface_bracket":
            if not isinstance(value, (list, tuple)) or len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value) or value[0] >= value[1]:
                raise ConfigurationError("initialization.values.interface_bracket must be an increasing finite pair")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigurationError(f"initialization.values.{key} must be finite numeric")
        if key in {"loading_anchor", "max_log_loading_step"} and value <= 0:
            raise ConfigurationError(f"initialization.values.{key} must be positive")
        if key == "max_loading_steps" and (not isinstance(value, int) or value < 1):
            raise ConfigurationError("initialization.values.max_loading_steps must be a positive integer")
    initialization = InitializationConfig(policy, _freeze(init_payload))
    engine_request = _section(request, "engine")
    execution = _execution(_section(request, "execution"))
    default_engine_record = _default_engine().as_dict()
    wire_roundtrip_default = (
        request.get("resolved_config_sha256") is not None
        and engine_request.get("required", False) in {False, True if preset == TWELVE_PRESET else False}
        and all(engine_request.get(key, default_engine_record[key]) == default_engine_record[key]
                for key in ("wheel", "sha256", "commit", "python"))
    )
    engine_explicit = any(key in engine_request for key in ("wheel", "sha256", "commit", "python")) and not wire_roundtrip_default
    engine_required = bool(engine_request.get("required", engine_explicit or preset == TWELVE_PRESET))
    engine = _engine(engine_request, required=engine_required)
    if engine_explicit and not execution.process_isolation:
        raise ConfigurationError("explicit Engine selection requires execution.process_isolation=true")
    config = ColumnConfig(
        preset=preset,
        case=case,
        model=model,
        numerics=numerics,
        initialization=initialization,
        acceptance=(_conserved_acceptance(_section(request, "acceptance"))
                    if preset == TWELVE_PRESET else _acceptance(_section(request, "acceptance"))),
        execution=execution,
        dependencies=dependencies,
        engine=engine,
    )
    expected_fingerprint = request.get("resolved_config_sha256")
    if expected_fingerprint is not None:
        if not isinstance(expected_fingerprint, str) or expected_fingerprint != config.resolved_config_sha256:
            raise ConfigurationError("resolved_config_sha256 does not match the canonical configuration")
    return config


def verify_engine(engine: EngineConfig) -> dict[str, Any]:
    """Verify configured wheel bytes before a native worker is allowed to run."""
    result = {
        "wheel": engine.wheel,
        "expected_sha256": engine.sha256,
        "actual_sha256": None,
        "expected_commit": engine.commit,
        "commit": engine.commit,
        "python": engine.python,
        "required": engine.required,
        "platform": platform.platform(),
        "status": "not_required",
    }
    if not engine.required and (engine.wheel is None or engine.sha256 is None or not Path(engine.wheel).is_file()):
        result["status"] = "not_required"
        return result
    if engine.wheel is None or engine.sha256 is None:
        raise CapabilityRefusal("Configured native execution lacks an immutable wheel path and SHA-256")
    path = Path(engine.wheel)
    if not path.is_file():
        raise CapabilityRefusal(f"Configured Engine wheel is unavailable: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    result["actual_sha256"] = actual
    if actual != engine.sha256:
        raise CapabilityRefusal(f"Engine wheel SHA-256 mismatch: expected {engine.sha256}, got {actual}")
    result["status"] = "verified"
    return result


def default_engine(*, required: bool = False) -> EngineConfig:
    return replace(_default_engine(), required=required)


def _normalise_archive_sha256(value: Any) -> str:
    if not isinstance(value, str):
        raise CapabilityRefusal("Installed Engine archive hash is not a string")
    value = value.lower()
    if value.startswith("sha256="):
        value = value.removeprefix("sha256=")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CapabilityRefusal("Installed Engine archive hash is not a SHA-256 digest")
    return value


def _installed_wheel_contents(wheel: Path, distribution) -> int:
    """Verify every non-RECORD wheel member against the installed distribution."""
    package_root = Path(distribution.locate_file("")).resolve()
    checked = 0
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = [
                name for name in archive.namelist()
                if not name.endswith("/") and not name.endswith(".dist-info/RECORD")
            ]
            for name in members:
                installed = package_root.joinpath(*name.split("/"))
                if not installed.is_file() or installed.read_bytes() != archive.read(name):
                    raise CapabilityRefusal(
                        f"Installed epcsaft artifact differs from selected wheel member: {name}"
                    )
                checked += 1
    except (OSError, zipfile.BadZipFile) as exc:
        raise CapabilityRefusal("Selected Engine wheel cannot establish installed artifact contents") from exc
    if checked == 0:
        raise CapabilityRefusal("Selected Engine wheel contains no installable artifact members")
    return checked


def verify_worker_identity(expected: Mapping[str, Any]) -> dict[str, Any]:
    """Check the interpreter and wheel selected by a worker before native use."""
    required = {"wheel", "sha256", "python"} - set(expected)
    if required:
        raise CapabilityRefusal(f"Worker identity is incomplete: missing {sorted(required)}")
    expected_python_path = Path(expected["python"]).expanduser()
    expected_python = expected_python_path.resolve()
    if expected_python != Path(sys.executable).resolve():
        raise CapabilityRefusal(f"Prepared interpreter mismatch: expected {expected['python']}, running {sys.executable}")
    selected_prefix = (
        expected_python_path.parent.parent
        if expected_python_path.parent.name.lower() in {"bin", "scripts"}
        else expected_python_path.parent
    )
    selected_prefix = selected_prefix.resolve()
    if Path(sys.prefix).resolve() != selected_prefix or Path(sys.exec_prefix).resolve() != selected_prefix:
        raise CapabilityRefusal(
            f"Prepared interpreter environment mismatch: expected prefix {selected_prefix}, "
            f"running prefix {Path(sys.prefix).resolve()}"
        )
    identity = verify_engine(EngineConfig(
        expected.get("wheel"), expected.get("sha256"), expected.get("commit"), expected.get("python"), True
    ))
    try:
        module = importlib.import_module("epcsaft")
        distribution = importlib.metadata.distribution("epcsaft")
        version = distribution.version
        module_path = getattr(module, "__file__", None)
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise CapabilityRefusal("Selected worker cannot import epcsaft") from exc
    if module_path is None:
        raise CapabilityRefusal("Selected worker loaded epcsaft without a module path")
    package_root = Path(distribution.locate_file("" )).resolve()
    try:
        Path(module_path).resolve().relative_to(package_root)
    except ValueError as exc:
        raise CapabilityRefusal(f"Loaded epcsaft module is outside the selected environment: {module_path}") from exc
    direct_url = distribution.read_text("direct_url.json")
    declared_wheel = None
    archive_hashes = []
    if direct_url:
        try:
            direct_url_record = json.loads(direct_url)
            declared_wheel = unquote(urlparse(direct_url_record["url"]).path)
            archive_info = direct_url_record.get("archive_info") or {}
            for candidate in (
                archive_info.get("hash"),
                archive_info.get("sha256"),
                (archive_info.get("hashes") or {}).get("sha256"),
            ):
                if candidate is not None:
                    archive_hashes.append(_normalise_archive_sha256(candidate))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError):
            declared_wheel = None
    if declared_wheel is None or Path(declared_wheel).resolve() != Path(expected["wheel"]).resolve():
        raise CapabilityRefusal("Loaded epcsaft distribution does not identify the selected wheel")
    if len(set(archive_hashes)) > 1:
        raise CapabilityRefusal("Installed Engine archive hash fields conflict")
    archive_hash = archive_hashes[0] if archive_hashes else None
    if archive_hash is not None and archive_hash != identity["actual_sha256"]:
        raise CapabilityRefusal("Loaded epcsaft distribution archive hash does not match the selected wheel")
    installed_members = _installed_wheel_contents(Path(expected["wheel"]), distribution)
    distribution_files = distribution.files or ()
    try:
        relative_module = Path(module_path).resolve().relative_to(package_root)
    except ValueError as exc:
        raise CapabilityRefusal(f"Loaded epcsaft module is outside the selected environment: {module_path}") from exc
    if distribution_files and relative_module.as_posix() not in {Path(item).as_posix() for item in distribution_files}:
        raise CapabilityRefusal("Loaded epcsaft module is not owned by the selected distribution")
    wheel_name = Path(expected["wheel"]).name
    wheel_parts = wheel_name.removesuffix(".whl").split("-")
    if len(wheel_parts) < 2 or wheel_parts[0].lower() != "epcsaft" or wheel_parts[1].replace("_", ".") != version:
        raise CapabilityRefusal(f"Loaded epcsaft version {version} disagrees with selected wheel {wheel_name}")
    identity.update({
        "loaded_version": version,
        "loaded_module": str(Path(module_path).resolve()),
        "loaded_distribution_root": str(package_root),
        "declared_wheel": str(Path(declared_wheel).resolve()),
        "declared_archive_sha256": archive_hash,
        "archive_hash_status": "verified" if archive_hash is not None else "verified_from_installed_contents",
        "installed_artifact_members": installed_members,
        "declared_commit": expected.get("commit"),
        "verified_commit": None,
        "commit_evidence": "not_available_from_installed_distribution",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": sys.implementation.name,
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_exec_prefix": str(Path(sys.exec_prefix).resolve()),
    })
    return identity
