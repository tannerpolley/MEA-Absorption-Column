"""Common selectable-column execution facade."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .benchmark import BenchmarkSettings
from .config.column import (
    CapabilityRefusal,
    ColumnConfig,
    ConfigurationError,
    resolve_column_config,
    TWELVE_PRESET,
    verify_engine,
    verify_worker_identity,
)
from .misc.Convert_Data import convert_data, physical_input_fingerprint


_ROOT = Path(__file__).resolve().parents[2]


def _physical_payload(config: ColumnConfig) -> Mapping[str, Any] | None:
    path = config.case.physical_input_file
    if path is None or Path(path).suffix.lower() != ".json":
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("physical_inputs"), Mapping):
        return None
    return payload


def _conserved_case_policy(config: ColumnConfig, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve case-owned conserved inputs; explicit solver settings win."""
    physical = payload.get("physical_inputs")
    if not isinstance(physical, Mapping):
        raise ConfigurationError("twelve_state_conserved requires physical_inputs JSON")
    branch = physical.get("liquid_branch_policy")
    diffusion = physical.get("species_diffusivity_model")
    if not isinstance(branch, Mapping) or not isinstance(diffusion, Mapping):
        raise ConfigurationError("Conserved physical inputs must declare branch and diffusivity policies")
    settings = dict(config.numerics.as_dict()["solver_settings"])
    initialization = dict(config.initialization.as_dict()["values"])
    if "interface_bracket" in initialization:
        raise ConfigurationError(
            "initialization.values.interface_bracket is not consumed by conserved preparation"
        )
    anchor = settings.get("reactive_loading_anchor", initialization.get("loading_anchor", branch.get("loading_anchor")))
    step = settings.get(
        "reactive_max_log_loading_step",
        initialization.get("max_log_loading_step", branch.get("max_log_loading_step")),
    )
    steps = settings.get("reactive_max_loading_steps", initialization.get("max_loading_steps", branch.get("max_loading_steps")))
    if (isinstance(anchor, bool) or not isinstance(anchor, (int, float)) or not math.isfinite(anchor) or anchor <= 0
            or isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0
            or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1):
        raise ConfigurationError("Conserved loading policy must be finite positive with an integer step budget")
    required = ("co2_prefactor_m2_s", "co2_activation_j_mol", "gas_constant_j_mol_k", "other_species_m2_s")
    if any(name not in diffusion for name in required):
        raise ConfigurationError("Conserved species_diffusivity_model is incomplete")
    if diffusion.get("pair_closure") != "harmonic mean, common Reactive_Film.binary_diffusivities_from_species":
        raise ConfigurationError("Conserved species diffusivities require the declared harmonic-mean pair closure")
    other = diffusion["other_species_m2_s"]
    if (not isinstance(other, (list, tuple)) or len(other) != 8
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in other)):
        raise ConfigurationError("Conserved diffusivity model requires eight positive finite non-CO2 values")
    values = {name: diffusion[name] for name in required if name != "other_species_m2_s"}
    if any(isinstance(values[name], bool) or not isinstance(values[name], (int, float)) or not math.isfinite(values[name]) or values[name] <= 0 for name in values):
        raise ConfigurationError("Conserved diffusivity model coefficients must be positive finite numbers")
    values["other_species_m2_s"] = tuple(float(value) for value in other)
    return {
        "loading_anchor": float(anchor),
        "max_log_loading_step": float(step),
        "max_loading_steps": int(steps),
        "species_diffusivity_model": values,
        "precedence": "physical_inputs defaults, overridden by initialization.values, then numerics.solver_settings reactive_loading_* values",
    }


def _source_identity() -> dict[str, Any]:
    """Record executable source content plus a replayable Git dirty patch."""
    source_root = _ROOT / "src" / "mea_absorption_column"
    source_files = {}
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(_ROOT).as_posix()
        source_files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree_payload = json.dumps(source_files, sort_keys=True, separators=(",", ":"))
    tree_sha256 = hashlib.sha256(tree_payload.encode()).hexdigest()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        status_lines = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "src/mea_absorption_column"],
            cwd=_ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        dirty_paths = [line[3:] for line in status_lines]
        tracked_patch = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "src/mea_absorption_column"],
            cwd=_ROOT, check=True, capture_output=True, text=True,
        ).stdout
        untracked_patches = []
        for line in status_lines:
            if line.startswith("?? "):
                path = _ROOT / line[3:]
                if path.is_file():
                    patch = subprocess.run(
                        ["git", "diff", "--no-index", "--binary", "/dev/null", line[3:]],
                        cwd=_ROOT, capture_output=True, text=True,
                    ).stdout
                    untracked_patches.append(patch)
        dirty_patch = tracked_patch + "\n".join(untracked_patches)
        return {
            "repository_root": str(_ROOT),
            "commit": commit,
            "dirty_paths": dirty_paths,
            "dirty": bool(dirty_paths),
            "source_tree_sha256": tree_sha256,
            "dirty_patch": dirty_patch,
            "dirty_patch_sha256": hashlib.sha256(dirty_patch.encode()).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "repository_root": str(_ROOT),
            "status": "unavailable",
            "reason": str(exc),
            "source_tree_sha256": tree_sha256,
        }


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "index": [_safe(item) for item in value.index.tolist()],
            "data": _safe(value.to_numpy()),
        }
    if isinstance(value, pd.Series):
        return {"name": str(value.name), "index": [_safe(item) for item in value.index.tolist()], "data": _safe(value.to_numpy())}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"value_class": "nan" if math.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")}
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _case_frame(config: ColumnConfig) -> tuple[pd.DataFrame, str]:
    from .benchmark import _filter_case_ids, load_case_data

    if config.case.physical_input_file is not None:
        path = Path(config.case.physical_input_file)
        if not path.is_file():
            raise ConfigurationError(f"Physical input file is unavailable: {path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, index_col=0)
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and isinstance(payload.get("model_input_row"), Mapping):
                row = dict(payload["model_input_row"])
                frame = pd.DataFrame(
                    [row],
                    index=[payload.get("physical_inputs", {}).get("case_id", config.case.case_id)],
                )
                if len(frame) == 1 and config.case.case_id not in frame.index:
                    frame.index = [config.case.case_id]
            elif isinstance(payload, Mapping) and isinstance(payload.get("values"), Mapping):
                frame = pd.DataFrame([payload["values"]], index=[payload.get("case_id", config.case.case_id)])
            elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
                frame = pd.DataFrame(payload["rows"])
            elif isinstance(payload, list):
                frame = pd.DataFrame(payload)
            else:
                raise ConfigurationError("Physical JSON input must contain values or rows")
            if "case_id" in frame.columns:
                frame = frame.set_index("case_id")
        else:
            raise ConfigurationError("physical_input_file must be CSV or JSON")
        return _filter_case_ids(frame, (config.case.case_id,), path.stem), path.stem

    source = config.case.source
    aliases = {
        "C_cases_data": ("C_cases_data", "legacy", "legacy"),
        "C_cases_campaign_inputs": ("C_cases_campaign_inputs", "campaign", "legacy"),
        "NCCC_Data": ("NCCC_Data", "legacy", "legacy"),
        "NCCC_2014_cases": ("NCCC_2014_cases", "legacy", "2014"),
        "NCCC_2017_cases": ("NCCC_2017_cases", "legacy", "2017"),
        "SRP_method_cases": ("SRP_method_cases", "legacy", "legacy"),
    }
    if source in aliases:
        label, c_dataset, nccc_dataset = aliases[source]
        c_cases, nccc_cases, srp_cases = load_case_data(c_dataset, nccc_dataset)
        frame = {
            "C_cases_data": c_cases,
            "C_cases_campaign_inputs": c_cases,
            "NCCC_Data": nccc_cases,
            "NCCC_2014_cases": nccc_cases,
            "NCCC_2017_cases": nccc_cases,
            "SRP_method_cases": srp_cases,
        }[label]
        return _filter_case_ids(frame, (config.case.case_id,), label), label
    path = Path(source)
    if not path.is_file():
        path = _ROOT / source
    if not path.is_file():
        raise ConfigurationError(f"Case source is unavailable: {source}")
    frame = pd.read_csv(path, index_col=0)
    return _filter_case_ids(frame, (config.case.case_id,), path.stem), path.stem


def _resolved_inputs(config: ColumnConfig, frame: pd.DataFrame) -> dict[str, Any]:
    physical_payload = _physical_payload(config)
    if physical_payload is None:
        parameters, raw, metadata = convert_data(
            frame,
            0,
            config.case.data_type,
            return_metadata=True,
            vapor_composition_mode=config.case.vapor_composition_mode,
            gas_flow_basis=config.case.gas_flow_basis,
        )
    else:
        parameters, raw, metadata = _direct_physical_inputs(physical_payload, config)
    _validate_si_inputs(parameters, raw, metadata)
    payload = {
        "case_id": config.case.case_id,
        "data_type": config.case.data_type,
        "vapor_composition_mode": config.case.vapor_composition_mode,
        "gas_flow_basis": config.case.gas_flow_basis,
        "parameters": parameters,
        "raw_input": raw,
        "metadata": metadata,
        "source_lineage": {
            "source": config.case.source,
            "source_row_sha256": _digest(frame.iloc[0].to_dict()),
            "physical_input_file": config.case.physical_input_file,
        },
    }
    if config.preset == "twelve_state_conserved":
        physical_payload = _physical_payload(config)
        if physical_payload is None:
            raise ConfigurationError("twelve_state_conserved requires a physical_input_file")
        payload["conserved_policy"] = _conserved_case_policy(config, physical_payload)
    source_path = Path(config.case.source)
    alias_files = {
        "C_cases_data": "C_cases_data.csv",
        "C_cases_campaign_inputs": "C_cases_campaign_inputs.csv",
        "NCCC_Data": "NCCC_Data.csv",
        "NCCC_2014_cases": "NCCC_2014_model_inputs_mass.csv",
        "NCCC_2017_cases": "NCCC_2017_model_inputs_mass.csv",
        "SRP_method_cases": "SRP_method_cases.csv",
    }
    if not source_path.is_file() and config.case.source in alias_files:
        source_path = _ROOT / "src" / "mea_absorption_column" / "data" / alias_files[config.case.source]
    if source_path.is_file():
        payload["source_lineage"]["source_file_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    else:
        payload["source_lineage"]["source_file_sha256"] = None
    if config.case.physical_input_file is not None:
        physical_path = Path(config.case.physical_input_file)
        payload["source_lineage"]["physical_input_file_sha256"] = hashlib.sha256(physical_path.read_bytes()).hexdigest()
    else:
        payload["source_lineage"]["physical_input_file_sha256"] = None
    payload["physical_input_sha256"] = physical_input_fingerprint(
        parameters,
        data_type=config.case.data_type,
        vapor_composition_mode=config.case.vapor_composition_mode,
        gas_flow_basis=config.case.gas_flow_basis,
    )
    return _safe(payload)


def _direct_physical_inputs(payload: Mapping[str, Any], config: ColumnConfig):
    values = payload["physical_inputs"]
    row = payload.get("model_input_row") or {}
    liquid = np.asarray(values.get("liquid_feed_mol_s"), dtype=float)
    vapor = np.asarray(values.get("vapor_feed_mol_s"), dtype=float)
    beds = row.get("Beds", values.get("beds"))
    intercoolers = row.get("Intercoolers", values.get("intercoolers", 0))
    if (
        isinstance(beds, bool)
        or not isinstance(beds, (int, float))
        or not math.isfinite(float(beds))
        or int(beds) != float(beds)
        or int(beds) < 1
    ):
        raise ConfigurationError("physical_inputs must declare a positive integer bed count")
    beds = int(beds)
    if (
        isinstance(intercoolers, bool)
        or not isinstance(intercoolers, (int, float))
        or not math.isfinite(float(intercoolers))
        or int(intercoolers) != float(intercoolers)
        or int(intercoolers) < 0
    ):
        raise ConfigurationError("physical_inputs must declare a nonnegative integer intercooler count")
    if liquid.shape != (3,) or vapor.shape != (4,):
        raise ConfigurationError("physical_inputs must provide three liquid and four vapor molar feeds")
    total_vapor = float(vapor.sum())
    raw = np.asarray([
        float(liquid.sum()) / total_vapor,
        total_vapor,
        float(liquid[0] / liquid[1]),
        float(liquid[1] * 0.061084 / (liquid[1] * 0.061084 + liquid[2] * 0.01801528)),
        float(vapor[0] / total_vapor),
        float(values["liquid_temperature_k"]),
        float(values["vapor_temperature_k"]),
        float(values["bottom_pressure_pa"]),
        float(beds),
    ])
    total_height = float(values["height_m"])
    parameters = [
        liquid.tolist(),
        vapor.tolist(),
        float(values["liquid_temperature_k"]),
        float(values["vapor_temperature_k"]),
        np.linspace(0.0, 1.0, 101),
        total_height,
        float(values["area_m2"]),
        float(values["bottom_pressure_pa"]),
        list(values["packing"]),
    ]
    metadata = {
        "case_id": str(values.get("case_id", config.case.case_id)),
        "beds": beds,
        "intercoolers": int(intercoolers),
        "single_bed_height_m": total_height / beds,
        "total_packed_height_m": total_height,
        "diameter_m": math.sqrt(4.0 * float(values["area_m2"]) / math.pi),
        "vapor_composition_mode": config.case.vapor_composition_mode,
        "gas_flow_basis": config.case.gas_flow_basis,
        "physical_input_basis": "direct_canonical_SI",
    }
    return parameters, raw, metadata


def _validate_si_inputs(inputs, raw, metadata) -> None:
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 9:
        raise ConfigurationError("Converted case must contain the nine canonical SI input records")
    liquid, vapor, liquid_temperature, vapor_temperature, coordinate, height, area, pressure, packing = inputs
    liquid = np.asarray(liquid, dtype=float)
    vapor = np.asarray(vapor, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float)
    packing = np.asarray(packing, dtype=float)
    if liquid.shape != (3,) or vapor.shape != (4,) or packing.shape != (7,):
        raise ConfigurationError("Converted case has an invalid SI feed or packing shape")
    if any(not np.all(np.isfinite(values)) or np.any(values <= 0.0) for values in (liquid, vapor, packing)):
        raise ConfigurationError("Converted case contains non-finite or nonpositive SI flows/packing values")
    if not np.isfinite([liquid_temperature, vapor_temperature, height, area, pressure]).all() or any(
        float(value) <= 0.0 for value in (liquid_temperature, vapor_temperature, height, area, pressure)
    ):
        raise ConfigurationError("Converted case contains invalid SI temperature, geometry, or pressure")
    if coordinate.ndim != 1 or coordinate.size < 2 or not np.all(np.isfinite(coordinate)):
        raise ConfigurationError("Converted case coordinate must be a finite one-dimensional grid")
    if coordinate[0] != 0.0 or coordinate[-1] != 1.0 or not np.all(np.diff(coordinate) > 0.0):
        raise ConfigurationError("Converted case coordinate must increase strictly from 0 to 1")
    if not 0.0 < float(packing[1]) < 1.0:
        raise ConfigurationError("Converted case packing void fraction must lie strictly between zero and one")
    raw_values = np.asarray(raw, dtype=float).reshape(-1)
    if raw_values.size < 9 or not np.all(np.isfinite(raw_values[:9])):
        raise ConfigurationError("Raw case record must contain nine finite values")
    beds = metadata.get("beds")
    if isinstance(beds, bool) or not isinstance(beds, (int, np.integer)) or int(beds) < 1:
        raise ConfigurationError("Case bed identity must be a positive integer")
    single_height = float(metadata.get("single_bed_height_m", 0.0))
    if not math.isfinite(single_height) or single_height <= 0.0 or not math.isclose(
        float(height), single_height * int(beds), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ConfigurationError("Case bed identity is inconsistent with packed height")


def _record(path: Path | None, name: str, value: Any) -> None:
    if path is None:
        return
    temporary = path / f".{name}.tmp"
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path / name)


def _settings(config: ColumnConfig, output_dir: Path | None, resolved_inputs: dict[str, Any] | None = None) -> BenchmarkSettings:
    solver = dict(config.numerics.as_dict()["solver_settings"])
    init_values = dict(config.initialization.as_dict()["values"])
    solver.update(init_values)
    if config.acceptance.boundary_residual_max is not None:
        solver["success_boundary_residual_max"] = config.acceptance.boundary_residual_max
    solver["vapor_composition_mode"] = config.case.vapor_composition_mode
    solver["gas_flow_basis"] = config.case.gas_flow_basis
    solver["return_internal_profile"] = True
    if solver.get("profile_csvs") or solver.get("profile_pngs"):
        solver["return_profiles"] = True
    return BenchmarkSettings(
        methods=(config.numerics.method,),
        thermo_models=(config.model.thermo_model,),
        output_dir=output_dir or Path(".tmp_column_runs"),
        write_artifacts=output_dir is not None,
        data_type=config.case.data_type,
        staged_beds="auto",
        solver_settings=solver,
        profile_csvs=bool(solver.get("profile_csvs", False)),
        profile_pngs=bool(solver.get("profile_pngs", False)),
        subprocess_timeout_s=config.execution.wall_limit_s,
        process_isolation=config.execution.process_isolation,
        worker_python=config.engine.python if config.execution.process_isolation else None,
        engine_wheel=config.engine.wheel if config.execution.process_isolation else None,
        engine_sha256=config.engine.sha256 if config.execution.process_isolation else None,
        engine_commit=config.engine.commit if config.execution.process_isolation else None,
        resolved_inputs=resolved_inputs,
        cache_policy=config.execution.cache_policy,
    )


def _dependency_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _ROOT / path


def _build_conserved_assembly(config: ColumnConfig, resolved_inputs: Mapping[str, Any]):
    """Build the native twelve-state graph and retain every callback owner."""
    import casadi as ca

    from .BVP.Coupled_Column import build_coupled_column_functions
    from .Thermodynamics.casadi_reactive import FixedCompositionVaporCallback, ReactiveLiquidCallback
    from .Thermodynamics.reactive_bundle import ReactiveLiquid, load_reference_thermochemistry
    from .Thermodynamics.thermo_models import ensure_epcsaft_importable
    epcsaft = ensure_epcsaft_importable()

    liquid_feed, vapor_feed, liquid_temperature, vapor_temperature, coordinate, height, area, pressure, packing = (
        resolved_inputs["parameters"]
    )
    policy = resolved_inputs.get("conserved_policy")
    if policy is None:
        raise ConfigurationError("Resolved conserved inputs lack their case-owned policy")
    dataset = _dependency_path(config.dependencies.dataset)
    liquid_reference_path = _dependency_path(config.dependencies.thermal_reference)
    liquid_reference = load_reference_thermochemistry(liquid_reference_path)
    reactive = ReactiveLiquid(
        dataset,
        thermochemistry=liquid_reference,
        loading_anchor=policy["loading_anchor"],
        max_log_loading_step=policy["max_log_loading_step"],
        max_loading_steps=policy["max_loading_steps"],
        reuse_states=False,
    )
    liquid = ReactiveLiquidCallback("column_liquid", reactive)
    neutral_parameters = _dependency_path(config.dependencies.references[0])
    neutral_reference_path = _dependency_path(config.dependencies.references[1])
    vapor = FixedCompositionVaporCallback(
        "column_vapor",
        epcsaft.Parameters.from_json(neutral_parameters),
        load_reference_thermochemistry(neutral_reference_path, liquid_reference=liquid_reference),
        packing_interval=(1.0e-6, .1),
    )
    diffusion = policy["species_diffusivity_model"]

    def species_diffusivities(temperature):
        return ca.vertcat(
            diffusion["co2_prefactor_m2_s"] * ca.exp(
                -diffusion["co2_activation_j_mol"]
                / (diffusion["gas_constant_j_mol_k"] * temperature)
            ),
            *diffusion["other_species_m2_s"],
        )

    settings = dict(config.numerics.as_dict()["solver_settings"])
    node, boundary, diagnostics = build_coupled_column_functions(
        liquid,
        vapor,
        species_diffusivities=species_diffusivities,
        quadrature_points=settings["quadrature_points"],
        co2_model={
            "equilibrium_manifold": "reactive_film",
            "enhancement_reference": "enhancement_reference",
        }[config.model.film_model],
        liquid_feed_mol_s=liquid_feed,
        vapor_feed_mol_s=vapor_feed,
        liquid_temperature_k=liquid_temperature,
        vapor_temperature_k=vapor_temperature,
        bottom_pressure_pa=pressure,
        area_m2=area,
        packing=packing,
    )
    balance = node._thermodynamic_callbacks[0]
    return {
        "liquid": liquid,
        "vapor": vapor,
        "reactive_liquid": reactive,
        "node": node,
        "balance": balance,
        "boundary": boundary,
        "diagnostics": diagnostics,
        "balance_count": int(node.size1_out(0)),
        "algebraic_count": int(node.size1_out(2)),
        "state_count": int(node.size1_in(1)),
        "boundary_count": int(boundary.size1_out(0)),
        "height_m": float(height),
        "coordinate": np.linspace(0.0, float(height), settings["nodes"]),
        "diffusion_model": diffusion,
        "asset_paths": [str(dataset), str(liquid_reference_path), str(neutral_parameters), str(neutral_reference_path)],
    }


def _prepare_conserved_column_in_process(config: ColumnConfig) -> dict[str, Any]:
    """Internal/test seam; callers use the configured worker wrapper below."""
    frame, source = _case_frame(config)
    resolved_inputs = _resolved_inputs(config, frame)
    engine = verify_engine(config.engine)
    worker_identity = verify_worker_identity({
        "wheel": config.engine.wheel,
        "sha256": config.engine.sha256,
        "commit": config.engine.commit,
        "python": config.engine.python,
    })
    engine.update(worker_identity)
    assembly = _build_conserved_assembly(config, resolved_inputs)
    assets = {}
    for path in assembly["asset_paths"]:
        asset = Path(path)
        if asset.is_file():
            assets[path] = hashlib.sha256(asset.read_bytes()).hexdigest()
        elif asset.is_dir():
            assets[path] = _digest({item.relative_to(asset).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
                                    for item in sorted(asset.rglob("*")) if item.is_file()})
    return {
        "preset": config.preset,
        "source": config.case.source,
        "case_source_label": source,
        "source_identity": _source_identity(),
        "config_sha256": config.resolved_config_sha256,
        "engine": engine,
        "resolved_inputs": resolved_inputs,
        "assets": assets,
        "layout": {
            "states": assembly["state_count"],
            "conserved_balances": assembly["balance_count"],
            "algebraic_equations": assembly["algebraic_count"],
            "boundary_equations": assembly["boundary_count"],
            "coordinate": "physical_height",
        },
        "assembly": assembly,
        "capabilities": {
            "a1_equilibrium_values": "required_on_evaluation",
            "a2_equilibrium_actions": (
                "required_on_outer_derivative"
                if config.model.film_model == "equilibrium_manifold"
                else "not_required_by_selected_film"
            ),
            "caloric_actions": "required_on_outer_derivative",
            "finite_difference_fallback": False,
        },
    }


def prepare_conserved_column(config: ColumnConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Prepare through the configured interpreter; no full column solve."""
    config = resolve_column_config(config.as_dict() if isinstance(config, ColumnConfig) else config)
    if config.preset != TWELVE_PRESET:
        raise ConfigurationError("prepare_conserved_column requires the twelve_state_conserved preset")
    with tempfile.TemporaryDirectory(prefix="conserved_prepare_") as temporary:
        temporary = Path(temporary)
        input_path, output_path = temporary / "input.json", temporary / "output.json"
        input_path.write_text(json.dumps(_safe({
            "task": "conserved_preparation",
            "config": config.as_dict(),
            "output_path": str(output_path),
            "runtime_identity": {
                "wheel": config.engine.wheel,
                "sha256": config.engine.sha256,
                "commit": config.engine.commit,
                "python": config.engine.python,
            },
        })), encoding="utf-8")
        environment = os.environ.copy()
        source_root = str(_ROOT / "src")
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        try:
            completed = subprocess.run(
                [config.engine.python, "-m", "mea_absorption_column.benchmark_worker", str(input_path)],
                cwd=_ROOT, env=environment, capture_output=True, text=True,
                timeout=config.execution.wall_limit_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("Conserved preparation worker exceeded its wall limit") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            raise CapabilityRefusal(f"Conserved preparation worker failed: {detail[-1] if detail else completed.returncode}")
        if not output_path.is_file():
            raise CapabilityRefusal("Conserved preparation worker returned no result")
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result.get("failure_kind") == "capability_refusal":
            raise CapabilityRefusal(result.get("message", "Conserved worker capability refusal"))
        if result.get("failure_kind") == "preparation_failed":
            raise ConfigurationError(result.get("message", "Conserved preparation failed"))
        return result


def _run_conserved_column_in_process(config: ColumnConfig, checkpoint) -> dict[str, Any]:
    """Run the case-owned twelve-state collocation path in the verified worker."""
    if config.numerics.method not in {"trapezoidal", "central"}:
        raise CapabilityRefusal(
            f"Conserved method {config.numerics.method!r} is configured but unavailable: "
            "the reduced-method controls have not been migrated to this execution boundary"
        )
    import casadi as ca
    from scipy.optimize import brentq

    payload = _physical_payload(config)
    physical = payload["physical_inputs"] if payload is not None else None
    if not isinstance(physical, Mapping):
        raise ConfigurationError("Conserved execution requires the case physical_inputs record")
    bulk = np.asarray(payload.get("initial_bulk_state"), dtype=float)
    bracket = np.asarray(payload.get("initial_interface_bracket"), dtype=float)
    tolerance = payload.get("physical_residual_tolerance")
    if bulk.shape != (8,) or np.any(~np.isfinite(bulk)):
        raise ConfigurationError("Conserved execution needs a finite eight-state initial_bulk_state")
    if bracket.shape != (2,) or np.any(~np.isfinite(bracket)) or bracket[0] >= bracket[1]:
        raise ConfigurationError("Conserved execution needs an increasing initial_interface_bracket")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ConfigurationError("Conserved execution needs a positive physical_residual_tolerance")
    liquid_feed = np.asarray(physical.get("liquid_feed_mol_s"), dtype=float)
    vapor_feed = np.asarray(physical.get("vapor_feed_mol_s"), dtype=float)
    expected_seed = np.r_[liquid_feed[0], liquid_feed[2], vapor_feed[:2],
                          physical.get("liquid_temperature_k"), physical.get("vapor_temperature_k"),
                          physical.get("bottom_pressure_pa")]
    packing = np.asarray(physical.get("packing"), dtype=float)
    if (liquid_feed.shape != (3,) or vapor_feed.shape != (4,) or packing.shape != (7,)
            or not np.allclose(bulk[:7], expected_seed, rtol=0.0, atol=1e-12)
            or packing[1] != .97 or not np.all((293.15 <= bulk[4:6]) & (bulk[4:6] <= 393.15))):
        raise ConfigurationError(
            "Only the case-declared 3C policy is supported: matching feed-seeded bulk state, "
            "0.97 packing void fraction, and 293.15--393.15 K inlet/initial temperatures"
        )
    checkpoint.update(stage="policy_checked", support_limits={
        "policy": "case_declared_native_inputs: 3C fixed 293.15--393.15 K bounds, 300 K state scale, 100 K span, 0.97 holdup/packing void fraction",
        "unsupported": "arbitrary packing void fractions, feed-seed mismatches, and temperature policies",
    })
    prepared = _prepare_conserved_column_in_process(config)
    assembly, inputs = prepared["assembly"], prepared["resolved_inputs"]
    checkpoint.update(resolved_inputs=inputs, engine=prepared["engine"], assets=prepared["assets"],
                      capabilities=prepared["capabilities"], layout=prepared["layout"])
    settings = dict(config.numerics.as_dict()["solver_settings"])
    node, boundary, diagnostics, balance = (assembly[name] for name in ("node", "boundary", "diagnostics", "balance"))
    liquid, vapor = assembly["liquid"], assembly["vapor"]
    liquid_feed, vapor_feed = (np.asarray(value, dtype=float) for value in inputs["parameters"][:2])
    checkpoint.update(stage="initializing", native_calls={})

    def instrument(owner, name, label):
        original = getattr(owner, name)
        counts = checkpoint["native_calls"][label] = {"started": 0, "returned": 0, "failed": 0, "wall_s": 0.0}
        def observed(*args, **kwargs):
            counts["started"] += 1
            started = time.perf_counter()
            try:
                value = original(*args, **kwargs)
                counts["returned"] += 1
                return value
            except Exception:
                counts["failed"] += 1
                raise
            finally:
                counts["wall_s"] += time.perf_counter() - started
        setattr(owner, name, observed)

    instrument(assembly["reactive_liquid"], "solve", "liquid_value_A1")
    instrument(assembly["reactive_liquid"], "solve_actions", "liquid_A2")
    instrument(vapor, "_state", "vapor_value_A1_H2")
    np.testing.assert_array_equal(liquid.molar_masses[:3], payload["liquid_molar_masses_kg_mol"])
    np.testing.assert_array_equal(vapor.molar_masses, payload["vapor_molar_masses_kg_mol"])
    bulk[7] -= float(balance(bulk, [0.0, 0.0, 0.0])[2])
    gas = np.asarray(balance(bulk, [0.0, 0.0, 0.0])[4]).ravel()
    evaluations = []

    def interface_residual(loading):
        value = diagnostics(np.r_[bulk, 0.0, 0.0, 0.0, loading])
        residual = float(value[1] / value[2] - value[3][0] * (gas[0] - value[7]))
        evaluations.append({"loading": float(loading), "original_co2_residual": residual})
        checkpoint.update(stage="interface_initialization", initialization_evaluations=evaluations)
        return residual

    loading, root = brentq(interface_residual, *bracket, full_output=True, disp=False)
    diagnostic = diagnostics(np.r_[bulk, 0.0, 0.0, 0.0, loading])
    point = np.r_[bulk, float(diagnostic[1] / diagnostic[2]),
                  float(diagnostic[3][1] * (gas[1] - balance(bulk, [0.0, 0.0, 0.0])[3][20])), 0.0, loading]
    point[10] = float(diagnostic[4]) * (bulk[5] - bulk[4]) + point[8:10] @ np.asarray(diagnostic[6]).ravel()
    height, span = assembly["height_m"], 393.15 - 293.15
    capacity = np.array([
        liquid_feed.sum() * liquid.input_jacobian(np.r_[bulk[4], bulk[6], liquid_feed])[-1, 0],
        vapor_feed.sum() * vapor.input_jacobian(np.r_[bulk[5], bulk[6], vapor_feed])[-1, 0],
    ])
    if np.any(~np.isfinite(capacity)) or np.any(capacity <= 0):
        raise RuntimeError("Native capacity scales must be finite positive")
    flux_scale = np.asarray(diagnostic[3]).ravel() * gas[:2]
    heat_scale = float(diagnostic[4]) * span + np.abs(np.asarray(diagnostic[6]).ravel()) @ flux_scale
    state_scale = np.r_[bulk[:4], 300.0, 300.0, bulk[6], .97, flux_scale, heat_scale, 1.0]
    balance_scale = np.r_[bulk[:4] / height, capacity * span / height, bulk[6] / height]
    algebraic_scale = np.r_[.97, flux_scale[0] * float(diagnostic[2]), flux_scale, heat_scale]
    boundary_scale = np.r_[bulk[:4], span, span, bulk[6]]
    if any(np.any(~np.isfinite(scale)) or np.any(scale <= 0) for scale in (state_scale, balance_scale, algebraic_scale, boundary_scale)):
        raise RuntimeError("Conserved state, balance, algebraic and boundary scales must be finite positive")
    margin = .97 * np.finfo(float).eps
    lower = np.r_[[0.0] * 4, 293.15, 293.15, 1.0, margin, [-np.inf] * 4]
    upper = np.r_[[np.inf] * 4, 393.15, 393.15, 1e7, .97-margin, [np.inf] * 4]
    algebraic = np.asarray(node(0.0, point)[2]).ravel()
    initialized = bool(root.converged and np.max(abs(algebraic / algebraic_scale)) <= tolerance)
    checkpoint.update(stage="initialized", initialization={"state": point, "algebraic_residual": algebraic, "accepted": initialized},
                      scaling={"state_scale": state_scale, "balance_scale": balance_scale, "algebraic_scale": algebraic_scale,
                               "boundary_scale": boundary_scale, "capacity_rates_w_k": capacity,
                               "policy": "case-specific 293.15--393.15 K (300 K scale, 100 K span) and 0.97 holdup bounds"})
    if not initialized:
        raise RuntimeError("Full original algebraic initialization check failed")
    grid = assembly["coordinate"]
    initial = np.tile(point[:, None], (1, len(grid)))
    checkpoint.update(stage="global_solve", initial_profile=initial)
    from .BVP.Methods.Casadi_Collocation import solve_conservative_collocation
    result = solve_conservative_collocation(
        node, boundary, grid, initial, lower, upper, state_scale=state_scale, balance_scale=balance_scale,
        algebraic_scale=algebraic_scale, boundary_scale=boundary_scale, tolerance=settings["tolerance"],
        max_iterations=settings["max_iterations"], scheme=config.numerics.method,
        boundary_slots=[(0, -1), (1, -1), (2, 0), (3, 0), (4, -1), (5, 0), (6, 0)] if config.numerics.method == "central" else None,
    )
    checkpoint.update(stage="physical_verification", result=result)
    missing_physical_checks = (
        "film-quadrature charge and capture"
        if config.model.film_model == "equilibrium_manifold" else "capture"
    )
    physical = {"accepted": False, "reason": f"Subset checks omit {missing_physical_checks} evidence"}
    if result["profile"] is not None:
        profile = np.asarray(result["profile"])
        evaluated = [tuple(np.asarray(value).ravel() for value in node(z, state)) for z, state in zip(result["grid"], profile.T)]
        conserved = np.column_stack([value[0] for value in evaluated])
        interface = np.column_stack([value[2] for value in evaluated])
        drift = conserved[[2, 3, 5]] - conserved[[0, 1, 4]]
        drift -= drift[:, :1]
        residual = {
            "material": drift[:2], "energy": drift[2], "interface": interface,
            "boundary": np.asarray(boundary(profile[:, 0], profile[:, -1])).ravel(),
        }
        scaled = {
            "material": float(np.max(abs(drift[:2] / boundary_scale[:2, None]))),
            "energy": float(np.max(abs(drift[2] / (balance_scale[4] * height)))),
            "interface": float(np.max(abs(interface / algebraic_scale[:, None]))),
            "boundary": float(np.max(abs(residual["boundary"] / boundary_scale))),
        }
        physical = {"accepted": False,
            "original_residuals": residual, "scaled_residual_inf": scaled,
            "criteria": {key: tolerance for key in scaled},
            "scope": (
                "Subset: native grid conservation, interface, boundary, and original bounds; "
                f"{missing_physical_checks} checks remain unavailable"
            ),
            "reason": "Subset checks cannot establish physical certification"}
    result_record = {key: value for key, value in prepared.items() if key != "assembly"}
    result_record.update(stage="finished", execution_status="completed", initialization=checkpoint["initialization"],
                         scaling=checkpoint["scaling"], initial_profile=initial, result=result,
                         physical_certification=physical, native_calls=checkpoint["native_calls"])
    return result_record


def execute_conserved_column(config: ColumnConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Execute one twelve-state case through its selected verified interpreter."""
    config = resolve_column_config(config.as_dict() if isinstance(config, ColumnConfig) else config)
    if config.preset != TWELVE_PRESET:
        raise ConfigurationError("execute_conserved_column requires the twelve_state_conserved preset")
    with tempfile.TemporaryDirectory(prefix="conserved_execute_") as temporary:
        temporary = Path(temporary)
        input_path, output_path = temporary / "input.json", temporary / "output.json"
        input_path.write_text(json.dumps(_safe({"task": "conserved_execution", "config": config.as_dict(),
            "output_path": str(output_path), "runtime_identity": config.engine.as_dict()})), encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        completed = None
        cause = None
        try:
            completed = subprocess.run(
                [config.engine.python, "-m", "mea_absorption_column.benchmark_worker", str(input_path)],
                cwd=_ROOT, env=environment, capture_output=True, text=True,
                timeout=config.execution.wall_limit_s, check=False,
            )
        except BaseException as error:
            cause = error
        raw_output = output_path.read_text(encoding="utf-8") if output_path.is_file() else None
        transport = {
            "returncode": getattr(completed, "returncode", None),
            "stdout": getattr(completed, "stdout", None) if completed is not None else getattr(cause, "stdout", None),
            "stderr": getattr(completed, "stderr", None) if completed is not None else getattr(cause, "stderr", None),
            "cause": type(cause).__name__ if cause is not None else None,
            "message": str(cause) if cause is not None else None,
        }
        parsed = None
        output_error = None
        if raw_output is not None:
            try:
                parsed = json.loads(raw_output)
                if not isinstance(parsed, Mapping):
                    raise ValueError("worker output must be a JSON object")
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                output_error = str(error)
        checkpoint = parsed.get("last_checkpoint") if isinstance(parsed, Mapping) else None
        worker_failure = parsed.get("failure_kind") if isinstance(parsed, Mapping) else None
        timed_out = isinstance(cause, subprocess.TimeoutExpired)
        interrupted = isinstance(cause, KeyboardInterrupt)
        if output_error is not None or raw_output is None:
            status = ("interrupted" if interrupted else "timed_out" if timed_out else
                      "launch_failed" if isinstance(cause, OSError) else "output_invalid")
            return {"failure_kind": status, "execution_status": status, "message": output_error or "worker returned no output",
                    "raw_output": raw_output, "last_checkpoint": checkpoint or {}, "transport_failure":
                    "launch_failed" if isinstance(cause, OSError) else None, "transport": transport}
        if timed_out or interrupted:
            status = "interrupted" if interrupted else "timed_out"
            return {**parsed, "failure_kind": status, "execution_status": status,
                    "message": "Conserved execution worker was interrupted" if interrupted else "Conserved execution worker exceeded its wall limit",
                    "completed_payload": parsed if parsed.get("stage") == "finished" and isinstance(parsed.get("result"), Mapping) else None,
                    "last_checkpoint": checkpoint or {}, "transport": transport}
        if parsed.get("stage") == "finished" and not isinstance(parsed.get("result"), Mapping):
            return {"failure_kind": "output_invalid", "execution_status": "output_invalid",
                    "message": "Worker completed payload has the wrong shape", "raw_output": raw_output,
                    "last_checkpoint": checkpoint or {}, "transport": transport}
        transport_failure = "nonzero_exit" if transport["returncode"] not in (None, 0) else None
        status = worker_failure or transport_failure
        if status is not None:
            return {**parsed, "failure_kind": status, "execution_status": "failed",
                    "worker_failure_kind": worker_failure, "transport_failure": transport_failure,
                    "transport": transport}
        if parsed.get("stage") != "finished":
            return {**parsed, "failure_kind": "incomplete", "execution_status": "incomplete",
                    "message": "Worker returned a checkpoint without a completed payload",
                    "last_checkpoint": checkpoint or {}, "transport": transport}
        return {**parsed, "execution_status": "completed", "transport": transport}


def _structured_solver_success(method: str, stages: Mapping[str, Any]) -> bool:
    required = {"single": ("root", "ivp"), "scipy-bvp": ("outer",), "finite": ("outer",)}.get(method, ("outer",))
    if not stages or any(
        isinstance(stage, Mapping) and (
            stage.get("success") is False or stage.get("status") == "timed_out"
        )
        for stage in stages.values()
    ):
        return False
    return all(
        isinstance(stages.get(name), Mapping) and stages[name].get("success") is True
        for name in required
    )


def _run_conserved_preparation(config: ColumnConfig) -> dict[str, Any]:
    """Run the configured conserved column and retain the actual worker result."""
    output_dir = Path(config.execution.output_dir) if config.execution.output_dir else None
    if output_dir is not None:
        if output_dir.exists():
            raise FileExistsError(f"Refusing overwrite existing output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    base = {
        "schema_version": 1,
        "attempt_id": f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}",
        "preset": config.preset,
        "formulation": config.model.formulation,
        "native_layout": config.model.layout,
        "coordinate": config.model.coordinate,
        "config": config.as_dict(),
        "config_sha256": config.resolved_config_sha256,
        "source_identity": _source_identity(),
        "execution": {"status": "prepared", "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "numerical_acceptance": "not_evaluated",
        "physical_acceptance": "not_evaluated",
        "scientific_acceptance": "not_evaluated",
        "native_profile": {"available": False, "reason": "worker has not returned a native profile"},
        "solver": {"method": config.numerics.method, "stages": {}},
        "limitations": [
            "exploratory numerical attempt only; no physical certification or scientific promotion claim",
            (
                "species mobilities are case-estimated and use the declared harmonic-mean closure"
                if config.model.film_model == "equilibrium_manifold"
                else "the enhancement reference retains its declared empirical kinetics and diffusivity basis"
            ),
            "native vapor caloric reference remains provisional for thermal interpretation",
        ],
        "failure": None,
    }
    _record(output_dir, "resolved_config.json", config.as_dict())
    _record(output_dir, "attempt.json", base)
    try:
        prepared = execute_conserved_column(config)
        if prepared.get("failure_kind") or prepared.get("execution_status") in {"incomplete", "output_invalid", "interrupted"}:
            kind = prepared["failure_kind"]
            base["execution"].update(status=prepared.get("execution_status", "failed"), runtime_s=time.perf_counter() - started)
            base["failure"] = {"kind": kind, "phase": prepared.get("last_checkpoint", {}).get("stage", "worker"),
                               "message": prepared.get("message"), "last_checkpoint": prepared.get("last_checkpoint"),
                               "worker_failure_kind": prepared.get("worker_failure_kind"),
                               "transport_failure": prepared.get("transport_failure"),
                               "transport": prepared.get("transport"), "raw_output": prepared.get("raw_output"),
                               "completed_payload": prepared.get("completed_payload")}
            _record(output_dir, "attempt.json", base)
            return base
        base.update({
            "resolved_inputs": prepared["resolved_inputs"],
            "engine": prepared["engine"],
            "transport": prepared.get("transport"),
            "dependencies": {"assets": prepared["assets"], "capabilities": prepared["capabilities"]},
            "assembly": prepared["layout"],
            "result": prepared.get("result"),
            "initialization": prepared.get("initialization"), "scaling": prepared.get("scaling"),
            "initial_profile": prepared.get("initial_profile"), "native_calls": prepared.get("native_calls"),
            "diagnostics": {"assembly": "complete", "solver_failure": prepared.get("result", {}).get("failure"),
                            "physical_certification": prepared.get("physical_certification")},
            "native_profile": {"available": prepared.get("result", {}).get("profile") is not None,
                               "grid": prepared.get("result", {}).get("grid"),
                               "state_matrix": prepared.get("result", {}).get("profile")},
            "solver": {"method": config.numerics.method, "stages": {"outer": {
                "success": bool((prepared.get("result", {}).get("solver_statistics") or {}).get("success", False)),
                "status": prepared.get("result", {}).get("status")}}},
        })
    except CapabilityRefusal as error:
        base["execution"].update(status="failed", runtime_s=time.perf_counter() - started)
        base["failure"] = {"kind": "capability_refusal", "phase": "preparation", "message": str(error)}
    except TimeoutError as error:
        base["execution"].update(status="timed_out", runtime_s=time.perf_counter() - started)
        base["failure"] = {"kind": "timed_out", "phase": "preparation", "message": str(error)}
    except (ConfigurationError, OSError, RuntimeError, ValueError) as error:
        base["execution"].update(status="failed", runtime_s=time.perf_counter() - started)
        base["failure"] = {"kind": "preparation_failed", "phase": "preparation", "message": str(error)}
    else:
        base["execution"]["runtime_s"] = time.perf_counter() - started
        base["execution"]["status"] = "completed"
        base["numerical_acceptance"] = "accepted" if prepared.get("result", {}).get("accepted") else "rejected"
        base["physical_acceptance"] = "not_evaluated"
    if output_dir is not None:
        _record(output_dir, "resolved_config.json", config.as_dict())
        _record(output_dir, "attempt.json", base)
    return base

def run_column(config: ColumnConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Run one resolved seven-state case and retain stage/failure provenance."""
    if not isinstance(config, ColumnConfig):
        if hasattr(config, "as_dict"):
            config = config.as_dict()
        if not isinstance(config, Mapping):
            raise ConfigurationError("Column configuration must be a mapping or resolved config record")
        config = resolve_column_config(config)
    else:
        # Re-resolve even a dataclass instance so dataclasses.replace cannot bypass
        # the same wire-level validation used for TOML/JSON configurations.
        config = resolve_column_config(config.as_dict())
    if config.preset == "twelve_state_conserved":
        return _run_conserved_preparation(config)
    if config.preset != "seven_state_legacy":
        raise ConfigurationError(f"Unsupported column preset: {config.preset}")

    output_dir = None
    parent_attempt_id = None
    prepared = None
    if config.execution.output_dir is not None:
        output_dir = Path(config.execution.output_dir)
        if output_dir.exists():
            existing_path = output_dir / "attempt.json"
            existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.is_file() else None
            if not config.execution.resume or existing is None:
                raise FileExistsError(f"Refusing overwrite existing output directory: {output_dir}")
            if existing.get("config_sha256") != config.resolved_config_sha256:
                raise ConfigurationError("Resume configuration fingerprint does not match retained attempt")
            if existing is None or existing.get("resolved_inputs") is None:
                raise ConfigurationError("Retained attempt lacks resolved inputs required for safe resume")
            retained_source = existing.get("source_identity") or {}
            current_source = _source_identity()
            for key in ("commit", "dirty_paths", "source_tree_sha256", "dirty_patch_sha256"):
                if retained_source.get(key) != current_source.get(key):
                    raise ConfigurationError(f"Resume source identity mismatch: {key}")
            prepared_frame, prepared_source = _case_frame(config)
            prepared_inputs = _resolved_inputs(config, prepared_frame)
            prepared_engine = verify_engine(config.engine)
            stored_lineage = (existing.get("resolved_inputs") or {}).get("source_lineage", {})
            current_lineage = prepared_inputs.get("source_lineage", {})
            for key in ("source_file_sha256", "source_row_sha256", "physical_input_file_sha256"):
                if stored_lineage.get(key) != current_lineage.get(key):
                    raise ConfigurationError(f"Resume input identity mismatch: {key}")
            stored_engine = existing.get("engine") or {}
            for key in ("wheel", "expected_sha256", "actual_sha256", "expected_commit", "python", "status"):
                if stored_engine.get(key) != prepared_engine.get(key):
                    raise ConfigurationError(f"Resume Engine identity mismatch: {key}")
            prepared = (prepared_frame, prepared_source, prepared_inputs, prepared_engine)
            if existing.get("numerical_acceptance") == "accepted" and existing.get("execution", {}).get("status") == "completed":
                return existing
            if not config.execution.retry:
                return existing
            parent_attempt_id = existing.get("attempt_id")
            output_dir = output_dir.parent / f"{output_dir.name}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        output_dir.mkdir(parents=True, exist_ok=False)

    attempt_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}"

    base = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "parent_attempt_id": parent_attempt_id,
        "preset": config.preset,
        "formulation": config.model.formulation,
        "native_layout": config.model.layout,
        "coordinate": config.model.coordinate,
        "config": config.as_dict(),
        "config_sha256": config.resolved_config_sha256,
        "source_identity": _source_identity(),
        "resolved_inputs": None,
        "engine": None,
        "execution": {
            "status": "prepared",
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "solver": {"method": config.numerics.method, "stages": {}},
        "legacy_outcome": None,
        "numerical_acceptance": "not_evaluated",
        "physical_acceptance": "not_evaluated",
        "scientific_acceptance": "not_evaluated",
        "native_profile": {"available": False, "reason": "preparation has not completed"},
        "observables": {},
        "diagnostics": {},
        "failure": None,
    }
    _record(output_dir, "resolved_config.json", config.as_dict())
    _record(output_dir, "attempt.json", base)
    started = time.perf_counter()

    def finish() -> dict[str, Any]:
        _record(output_dir, "attempt.json", base)
        return base

    try:
        if prepared is None:
            frame, case_source = _case_frame(config)
            inputs = _resolved_inputs(config, frame)
            engine = verify_engine(config.engine)
        else:
            frame, case_source, inputs, engine = prepared
        base["resolved_inputs"] = inputs
        base["engine"] = engine
        _record(output_dir, "attempt.json", base)

        from . import benchmark

        result = benchmark._run_one_case(
            frame, 0, case_source, config.numerics.method, config.model.thermo_model,
            _settings(config, output_dir, inputs),
        )
        stage_status = result.get("solver_stage_status") or {}
        message = str(result.get("message") or "")
        timed_out = (
            result.get("jacobian_status") == "subprocess_timeout"
            or any(stage.get("status") == "timed_out" for stage in stage_status.values() if isinstance(stage, Mapping))
            or "exceeded subprocess_timeout_s" in message
        )
        failed_stage = any(
            isinstance(stage, Mapping) and stage.get("success") is False
            for stage in stage_status.values()
        )
        solver_success = bool(result.get("success", False))
        worker_failure_kind = result.get("failure_kind")
        if worker_failure_kind == "capability_refusal":
            execution_status = "failed"
            failure_kind = "capability_refusal"
        elif timed_out:
            execution_status = "timed_out"
            failure_kind = "timed_out"
        elif failed_stage or not solver_success:
            execution_status = "failed"
            failure_kind = "solver_failed"
        else:
            execution_status = "completed"
            failure_kind = None

        base["execution"].update({"status": execution_status, "runtime_s": time.perf_counter() - started})
        base["solver"]["stages"] = stage_status
        base["solver"]["legacy_message"] = result.get("message")
        base["solver"]["iterations"] = result.get("solver_iterations")
        base["solver"]["final_mesh_nodes"] = result.get("final_mesh_nodes")
        base["legacy_outcome"] = {"success": solver_success, "message": result.get("message")}

        boundary = result.get("boundary_residual_norm")
        capture_error = result.get("capture_error_pct")
        numerical_criteria = []
        if config.acceptance.boundary_residual_max is not None:
            numerical_criteria.append(
                boundary is not None and math.isfinite(float(boundary))
                and float(boundary) <= config.acceptance.boundary_residual_max
            )
        observation_criteria = []
        if config.acceptance.capture_error_max_pct is not None:
            observation_criteria.append(
                capture_error is not None and math.isfinite(float(capture_error))
                and abs(float(capture_error)) <= config.acceptance.capture_error_max_pct
            )
        if not observation_criteria:
            base["observation_agreement"] = "not_evaluated"
        elif execution_status != "completed":
            base["observation_agreement"] = "rejected"
        else:
            base["observation_agreement"] = "accepted" if all(observation_criteria) else "rejected"
        base["observation_agreement_scope"] = "capture agreement only"
        if execution_status != "completed":
            base["numerical_acceptance"] = "rejected"
        elif numerical_criteria:
            if not stage_status:
                # A passing scalar criterion cannot turn an unstructured legacy
                # row into a numerical acceptance. A failing criterion can still
                # reject the row without inventing solver-stage evidence.
                base["numerical_acceptance"] = "not_evaluated" if all(numerical_criteria) else "rejected"
            else:
                structured = _structured_solver_success(config.numerics.method, stage_status)
                base["numerical_acceptance"] = (
                    "accepted" if all(numerical_criteria) and structured and not failed_stage else "rejected"
                )
        else:
            base["numerical_acceptance"] = "not_evaluated"

        base["observables"] = {
            "capture": {
                "value": result.get("capture_pct"),
                "unit": "percent",
                "basis": "specified inlet vapor CO2",
                "model_value": result.get("raw_capture_pct", result.get("capture_pct")),
                "corrected_value": result.get("capture_pct"),
            },
            "temperature_rmse": {
                "value": result.get("temperature_rmse_K"),
                "unit": "K",
                "basis": "case temperature taps",
            },
            "boundary_residual_norm": {
                "value": boundary,
                "unit": "percent-norm",
                "basis": "legacy seven-state boundary equations",
            },
        }
        base["diagnostics"] = {
            "solver_stage_status": stage_status,
            "failure_kind": worker_failure_kind,
            "boundary_residual_components": result.get("boundary_residual_components"),
            "max_rms_residual": result.get("max_rms_residual"),
            "max_scaled_boundary_residual": result.get("max_scaled_boundary_residual"),
            "first_failed_domain": result.get("first_failed_domain"),
            "domain_guard_counts": result.get("domain_guard_counts"),
        }
        if result.get("_native_grid") is not None and result.get("_native_state_scaled") is not None:
            base["native_profile"] = {
                "available": True,
                "grid": {
                    "values": result["_native_grid"],
                    "unit": "normalized_height",
                    "identity": "native_solver_grid",
                },
                "state_matrix_scaled": result["_native_state_scaled"],
                "layout": result.get("_native_state_layout"),
                "sampled_profiles": result.get("_profiles"),
            }
        else:
            base["native_profile"] = {
                "available": False,
                "reason": "solver did not return an internal native profile",
            }
        if result.get("worker_identity") is not None:
            base["engine"]["worker"] = result["worker_identity"]
        base["result"] = result
        if failure_kind:
            base["failure"] = {
                "kind": failure_kind,
                "phase": "worker_preparation" if failure_kind == "capability_refusal" else "solver",
                "cause": result.get("message"),
                "diagnostics": base["diagnostics"],
            }
    except KeyboardInterrupt as exc:
        base["execution"].update({"status": "interrupted", "runtime_s": time.perf_counter() - started})
        base["failure"] = {"kind": "interrupted", "cause": str(exc), "diagnostics": base["diagnostics"]}
        base["result"] = None
    except TimeoutError as exc:
        base["execution"].update({"status": "timed_out", "runtime_s": time.perf_counter() - started})
        base["failure"] = {"kind": "timed_out", "cause": str(exc), "diagnostics": base["diagnostics"]}
        base["result"] = None
    except Exception as exc:
        base["execution"].update({"status": "failed", "runtime_s": time.perf_counter() - started})
        base["failure"] = {"kind": type(exc).__name__, "cause": str(exc), "diagnostics": base["diagnostics"]}
        base["result"] = None
    return finish()
