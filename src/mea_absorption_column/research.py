"""Explicit research selections; preview by default, execute only with --run.

Run this file directly for a preview without importing the package runtime.
Solver settings remain the runner's extensible numerical option mapping.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import importlib
import inspect
from pathlib import Path
import tomllib

OPTIONS = {
    "formulation": ("seven_state", "conserved"),
    "thermo_model": ("ideal_henry", "epcsaft_neutral", "epcsaft_ionic", "epcsaft_reactive_nine"),
    "film_model": ("enhancement_factor", "reactive_film_linearization"),
    "method": ("single", "scipy-bvp", "finite"),
}
CASE_KEYS = {"c_case_limit", "nccc_case_limit", "srp_case_limit", "c_case_ids", "nccc_case_ids", "srp_case_ids", "c_case_dataset", "nccc_dataset"}
SETTING_KEYS = {"data_type", "staged_beds", "profile_pngs", "profile_csvs", "subprocess_timeout_s", "write_artifacts"}
LIMITATION = (
    "Selections identify implemented paths, not a validated Cartesian product. "
    "Native derivative, chemistry and numerical compatibility checks still apply. "
    "Conserved formulation requires a user problem_factory owning the physics; "
    "the twelve-state column is not integrated. Built-in eNRTL and MDEA are unavailable."
)


def resolve_config(config):
    """Validate and copy a mapping without importing or evaluating any models."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    unknown = set(config) - set(OPTIONS) - {"cases", "settings", "solver_settings", "output_dir", "problem_factory"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    resolved = deepcopy(config)
    conserved = resolved.get("formulation") == "conserved"
    for name, choices in OPTIONS.items():
        if conserved and name in {"thermo_model", "film_model"}:
            if not isinstance(resolved.get(name), str) or not resolved[name].strip():
                raise ValueError(f"{name} requires an explicit builder-owned label")
            continue
        if conserved and name == "method":
            choices = ("trapezoidal", "central", "shooting", "collocation")
        if resolved.get(name) not in choices:
            raise ValueError(f"{name} must explicitly select one of {choices}; got {resolved.get(name)!r}. {LIMITATION}")
    if conserved:
        factory = resolved.get("problem_factory", "")
        if not isinstance(factory, str) or factory.count(":") != 1 or not all(factory.split(":")):
            raise ValueError("conserved requires problem_factory='module:function'")
    elif "problem_factory" in resolved:
        raise ValueError("problem_factory applies only to conserved formulation")
    for table, keys in (("cases", None if conserved else CASE_KEYS), ("settings", None if conserved else SETTING_KEYS), ("solver_settings", None)):
        value = resolved.setdefault(table, {})
        if not isinstance(value, dict):
            raise ValueError(f"{table} must be a table")
        if keys is not None and set(value) - keys:
            raise ValueError(f"Unknown {table} keys: {sorted(set(value) - keys)}")
    solver = resolved["solver_settings"]
    if not conserved and "co2_mass_transfer_model" in solver and solver["co2_mass_transfer_model"] != resolved["film_model"]:
        raise ValueError("film_model conflicts with solver_settings.co2_mass_transfer_model")
    if not conserved:
        solver["co2_mass_transfer_model"] = resolved["film_model"]
        if resolved["film_model"] == "reactive_film_linearization" and "reactive_film_linearization" not in solver:
            raise ValueError("reactive_film_linearization requires solver_settings.reactive_film_linearization = [positions, conductances, bulk_fugacities]")
    for name, value in resolved["cases"].items():
        if name.endswith("_limit") and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
        if name.endswith("_ids") and (not isinstance(value, (list, tuple)) or not value or any(not isinstance(item, str) for item in value)):
            raise ValueError(f"{name} must be a nonempty list of case IDs")
    if "output_dir" in resolved:
        if not isinstance(resolved["output_dir"], str) or not resolved["output_dir"].strip():
            raise ValueError("output_dir must be a nonempty path string")
    return resolved


def run_research(config):
    """Run one selection using the existing benchmark and retain its settings."""
    resolved = resolve_config(config)
    if "output_dir" not in resolved:
        raise ValueError("An explicit new output_dir is required for --run")
    output = Path(resolved["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    if resolved["formulation"] == "conserved":
        output.mkdir(parents=True, exist_ok=False)
        (output / "research_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        try:
            module, function = resolved["problem_factory"].split(":")
            problem = getattr(importlib.import_module(module), function)(deepcopy(resolved))
            result = run_conserved(problem, resolved["method"], resolved["solver_settings"])
            (output / "result.json").write_text(json.dumps(result, indent=2, default=_json_value) + "\n")
            return result
        except Exception as exc:
            (output / "failure.json").write_text(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2) + "\n")
            raise
    from mea_absorption_column.benchmark import BenchmarkSettings, run_benchmark

    settings = BenchmarkSettings(
        methods=(resolved["method"],), thermo_models=(resolved["thermo_model"],),
        output_dir=output, solver_settings=resolved["solver_settings"],
        **resolved["cases"], **resolved["settings"],
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {"selection": resolved, "benchmark_settings": asdict(settings), "limitations": LIMITATION}
    (output / "research_config.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return run_benchmark(settings)


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "full"):
        return value.full().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot retain result value of type {type(value).__name__}")


def run_conserved(problem, method, settings=None):
    """Dispatch exact solver kwargs supplied by the caller; no physics substitution.

    For trapezoidal/central supply solve_conservative_collocation kwargs
    (node, boundary, grid, initial, bounds and scales). For shooting/collocation
    supply solve_reduced_bvp kwargs, including a constructed ConservedReduction
    as model. Settings supply additional solver kwargs; duplicate keys fail.
    """
    if not isinstance(problem, dict) or not isinstance(settings or {}, dict):
        raise ValueError("problem and settings must be mappings of solver keyword arguments")
    if set(problem) & set(settings or {}):
        raise ValueError("Problem and solver_settings contain duplicate keys")
    kwargs = {**problem, **(settings or {})}
    if method in {"trapezoidal", "central"}:
        from mea_absorption_column.BVP.Methods.Casadi_Collocation import solve_conservative_collocation as solve
        selector = "scheme"
    elif method in {"shooting", "collocation"}:
        from mea_absorption_column.BVP.Methods.Conserved_Reduction import solve_reduced_bvp as solve
        selector = "method"
    else:
        raise ValueError(f"Unknown conserved method: {method}")
    if selector in kwargs:
        raise ValueError(f"Select {selector} only through method")
    kwargs[selector] = method
    inspect.signature(solve).bind(**kwargs)
    return solve(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--run", action="store_true", help="Execute the selected study; otherwise only preview")
    parser.add_argument("--list-options", action="store_true")
    args = parser.parse_args(argv)
    if args.list_options:
        print(json.dumps({"options": OPTIONS, "conserved_methods": ["trapezoidal", "central", "shooting", "collocation"], "conserved_physics": "Explicit thermo_model and film_model labels interpreted by problem_factory(config)", "limitations": LIMITATION}, indent=2))
        return
    if args.config is None:
        parser.error("Provide a TOML config or --list-options")
    with args.config.open("rb") as stream:
        config = resolve_config(tomllib.load(stream))
    print(json.dumps({"selection": config, "limitations": LIMITATION}, indent=2))
    if args.run:
        run_research(config)


if __name__ == "__main__":
    main()
