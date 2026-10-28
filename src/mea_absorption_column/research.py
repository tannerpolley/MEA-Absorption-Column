"""Explicit research selections; preview by default, execute only with --run.

Run this file directly for a preview without importing the package runtime.
Solver settings remain the runner's extensible numerical option mapping.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import inspect
import importlib.util
from pathlib import Path
import sys
import tomllib

OPTIONS = {
    "seven_state_legacy": {
        "thermo_model": ["ideal_henry", "epcsaft_neutral", "epcsaft_ionic", "epcsaft_reactive_nine"],
        "film_model": ["enhancement_factor", "reactive_film_linearization"],
        "method": ["single", "scipy-bvp", "finite"],
        "thermal_state_mode": ["enthalpy", "temperature"],
    },
    "twelve_state_conserved": {
        "thermo_model": ["reactive_epcsaft"],
        "film_model": ["equilibrium_manifold", "enhancement_reference"],
        "method": ["trapezoidal", "central", "upwind"],
        "unavailable_methods": ["shooting", "collocation"],
    },
}
LIMITATION = (
    "Selections identify existing calculations, not a validated Cartesian product. "
    "Seven-state ePC-SAFT adapters require retired Engine interfaces and are unavailable "
    "on the current pin; execution checks the selected prepared runtime before the outer solver. "
    "The twelve-state column retains its fixed case policy and exploratory physical limits; "
    "reduced shooting/collocation are unavailable. Built-in eNRTL and MDEA are unavailable."
)

_COLUMN_API = None


def _column_api():
    global _COLUMN_API
    if _COLUMN_API is not None:
        return _COLUMN_API
    if __package__:
        from .config.column import ColumnConfig, resolve_column_config
    else:
        path = Path(__file__).with_name("config") / "column.py"
        spec = importlib.util.spec_from_file_location("_mea_absorption_column_config", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load configuration module: {path}")
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        ColumnConfig, resolve_column_config = module.ColumnConfig, module.resolve_column_config
    _COLUMN_API = (ColumnConfig, resolve_column_config)
    return _COLUMN_API


def resolve_config(config):
    """Resolve the ordinary absorber request through its one configuration owner."""
    ColumnConfig, resolve_column_config = _column_api()
    return config if isinstance(config, ColumnConfig) else resolve_column_config(config)


def run_research(config):
    """Execute the resolved single-case selection and retain the native attempt."""
    resolved = resolve_config(config)
    if resolved.execution.output_dir is None:
        raise ValueError("An explicit new execution.output_dir is required for --run")
    from mea_absorption_column.column import run_column

    return run_column(resolved)


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
        print(json.dumps({"options": OPTIONS, "limitations": LIMITATION}, indent=2))
        return 0
    try:
        if args.config is None:
            raise ValueError("Provide a TOML or JSON config or --list-options")
        with args.config.open("rb") as stream:
            request = json.load(stream) if args.config.suffix.lower() == ".json" else tomllib.load(stream)
        config = resolve_config(request)
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"error": {"kind": type(error).__name__, "cause": str(error)}}, indent=2))
        return 2
    if not args.run:
        print(json.dumps({"selection": config.as_dict(), "limitations": LIMITATION}, indent=2))
        return 0
    try:
        with redirect_stdout(sys.stderr):
            attempt = run_research(config)
    except (OSError, ValueError) as error:
        print(json.dumps({"error": {"kind": type(error).__name__, "cause": str(error)}}, indent=2))
        return 2
    print(json.dumps(attempt, indent=2, allow_nan=False))
    return 0 if attempt["execution"]["status"] == "completed" and attempt["failure"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
