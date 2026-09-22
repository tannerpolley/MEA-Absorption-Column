"""Reduce explicitly certified coupled-column records; never run a solver."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median


STATE_LAYOUT = ["Fl_CO2_mol_s", "Fl_H2O_mol_s", "Fv_CO2_mol_s", "Fv_H2O_mol_s",
                "Tl_K", "Tv_K", "P_Pa", "holdup", "j_CO2_mol_m2_s",
                "j_H2O_mol_m2_s", "energy_flux_W_m2", "interface_log_loading"]
MEASUREMENTS = ("wall_s", "cpu_s", "peak_rss_bytes", "setup_wall_s",
                "thermo_wall_s", "film_wall_s", "global_wall_s")
PHYSICAL_INPUTS = ("height_m", "liquid_feed_mol_s", "vapor_feed_mol_s", "liquid_temperature_k",
                   "vapor_temperature_k", "bottom_pressure_pa", "area_m2", "packing",
                   "gas_mass_flow_basis", "liquid_mass_flow_basis", "humidity_assumption", "case_id")
RESIDUAL_GROUPS = ("material", "energy", "charge", "interface", "boundary")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def eligible(run):
    return (run.get("execution_status") == "completed"
            and run.get("termination") not in ("timeout", "external_wall_time_limit", "interrupted")
            and run["result"] is not None and run["result"].get("accepted") is True
            and run["physical_certification"].get("accepted") is True)


def validate(run):
    for key in ("run_id", "method", "settings", "physical_certification"):
        if not run.get(key):
            raise ValueError(f"Missing {key}")
    if run.get("execution_status") not in ("completed", "failed", "timeout", "interrupted", "not_run"):
        raise ValueError("An explicit execution_status is required")
    for key in MEASUREMENTS:
        value = run.get("measurements", {}).get(key)
        if value is not None and (not finite(value) or value < 0):
            raise ValueError(f"Invalid measured {key}; use null for unmeasured values")
    if not run.get("problem"):
        if (run["execution_status"] in ("failed", "timeout", "interrupted")
                and run.get("result") is None and run.get("failure")
                and run["physical_certification"].get("accepted") is not True):
            return  # Preserve setup failures, explicitly outside physical comparison.
        raise ValueError("Missing problem identity")
    method_hash = run["settings"].get("method_source_sha256", "")
    if not isinstance(method_hash, str) or len(method_hash) != 64 or any(c not in "0123456789abcdef" for c in method_hash):
        raise ValueError("A method source SHA-256 is required in settings")
    problem = run["problem"]
    if (problem.get("state_layout") != STATE_LAYOUT
            or problem.get("coordinate") != "height_m_from_gas_inlet"):
        raise ValueError("Expected unscaled twelve-state layout and upward physical height")
    for key, length in (("engine_commit", 40), ("wheel_sha256", 64),
                        ("parameters_sha256", 64), ("reference_sha256", 64)):
        value = problem.get(key, "")
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing or invalid {key}")
    sources = problem.get("model_source_sha256", {})
    if not sources or any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in sources.values()):
        raise ValueError("Model/transport source hashes are required")
    inputs = problem["physical_inputs"]
    if any(key not in inputs or inputs[key] is None for key in PHYSICAL_INPUTS):
        raise ValueError("Incomplete resolved feeds, geometry, boundary or flow-basis inputs")
    for key in ("height_m", "liquid_temperature_k", "vapor_temperature_k", "bottom_pressure_pa", "area_m2"):
        if not finite(inputs[key]) or inputs[key] <= 0:
            raise ValueError(f"Invalid physical input {key}")
    for key, size in (("liquid_feed_mol_s", 3), ("vapor_feed_mol_s", 4), ("packing", 7)):
        if len(inputs[key]) != size or any(not finite(v) or v <= 0 for v in inputs[key]):
            raise ValueError(f"Invalid physical input {key}")
    if not 0 < inputs["packing"][1] < 1:
        raise ValueError("Packing void fraction must be between zero and one")
    for key in ("gas_mass_flow_basis", "liquid_mass_flow_basis", "humidity_assumption", "case_id"):
        if not isinstance(inputs[key], str) or not inputs[key].strip():
            raise ValueError(f"Missing physical input description {key}")
    height = inputs["height_m"]
    if not finite(height) or height <= 0:
        raise ValueError("Packed height must be finite and positive")
    if run["physical_certification"].get("accepted") is True:
        for field in ("original_residuals", "criteria"):
            evidence = run["physical_certification"].get(field, {})
            if any(key not in evidence or evidence[key] is None or evidence[key] == {} or evidence[key] == []
                   for key in RESIDUAL_GROUPS):
                raise ValueError(f"Physical certification needs material/energy/charge/interface/boundary {field}")
    if not eligible(run):
        return
    grid, profile = run["result"]["grid"], run["result"]["profile"]
    if (len(grid) < 2 or any(not finite(z) for z in grid)
            or any(b <= a for a, b in zip(grid, grid[1:]))
            or grid[0] != 0 or grid[-1] != height):
        raise ValueError("Grid must strictly increase from zero to the declared packed height")
    if (len(profile) != len(STATE_LAYOUT) or any(len(row) != len(grid) for row in profile)
            or any(not finite(value) for row in profile for value in row)):
        raise ValueError("Accepted profile must be finite, unscaled and 12 by grid-size")
    if profile[2][0] <= 0:
        raise ValueError("Capture requires positive inlet vapor CO2 flow")


def interpolate(grid, values, height):
    # ponytail: piecewise-linear exports; retain native dense output before claiming spline extrema.
    i = min(bisect_right(grid, height) - 1, len(grid) - 2)
    fraction = (height - grid[i]) / (grid[i + 1] - grid[i])
    return values[i] + fraction * (values[i + 1] - values[i])


def profile_metrics(run):
    grid, profile = run["result"]["grid"], run["result"]["profile"]
    metrics = {"capture_pct": 100 * (1 - profile[2][-1] / profile[2][0])}
    for name, values in (("Tl", profile[4]), ("Tv", profile[5])):
        maximum = max(values)
        locations = [z for z, value in zip(grid, values) if value == maximum]
        metrics.update({f"{name}_peak_K": maximum,
                        f"{name}_peak_first_height_m": locations[0],
                        f"{name}_peak_last_height_m": locations[-1]})
    return metrics


def reduce_records(records, reference_id):
    """Return pairwise differences, never an automatic accuracy or method ranking."""
    for run in records:
        validate(run)
    by_id = {run["run_id"]: run for run in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate run_id would hide an attempt")
    reference = by_id[reference_id]
    if not eligible(reference):
        raise ValueError("Reference needs numerical acceptance and explicit physical certification")
    if any(run.get("problem") and run["problem"] != reference["problem"] for run in records):
        raise ValueError("Problem identities or physical inputs differ; split the comparison")
    accepted = [run for run in records if eligible(run)]
    heights = sorted({z for run in accepted for z in run["result"]["grid"]})
    reference_metrics = profile_metrics(reference)
    reference_temperatures = [[interpolate(reference["result"]["grid"],
                              reference["result"]["profile"][index], z) for z in heights]
                             for index in (4, 5)]
    rows, profiles, groups = [], [], {}
    for run in records:
        result = run["result"] or {}
        row = dict(run_id=run["run_id"], method=run["method"], eligible=eligible(run),
                   problem_identity_available=bool(run.get("problem")),
                   solver_accepted=result.get("accepted"),
                   physical_accepted=run["physical_certification"].get("accepted"),
                   execution_status=run["execution_status"],
                   termination=run.get("termination", result.get("status")),
                   failure=run.get("failure") or result.get("failure"),
                   final_nodes=len(result.get("solver_grid", result["grid"])) if result.get("grid") is not None else None,
                   profile_samples=len(result["grid"]) if result.get("grid") is not None else None,
                   iterations=result.get("iterations"), capture_delta_pp=None,
                   Tl_difference_inf_K=None, Tv_difference_inf_K=None)
        row.update({key: None for key in reference_metrics})
        if eligible(run):
            row.update(profile_metrics(run))
            row["capture_delta_pp"] = row["capture_pct"] - reference_metrics["capture_pct"]
            temperatures = [[interpolate(result["grid"], result["profile"][index], z)
                             for z in heights] for index in (4, 5)]
            for name, values, ref_values in zip(("Tl", "Tv"), temperatures, reference_temperatures):
                row[f"{name}_difference_inf_K"] = max(abs(a - b) for a, b in zip(values, ref_values))
            profiles.extend(dict(run_id=run["run_id"], height_m=z, Tl_K=tl, Tv_K=tv)
                            for z, tl, tv in zip(heights, *temperatures))
        rows.append(row)
        measurements = run.get("measurements", {})
        context = measurements.get("context", {})
        complete_context = all(context.get(key) for key in ("hardware", "threads", "software", "scope"))
        key = json.dumps([run["method"], run["settings"], context,
                          None if complete_context else run["run_id"]], sort_keys=True)
        groups.setdefault(key, []).append(run)
    timings = []
    for key, runs in groups.items():
        method, settings, context, isolated_id = json.loads(key)
        timing = dict(method=method, settings=settings, context=context,
                      run_ids=[r["run_id"] for r in runs], attempts=len(runs),
                      eligible_attempts=sum(eligible(r) for r in runs),
                      ineligible_attempts=sum(not eligible(r) for r in runs),
                      comparable_timing_context=isolated_id is None)
        for metric in MEASUREMENTS:
            samples = [r.get("measurements", {}).get(metric) for r in runs if eligible(r)]
            samples = [v for v in samples if v is not None] if isolated_id is None else []
            timing[metric] = {"samples": len(samples), "median": median(samples) if samples else None,
                              "min": min(samples) if samples else None, "max": max(samples) if samples else None}
        timings.append(timing)
    return {"reference_id": reference_id, "profile_representation": "piecewise_linear_on_exported_grid",
            "thermal_extrema_representation": "sampled_on_exported_grid; continuous thermal peak accuracy requires sampling/refinement evidence",
            "claim_limit": "Pairwise differences and certified-success timing samples; no matched-accuracy ranking.",
            "rows": rows, "profiles": profiles, "timing_groups": timings,
            "records": records}  # Preserve original dimensional residuals, candidates, counts and failures.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--reference", required=True, help="Explicit run_id; no automatic best-run selection")
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing output is never overwritten")
    args = parser.parse_args()
    data = [path.read_bytes() for path in args.records]
    report = reduce_records([json.loads(value) for value in data], args.reference)
    report["input_files"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(value).hexdigest()}
                             for path, value in zip(args.records, data)]
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(encoded)
    for name, rows in (("comparison", report["rows"]), ("profiles", report["profiles"])):
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
