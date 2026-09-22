"""Render retained certified comparisons; never import or execute the model."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"trapezoidal": "#0072B2", "central": "#D55E00", "shooting": "#009E73", "collocation": "#CC79A7"}
LABELS = {"trapezoidal": "Trapezoidal", "central": "Centered FD", "shooting": "Shooting", "collocation": "Adaptive collocation"}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(report, output, *, refinement=None, capture_limit=None, temperature_limit=None):
    """An explicit reference is already selected and certified by compare.py."""
    import sys
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts))
    from compare import reduce_records
    # Recompute the cheap reducer: do not trust edited eligibility flags/metrics.
    verified = reduce_records(report["records"], report["reference_id"])
    rows = {r["run_id"]: r for r in verified["rows"]}
    records = {r["run_id"]: r for r in verified["records"]}
    methods = list(dict.fromkeys(r["method"] for r in verified["records"]))
    if any(method not in COLORS for method in methods):
        raise ValueError("Use the four explicit full-reactive method identifiers")
    if capture_limit is not None:
        contexts = [r.get("measurements", {}).get("context", {}) for r in records.values()]
        if any(not all(c.get(k) for k in ("hardware", "threads", "software", "scope")) for c in contexts) or any(c != contexts[0] for c in contexts):
            raise ValueError("Cost comparison requires identical complete measurement contexts")
    output.mkdir(parents=True, exist_ok=False)
    files = []

    def save(fig, name):
        for ax in fig.axes:
            ax.grid(alpha=.2)
        for extension in ("svg", "png", "pdf"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=180)
            files.append(path.name)
        plt.close(fig)

    profiles = [dict(p, method=records[p["run_id"]]["method"]) for p in verified["profiles"]]
    write_csv(output / "profiles.csv", profiles)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    for run_id, row in rows.items():
        if not row["eligible"]:
            continue
        data = [p for p in profiles if p["run_id"] == run_id]
        for axis, key in zip(axes, ("Tl_K", "Tv_K")):
            axis.plot([p["height_m"] for p in data], [p[key] for p in data],
                      color=COLORS[row["method"]], label=run_id)
    for axis, phase in zip(axes, ("Liquid", "Vapor")):
        axis.set(xlabel="Height from gas inlet (m)", ylabel=f"{phase} temperature (K)")
    axes[-1].legend(frameon=False, fontsize=7)
    save(fig, "temperature_profiles")

    if refinement:
        varying = {"axial": "nodes", "film": "film_points", "ode": "ivp_rtol", "bvp": "tolerance"}[refinement]
        allowed = {"axial": ("trapezoidal", "central"), "film": ("trapezoidal", "central"),
                   "ode": ("shooting",), "bvp": ("collocation",)}[refinement]
        invariant_keys = set().union(*(r["settings"] for r in records.values())) - {varying, "initial_state"}
        if refinement == "ode":
            invariant_keys.discard("ivp_atol")
        groups = {}
        for run in records.values():
            if run["method"] not in allowed:
                continue
            settings = run["settings"]
            if varying not in settings:
                raise ValueError(f"Missing {varying} for refinement")
            fixed = {k: settings.get(k) for k in sorted(invariant_keys)}
            if refinement == "ode":
                fixed["ivp_absolute_relative_ratio"] = str(Decimal(str(settings["ivp_atol"]))/Decimal(str(settings["ivp_rtol"])))
            key = json.dumps([run["method"], fixed], sort_keys=True)
            groups.setdefault(key, []).append(run)
        data = []
        # Differences use each same-settings series' finest certified run.
        for group in groups.values():
            accepted = [r for r in group if rows[r["run_id"]]["eligible"]]
            levels = {r["settings"][varying] for r in accepted}
            if len(levels) < 2:
                continue
            finest = min if refinement in ("ode", "bvp") else max
            reference = finest(accepted, key=lambda r: r["settings"][varying])
            reduced = reduce_records(group, reference["run_id"])
            for row in reduced["rows"]:
                run = records[row["run_id"]]
                data.append(dict(run_id=row["run_id"], method=run["method"],
                    level=run["settings"][varying], reference_id=reference["run_id"], eligible=row["eligible"],
                    capture_difference_pp=abs(row["capture_delta_pp"]) if row["eligible"] else None,
                    temperature_difference_K=max(row["Tl_difference_inf_K"], row["Tv_difference_inf_K"]) if row["eligible"] else None))
        if not data:
            raise ValueError("No certified same-settings series with at least two refinement levels")
        write_csv(output / f"{refinement}_refinement.csv", data)
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
        for method in methods:
            values = [r for r in data if r["method"] == method and r["eligible"] and r["run_id"] != r["reference_id"]]
            if not values:
                continue
            for axis, metric in zip(axes, ("capture_difference_pp", "temperature_difference_K")):
                axis.scatter([r["level"] for r in values], [r[metric] for r in values],
                             color=COLORS[method], label=LABELS[method])
        for axis, label in zip(axes, ("Capture difference (percentage points)", "Temperature sup difference (K)")):
            axis.set(xlabel={"axial": "Axial nodes", "film": "Film quadrature points",
                "ode": "IVP relative tolerance", "bvp": "Adaptive BVP tolerance"}[refinement], ylabel=label)
            if refinement in ("ode", "bvp"):
                axis.set_xscale("log")
        axes[-1].legend(frameon=False, fontsize=7)
        save(fig, f"{refinement}_refinement")

    if capture_limit is not None and temperature_limit is not None:
        costs = []
        for run_id, row in rows.items():
            run = records[run_id]
            matched = bool(row["eligible"] and abs(row["capture_delta_pp"]) <= capture_limit
                and max(row["Tl_difference_inf_K"], row["Tv_difference_inf_K"]) <= temperature_limit)
            costs.append(dict(run_id=run_id, method=row["method"], matched=matched, eligible=row["eligible"],
                execution_status=run["execution_status"], wall_s=run.get("measurements", {}).get("wall_s"),
                capture_delta_pp=row["capture_delta_pp"], Tl_difference_inf_K=row["Tl_difference_inf_K"],
                Tv_difference_inf_K=row["Tv_difference_inf_K"], failure=row["failure"]))
        write_csv(output / "attempt_costs.csv", costs)
        fig, axis = plt.subplots(figsize=(6.7, 3.5), constrained_layout=True)
        for index, method in enumerate(methods):
            for row in (r for r in costs if r["method"] == method and r["wall_s"] is not None):
                marker = "o" if row["eligible"] else "x"
                axis.scatter(index, row["wall_s"], marker=marker, color=COLORS[method],
                             facecolors=COLORS[method] if row["matched"] or marker == "x" else "none")
        axis.set(xticks=range(len(methods)), xticklabels=[LABELS[m] for m in methods], ylabel="Measured attempt wall time (s)")
        axis.text(.01, .99, "Filled: within declared differences; open: outside; ×: failed/uncertified",
                  transform=axis.transAxes, va="top", fontsize=7)
        save(fig, "accuracy_and_cost")
    write_csv(output / "attempt_summary.csv", verified["rows"])
    (output / "render.json").write_text(json.dumps(dict(figures=files, reference_id=report["reference_id"],
        refinement=refinement, capture_limit_pp=capture_limit, temperature_limit_K=temperature_limit,
        limits="Differences from certified numerical reference, not known absolute errors. Exported-grid profiles are piecewise linear. Cost points retain individual attempts; no cross-setting averages."), indent=2)+"\n")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"output")
    parser.add_argument("--refinement", choices=("axial", "film", "ode", "bvp"))
    parser.add_argument("--capture-limit-pp", type=float)
    parser.add_argument("--temperature-limit-k", type=float)
    args = parser.parse_args()
    if (args.capture_limit_pp is None) != (args.temperature_limit_k is None):
        parser.error("Supply both achieved-difference limits for cost comparison")
    if any(v is not None and v <= 0 for v in (args.capture_limit_pp, args.temperature_limit_k)):
        parser.error("Comparison limits must be positive")
    data = args.comparison.read_bytes()
    files = render(json.loads(data), args.output, refinement=args.refinement,
                   capture_limit=args.capture_limit_pp, temperature_limit=args.temperature_limit_k)
    shutil.copyfile(args.comparison, args.output/"comparison.json")
    (args.output/"input.sha256").write_text(hashlib.sha256(data).hexdigest()+"\n")
    print("\n".join(files))


if __name__ == "__main__":
    main()
