"""Case 3C column attempts through the public twelve-state path (contract K1-K2, Engine #148).

`attempt` runs one bounded `run_column` worker and retains its attempt.json (K1 is its
native-grid physical certification). `film-control` evaluates the film integral of an
accepted attempt at its own and a finer quadrature. `diagnose` checks the node Jacobian,
collocation conditioning and stiff axial modes of a trapezoidal attempt. `summarize` reduces
attempts to K2.

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python \
    analyses/bvp_solution_methods/scripts/run_case.py attempt --method trapezoidal --nodes 2 --output DIR
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

CASE = "analyses/bvp_solution_methods/input/case_3c.json"
CAPTURE_CHANGE, FILM_CHANGE = 0.5, 1e-3  # percentage points; relative (contract K2)


def clean(value):
    """Preserve non-finite failure evidence as explicit strings, never zeros."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def request(method, nodes, points, *, output=None, wall=1200.0, profile=None):
    values = {"retained_profile": str(profile)} if profile else {}
    return {"preset": "twelve_state_conserved", "case": {"physical_input_file": CASE},
            "numerics": {"method": method, "solver_settings": {"nodes": nodes, "quadrature_points": points,
                                                               "max_iterations": 20, "tolerance": 1e-7}},
            "initialization": {"policy": "case_declared_native_inputs", "values": values},
            "execution": {"wall_limit_s": wall, **({"output_dir": str(output)} if output else {})}}


def outcome(attempt):
    """Status, iterations, K1 scaled residuals and CO2 capture of one retained attempt."""
    result, certificate = attempt.get("result") or {}, attempt.get("physical_certification") or {}
    profile = (attempt.get("native_profile") or {}).get("state_matrix")
    return {"attempt_id": attempt["attempt_id"], "method": attempt["config"]["numerics"]["method"],
            "nodes": attempt["config"]["numerics"]["solver_settings"]["nodes"],
            "quadrature_points": attempt["config"]["numerics"]["solver_settings"]["quadrature_points"],
            "execution": attempt["execution"]["status"], "runtime_s": attempt["execution"].get("runtime_s"),
            "iterations": result.get("iterations"),
            "scaled_residual_inf": result.get("scaled_residual_inf"),
            "numerical_acceptance": attempt["numerical_acceptance"], "k1_accepted": certificate.get("accepted"),
            "k1_scaled_residual_inf": certificate.get("scaled_residual_inf"),
            "capture_pct": None if profile is None else 100 * (1 - profile[2][-1] / profile[2][0]),
            "failure": attempt.get("failure")}


def film_control(path, points):
    from mea_absorption_column.column import _prepare_conserved_column_in_process
    from mea_absorption_column.config.column import resolve_column_config

    attempt = json.loads(Path(path).read_text())
    if attempt["physical_acceptance"] != "accepted":
        raise ValueError("The quadrature control needs a physically accepted attempt")
    settings = attempt["config"]["numerics"]["solver_settings"]
    states = np.asarray(attempt["native_profile"]["state_matrix"], float).T
    integrals = {}
    for count in (settings["quadrature_points"], points):
        prepared = _prepare_conserved_column_in_process(resolve_column_config(
            request(attempt["config"]["numerics"]["method"], settings["nodes"], count)))
        integrals[count] = [float(prepared["assembly"]["diagnostics"](state)[1]) for state in states]
    coarse, fine = (np.asarray(integrals[k]) for k in (settings["quadrature_points"], points))
    change = float(np.max(np.abs(coarse - fine) / np.abs(fine)))
    return {"source_attempt_id": attempt["attempt_id"], "film_integral_by_points": integrals,
            "max_relative_change": change, "criterion": FILM_CHANGE, "accepted": change <= FILM_CHANGE}


def diagnose(path):
    """Trapezoidal attempt: node-Jacobian columns vs centred differences and collocation SVD at the
    final iterate; eigenvalues of dB/dz = R on the interface-equation manifold at each node."""
    import casadi as ca
    import scipy.linalg
    from mea_absorption_column.column import _prepare_conserved_column_in_process
    from mea_absorption_column.config.column import resolve_column_config

    attempt = json.loads(Path(path).read_text())
    settings, scaling = attempt["config"]["numerics"]["solver_settings"], attempt["scaling"]
    node, boundary = (_prepare_conserved_column_in_process(resolve_column_config(
        request("trapezoidal", settings["nodes"], settings["quadrature_points"])))["assembly"][k] for k in ("node", "boundary"))
    su, sb, sa, sc = (np.asarray(scaling[k]) for k in ("state_scale", "balance_scale", "algebraic_scale", "boundary_scale"))
    grid, profile = np.asarray(attempt["result"]["grid"]), np.asarray(attempt["result"]["profile"])
    x, z = ca.MX.sym("x", 12), ca.MX.sym("z")
    b, r, a = node.call([z, x], True, False)
    parts = ca.Function("parts", [z, x], [ca.jacobian(ca.vertcat(b, r, a), x), ca.jacobian(b, x), ca.jacobian(r, x), ca.jacobian(a, x)])
    value = lambda k, u: np.concatenate([np.asarray(v).ravel() for v in node(grid[k], u)])
    report = {"source_attempt_id": attempt["attempt_id"], "nodes": []}
    for k, u in enumerate(profile.T):
        full, bu, ru, au = (np.asarray(v) for v in parts(grid[k], u))
        worst, centre = 0.0, value(k, u)
        for j in range(12):
            h = 1e-4 * max(abs(u[j]), 1e-3 * su[j])
            d4, d2, d1 = ((value(k, u + t * h * np.eye(12)[j]) - value(k, u - t * h * np.eye(12)[j])) / (2 * t * h) for t in (4, 2, 1))
            coarse, fine = (4 * d2 - d4) / 3, (4 * d1 - d2) / 3  # Richardson, as in the node ladders
            # Resolved: Richardson estimates agree to 10 %, the finest two-sided change exceeds 1e-8 of the
            # value and the entry exceeds 1e-9 of its column's largest entry (below either is round-off).
            resolved = ((np.abs(coarse - fine) <= 0.1 * np.abs(fine)) & (np.abs(2 * h * d1) >= 1e-8 * np.abs(centre))
                        & (np.abs(fine) >= 1e-9 * np.abs(full[:, j]).max()))
            defect = np.abs(full[:, j] - fine) / np.maximum(np.abs(full[:, j]), np.abs(fine)).clip(1e-300)
            worst = max(worst, float(np.max(defect[resolved], initial=0.0)))
        manifold = scipy.linalg.null_space(au)
        rates = np.linalg.eigvals(np.linalg.solve(bu @ manifold, ru @ manifold))
        stiff = rates[np.argmax(np.abs(rates))]
        report["nodes"].append({"z_m": float(grid[k]), "node_jacobian_max_relative_defect": worst,
                                "eigenvalues_per_m": sorted(rates.real.tolist(), key=abs, reverse=True),
                                "trapezoidal_amplification_of_stiffest_mode": {
                                    str(h): float(((1 + stiff * h / 2) / (1 - stiff * h / 2)).real) for h in (3.0, 1.5, 0.16)}})
    scaled = ca.MX.sym("scaled", 12, grid.size)
    physical = ca.repmat(ca.DM(su), 1, grid.size) * scaled
    at = [node.call([ca.MX(float(g)), physical[:, k]], True, False) for k, g in enumerate(grid)]
    residual = ca.vertcat(
        ca.vec(ca.horzcat(*[at[k + 1][0] - at[k][0] - .5 * h * (at[k + 1][1] + at[k][1])
                            for k, h in enumerate(np.diff(grid))]) / (sb[:, None] * np.diff(grid)[None, :])),
        ca.vec(ca.horzcat(*[v[2] for v in at]) / ca.repmat(ca.DM(sa), 1, grid.size)),
        boundary(physical[:, 0], physical[:, -1]) / sc)
    jacobian = np.asarray(ca.Function("collocation", [ca.vec(scaled)], [ca.jacobian(residual, ca.vec(scaled))])(
        (profile / su[:, None]).ravel(order="F")))
    _, singular, right = np.linalg.svd(jacobian)
    null = right[-1].reshape((12, grid.size), order="F")
    report["collocation"] = {"singular_values_smallest": singular[-4:].tolist(), "condition": float(singular[0] / singular[-1]),
                             "null_vector_by_state_and_node": null.tolist()}
    return report


def summarize(paths, control):
    rows = [outcome(json.loads(Path(p).read_text())) for p in paths]
    k2 = []
    for scheme in ("trapezoidal", "central"):
        accepted = [r for r in rows if r["method"] == scheme and r["k1_accepted"] and r["numerical_acceptance"] == "accepted"]
        for coarse, fine in zip(accepted, accepted[1:]):
            change = abs(fine["capture_pct"] - coarse["capture_pct"])
            k2.append({"scheme": scheme, "nodes": [coarse["nodes"], fine["nodes"]], "capture_change_pp": change,
                       "criterion_pp": CAPTURE_CHANGE, "accepted": change <= CAPTURE_CHANGE})
    return {"attempts": rows, "k2_capture_refinement": k2,
            "k2_film_quadrature": json.loads(Path(control).read_text()) if control else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    attempt = commands.add_parser("attempt")
    attempt.add_argument("--method", choices=("trapezoidal", "central"), required=True)
    attempt.add_argument("--nodes", type=int, required=True)
    attempt.add_argument("--film-points", type=int, default=9)
    attempt.add_argument("--initial-profile", type=Path, help="Accepted attempt.json interpolated as the initial guess")
    attempt.add_argument("--wall-limit", type=float, default=1200.0)
    attempt.add_argument("--output", type=Path, required=True)
    control = commands.add_parser("film-control")
    control.add_argument("attempt", type=Path)
    control.add_argument("--film-points", type=int, default=17)
    control.add_argument("--output", type=Path, required=True)
    diagnosis = commands.add_parser("diagnose")
    diagnosis.add_argument("attempt", type=Path)
    diagnosis.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("attempts", type=Path, nargs="+")
    summary.add_argument("--film-control", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "attempt":
        from mea_absorption_column.column import run_column

        record = run_column(request(args.method, args.nodes, args.film_points, output=args.output.resolve(),
                                    wall=args.wall_limit, profile=args.initial_profile and args.initial_profile.resolve()))
        print(json.dumps(clean(outcome(record)), indent=1))
    elif args.command == "film-control":
        result = film_control(args.attempt, args.film_points)
        save(args.output, result)
        print(json.dumps(clean(result), indent=1))
    elif args.command == "diagnose":
        result = diagnose(args.attempt)
        save(args.output, result)
        print(json.dumps(clean(result), indent=1))
    else:
        result = summarize(args.attempts, args.film_control)
        save(args.output, result)
        print(json.dumps(clean(result), indent=1))


if __name__ == "__main__":
    main()
