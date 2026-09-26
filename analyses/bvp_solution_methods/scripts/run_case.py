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


def request(method, nodes, points, *, output=None, wall=1200.0, profile=None, clustering=None):
    values = {"retained_profile": str(profile)} if profile else {}
    extra = {"end_clustering": clustering} if clustering else {}
    return {"preset": "twelve_state_conserved", "case": {"physical_input_file": CASE},
            "numerics": {"method": method, "solver_settings": {"nodes": nodes, "quadrature_points": points,
                                                               "max_iterations": 20, "tolerance": 1e-7, **extra}},
            "initialization": {"policy": "case_declared_native_inputs", "values": values},
            "execution": {"wall_limit_s": wall, **({"output_dir": str(output)} if output else {})}}


def outcome(attempt):
    """Status, iterations, K1 scaled residuals and CO2 capture of one retained attempt."""
    result, certificate = attempt.get("result") or {}, attempt.get("physical_certification") or {}
    profile = (attempt.get("native_profile") or {}).get("state_matrix")
    return {"attempt_id": attempt["attempt_id"], "method": attempt["config"]["numerics"]["method"],
            "nodes": attempt["config"]["numerics"]["solver_settings"]["nodes"],
            "end_clustering": attempt["config"]["numerics"]["solver_settings"].get("end_clustering"),
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


LIQUID, GAS = (0, 1, 4), (2, 3, 5, 6)  # bulk rows leaving a cell at its lower / upper node


def linear_scheme(path, node_counts):
    """Frozen-coefficient column linearization at each node of an accepted attempt (#176 design).

    Per-cell transfer-matrix eigenvalues of the upwind cells and trapezoidal rule against exp(A h);
    capture of the affine frozen BVP (native inlets) on each ladder against its exact solution."""
    import casadi as ca
    import scipy.linalg
    from mea_absorption_column.column import _prepare_conserved_column_in_process
    from mea_absorption_column.config.column import resolve_column_config

    attempt = json.loads(Path(path).read_text())
    settings = attempt["config"]["numerics"]["solver_settings"]
    node = _prepare_conserved_column_in_process(resolve_column_config(
        request("trapezoidal", settings["nodes"], settings["quadrature_points"])))["assembly"]["node"]
    grid, profile = np.asarray(attempt["result"]["grid"]), np.asarray(attempt["result"]["profile"])
    height, (bottom, top) = grid[-1], (profile[:, 0], profile[:, -1])
    x, z = ca.MX.sym("x", 12), ca.MX.sym("z")
    parts = ca.Function("parts", [z, x], [*node.call([z, x]), *(ca.jacobian(v, x) for v in node.call([z, x]))])
    cell = np.zeros((12, 24))  # u_cell = cell @ [u_k; u_k+1] + (algebraic rows free)
    for i in LIQUID:
        cell[i, i] = 1
    for i in GAS:
        cell[i, 12 + i] = 1
    report = {"source_attempt_id": attempt["attempt_id"], "nodes": []}
    for k, u0 in enumerate(profile.T):
        _, r0, a0, bu, ru, au = (np.asarray(v) for v in parts(grid[k], u0))
        r0, a0 = r0.ravel(), a0.ravel()
        manifold = scipy.linalg.null_space(au)
        particular = u0 - np.linalg.pinv(au) @ a0
        rate = np.linalg.solve(bu @ manifold, ru @ manifold)
        forcing = np.linalg.solve(bu @ manifold, r0 + ru @ (particular - u0))

        def transfer(h, scheme):
            """xi_k+1 = T xi_k + c on the node manifold; cell algebraics eliminated."""
            if scheme == "trapezoidal":
                left = bu @ manifold - h / 2 * ru @ manifold
                return np.linalg.solve(left, bu @ manifold + h / 2 * ru @ manifold)
            free = np.zeros((12, 5)); free[7:, :] = np.eye(5)
            # unknowns [xi_k+1, cell algebraics]; balances and cell algebraic equations
            upper, lower = cell[:, 12:] @ manifold, cell[:, :12] @ manifold
            left = np.block([[bu @ manifold - h * ru @ upper, -h * ru @ free], [au @ upper, au @ free]])
            right = np.vstack([bu @ manifold + h * ru @ lower, -au @ lower])
            return np.linalg.solve(left, right)[:7]

        def capture(count, scheme):
            if scheme == "exact":
                flow = scipy.linalg.expm(np.block([[rate, forcing[:, None]], [np.zeros((1, 8))]]) * height)
                rows = np.vstack([manifold[GAS, :], (manifold @ flow[:7, :7])[LIQUID, :]])
                rhs = np.r_[bottom[list(GAS)] - particular[list(GAS)],
                            top[list(LIQUID)] - particular[list(LIQUID)] - (manifold @ flow[:7, 7])[list(LIQUID)]]
                xi0 = np.linalg.solve(rows, rhs)
                ends = particular + manifold @ xi0, particular + manifold @ (flow[:7, :7] @ xi0 + flow[:7, 7])
            else:  # the same affine cells as a dense global linear system over node manifold coordinates
                h, n = height / (count - 1), count
                size = 7 * n + 5 * (n - 1)
                matrix, rhs = np.zeros((size, size)), np.zeros(size)
                free = np.zeros((12, 5)); free[7:, :] = np.eye(5)
                for j in range(n - 1):
                    rows, lo, up, cc = slice(12 * j, 12 * j + 7), slice(7 * j, 7 * j + 7), slice(7 * j + 7, 7 * j + 14), slice(7 * n + 5 * j, 7 * n + 5 * j + 5)
                    base = cell[:, :12] @ particular + cell[:, 12:] @ particular + free @ particular[7:]
                    matrix[rows, lo] = -bu @ manifold - h * ru @ cell[:, :12] @ manifold
                    matrix[rows, up] = bu @ manifold - h * ru @ cell[:, 12:] @ manifold
                    matrix[rows, cc] = -h * ru @ free
                    rhs[rows] = h * (r0 + ru @ (base - u0))
                    arow = slice(12 * j + 7, 12 * j + 12)
                    matrix[arow, lo], matrix[arow, up], matrix[arow, cc] = au @ cell[:, :12] @ manifold, au @ cell[:, 12:] @ manifold, au @ free
                    rhs[arow] = -a0 - au @ (base - u0)
                last = 12 * (n - 1)
                matrix[last:last + 4, :7] = manifold[GAS, :]
                rhs[last:last + 4] = bottom[list(GAS)] - particular[list(GAS)]
                matrix[last + 4:last + 7, 7 * (n - 1):7 * n] = manifold[LIQUID, :]
                rhs[last + 4:last + 7] = top[list(LIQUID)] - particular[list(LIQUID)]
                xi = np.linalg.solve(matrix, rhs)
                ends = particular + manifold @ xi[:7], particular + manifold @ xi[7 * (n - 1):7 * n]
            return 100 * (1 - ends[1][2] / ends[0][2])

        exact = capture(None, "exact")
        spectrum = lambda m: [[v.real, v.imag] for v in sorted(np.linalg.eigvals(m), key=abs, reverse=True)]
        report["nodes"].append({
            "z_m": float(grid[k]), "eigenvalues_per_m": sorted(np.linalg.eigvals(rate).real.tolist(), key=abs, reverse=True),
            "transfer_eigenvalues_re_im": {scheme: {str(h): spectrum(transfer(h, scheme))
                                              for h in height / (np.asarray(node_counts) - 1)}
                                     for scheme in ("upwind", "trapezoidal")},
            "exp_lambda_h_re_im": {str(h): spectrum(scipy.linalg.expm(rate * h))
                             for h in height / (np.asarray(node_counts) - 1)},
            "frozen_capture_pct": {"exact": exact, **{str(n): capture(n, "upwind") for n in node_counts}}})
    return report


def summarize(paths, control):
    rows = sorted((outcome(json.loads(Path(p).read_text())) for p in paths),
                  key=lambda r: (r["method"], r["end_clustering"] or 0, r["nodes"]))
    k2 = []
    for ladder in sorted({(r["method"], r["end_clustering"]) for r in rows}, key=lambda v: (v[0], v[1] or 0)):
        accepted = [r for r in rows if (r["method"], r["end_clustering"]) == ladder
                    and r["k1_accepted"] and r["numerical_acceptance"] == "accepted"]
        for coarse, fine in zip(accepted, accepted[1:]):
            change = abs(fine["capture_pct"] - coarse["capture_pct"])
            k2.append({"scheme": ladder[0], "end_clustering": ladder[1], "nodes": [coarse["nodes"], fine["nodes"]],
                       "capture_change_pp": change,
                       "criterion_pp": CAPTURE_CHANGE, "accepted": change <= CAPTURE_CHANGE})
    return {"attempts": rows, "k2_capture_refinement": k2,
            "k2_film_quadrature": json.loads(Path(control).read_text()) if control else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    attempt = commands.add_parser("attempt")
    attempt.add_argument("--method", choices=("trapezoidal", "central", "upwind"), required=True)
    attempt.add_argument("--nodes", type=int, required=True)
    attempt.add_argument("--film-points", type=int, default=9)
    attempt.add_argument("--end-clustering", type=float, help="0 < c <= 1 blend toward cosine nodes")
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
    linear = commands.add_parser("linear-scheme")
    linear.add_argument("attempt", type=Path)
    linear.add_argument("--nodes", type=int, nargs="+", default=[3, 5, 9, 17, 33, 65])
    linear.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("attempts", type=Path, nargs="+")
    summary.add_argument("--film-control", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "attempt":
        from mea_absorption_column.column import run_column

        record = run_column(request(args.method, args.nodes, args.film_points, output=args.output.resolve(),
                                    wall=args.wall_limit, profile=args.initial_profile and args.initial_profile.resolve(),
                                    clustering=args.end_clustering))
        print(json.dumps(clean(outcome(record)), indent=1))
    elif args.command == "film-control":
        result = film_control(args.attempt, args.film_points)
        save(args.output, result)
        print(json.dumps(clean(result), indent=1))
    elif args.command in ("diagnose", "linear-scheme"):
        result = diagnose(args.attempt) if args.command == "diagnose" else linear_scheme(args.attempt, args.nodes)
        save(args.output, result)
        print(json.dumps(clean(result), indent=1))
    else:
        result = summarize(args.attempts, args.film_control)
        save(args.output, result)
        print(json.dumps(clean(result), indent=1))


if __name__ == "__main__":
    main()
