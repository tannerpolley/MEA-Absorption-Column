"""Bounded conservative collocation with explicit thermodynamic variables."""
from __future__ import annotations

import casadi as ca
import numpy as np


def solve_conservative_collocation(
    node, boundary, grid, initial, lower, upper, *, state_scale,
    balance_scale, algebraic_scale, boundary_scale, tolerance=1e-8,
    max_iterations=200, iteration_callback=None, scheme="trapezoidal", boundary_slots=None,
):
    """Solve dB(z,u)/dz=R(z,u), a(z,u)=0 with countercurrent boundaries.

    ``node(z,u)`` returns column vectors B, R and a; ``boundary(bottom,top)``
    returns the physical boundary residual. For m balances and q algebraic
    equations, u has m+q entries and boundary has m entries. The supplied model
    must establish local regularity of (B,a) with respect to u.

    Temperatures belong in u; evaluate total enthalpy in B, avoiding an
    independent enthalpy/temperature inversion. Bounds apply at nodes, not to
    every coupled thermodynamic constraint or an interpolated nonlinear state.
    Scales have the units of u, dB/dz, a and boundary respectively. Failed
    returned candidates remain available but rejected; exceptions never
    substitute the initial profile for a candidate. No numerical derivatives
    or alternative-solver recovery are enabled.

    ``central`` instead differentiates B on the supplied nonuniform grid,
    retaining every nodal algebraic equation. ``boundary_slots`` lists the
    (balance row, endpoint 0 or -1) differential equations replaced by the m
    inlet conditions. All raw endpoint differential residuals remain reported.
    The default trapezoidal discretization is unchanged.
    """
    grid = np.asarray(grid, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("Collocation grid must be finite and strictly increasing")
    if node.n_in() != 2 or node.n_out() != 3 or boundary.n_in() != 2 or boundary.n_out() != 1:
        raise ValueError("Expected node(z,u)->(B,R,a) and boundary(bottom,top)")
    n, m, q = node.size1_in(1), node.size1_out(0), node.size1_out(2)
    if (m < 1 or n != m + q or node.size_in(0) != (1, 1)
            or node.size_in(1) != (n, 1) or node.size_out(0) != (m, 1)
            or node.size_out(1) != (m, 1) or node.size_out(2) != (q, 1)
            or boundary.size_out(0) != (m, 1)
            or any(boundary.size_in(i) != (n, 1) for i in (0, 1))):
        raise ValueError("Differential/algebraic dimensions or boundary equation count disagree")
    scales = [np.asarray(v, dtype=float).reshape(-1) for v in (
        state_scale, balance_scale, algebraic_scale, boundary_scale,
    )]
    for scale, size in zip(scales, (n, m, q, m), strict=True):
        if scale.size != size or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("Every physical scale must have the required size and be finite positive")
    state_scale, balance_scale, algebraic_scale, boundary_scale = scales
    lower, upper = (np.asarray(v, dtype=float).reshape(-1) for v in (lower, upper))
    if (lower.size != n or upper.size != n or np.any(np.isnan(lower)) or np.any(np.isnan(upper))
            or np.any(lower >= upper) or initial.shape != (n, grid.size)
            or np.any(~np.isfinite(initial)) or np.any(initial < lower[:, None])
            or np.any(initial > upper[:, None])):
        raise ValueError("Initial profile and original physical bounds are inconsistent")
    if not np.isfinite(tolerance) or tolerance <= 0 or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("Tolerance and iteration limit must be positive")
    if scheme not in ("trapezoidal", "central"):
        raise ValueError("Unknown conservative difference scheme")
    if scheme == "central":
        if (grid.size < 3 or boundary_slots is None or len(boundary_slots) != m
                or sorted(row for row, _ in boundary_slots) != list(range(m))
                or any(end not in (0, -1) for _, end in boundary_slots)):
            raise ValueError("Central differences need >=3 nodes and one endpoint replacement per balance row")
    elif boundary_slots is not None:
        raise ValueError("Boundary row replacements apply only to central differences")

    variables = ca.MX.sym("scaled_nodes", n, grid.size)
    physical = ca.repmat(ca.DM(state_scale), 1, grid.size) * variables
    evaluated = [node.call([ca.MX(float(z)), physical[:, k]], True, False) for k, z in enumerate(grid)]
    if scheme == "trapezoidal":
        defects = ca.horzcat(*[
            evaluated[k + 1][0] - evaluated[k][0]
            - .5 * h * (evaluated[k + 1][1] + evaluated[k][1])
            for k, h in enumerate(np.diff(grid))])
        balance_denominator = balance_scale[:, None] * np.diff(grid)[None, :]
        balance_constraints = ca.vec(defects / balance_denominator)
    else:
        derivative = [(evaluated[1][0]-evaluated[0][0])/(grid[1]-grid[0])]
        for k in range(1, grid.size-1):
            hm, hp = grid[k]-grid[k-1], grid[k+1]-grid[k]
            derivative.append(-hp/(hm*(hm+hp))*evaluated[k-1][0]
                              +(hp-hm)/(hm*hp)*evaluated[k][0]
                              +hm/(hp*(hm+hp))*evaluated[k+1][0])
        derivative.append((evaluated[-1][0]-evaluated[-2][0])/(grid[-1]-grid[-2]))
        defects = ca.horzcat(*[d-v[1] for d, v in zip(derivative, evaluated)])
        scaled_defects = ca.vec(defects / ca.repmat(ca.DM(balance_scale), 1, grid.size))
        replaced = {row+m*(0 if end == 0 else grid.size-1) for row, end in boundary_slots}
        balance_constraints = scaled_defects[[i for i in range(m*grid.size) if i not in replaced]]
    algebraic = ca.horzcat(*[v[2] for v in evaluated])
    boundary_residual = boundary(physical[:, 0], physical[:, -1])
    scaled = ca.vertcat(balance_constraints,
                       ca.vec(algebraic / ca.repmat(ca.DM(algebraic_scale), 1, grid.size)),
                       boundary_residual / boundary_scale)
    residuals = ca.Function("physical_residuals", [variables], [defects, algebraic, boundary_residual, scaled])
    result = {"accepted": False, "profile": None, "grid": grid.copy(), "status": None,
              "failure": None, "profile_finite": None, "defects": None, "algebraic_residual": None,
              "boundary_residual": None, "bound_violation": None,
              "balance_residual_per_length": None, "scaled_residual_inf": None,
              "scaled_bound_violation_inf": None, "iterations": None, "solver_statistics": None,
              "scheme": scheme, "boundary_slots": boundary_slots,
              "nlp_scaling_method": "none",
              "defect_units": "B" if scheme == "trapezoidal" else "B per metre"}
    try:
        # Share identical native callback expressions between g and its exact
        # Jacobian; the autogenerated graph repeats their value work.
        jacobian = ca.Function("column_constraint_jacobian", [ca.vec(variables), ca.MX.sym("p", 0)],
                               [scaled, ca.jacobian(scaled, ca.vec(variables))], {"cse": True})
        solver = ca.nlpsol("conservative_column", "ipopt", {
            "x": ca.vec(variables), "f": 0., "g": scaled,
        }, {"print_time": False, "record_time": True, "error_on_fail": False, "enable_fd": False,
            "jac_g": jacobian,
            **({"iteration_callback": iteration_callback} if iteration_callback is not None else {}),
            "ipopt.print_level": 0, "ipopt.max_iter": int(max_iterations),
            # Physical variables/equations are already scaled above. Automatic
            # gradient scaling repeats the costly initial constraint Jacobian.
            "ipopt.nlp_scaling_method": "none",
            "ipopt.tol": tolerance, "ipopt.constr_viol_tol": tolerance,
            "ipopt.bound_relax_factor": 0., "ipopt.hessian_approximation": "limited-memory"})
        candidate = solver(x0=ca.vec(initial / state_scale[:, None]),
                           lbx=np.tile(lower / state_scale, grid.size),
                           ubx=np.tile(upper / state_scale, grid.size), lbg=0., ubg=0.)
        stats = solver.stats()
        result.update(status=stats["return_status"], iterations=stats.get("iter_count"), solver_statistics=stats)
        coordinates = np.asarray(candidate["x"]).reshape((n, grid.size), order="F")
        profile = coordinates * state_scale[:, None]
        result["profile"] = profile
        result["profile_finite"] = bool(np.all(np.isfinite(profile)))
        violation = np.maximum(np.maximum(lower[:, None] - profile, profile - upper[:, None]), 0.)
        result.update(bound_violation=violation,
                      scaled_bound_violation_inf=float(np.max(violation / state_scale[:, None])))
        if not result["profile_finite"]:
            result["failure"] = "Solver returned a non-finite candidate"
            return result
        defect, alg, bc, scaled_value = (np.asarray(v) for v in residuals(coordinates))
        result.update(defects=defect, algebraic_residual=alg, boundary_residual=bc,
                      balance_residual_per_length=defect / np.diff(grid)[None, :] if scheme == "trapezoidal" else defect,
                      scaled_residual_inf=float(np.max(np.abs(scaled_value))))
        finite = all(np.all(np.isfinite(v)) for v in (profile, defect, alg, bc, scaled_value, violation))
        result["accepted"] = bool(stats["success"] and finite
                                  and result["scaled_residual_inf"] <= tolerance
                                  and result["scaled_bound_violation_inf"] <= tolerance)
        if not result["accepted"]:
            result["failure"] = "Solver termination or original residual/bound checks failed"
    except RuntimeError as error:
        result["failure"] = str(error)
    return result
