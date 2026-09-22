"""Exact local DAE reduction shared by shooting and adaptive collocation."""
from __future__ import annotations

import casadi as ca
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import least_squares, root


def dop853_dense_derivative(dense, mesh, balances):
    """Differentiate the documented degree-seven DOP853 interpolant, not the EOS.

    Reconstruct on Chebyshev nodes in each native IVP interval and check the
    reconstruction at separate points before evaluating its analytic derivative.
    Only the public dense-output callable and returned IVP mesh are consumed.
    """
    x = np.cos(np.arange(8)*np.pi/7)
    check = np.array([-.8, -.3, .2, .7])
    heights, values, derivatives, errors = [], [], [], []
    for left, right in zip(mesh[:-1], mesh[1:]):
        half, middle = .5*(right-left), .5*(right+left)
        sampled = dense(middle+half*x)[:balances]
        offset = sampled[:, :1]
        coefficients = np.polynomial.chebyshev.chebfit(x, (sampled-offset).T, 7)
        z = middle+half*check
        actual = dense(z)[:balances]
        recovered = np.polynomial.chebyshev.chebval(check, coefficients)+offset
        error = float(np.max(abs(recovered-actual)/np.maximum(1., abs(actual))))
        if not np.isfinite(error) or error > 128*np.finfo(float).eps:
            raise RuntimeError(f"DOP853 degree-seven dense reconstruction failed: {error}")
        heights.extend(z)
        values.append(actual)
        derivatives.append(np.polynomial.chebyshev.chebval(check,
            np.polynomial.chebyshev.chebder(coefficients))/half)
        errors.append(error)
    return np.asarray(heights), np.hstack(values), np.hstack(derivatives), max(errors)


class ConservedReduction:
    """Recover u from B(z,u)=q, a(z,u)=0 and integrate w=q/conserved_scale.

    The caller supplies physical bounds, a fixed branch seed, scales and local
    admission limits. Native refusals propagate. Local numerical admission is
    not a physical column certificate or proof of global branch uniqueness.
    Retain this object: its node owns the native derivative callbacks.
    """

    def __init__(self, node, boundary, *, initial_state, lower, upper, state_scale,
                 conserved_scale, algebraic_scale, boundary_scale, tolerance, solver_tolerance,
                 max_evaluations, max_condition):
        self.node, self.boundary = node, boundary
        if node.n_in() != 2 or node.n_out() != 3 or boundary.n_in() != 2 or boundary.n_out() != 1:
            raise ValueError("Expected node(z,u)->(B,R,a) and boundary(left,right)")
        self.n, self.m, self.q = node.size1_in(1), node.size1_out(0), node.size1_out(2)
        if (self.m < 1 or self.n != self.m+self.q or node.size_in(0) != (1, 1)
                or node.size_in(1) != (self.n, 1) or node.size_out(0) != (self.m, 1)
                or node.size_out(1) != (self.m, 1) or node.size_out(2) != (self.q, 1)
                or boundary.size_out(0) != (self.m, 1)
                or any(boundary.size_in(i) != (self.n, 1) for i in (0, 1))):
            raise ValueError("Expected n=m+q states, m balances/boundaries and q algebraic equations")
        for name, values, size in (("state_scale", state_scale, self.n),
                                   ("conserved_scale", conserved_scale, self.m),
                                   ("algebraic_scale", algebraic_scale, self.q),
                                   ("boundary_scale", boundary_scale, self.m)):
            values = np.asarray(values, dtype=float).reshape(-1)
            if values.size != size or np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"Invalid {name}")
            setattr(self, name, values)
        self.initial_state, self.lower, self.upper = (np.asarray(v, dtype=float).reshape(-1)
                                                     for v in (initial_state, lower, upper))
        if (any(v.size != self.n for v in (self.initial_state, self.lower, self.upper))
                or np.any(~np.isfinite(self.initial_state)) or np.any(np.isnan(self.lower))
                or np.any(np.isnan(self.upper)) or np.any(self.lower >= self.upper)
                or np.any(self.initial_state < self.lower) or np.any(self.initial_state > self.upper)):
            raise ValueError("Invalid local initial state or physical bounds")
        if (not np.isfinite(tolerance) or tolerance <= np.finfo(float).eps
                or not np.isfinite(solver_tolerance) or solver_tolerance <= np.finfo(float).eps
                or int(max_evaluations) != max_evaluations or max_evaluations < 1
                or not np.isfinite(max_condition) or max_condition <= 1):
            raise ValueError("Local tolerance, evaluation budget and condition limit must be valid")
        self.tolerance, self.max_evaluations, self.max_condition = tolerance, int(max_evaluations), max_condition
        self.solver_tolerance = solver_tolerance
        z, u = ca.MX.sym("z"), ca.MX.sym("u", self.n)
        b, r, a = node.call([z, u], True, False)
        self.height_independent = ca.jacobian(ca.vertcat(b, r, a), z).sparsity().nnz() == 0
        self.jacobian = ca.Function("local_dae_jacobian", [z, u],
            [ca.jacobian(b, u), ca.jacobian(r, u), ca.jacobian(a, u)], {"cse": True})
        left, right = ca.MX.sym("left", self.n), ca.MX.sym("right", self.n)
        bc = boundary(left, right)
        self.boundary_jacobian = ca.Function("local_boundary_jacobian", [left, right],
                                             [ca.jacobian(bc, left), ca.jacobian(bc, right)])
        self.counts = dict(local_solves=0, node_values=0, node_jacobians=0)
        self.last_local = None
        self._last = None

    def evaluate(self, z, w):
        w = np.asarray(w, dtype=float).reshape(-1)
        if not np.isfinite(z) or w.size != self.m or np.any(~np.isfinite(w)):
            raise ValueError("Reduced state and physical height must be finite")
        key = (0. if self.height_independent else float(z), w.tobytes())
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        self.counts["local_solves"] += 1
        self.last_local = dict(accepted=False, height_m=float(z), target_conserved=w*self.conserved_scale,
                               state=None, scaled_residual=None, scaled_condition=None)
        value_cache, jacobian_cache = None, None

        def values(v):
            nonlocal value_cache
            self.last_local["state"] = v*self.state_scale
            if value_cache is None or not np.array_equal(v, value_cache[0]):
                self.counts["node_values"] += 1
                evaluated = tuple(np.asarray(x).ravel() for x in self.node(z, v*self.state_scale))
                value_cache = v.copy(), evaluated
            b, _, a = value_cache[1]
            return np.r_[b/self.conserved_scale-w, a/self.algebraic_scale]

        def jac(v):
            nonlocal jacobian_cache
            if jacobian_cache is None or not np.array_equal(v, jacobian_cache[0]):
                self.counts["node_jacobians"] += 1
                evaluated = tuple(np.asarray(x) for x in self.jacobian(z, v*self.state_scale))
                jacobian_cache = v.copy(), evaluated
            bu, _, au = jacobian_cache[1]
            return np.vstack((bu/self.conserved_scale[:, None], au/self.algebraic_scale[:, None]))*self.state_scale

        solved = least_squares(values, self.initial_state/self.state_scale, jac=jac,
            bounds=(self.lower/self.state_scale, self.upper/self.state_scale),
            ftol=self.solver_tolerance, xtol=self.solver_tolerance, gtol=self.solver_tolerance,
            max_nfev=self.max_evaluations)
        state = solved.x*self.state_scale
        residual = values(solved.x)
        matrix = jac(solved.x)
        condition = float(np.linalg.cond(matrix))
        self.last_local.update(state=state, scaled_residual=residual, scaled_condition=condition,
                               status=solved.message, nfev=solved.nfev, njev=solved.njev)
        if (not solved.success or not np.all(np.isfinite(residual))
                or np.max(abs(residual)) > self.tolerance or not np.isfinite(condition)
                or condition > self.max_condition or np.any(state < self.lower) or np.any(state > self.upper)):
            raise RuntimeError(f"Local DAE recovery rejected: {solved.message}; "
                               f"scaled residual={np.max(abs(residual))}, condition={condition}")
        # M_scaled d(u/Su)/dw = [I;0]; no differentiated matrix inverse or EOS finite differences.
        state_jacobian = self.state_scale[:, None]*np.linalg.solve(matrix,
            np.vstack((np.eye(self.m), np.zeros((self.q, self.m)))))
        # The original residual/conditioning checks above evaluated this exact
        # solved coordinate; reuse those native values and full derivatives.
        b, r, a = value_cache[1]
        _, ru, _ = jacobian_cache[1]
        result = dict(state=state, rhs=r/self.conserved_scale,
                      rhs_jacobian=(ru@state_jacobian)/self.conserved_scale[:, None],
                      state_jacobian=state_jacobian, algebraic_residual=a, conserved=b)
        if any(np.any(~np.isfinite(v)) for v in result.values()):
            raise RuntimeError("Non-finite implicit DAE values or derivatives")
        self.last_local["accepted"] = True
        # Exact last-input reuse only; the root seed never follows a preceding trial.
        self._last = key, result
        return result

    def boundary_values(self, left_z, right_z, left_w, right_w):
        left, right = self.evaluate(left_z, left_w), self.evaluate(right_z, right_w)
        residual = np.asarray(self.boundary(left["state"], right["state"])).ravel()
        ja, jb = (np.asarray(v) for v in self.boundary_jacobian(left["state"], right["state"]))
        return (residual/self.boundary_scale,
                (ja@left["state_jacobian"])/self.boundary_scale[:, None],
                (jb@right["state_jacobian"])/self.boundary_scale[:, None])


def solve_reduced_bvp(model, grid, initial, *, method, tolerance, boundary_tolerance,
                      max_nodes, max_evaluations, ivp_rtol, ivp_atol):
    """Connect a shared conserved DAE reduction to SciPy BVP or shooting.

    Shooting uses explicit DOP853 for state plus variational equations and an
    exact boundary-root Jacobian. Failed runs never substitute the initial
    profile. `accepted` describes numerical termination/local/boundary checks
    and the sampled scaled differential defect;
    original column physical certification remains an independent operation.
    """
    grid, initial = np.asarray(grid, dtype=float), np.asarray(initial, dtype=float)
    if (method not in ("shooting", "collocation") or grid.ndim != 1 or len(grid) < 2
            or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0)
            or initial.shape != (model.n, len(grid)) or np.any(~np.isfinite(initial))
            or np.any(initial < model.lower[:, None]) or np.any(initial > model.upper[:, None])):
        raise ValueError("Invalid method, initial profile, bounds or increasing physical grid")
    if (any(not np.isfinite(v) or v <= 0 for v in (tolerance, boundary_tolerance, ivp_rtol, ivp_atol))
            or int(max_nodes) != max_nodes or max_nodes < len(grid)
            or int(max_evaluations) != max_evaluations or max_evaluations < 1):
        raise ValueError("Invalid BVP tolerance, mesh or evaluation budget")
    output = dict(accepted=False, grid=grid.copy(), profile=None, reduced_profile=None,
                  failure=None, status=None, iterations=None, boundary_residual=None,
                  algebraic_residual=None, differential_residual=None)
    try:
        guess = np.column_stack([np.asarray(model.node(z, initial[:, i])[0]).ravel()/model.conserved_scale
                                 for i, z in enumerate(grid)])

        def bc(a, b):
            return model.boundary_values(grid[0], grid[-1], a, b)[0]

        def bc_jac(a, b):
            return model.boundary_values(grid[0], grid[-1], a, b)[1:]

        if method == "collocation":
            def rhs(z, w):
                return np.column_stack([model.evaluate(zi, wi)["rhs"] for zi, wi in zip(z, w.T)])

            def rhs_jac(z, w):
                return np.stack([model.evaluate(zi, wi)["rhs_jacobian"] for zi, wi in zip(z, w.T)], axis=2)

            solved = solve_bvp(rhs, bc, grid, guess, fun_jac=rhs_jac, bc_jac=bc_jac,
                               tol=tolerance, bc_tol=boundary_tolerance, max_nodes=int(max_nodes))
            output.update(grid=solved.x, reduced_profile=solved.y, status=solved.message, iterations=solved.niter,
                          solver_grid=solved.x, native_rms_residuals=solved.rms_residuals)
            check_z = np.sort(np.r_[solved.x, .5*(solved.x[1:]+solved.x[:-1])])
            checked = [model.evaluate(z, w) for z, w in zip(check_z, solved.sol(check_z).T)]
            defect = (solved.sol(check_z, 1)-np.column_stack([v["rhs"] for v in checked]))*model.conserved_scale[:, None]
            output.update(differential_residual=defect, differential_residual_height_m=check_z,
                          scaled_differential_residual_inf=float(np.max(abs(defect/model.conserved_scale[:, None]))),
                          differential_residual_criterion="max(abs((q_prime-R)/conserved_scale)) <= tolerance, per height unit")
        else:
            last = None
            shooting_work = dict(ivp_solves_started=0, ivp_solves_returned=0, ivp_nfev_total=0)
            output["shooting_work"] = shooting_work

            def integrate(a):
                nonlocal last
                if last is not None and np.array_equal(a, last[0]):
                    return last[1]

                def augmented_rhs(z, augmented):
                    shooting_work["ivp_nfev_total"] += 1
                    local = model.evaluate(z, augmented[:model.m])
                    sensitivity = augmented[model.m:].reshape(model.m, model.m)
                    return np.r_[local["rhs"], (local["rhs_jacobian"]@sensitivity).ravel()]

                shooting_work["ivp_solves_started"] += 1
                ivp = solve_ivp(augmented_rhs, (grid[0], grid[-1]), np.r_[a, np.eye(model.m).ravel()],
                                method="DOP853", rtol=ivp_rtol, atol=ivp_atol, dense_output=True)
                shooting_work["ivp_solves_returned"] += 1
                if not ivp.success:
                    output.update(partial_reduced_grid=ivp.t, partial_reduced_profile=ivp.y[:model.m])
                    raise RuntimeError(f"Shooting IVP failed: {ivp.message}")
                last = a.copy(), ivp
                return ivp

            def boundary_root(a):
                return bc(a, integrate(a).y[:model.m, -1])

            def boundary_root_jac(a):
                ivp = integrate(a)
                ja, jb = bc_jac(a, ivp.y[:model.m, -1])
                return ja+jb@ivp.y[model.m:, -1].reshape(model.m, model.m)

            solved = root(boundary_root, guess[:, 0], jac=boundary_root_jac,
                          tol=tolerance, options={"maxfev": int(max_evaluations)})
            ivp = integrate(solved.x)
            output.update(grid=np.unique(np.r_[grid, ivp.t]), status=solved.message,
                          root_nfev=solved.nfev, root_njev=solved.njev, final_ivp_nfev=ivp.nfev,
                          ivp_mesh=ivp.t, solver_grid=ivp.t)
            output["reduced_profile"] = ivp.sol(output["grid"])[:model.m]
            check_z, check_w, derivative, reconstruction = dop853_dense_derivative(ivp.sol, ivp.t, model.m)
            checked = [model.evaluate(z, w) for z, w in zip(check_z, check_w.T)]
            sources = np.column_stack([v["rhs"] for v in checked])
            defect = (derivative-sources)*model.conserved_scale[:, None]
            output.update(differential_residual=defect, differential_residual_height_m=check_z,
                          scaled_differential_residual_inf=float(np.max(abs(derivative-sources))),
                          dense_reconstruction_relative_error_inf=reconstruction,
                          differential_residual_criterion="max(abs((q_prime-R)/conserved_scale)) <= tolerance, per height unit",
                          differential_reconstruction="Analytic derivative of degree-seven DOP853 dense polynomial; four independent check points per native IVP interval")
        # Export physical states already recovered for the dense residual check,
        # keeping the numerical mesh separate from the profile sampling grid.
        checked = dict(zip(check_z, checked))
        output["grid"] = np.unique(np.r_[output["grid"], check_z])
        dense = ivp.sol if method == "shooting" else solved.sol
        output["reduced_profile"] = dense(output["grid"])[:model.m]
        local = [checked[z] if z in checked else model.evaluate(z, w)
                 for z, w in zip(output["grid"], output["reduced_profile"].T)]
        output["profile_sampling"] = "Numerical mesh plus dense residual-check points; reported thermal extrema are sampled, not continuous stationary-point extrema"
        output["profile"] = np.column_stack([v["state"] for v in local])
        output["algebraic_residual"] = np.column_stack([v["algebraic_residual"] for v in local])
        output["boundary_residual"] = np.asarray(model.boundary(output["profile"][:, 0], output["profile"][:, -1])).ravel()
        differential_ok = (np.isfinite(output["scaled_differential_residual_inf"])
            and output["scaled_differential_residual_inf"] <= tolerance)
        output["accepted"] = bool(solved.success and differential_ok and
            np.max(abs(output["boundary_residual"]/model.boundary_scale)) <= boundary_tolerance)
        if not output["accepted"]:
            output["failure"] = "Global termination, sampled differential or original boundary residual check failed"
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        output.update(failure=str(error), failure_type=type(error).__name__)
    output.update(local_counts=model.counts.copy(), last_local=model.last_local,
                  local_counts_scope="Cumulative local recovery calls only; excludes initial-guess node and boundary evaluations; not total native work")
    return output
