"""Run issue #93 problem P and check the assembled NLP at its accepted solution."""
import json
import sys
from pathlib import Path

import casadi as ca
import numpy as np

from mea_absorption_column.benchmark_worker import main as worker
from mea_absorption_column.column import _run_conserved_preparation
from mea_absorption_column.config.column import resolve_column_config


def checks(balance, inlet, solver, candidate, lower, upper, p, profile):
    x, lam_g, lam_x = (np.asarray(candidate[k]).ravel() for k in ("x", "lam_g", "lam_x"))
    variables, parameters = ca.MX.sym("x", x.size), ca.MX.sym("p", p.size)
    residual = solver.get_function("nlp_g")
    gradient = ca.jacobian(residual(variables, parameters), ca.vertcat(variables, parameters))
    jac = np.asarray(ca.Function("constraint_gradient", [variables, parameters], [gradient], {"cse": True})(x, p))
    point, exact = np.r_[x[-1], p], jac[:, x.size - 1:]
    steps, errors = [1e-3, 1e-4, 1e-5], []
    for relative_step in steps:
        signed_steps = np.diag(relative_step * np.abs(point))
        def evaluate(values):
            return np.asarray(residual(np.r_[x[:-1], values[0]], values[1:])).ravel()
        difference = np.column_stack([(evaluate(point + h) - evaluate(point - h)) / (2 * h[i])
                                      for i, h in enumerate(signed_steps)])
        errors.append(np.max(abs(difference - exact), axis=0) / np.max(abs(exact), axis=0))
    sensitivity = -np.asarray(candidate["lam_p"]).ravel()
    scale_identity = p[:3] @ sensitivity[:3]
    tol = 1e-7  # Case 3C physical residual tolerance and problem P feasibility limit.
    slacks = np.array([x - lower, upper - x])
    multipliers = np.array([np.maximum(-lam_x, 0.), np.maximum(lam_x, 0.)])
    finite = np.isfinite(slacks)
    active = np.any(slacks <= tol, axis=0)
    capture_slack = -float(candidate["g"][-1])
    jac_x = jac[:, :x.size]
    active_jac = np.vstack((jac_x if capture_slack <= tol else jac_x[:-1], np.eye(x.size)[active]))
    inlets = inlet(x[-1], p)
    pressure_gradient = [float(balance(state[:8], state[8:11], inlets)[5][-1]) for state in profile.T]
    return dict(relative_steps=steps, relative_errors=errors, derivative_pass=bool(np.all(np.asarray(errors) <= 1e-5)),
                scale_identity=scale_identity, scale_identity_relative_error=abs(scale_identity + x[-1]) / abs(x[-1]),
                peak_liquid_temperature_k=max(profile[4]), peak_holdup=max(profile[7]), max_pressure_gradient_pa_per_m=max(pressure_gradient),
                capture_slack_scaled=capture_slack, bound_slacks_scaled=slacks, bound_multipliers_scaled=multipliers,
                active_bound_indices=np.flatnonzero(active), active_jacobian_sigma_min=np.linalg.svd(active_jac, compute_uv=False)[-1],
                complementarity=max(abs(capture_slack * lam_g[-1]), np.max(abs(slacks[finite] * multipliers[finite]))))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        sys.exit(worker(sys.argv[1:], design_check=checks))
    request = json.loads(Path(sys.argv[1]).read_text())
    output = Path(__file__).resolve().parents[1] / "results" / "runs" / sys.argv[2]
    request.setdefault("execution", {})["output_dir"] = str(output)
    result = _run_conserved_preparation(resolve_column_config(request), worker_args=("-u", str(Path(__file__).resolve())))
    transport = result.get("transport") or (result.get("failure") or {}).get("transport") or {}
    (output / "ipopt.log").write_text((transport.get("stdout") or "") + (transport.get("stderr") or ""))
    verified = (result.get("result") or {}).get("design_checks", {})
    sys.exit(0 if result["numerical_acceptance"] == result["physical_acceptance"] == "accepted"
             and verified.get("derivative_pass") and verified["scale_identity_relative_error"] <= 1e-6 else 1)
