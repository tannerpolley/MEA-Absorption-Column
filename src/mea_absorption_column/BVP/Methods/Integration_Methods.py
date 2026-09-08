import numpy as np


def eulers(fxn, y, t_eval, args=None):
    """Explicit Euler comparison integrator on the supplied mesh."""
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or len(t_eval) < 2 or not np.all(np.isfinite(t_eval)):
        raise ValueError("Euler integration requires at least two finite mesh points")
    steps = np.diff(t_eval)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError("Euler integration requires a strictly monotonic mesh")
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("Euler integration requires a finite initial state vector")
    results = np.zeros((len(t_eval), len(y)))
    results[0] = y
    for i, step in enumerate(steps):
        with np.errstate(over="raise", invalid="raise"):
            derivative = np.asarray(fxn(t_eval[i], results[i], args), dtype=float)
            if derivative.shape != y.shape or not np.all(np.isfinite(derivative)):
                raise ValueError("Euler integration requires a finite derivative vector")
            results[i + 1] = results[i] + step * derivative
        if not np.all(np.isfinite(results[i + 1])):
            raise ValueError("Euler integration produced a non-finite state")
    return results.T, t_eval, True, "Euler integration completed"
