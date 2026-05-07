from mea_absorption_column.BVP.Methods.Integration_Methods import eulers
from scipy.integrate import solve_ivp
from scipy.optimize import root
from mea_absorption_column.BVP.ABS_Column import abs_column
from mea_absorption_column.BVP.robust_core import guard_column_rhs
import numpy as np
import time


DEFAULT_SINGLE_SHOOT_SETTINGS = {
    'integrator': 'euler',
    'ivp_method': 'BDF',
    'ivp_rtol': 1e-5,
    'ivp_atol': 1e-8,
    'root_method': 'Krylov',
    'fatol': 0.1,
    'maxiter': 50,
    'line_search': 'armijo',
    'display': False,
}


def single_shoot_solve(Y_a_scaled, Y_b_scaled, z, parameters, settings=None):
    settings = {**DEFAULT_SINGLE_SHOOT_SETTINGS, **(settings or {})}
    settings["_runtime_start_s"] = time.time()
    rhs = _guarded_abs_column if settings.get("guard_rhs", True) else _raw_abs_column

    Fl_CO2_a_guess, Fl_H2O_a_guess, Fv_CO2_a, Fv_H2O_a, Hlf_a_guess, Hvf_a, P_a = Y_a_scaled
    Fl_CO2_b, Fl_H2O_b, Fv_CO2_b_guess, Fv_H2O_b_guess, Hlf_b, Hvf_b_guess, P_b = Y_b_scaled

    integrater = _select_integrator(settings)

    shoot = True

    if shoot:
        def shooter(X):
            _raise_if_timed_out(settings)

            Fl_CO2_a, Fl_H2O_a, Hlf_a = X

            Y_a_scaled = [Fl_CO2_a, Fl_H2O_a,
                          Fv_CO2_a, Fv_H2O_a,
                          Hlf_a,  Hvf_a, P_a]

            Y_scaled, _, _, _ = integrater(rhs, Y_a_scaled, z, args=parameters)
            _raise_if_timed_out(settings)

            Fl_CO2_b_sim, Fl_H2O_b_sim, Hlf_b_sim = Y_scaled[0, -1], Y_scaled[1, -1], Y_scaled[4, -1]

            eq1 = Fl_CO2_b_sim - Fl_CO2_b
            eq2 = Fl_H2O_b_sim - Fl_H2O_b
            eq3 = Hlf_b_sim - Hlf_b

            eqs = [eq1, eq2, eq3]

            return eqs

        Y_0_guess = np.array([Y_a_scaled[0], Y_a_scaled[1], Y_a_scaled[4]])

        method = settings['root_method']
        options = {
            'fatol': settings['fatol'],
            'maxiter': settings['maxiter'],
            'line_search': settings['line_search'],
            'disp': settings['display'],
        }
        root_output = root(shooter, Y_0_guess, method=method, options=options)
        _raise_if_timed_out(settings)
        if len(parameters) > 6 and isinstance(parameters[6], dict):
            parameters[6].get("solver_diagnostics", {})["jacobian_status"] = str(getattr(root_output, "status", ""))

        solved_initials_scaled, success, message, n_eval = root_output.x, root_output.success, root_output.message, root_output.nit

        Fl_CO2_a, Fl_H2O_a, Hlf_a = solved_initials_scaled
        Y_a_scaled = [Fl_CO2_a, Fl_H2O_a,
                      Fv_CO2_a, Fv_H2O_a,
                      Hlf_a, Hvf_a, P_a]

    Y_scaled, z, success, message = integrater(rhs, Y_a_scaled, z, args=parameters)

    return Y_scaled, z, 'Single Shooting Method', success, message


def _guarded_abs_column(zi, y_scaled, parameters):
    return guard_column_rhs(zi, y_scaled, parameters, evaluator=abs_column)


def _raw_abs_column(zi, y_scaled, parameters):
    return abs_column(zi, y_scaled, parameters)


def _select_integrator(settings):
    integrator = str(settings.get("integrator", "euler")).lower()
    if integrator == "euler":
        return eulers
    if integrator in {"solve_ivp", "ivp", "bdf", "radau", "rk45"}:
        method_map = {"bdf": "BDF", "radau": "Radau", "rk45": "RK45"}
        method = settings.get("ivp_method", method_map.get(integrator, "BDF"))
        rtol = float(settings.get("ivp_rtol", 1e-5))
        atol = float(settings.get("ivp_atol", 1e-8))

        def integrate(fxn, y0, t_eval, args=None):
            def rhs(t, y):
                _raise_if_timed_out(settings)
                return np.asarray(fxn(t, y, args), dtype=float)

            obj = solve_ivp(
                rhs,
                (float(t_eval[0]), float(t_eval[-1])),
                np.asarray(y0, dtype=float),
                method=method,
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
            )
            if obj.y.shape[1] != len(t_eval):
                return obj.y, obj.t, bool(obj.success), obj.message
            return obj.y, t_eval, bool(obj.success), obj.message

        return integrate
    raise ValueError(f"Unknown shooting integrator: {settings.get('integrator')}")


def _raise_if_timed_out(settings):
    max_runtime = settings.get("max_runtime_s")
    if max_runtime is None:
        return
    start = settings.get("_runtime_start_s")
    if start is None:
        return
    if time.time() - float(start) > float(max_runtime):
        raise TimeoutError(f"Shooting solve exceeded max_runtime_s={float(max_runtime):g}")
