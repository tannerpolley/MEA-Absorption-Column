import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pcsaft import InputError


def eulers(fxn, y, t_eval, scales, args=(), **kwargs):


    n_steps = len(t_eval)
    step_size = t_eval[1] - t_eval[0]
    # Initialize storage for results
    results = np.zeros((n_steps, len(y)))
    results[0] = y*scales  # Set initial values

    # Iteratively apply Euler's method
    for i in range(1, n_steps):
        t = t_eval[i - 1]
        y_prev = results[i-1]
        # Compute derivatives using the provided function
        dydt_scaled = np.array(fxn(t, y_prev/scales, *args, **kwargs))
        dydt = dydt_scaled*scales
        # Update the dependent variables
        results[i] = y_prev + step_size * dydt

    return results.T, t_eval, 'Success', 'Yay'


def runge_kutta(fxn, y0, t_eval, scales, args=(), **kwargs):
    """
    Solve an ODE using the 4th-order Runge-Kutta method.

    Parameters:
        f (function): The ODE function, y' = f(t, y).
        t0 (float): The initial time.
        y0 (float): The initial value of y at t0.
        t_end (float): The end time of the interval.
        h (float): The step size.

    Returns:
        t_values (numpy array): Array of time values.
        y_values (numpy array): Array of y values corresponding to t_values.
    """
    # Initialize variables
    t_values = [t_eval[0]]
    t_end = t_eval[-1]
    y_values = [y0]
    h = t_eval[1] - t_eval[0]
    t = t_eval[0]
    print(y0)
    y = np.array(y0)

    # Main RK4 loop
    while t < t_end:
        # Adjust step size if the next step exceeds t_end
        if t + h > t_end:
            h = t_end - t

        # Calculate slopes
        try:
            k1 = h * fxn(t, y, *args, **kwargs)
            k2 = h * fxn(t + h / 2, y + k1 / 2, *args, **kwargs)
            k3 = h * fxn(t + h / 2, y + k2 / 2, *args, **kwargs)
            k4 = h * fxn(t + h, y + k3, *args, **kwargs)
        except InputError:
            k1 = np.nan
            k2 = np.nan
            k3 = np.nan
            k4 = np.nan
        # Update y and t
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h

        # Store results
        t_values.append(t)
        y_values.append(list(y))

    Y = np.array(y_values)
    Y = Y.T

    return np.array(y_values).T, t_eval, 'Success', 'Yay'


def scipy_integrate(fxn, y0, t_eval, scales, args=(), **kwargs):

    obj = solve_ivp(fxn, [t_eval[0], t_eval[-1]], y0,
                    args=args,
                    method='Radau', t_eval=t_eval,
                    vectorized=False,
                    # options={'first_step': 1e-10,
                    #          'max_step': 1e-5,
                    #          'rtol': 1e-0,
                    #          'atol': 1e-0}
                    )

    Y_scaled = obj.y
    success = obj.success
    message = obj.message
    z = obj.t

    return Y_scaled, z, success, message