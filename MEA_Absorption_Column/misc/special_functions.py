import numpy as np
EPS = np.finfo(float).eps


def finite_difference(f, x, *args, h=1e-7):

    x_forward = np.copy(x)
    x_backward = np.copy(x)

    x_forward += h  # Increment the i-th component
    x_backward -= h  # Decrement the i-th component

    # Central difference formula for the partial derivative

    grad = (f(x_forward, *args) - f(x_backward, *args)) / (2 * h)

    return grad


def jac(f, x, y):

    n, m = y.shape
    dtype = y.dtype

    df_dy = np.empty((n, n, m), dtype=dtype)
    h = EPS ** 0.5 * (1 + np.abs(y))
    for i in range(n):
        y_new = y.copy()
        y_new2 = y.copy()
        y_new[i] += h[i]
        y_new2[i] -= h[i]
        hi = y_new[i] - y[i]
        f_new = f(x, y_new)
        f_new2 = f(x, y_new2)

        df_dy[:, i, :] = (f_new - f_new2) / (2*hi)

    return df_dy
