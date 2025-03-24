import numpy as np
EPS = np.finfo(float).eps


def finite_difference(f, x, *args, h=1e-5):

    x_forward = np.copy(x)
    x_backward = np.copy(x)

    x_forward += h  # Increment the i-th component
    x_backward -= h  # Decrement the i-th component

    # Central difference formula for the partial derivative

    grad = (f(x_forward, *args) - f(x, *args)) / (h)

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


# def complex_step(f, x, *args, h=1e-30):
#
#     imag = 1j
#     x_copy = np.copy(x)
#     print(x_copy)
#     x_copy += x_copy + imag * h
#     f_eval = f(x, *args)
#     dy_dt = np.imag(f_eval) / h
#     x_copy += x_copy - imag * h
#
#     return np.real(dy_dt)

def complex_step(f, x, *args, h=1e-200):

    return np.real(np.imag(f(x + 1j * h, *args)) / h)
