import numpy as np


def finite_difference(f, x, *args, h=1e-7):

    x_forward = np.copy(x)
    x_backward = np.copy(x)

    x_forward += h  # Increment the i-th component
    x_backward -= h  # Decrement the i-th component

    # Central difference formula for the partial derivative

    grad = (f(x_forward, *args) - f(x_backward, *args)) / (2 * h)

    return grad
