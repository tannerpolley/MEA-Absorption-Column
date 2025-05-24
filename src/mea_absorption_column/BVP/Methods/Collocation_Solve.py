import numpy as np
from scipy.optimize import root
from scipy.special import legendre
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def orthogonal_collocation(fun, y0, t_range, n=6):

    def get_N(a, b, n):
        points = np.sort(np.hstack([-1, legendre(n - 1).deriv().roots, 1]))
        t = 0.5 * (b - a) * points + 0.5 * (b + a)
        A = np.array([[(j + 1) * t[i + 1] ** j for j in range(n - 1)] for i in range(n - 1)])
        B = np.array([[t[i + 1] ** (j + 1) for j in range(n - 1)] for i in range(n - 1)])
        N = np.linalg.inv(np.dot(A, np.linalg.inv(B)))
        return t, N

    a, b = t_range[0], t_range[-1]
    t, N = get_N(a, b, n)
    k = len(y0)
    m = int(2 * len(y0))
    y_guess = np.zeros((int(m), n - 1))

    def solve(y):
        y = y.reshape(m, n - 1)
        diff = np.array([fun(t, y[:k, j]) for j in range(n - 1)]).T
        F1 = [np.dot(N, y[i + k]) - (y[i] - y0[i]) for i in range(k)]
        F2 = [y[i + k] - diff[i] for i in range(k)]
        return np.array(F1 + F2).flatten()

    y = root(solve, y_guess).x.reshape(m, n - 1)
    y_output = np.array([CubicSpline(t, [y0[i]] + list(y[i]))(t_range) for i in range(k)])

    return y_output