


def finite_difference(f, x, h):

    m = len(f(x))
    n = len(x)
    J = np.zeros((m, n))
    for i in range(m):
        f_eval_0 = f(x.copy())[i]
        for j in range(n):
            x_new = x.copy()
            Δx = h * (1 + np.abs(x_new[j]))
            x_new[j] += Δx
            f_eval_1 = f(x_new)[i]
            J[i, j] = (f_eval_1 - f_eval_0) / Δx
            # x[j] = x[j] - Δx

    return J