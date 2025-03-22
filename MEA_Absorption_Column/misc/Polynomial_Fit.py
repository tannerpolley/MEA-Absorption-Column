import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def polynomial_fit(z, y_int, i):

    df = pd.read_csv('data/fitted_coefficients.csv')

    p = df.iloc[:, 1+i].to_numpy()

    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    y = []
    for x in z:
        Σ = 0
        if i == 2:

            p = p[:2]
            y.append(exp_decay(x, *p))

        else:
            for j, pi in enumerate(p):
                Σ += pi * x ** (len(p) - j)
            y.append(Σ + y_int)

    return y

