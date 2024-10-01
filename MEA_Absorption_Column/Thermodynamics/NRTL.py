import numpy as np


def nrtl(x, Tl):

    # From Morgan 2017 Thermodynamic modeling and uncertainty quantification of CO2-loaded
    # aqueous MEA solutions
    a = np.array([[0, 0, 69.38507],
                  [0, 0, 3.25515],
                  [0, 4.33838, 0]])

    b = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, -2197.53, 0]])

    α = .3

    τ = a + b/Tl

    G = np.exp(-α*τ)
    γ = []
    for i in range(len(x)):

        Σ_1 = np.sum([x[j]*τ[i, j]*G[i, j] for j in range(len(x))])
        Σ_2 = np.sum([x[k]*G[k, i] for k in range(len(x))])
        Σ_3 = 0
        for j in range(len(x)):
            Σ_4 = np.sum([x[k]*G[k, i] for k in range(len(x))])
            Σ_5 = np.sum([x[m]*τ[m, j]*G[m, j] for m in range(len(x))])
            Σ_6 = np.sum([x[k]*G[k, j] for k in range(len(x))])

            Σ_3 += x[j]*G[i, j]/Σ_4*(τ[i, j] - Σ_5/Σ_6)

        np.array(γ.append(Σ_1/Σ_2 + Σ_3))

    return np.exp(γ)
