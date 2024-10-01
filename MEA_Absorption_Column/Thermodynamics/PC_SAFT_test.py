from PC_SAFT import flash
from MEA_Absorption_Column.Thermodynamics.Solve_ChemEQ import solve_ChemEQ
import matplotlib.pyplot as plt
import numpy as np
from MEA_Absorption_Column.Properties.Density import liquid_density
import pandas as pd


def get_x(α, w_MEA):

    MW_CO2 = 44.01 / 1000
    MW_MEA = 61.08 / 1000
    MW_H2O = 18.02 / 1000

    x_MEA = ((1 + α + (MW_MEA/MW_H2O))*(1-w_MEA)/w_MEA)**-1
    x_CO2 = x_MEA*α
    x_H2O = 1 - x_CO2 - x_MEA

    return [x_CO2, x_MEA, x_H2O]


Tl = 313 # K
α_range = np.linspace(.2, .5, 21)
P_CO2_range = np.zeros(len(α_range))

for i, α in enumerate(α_range):

    x = get_x(α, .3)
    rho_mol_l, _ = liquid_density(Tl, x, [])
    Cl = [x[i] * rho_mol_l for i in range(len(x))]
    Cl_true = solve_ChemEQ(Cl, Tl)
    x_true = Cl_true / (sum(Cl_true)).astype('float')
    x = x_true[:3]


    m = np.array([2.0729, 3.0353, 1.9599])  # Number of segments
    σ = np.array([2.7852, 3.0435, 2.362])  # Temperature-Independent segment diameter σ_i (Aᵒ)
    ϵ_k = np.array([169.21, 277.174, 279.42]) # Depth of pair potential / Boltzmann constant (K)
    k_ij = np.array([[0.0, .16, .065],
                     [.16, 0.0, -.18],
                     [.065, -.18, 0.0]])
    κ_AB = np.array([0, .037470, .2039])
    ϵ_AB_k = np.array([0, 2586.3, 2059.28])

    yg = [.01, .00001, .98]
    Pg = 1000000

    y, P = flash(x, yg, Tl, Pg, m, σ, ϵ_k, k_ij, flash_type='Bubble_T', κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)

    print(y[0]*P)

    P_CO2_range[i] = y[0]*P

P_CO2_data, α_data = pd.read_csv('Jou_1995.csv').to_numpy().T

plt.plot(α_data, P_CO2_data, 'x')
plt.plot(α_range, P_CO2_range/1000)
plt.yscale('log')
plt.show()

