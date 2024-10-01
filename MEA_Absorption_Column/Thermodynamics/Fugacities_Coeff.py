from MEA_Absorption_Column.Thermodynamics.PC_SAFT import PCSAFT
import numpy as np


def fugacity_coeff(z, phase, T, P):

    if phase == 'liquid':
        # 2B Association Scheme

        kij_CO2_MEA = .16
        kij_CO2_H2O = .15
        kij_MEA_H2O = -.18

        prop_dic = {
            'm': np.array([2.079, 3.0353, 1.9599, 1, 1, 1]),
            's': np.array([2.7852, 3.0435, 2.363, 3, 3, 3]),
            'e': np.array([169.21, 277.174, 279.42, 300, 300, 300]),
            'vol_a': np.array([0, .037470, .1750, 0, 0, 0]),
            'e_assoc': np.array([0, 2586.3, 2059.28, 0, 0, 0]),
            'k_ij': np.array([[0.0, kij_CO2_MEA, kij_CO2_H2O, 0., 0., 0.],
                              [kij_CO2_MEA, 0.0, kij_MEA_H2O, 0., 0., 0.],
                              [kij_CO2_H2O, kij_MEA_H2O, 0.0, 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.]]),
            'dielc': 75,
            # # 'dipm': np.array([0, 2, 9, 2, 2, 2]),
            'z': np.array([0, 0, 0, +1, -1, -1])
        }

        mix_l = Ppcsaft_fugcoef(T, z, m, σ, ϵ_k, k_ij, phase='liquid', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
        return mix_l.φ()

    elif phase == 'vapor':

        m = np.array([2.0729, 1.9599, 1.2053, 1.1335])  # Number of segments
        σ = np.array([2.7852, 2.362, 3.3130, 3.1947])  # Temperature-Independent segment diameter σ_i (Aᵒ)
        ϵ_k = np.array([169.21, 279.42, 90.96, 114.43]) # Depth of pair potential / Boltzmann constant (K)
        k_ij = np.array([[0.0,   .065, -.0149, -.04838],
                         [.065,   0.0,    0.0, 0.0],
                         [-.0149, 0.0,    0.0, 0.0],
                         [-.04838, 0.0, 0.0, -.00978]])

        κ_AB = np.array([0, 0, 0, 0])
        ϵ_AB_k = np.array([0, 0, 0, 0])

        mix_v = PCSAFT(T, z, m, σ, ϵ_k, k_ij, phase='vapor', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
        return mix_v.φ()
