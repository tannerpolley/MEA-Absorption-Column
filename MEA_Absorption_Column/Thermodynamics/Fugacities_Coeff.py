from MEA_Absorption_Column.Thermodynamics.PC_SAFT import PCSAFT
import numpy as np
from pcsaft import pcsaft_fugcoef, pcsaft_den


def fugacity_coeff(z, phase, T, P):
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
    if P < 0:
        P = 109180
    rho = pcsaft_den(t=T, p=P, x=z, params=prop_dic, phase=phase)
    φ = pcsaft_fugcoef(t=T, rho=rho, x=z, params=prop_dic)
    return φ[0], φ[2]
