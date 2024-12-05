import numpy as np
from MEA_Absorption_Column.Parameters import MWs_v

def thermal_conductivity(z, T, phase):

    if phase == 'vapor':

        A, B, C = 2.148e-6, 0.46, 290
        dyn_vis_CO2 = A * T ** B / (1 + C / T)
        A, B, C = 1.7096e-8, 1.1146, 0.0
        dyn_vis_H2O = A * T ** B / (1 + C / T)
        A, B, C = 0.01781e-3, 300.55, 111
        dyn_vis_N2 = A * (B + C)/(T + C) * (T/B)**1.5
        A, B, C = 0.02018e-3, 292.25, 127
        dyn_vis_O2 = A * (B + C)/(T + C) * (T/B)**1.5

        dyn_vis = [dyn_vis_CO2, dyn_vis_H2O, dyn_vis_N2, dyn_vis_O2]


        coefficients = {'CO2': np.array([3.69, -.3838, 964, 1860000]),
                        'H2O': np.array([6.2041e-6, 1.3973, 0, 0]),
                        'N2': np.array([.00033143, .7722, 16.323, 373.72]),
                        'O2': np.array([.00044994, .7456, 56.699, 0])
                        }

        kt_i = []
        for species in ['CO2', 'H2O', 'N2', 'O2']:
            A, B, C, D  = coefficients[species]

            kt_i.append((A * T ** B)/(1 + C / T + D / (T ** 2)))

        k_vap = 0
        for i in range(len(z)):
            sum_ij = 0
            for j in range(len(z)):
                Aij = (1 + (dyn_vis[i]/dyn_vis[j])**.5 * (MWs_v[j]/MWs_v[i])** .25)**2 * (8 * (1 + MWs_v[i]/MWs_v[j]))**-.5
                sum_ij += Aij * z[j]
            k_vap += z[i] * kt_i[i] / sum_ij
        return k_vap

    elif phase == 'liquid':

        coefficients = {'CO2': np.array([.4406, -.0012175, 0, 0]),
                        'MEA': np.array([-.0149, .0014816, -2.14e-6, 0]),
                        'H2O': np.array([-.432, .0057255, -8.078e-6, 1.861e-9]),
                        'N2': np.array([.2654, -.001677, 0, 0]),
                        'O2': np.array([.2741, -.00138, 0, 0])
                        }

        A, B, C, D = coefficients[species]

        k = A + B * T + C * T ** 2 + D * T ** 3

        return k
