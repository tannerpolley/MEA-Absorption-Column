import numpy as np
from MEA_Absorption_Column.Parameters import MWs_l, R


def heat_capacity(T, phase, x, w):
    if phase == 'vapor':

        coefficients = {'CO2': np.array([5.457, 1.045e-3, -1.157e5]),
                        'H2O': np.array([3.47, 1.45e-3, 0.121e5]),
                        'N2': np.array([3.28, 0.593e-3, 0.04e5]),
                        'O2': np.array([3.639, 0.506e-3, -0.227e5]),
                        }
        Cp = []
        for sp in coefficients.keys():
            C1, C2, C3 = coefficients[sp]
            Cp.append((C1 + C2 * T + C3 * T ** -2)*R)

        return Cp

    elif phase == 'liquid':

        coefficients = {'CO2': np.array([276370, -2090.1, 8.125, -.014116, 9.3701e-6]),
                        'MEA': np.array([2.6161, 3.706e-3, 3.787e-6, 0, 0]),
                        'H2O': np.array([4.2107, -1.696e-3, 2.568e-5, -1.095e-7, 3.038e-10]),
                        }
        Cp = []
        for i, sp in enumerate(coefficients.keys()):
            A, B, C, D, E = coefficients[sp]
            T_C = T - 273.15
            Cp.append(MWs_l[i]*(A + B * T_C + C * T_C ** 2 + D * T_C ** 3 + E * T_C ** 4)*1000)

        Cpl_CO2 = (x[2]*Cp[2] + x[1]*Cp[1])/x[0]*(1/(1 - w[0]) - 1)

        Cp[0] = Cpl_CO2

        return Cp

