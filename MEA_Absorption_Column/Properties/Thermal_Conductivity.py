import numpy as np


def thermal_conductivity(T, species, phase):

    if phase == 'vapor':
        coefficients = {'CO2': np.array([3.69, -.3838, 964, 1860000]),
                        'MEA': np.array([-11278, .4413, -2870000000, 0]),
                        'H2O': np.array([6.2041e-6, 1.3973, 0, 0]),
                        'N2': np.array([.00033143, .7722, 16.323, 373.72]),
                        'O2': np.array([.00044994, .7456, 56.699, 0])
                        }

        A, B, C, D  = coefficients[species]

        k = (A * T ** B)/(1 + C / T + D / (T ** 2))

        return k

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
