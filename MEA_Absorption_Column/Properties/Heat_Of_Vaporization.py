import numpy as np
from MEA_Absorption_Column.Parameters import data_EOS


def heat_of_vaporization(Tl, species):

    coefficients = {'CO2': np.array([0, 0, 0, 0, 0]),
                    'MEA': np.array([82393000, .59045, -.43602, .37843, 0]),
                    'H2O': np.array([56600000, .612041, -.625697, .398804, 0]),
                    'N2': np.array([0, 0, 0, 0, 0]),
                    'O2': np.array([0, 0, 0, 0, 0])
                    }
    Tc = data_EOS[0]

    Tc_dict = {'CO2': Tc[0],
               'MEA': Tc[1],
               'H2O': Tc[2],
               'N2': Tc[3],
               'O2': Tc[4]
                }

    A, B, C, D, E = coefficients[species]

    Tr = Tl / Tc_dict[species]
    delHlv = (A*(1 - Tr) ** (B + C * Tr + D * Tr ** 2 + E * Tr ** 3))/1000

    # print(delHlv)

    return delHlv  # J/mol


