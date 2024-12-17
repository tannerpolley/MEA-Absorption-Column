import numpy as np

from MEA_Absorption_Column.BVP.enhancement_model_analysis import rho_mol_l
from MEA_Absorption_Column.Parameters import MWs_l, R
from MEA_Absorption_Column.Properties.Density import liquid_density


def liquid_enthalpy(Tl, x):

    _, _, Volume = liquid_density(float(Tl), x)
    Vl = Volume[0]

    coefficients = {'CO2': np.array([276370, -2090.1, 8.125, -.014116, 9.3701e-6]),
                    'MEA': np.array([2.6161, 3.706e-3, 3.787e-6, 0, 0]),
                    'H2O': np.array([4.2107, -1.696e-3, 2.568e-5, -1.095e-7, 3.038e-10]),
                    }
    Tr = 298.15 - 273.15
    Pref = 101325.0
    P = 109180.0


    dh_vap_MEA = 58000
    dh_vap_H2O = 43.99e3
    Tl = float(Tl) - 273.15
    Hl_CO2 = -83999.8249763614

    Hl_MEA = MWs_l[1]*1000 * sum([coefficients['MEA'][i]/(i+1) * (Tl**(i+1) - Tr**(i+1)) for i in range(len(coefficients['MEA']))]) - dh_vap_MEA + (P - Pref)/(1/Vl)
    Hl_H2O = MWs_l[2]*1000 * np.sum([coefficients['H2O'][i]/(i+1) * (Tl**(i+1) - Tr**(i+1)) for i in range(len(coefficients['H2O']))]) - dh_vap_H2O + (P - Pref)/(1/Vl)
    return np.array([Hl_CO2, Hl_MEA, Hl_H2O])


def vapor_enthalpy(Tv):

    coefficients = {'CO2': np.array([5.457, 1.045e-3, -1.157e5]),
                    'H2O': np.array([3.47, 1.45e-3, 0.121e5]),
                    'N2': np.array([3.28, 0.593e-3, 0.04e5]),
                    'O2': np.array([3.639, 0.506e-3, -0.227e5]),
                    }
    Tr = 298.15

    def vapor_enthalpy(species):
        A, B, C = coefficients[species]
        return (A*(Tv - Tr) + .5*B*(Tv ** 2 - Tr ** 2) - C*(Tv ** -1 - Tr ** -1))*R
    Hv_i = [vapor_enthalpy(species) for species in coefficients.keys()]

    return Hv_i
