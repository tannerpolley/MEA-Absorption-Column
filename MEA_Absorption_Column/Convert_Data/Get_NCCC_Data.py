import pandas as pd
import numpy as np


def get_NCCC_data(index=0):

    data = pd.read_csv(r'data\NCCC_Data.csv', index_col=0)
    data = data.dropna(how='all')
    data = data.dropna(how='all', axis=1)

    i = index
    m_T_l = np.array(data['L'])[i]      # Liquid Mass Flow Rate (kg/s)
    m_T_v = np.array(data['G'])[i]      # Vapor Mass Flow Rate (kg/s)
    alpha = np.array(data['alpha'])[i]  # CO2 Loading in Lean Solvent
    w_MEA = np.array(data['w_MEA'])[i]  # MEA weight fraction in Lean Solvent
    y_CO2 = np.array(data['y_CO2'])[i]  # Vapor Mole Fraction of CO2
    Tl_0 = np.array(data['Tl'])[i]  # Inlet Liquid Temperature
    Tv_z = np.array(data['Tv'])[i]  # Inlet Vapor Temperature
    P = np.array(data['P'])[i] # Pressure of Absorber
    n_beds = np.array(data['Beds'])[i]

    X = [m_T_l, m_T_v, alpha, w_MEA, y_CO2, Tl_0, Tv_z, P, n_beds]
    return X
