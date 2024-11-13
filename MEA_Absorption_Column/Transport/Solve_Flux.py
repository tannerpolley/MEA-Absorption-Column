import numpy as np


def solve_flux(kv_CO2, kv_H2O,  DF_CO2, DF_H2O, KH):

    # Compute Fluxes
    N_CO2 = kv_CO2 * DF_CO2 * KH
    N_H2O = kv_H2O * DF_H2O

    return N_CO2, N_H2O
