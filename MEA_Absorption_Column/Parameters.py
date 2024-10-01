import numpy as np
import pandas as pd

# Molecular Weights
MWs_l = np.array([44.01, 61.08, 18.02]) / 1000  # kg/mol
MWs_v = np.array([44.01, 18.02, 28.01, 32]) / 1000  # kg/mol

# Packing Coefficients Mellapak Metal 250Y
a_p = 250  # Packing Surface Area/Volume (m^2/m^3)
a_p = 143.9
Clp = .5  # Packing Coefficient
Chp = .554  # Packing Coefficient
Cvp = .357  # Packing Coefficient
ϵ = .970  # Packing Void Fraction
S = .017 # Channel Side (m)

# Other Constants
g = 9.81  # Gravitational Constant
R = 8.314462618  # J/mol-K

# Integration Parameters
n = 51  # Number of points to evaluate for the integral
