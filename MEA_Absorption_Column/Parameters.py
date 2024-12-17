import numpy as np

# Molecular Weights
MWs_l = np.array([.04401, .06108, .01802]) # kg/mol
MWs_v = np.array([.04401, .01802, .02801, .032])  # kg/mol

column_params = {
    'NCCC': {
        'D': .64,
        'H': 6.0,
    },
    'SRP': {
        'D': .467,
        'H': 12.0,
    }
}

packing_params = {
    'MellapakPlus252Y2': {
        'a_e': 250.0, # Akula 2022 Table 14
        'eps': .970, # Akula 2022 Table 14
        'Cl': .203, # Regressed from Chinen 2018 Supporting Information Table S3
        'Cv': .35, # Regressed from Chinen 2018 Supporting Information Table S3
        'Cs': .017, # Akula 2022 Table 14
        'Cb': .241, # Akula 2022 Table 14
        'Ch': .119, # Akula 2022 Table 14
        'CS': .017, # Channel Side (m)
        'Cp_0': 0.292 # From Billet (1999) Table 2b.

    },
    'MellapakPlus252Y': {
        'a_e': 250.0,  # Akula 2022 Table 14
        'eps': .970,  # Akula 2022 Table 14
        'Cl': .5,  # Regressed from Chinen 2018 Supporting Information Table S3
        'Cv': .357,  # Regressed from Chinen 2018 Supporting Information Table S3
        'Cs': .017,  # Akula 2022 Table 14
        'Cb': .241,  # Akula 2022 Table 14
        'Ch': .119,  # Akula 2022 Table 14
        'CS': .017,  # Channel Side (m)
        'Cp_0': 0.292  # From Billet (1999) Table 2b.

    },
    'IMTP-40': {
        'a_e': 145.0,
        'eps': .98,
        'Cl': .203,  # Regressed from Chinen 2018 Supporting Information Table S3
        'Cv': .35,  # Regressed from Chinen 2018 Supporting Information Table S3
        'Cs': 3.157,
        'Cfl': 2.558,
        'Ch': .554,
        'Cp0': .292,
        'CS': .017  # Channel Side (m)
        }
}

# Packing Coefficients MellapakPlus 252Y
# a_p = 250  # Packing Surface Area/Volume (m^2/m^3)
# a_p = packing_params['MellapakPlus252Y']['a_e']
# a_p = 143.9
# 'Cfl': 2.558,
# 'Cp0': .292
# Clp = .5  # Packing Coefficient
# Chp = .554  # Packing Coefficient
# Cvp = .357  # Packing Coefficient
# ϵ = .970  # Packing Void Fraction
# ϵ = packing_params['MellapakPlus252Y']['eps']
# S = .017 # Channel Side (m)

# Other Constants
g = 9.80665  # Gravitational Constant
R = 8.314462618  # J/mol-K

# Integration Parameters
n = 51  # Number of points to evaluate for the integral
