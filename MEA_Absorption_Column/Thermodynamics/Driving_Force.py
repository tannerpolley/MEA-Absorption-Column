import numpy as np
from MEA_Absorption_Column.Thermodynamics.Fugacity_Coeff import fugacity_coeff


def driving_force(x, y, x_true, Cl_true, Tl, Tv, alpha, H_CO2_mix, P):

    y_CO2 = y[0]
    y_H2O = y[1]
    x_CO2_true = x_true[0]
    x_H2O_true = x_true[2]
    Cl_CO2_true = Cl_true[0]

    # IDAES Parameters for Psat H2O
    Psat_H2O = np.exp(72.55 + -7206.70 / Tl + -7.1385 * np.log(Tl) + 4.05e-6 * Tl ** 2)

    method = 'ideal'
    # method = 'ePC-SAFT'
    # method = 'surrogate'

    if method == 'ideal':

        # From Xu and Rochelle
        Pl_CO2 = Cl_CO2_true * H_CO2_mix
        Pv_CO2 = y_CO2 * P

        Pv_H2O = y_H2O * P
        Pl_H2O = x_H2O_true * Psat_H2O

    elif method == 'ePC-SAFT':

        # --------------- PC-SAFT Method ----------------------- #

        φl_CO2, φl_H2O = fugacity_coeff(x_true, 'liq', Tl, P)
        φv_CO2, φv_H2O = fugacity_coeff(y, 'vap', Tv, P)

        Pl_CO2 = P * φl_CO2 * x_CO2_true
        Pl_H2O = P * φl_H2O * x_H2O_true

        Pv_CO2 = P * φv_CO2 * y_CO2
        Pv_H2O = P * φv_H2O * y_H2O

    elif method == 'surrogate':

        # -------- Gabrielsen Approximation Method --------------

        # Combined Henry's Law and chemical equilibrium constant for MEA-CO2 Eq. 14 and Table 1
        # From Gabrielsen: A Model for Estimating CO2 Solubility in Aqueous Alkanolamines Eq. 11

        # K_CO2 = np.exp(30.96 + -10584 / Tl + -7.187 * a0 * alpha)
        # Pl_CO2 = K_CO2 * x[0] * a0 * alpha / (a0 * (1 - 2 * alpha)) ** 2

        Pv_CO2 = y_CO2 * P

        # From Xu and Rochelle
        Pl_CO2 = np.exp(39.3 - 12155 / Tl - 19.0 * alpha ** 2 + 1105 * alpha / Tl + 12800 * alpha ** 2 / Tl)

        Pv_H2O = y_H2O * P
        Pl_H2O = x_H2O_true * Psat_H2O

    else:
        raise ValueError('Choose ideal, ePC-SAFT, or surrogate')

    DF_CO2 = (Pv_CO2 - Pl_CO2)
    DF_H2O = (Pv_H2O - Pl_H2O)

    return DF_CO2, DF_H2O, [DF_CO2, Pv_CO2, Pl_CO2, H_CO2_mix], [DF_H2O, Pv_H2O, Pl_H2O, Psat_H2O]
