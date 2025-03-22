from MEA_Absorption_Column.misc.Polynomial_Fit import polynomial_fit
import numpy as np


def scaling(z, Y_A_unscaled):

    scales = []
    z_2 = np.linspace(z[0], z[-1], 200)

    for i, y in enumerate(Y_A_unscaled):

        scales.append(np.max(polynomial_fit(z_2, y, i)))
    return np.abs(np.rint(np.array(scales)))
