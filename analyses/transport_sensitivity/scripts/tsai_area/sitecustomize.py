"""Apply the requested coefficient pair in this analysis process and its workers."""
import os
if os.environ.get('ISSUE84_PAIR'):
    import numpy as np
    import casadi as ca
    from mea_absorption_column.Transport import Hydraulic_Variables_Correlations as hydraulic
    from mea_absorption_column.config.Constants import g
    coefficient, exponent = map(float, os.environ['ISSUE84_PAIR'].split(','))
    def area(rho, sigma, velocity, cross_section, packing):
        ap, eps = packing[:2]
        factor = rho / sigma * g ** (1 / 3) * (velocity * eps / ap) ** (4 / 3)
        value = coefficient * ap * factor ** exponent
        return value, value * cross_section
    hydraulic.interfacial_area_expression = area
