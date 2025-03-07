import numpy as np
from numpy import log, exp
from MEA_Absorption_Column.Parameters import MWs_l, MWs_v, R


def henrys_law(T, z):

    Tl = T
    x = z

    x_CO2, x_MEA, x_H2O = x

    m_MEA = x_MEA * MWs_l[1]
    m_H2O = x_H2O * MWs_l[2]
    wt_MEA = m_MEA / (m_MEA + m_H2O)
    wt_H2O = m_H2O / (m_MEA + m_H2O)

    H_N2O_MEA = 2.448e5 * exp(-1348 / Tl)
    H_CO2_H2O = 3.52e6 * exp(-2113 / Tl)
    H_N2O_H2O = 8.449e6 * exp(-2283 / Tl)
    H_CO2_MEA = H_N2O_MEA * (H_CO2_H2O / H_N2O_H2O)

    a1, a2, a3, b = -2.076073001, 0.037322205, -0.00032721, -0.111102655
    lwm = (a1 + a2 * (Tl-273.15) + a3 * (Tl-273.15) ** 2 + b * wt_H2O)

    Ra = lwm * wt_MEA * wt_H2O

    sigma = wt_MEA * log(H_CO2_MEA) + wt_H2O * log(H_CO2_H2O)

    log_H_CO2_mix = Ra + sigma

    H_CO2_mix = exp(log_H_CO2_mix)

    return H_CO2_mix


def density(T, z, P, phase='liquid'):

    if phase == 'liquid':
        Tl = T
        x = z
        x_CO2, x_MEA, x_H2O = x

        MWT_l = sum([x[i] * MWs_l[i] for i in range(len(x))])

        a1, b1, c1 = -10.5792012186177, 192.012600751473, -695.384861676286
        a2, b2, c2, d2, e2 = [-5.35162e-7, -4.51417e-4, 1.19451, -2.02049415703576, 3.1506793296904,]
        a3, b3, c3 = [-3.2484e-6, 0.00165, 0.793]

        V_CO2 = a1 + (b1 + c1 * x_MEA) * x_MEA
        V_MEA = MWs_l[1]*1000 / (a2 * Tl ** 2 + b2 * Tl + c2) + x_H2O*(d2 + e2*x_MEA) # mL/mol
        V_H2O = MWs_l[2]*1000 / (a3 * Tl ** 2 + b3 * Tl + c3)# mL/mol


        V_l = V_CO2 * x_CO2 + x_MEA * V_MEA + x_H2O * V_H2O # Liquid Molar Volume (mL/mol)
        V_l = V_l*1e-6  # Liquid Molar Volume (mL/mol --> m3/mol)

        rho_mol_l = V_l**-1  # Liquid Molar Density (m3/mol --> mol/m3)
        rho_mass_l = rho_mol_l*MWT_l  # Liquid Mass Density (mol/m3 --> kg/m3)

        volume = [V_l, V_CO2*1e-6, V_MEA*1e-6, V_H2O*1e-6]

        return rho_mol_l, rho_mass_l, volume

    elif phase == 'vapor':
        Tv = T
        y = z
        rho_mol_v = P/(R*Tv) # Vapor Molar Density (mol/m3)

        MWT_v = 0
        for i in range(len(y)):
            MWT_v += y[i] * MWs_v[i]  # kg/mol

        rho_mass_v = rho_mol_v * MWT_v  # Vapor Mass Density (mol/m3 --> kg/m3)

        return rho_mol_v, rho_mass_v


def surface_tension(T, z, w_MEA, w_H2O):
    Tl = T
    x = z

    x_CO2, x_MEA, x_H2O = x

    alpha = x_CO2/x_MEA

    r = w_MEA / (w_MEA + w_H2O)

    S1, S2, S3, S4, S5, S6 = -0.00589934906112609, 0.00175020536428591, 0.129650182728177, 0.0000126444768126308, -5.73954817199691E-06, -0.00018969005534195,

    c1_MEA, c2_MEA, c3_MEA, c4_MEA, Tc_MEA = 0.09945, 1.067, 0, 0, 614.45
    c1_H2O, c2_H2O, c3_H2O, c4_H2O, Tc_H2O = 0.18548, 2.717, -3.554, 2.047, 647.13

    a, b, c, d, e, f, g, h, i, j = (1070.65668317975,-2578.78134208703,3399.24113311222,-2352.47410135319,2960.24753687833,
                                    3.06684894924048,-1.79435372759593,-7.2124219075848,2.97502322396621,-10.5738529301824)

    sigma_CO2 = S1 * r ** 2 + S2 * r + S3 + Tl * (S4 * r ** 2 + S5 * r + S6)

    sigma_MEA = c1_MEA * (1 - Tl / Tc_MEA) ** (c2_MEA + c3_MEA * (Tl / Tc_MEA) + c4_MEA * (Tl / Tc_MEA) ** 2)
    sigma_H2O = c1_H2O * (1 - Tl / Tc_H2O) ** (c2_H2O + c3_H2O * (Tl / Tc_H2O) + c4_H2O * (Tl / Tc_H2O) ** 2)

    fxn_f = a + b * alpha + c * alpha ** 2 + d * r + e * r ** 2
    fxn_g = f + g * alpha + h * alpha ** 2 + i * r + j * r ** 2

    sigma_l = sigma_H2O + (sigma_CO2 - sigma_H2O) * fxn_f * x_CO2 + (sigma_MEA - sigma_H2O) * fxn_g * x_MEA

    return sigma_l


def heat_capacity(T, z, w, phase='liquid'):

    if phase == 'liquid':
        Tl = T
        x = z

        coefficients = {'CO2': np.array([276370, -2090.1, 8.125, -.014116, 9.3701e-6]),
                        'MEA': np.array([2.6161, 3.706e-3, 3.787e-6, 0, 0]),
                        'H2O': np.array([4.2107, -1.696e-3, 2.568e-5, -1.095e-7, 3.038e-10]),
                        }
        Cpl = []
        for j, sp in enumerate(coefficients.keys()):
            A, B, C, D, E = coefficients[sp]
            T_C = Tl - 273.15
            Cpl.append(MWs_l[j] * (A + B * T_C + C * T_C ** 2 + D * T_C ** 3 + E * T_C ** 4) * 1000)

        Cpl_CO2 = (x[2] * Cpl[2] + x[1] * Cpl[1])/ x[0] * (1 / (1 - w[0]) - 1)

        Cpl[0] = Cpl_CO2

        Cpl_T = sum([Cpl[i] * x[i] for i in range(len(x))])


        return Cpl, Cpl_T

    elif phase == 'vapor':
        Tv = T
        y = z

        coefficients = {'CO2': np.array([5.457, 1.045e-3, -1.157e5]),
                        'H2O': np.array([3.47, 1.45e-3, 0.121e5]),
                        'N2': np.array([3.28, 0.593e-3, 0.04e5]),
                        'O2': np.array([3.639, 0.506e-3, -0.227e5]),
                        }
        Cpv = []
        for sp in coefficients.keys():
            C1, C2, C3 = coefficients[sp]
            Cpv.append((C1 + C2 * T + C3 * T ** -2) * R)

        Cpv_T = sum([Cpv[i] * y[i] for i in range(len(y))])

        return Cpv, Cpv_T


def enthalpy(T, z, phase='liquid'):

    if phase == 'liquid':
        Tl = T
        x = z
        rho_mol_l, _, _ = density(float(Tl), x, 0, phase=phase)

        coefficients = {'CO2': np.array([276370, -2090.1, 8.125, -.014116, 9.3701e-6]),
                        'MEA': np.array([2.6161, 3.706e-3, 3.787e-6, 0, 0]),
                        'H2O': np.array([4.2107, -1.696e-3, 2.568e-5, -1.095e-7, 3.038e-10])
                        }
        Tr = 298.15 - 273.15
        Pref = 101325.0
        P = 109180.0

        dh_vap_MEA = 58000
        dh_vap_H2O = 43.99e3
        Tl = float(Tl) - 273.15
        Hl_CO2 = -83999.8249763614

        Hl_MEA = MWs_l[1] * 1000 * sum([coefficients['MEA'][i] / (i + 1) * (Tl ** (i + 1) - Tr ** (i + 1)) for i in
                                        range(len(coefficients['MEA']))]) - dh_vap_MEA + (P - Pref) / rho_mol_l
        Hl_H2O = MWs_l[2] * 1000 * sum([coefficients['H2O'][i] / (i + 1) * (Tl ** (i + 1) - Tr ** (i + 1)) for i in
                                        range(len(coefficients['H2O']))]) - dh_vap_H2O + (P - Pref) / rho_mol_l

        Hl = np.array([Hl_CO2, Hl_MEA, Hl_H2O])
        Hl_T = sum([x[i] * Hl[i] for i in range(len(x))])
        return Hl, Hl_T

    if phase == 'vapor':
        Tv = T
        y = z
        coefficients = {'CO2': np.array([5.457, 1.045e-3, -1.157e5]),
                        'H2O': np.array([3.47, 1.45e-3, 0.121e5]),
                        'N2': np.array([3.28, 0.593e-3, 0.04e5]),
                        'O2': np.array([3.639, 0.506e-3, -0.227e5]),
                        }
        Tr = 298.15

        def vapor_enthalpy(species):
            A, B, C = coefficients[species]
            return (A * (Tv - Tr) + .5 * B * (Tv ** 2 - Tr ** 2) - C * (Tv ** -1 - Tr ** -1)) * R

        Hv = [vapor_enthalpy(species) for species in coefficients.keys()]
        Hv_T = sum([y[i] * Hv[i] for i in range(len(y))])
        return Hv, Hv_T


def thermal_conductivity(T, z, muv):
    coefficients = {'CO2': np.array([3.69, -0.3838, 964., 1.86e6]),
                    'H2O': np.array([6.204e-6, 1.3973, 0, 0]),
                    'N2': np.array([.000331, .7722, 16.323, 373.72]),
                    'O2': np.array([.00045, .7456, 56.699, 0])
                    }

    kt_i = []
    for species in ['CO2', 'H2O', 'N2', 'O2']:
        A, B, C, D = coefficients[species]

        kt_i.append((A * T ** B) / (1 + C / T + D / (T ** 2)))

    k_vap = 0
    for i in range(len(z)):
        sum_ij = 0
        for j in range(len(z)):
            Aij = (1 + (muv[i] / muv[j]) ** .5 * (MWs_v[j] / MWs_v[i]) ** .25) ** 2 * (
                    8 * (1 + MWs_v[i] / MWs_v[j])) ** -.5
            sum_ij += Aij * z[j]
        k_vap += z[i] * kt_i[i] / sum_ij
    return k_vap
