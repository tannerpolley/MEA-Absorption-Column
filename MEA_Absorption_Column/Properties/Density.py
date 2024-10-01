from MEA_Absorption_Column.Parameters import MWs_l, MWs_v, R


def liquid_density(Tl, x, df_param):

    x_CO2, x_MEA, x_H2O = x

    MWT_l = sum([x[i] * MWs_l[i] for i in range(len(x))])

    a1, b1, c1 = [-5.35162e-7, -4.51417e-4, 1.19451]
    a2, b2, c2 = [-3.2484e-6, 0.00165, 0.793]

    V_MEA = MWs_l[1]*1000 / (a1 * Tl ** 2 + b1 * Tl + c1)  # mL/mol
    V_H2O = MWs_l[2]*1000 / (a2 * Tl ** 2 + b2 * Tl + c2)   # mL/mol

    # a, b, c, d, e = df_param['molar_volume'].values()
    a, b, c, d, e = 10.57920122, -2.020494157, 3.15067933, 192.0126008, -695.3848617

    V_CO2 = a + (b + c * x_MEA) * x_MEA * x_H2O + (d + e * x_MEA) * x_MEA * x_CO2

    V_l = V_CO2 * x_CO2 + x_MEA * V_MEA + x_H2O * V_H2O # Liquid Molar Volume (mL/mol)
    V_l = V_l*1e-6  # Liquid Molar Volume (mL/mol --> m3/mol)

    rho_mol_l = V_l**-1  # Liquid Molar Density (m3/mol --> mol/m3)
    rho_mass_l = rho_mol_l*MWT_l  # Liquid Mass Density (mol/m3 --> kg/m3)

    return rho_mol_l, rho_mass_l


def vapor_density(Tv, P, y):

    rho_mol_v = P/(R*Tv) # Vapor Molar Density (mol/m3)

    MWT_v = 0
    for i in range(len(y)):
        MWT_v += y[i] * MWs_v[i]  # kg/mol

    rho_mass_v = rho_mol_v * MWT_v  # Vapor Mass Density (mol/m3 --> kg/m3)

    return rho_mol_v, rho_mass_v
