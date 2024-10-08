from numpy import exp, log, array
import numpy as np
from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
from gekko import GEKKO

# From Akula Appendix of Model Development, Validation, and Part-Load Optimization of a
# MEA-Based Post-Combustion CO2 Capture Process Under SteadyState Flexible Capture Operation


def solve_ChemEQ(Cl_0, Tl, guesses=array([.005, 1800, 39500, 1300, 1300, 20])):

    a1, b1, c1 = 233.4, -3410, -36.8
    a2, b2, c2 = 176.72, -2909, -28.46

    K1 = exp(a1 + b1/Tl + c1*log(Tl))/1000 # kmol -> mol
    K2 = exp(a2 + b2/Tl + c2*log(Tl))/1000 # kmol -> mol

    def f(Cl):

        # Kee1 = log(Cl[3]) + log(Cl[4]) - log(Cl[0]) - 2*log(Cl[1])
        # Kee2 = log(Cl[3]) + log(Cl[5]) - log(Cl[0]) - log(Cl[1]) - log(Cl[2])

        Kee1 = Cl[3]*Cl[4]/(Cl[0]*Cl[1]**2)
        Kee2 = Cl[3]*Cl[5]/(Cl[0]*Cl[1]*Cl[2])

        eq1 = Kee1 - K1
        eq2 = Kee2 - K2
        eq3 = Cl_0[0] - (Cl[0] + Cl[3])
        eq4 = Cl_0[1] - (Cl[1] + Cl[3] + Cl[4])
        eq5 = Cl_0[2] - (Cl[2] + Cl[3] - Cl[4])
        eq6 = Cl[3] - (Cl[4] + Cl[5])

        eqs = array([eq1, eq2, eq3, eq4, eq5, eq6])
        # print(eqs)
        return eqs

    return array(root(f, guesses).x).astype('float')


def solve_ChemEQ_2(alpha, w_MEA, Tl):

    def solve_ChemEQ(guesses, Cl_0, Tl):

        Cl_CO2_0 = Cl_0[0]
        Cl_MEA_0 = Cl_0[1]
        Cl_H2O_0 = Cl_0[2]
        Cl_CO2 = guesses[0]
        Cl_MEA = guesses[1]
        Cl_H2O = guesses[2]
        Cl_MEAH = guesses[3]
        Cl_MEACOO = guesses[4]
        Cl_HCO3 = guesses[5]

        a1, b1, c1, d1 = 234.2, -1434.4, -36.8, -.0074
        a2, b2, c2, d2 = 176.8, -991.2, -29.5, .0129

        # a1, b1, c1, d1 = 233.4, -3410, -36.8, 0
        # a2, b2, c2, d2 = 176.72, -2909, -28.46, 0

        # a1, b1, c1, d1 = 233.4, -899.9, -37.5, 0
        # a2, b2, c2, d2 = 176.72, -1947.9, -28.2, 0

        K1 = np.exp(a1 + b1 / Tl + c1 * np.log(Tl) + d1 * Tl) / 1000  # kmol -> mol
        K2 = np.exp(a2 + b2 / Tl + c2 * np.log(Tl) + d2 * Tl) / 1000  # kmol -> mol

        Kee1 = (Cl_MEAH * Cl_MEACOO) / (Cl_CO2 * Cl_MEA ** 2)  # carbamate
        Kee2 = (Cl_MEAH * Cl_HCO3) / (Cl_CO2 * Cl_MEA * Cl_H2O)  # bicarbonate
        #
        if Cl_0[0] > 3800:
            Cl_CO2_scale = 20
        else:
            Cl_CO2_scale = 5

        eq1 = Kee1 / 100 - K1 / 100
        eq2 = Kee2 / 100 - K2 / 100
        eq3 = Cl_CO2_0 / Cl_CO2_scale - (Cl_CO2 + Cl_MEAH) / Cl_CO2_scale
        eq4 = Cl_MEA_0 / 3000 - (Cl_MEA + Cl_MEAH + Cl_MEACOO) / 3000
        eq5 = Cl_H2O_0 / 10000 - (Cl_H2O + Cl_MEAH - Cl_MEACOO) / 10000
        eq6 = Cl_MEAH - (Cl_MEACOO + Cl_HCO3)

        return eq1, eq2, eq3, eq4, eq5, eq6

    def get_x(CO2_loading, w_MEA):
        MW_MEA = 61.084
        MW_H2O = 18.02

        x_MEA_unloaded = w_MEA / (MW_MEA / MW_H2O + w_MEA * (1 - MW_MEA / MW_H2O))
        x_H2O_unloaded = 1 - x_MEA_unloaded

        n_MEA = 100 * x_MEA_unloaded
        n_H2O = 100 * x_H2O_unloaded

        n_CO2 = n_MEA * CO2_loading
        n_tot = n_MEA + n_H2O + n_CO2
        x_CO2, x_MEA, x_H2O = n_CO2 / n_tot, n_MEA / n_tot, n_H2O / n_tot

        return x_CO2, x_MEA, x_H2O

    def liquid_density(Tl, x):
        x_CO2, x_MEA, x_H2O = x

        MWs_l = np.array([44.01, 61.08, 18.02]) / 1000  # kg/mol

        MWT_l = sum([x[i] * MWs_l[i] for i in range(len(x))])

        a1, b1, c1 = [-5.35162e-7, -4.51417e-4, 1.19451]
        a2, b2, c2 = [-3.2484e-6, 0.00165, 0.793]

        V_MEA = MWs_l[1] * 1000 / (a1 * Tl ** 2 + b1 * Tl + c1)  # mL/mol
        V_H2O = MWs_l[2] * 1000 / (a2 * Tl ** 2 + b2 * Tl + c2)  # mL/mol

        # a, b, c, d, e = df_param['molar_volume'].values()
        a, b, c, d, e = 10.57920122, -2.020494157, 3.15067933, 192.0126008, -695.3848617

        V_CO2 = a + (b + c * x_MEA) * x_MEA * x_H2O + (d + e * x_MEA) * x_MEA * x_CO2

        V_l = V_CO2 * x_CO2 + x_MEA * V_MEA + x_H2O * V_H2O  # Liquid Molar Volume (mL/mol)
        V_l = V_l * 1e-6  # Liquid Molar Volume (mL/mol --> m3/mol)

        rho_mol_l = V_l ** -1  # Liquid Molar Density (m3/mol --> mol/m3)
        rho_mass_l = rho_mol_l * MWT_l  # Liquid Mass Density (mol/m3 --> kg/m3)

        return rho_mol_l, rho_mass_l

    alpha_range = np.linspace(0.002, .5, 21)
    data = {
        'CO2_loading': [],
        'temperature': []
    }
    comp = ['CO2', 'MEA', 'H2O', 'MEAH^+', 'MEACOO^-', 'HCO3^-']
    for c in comp:
        data[c] = []
    guesses = np.array([1.32885319e-10, 4.85942087e3, 3.92196901e4, 4.95855464e1, 4.95482234e1, 3.73230850e-2])
    for i, a in enumerate(alpha_range):
        x = get_x(a, w_MEA)
        rho_mol_l, _ = liquid_density(Tl, x)
        Cl_0 = [x[i] * rho_mol_l for i in range(len(x))]
        result = root(solve_ChemEQ, guesses, args=(Cl_0, Tl))
        Cl_true, solution = result.x, result.message
        guesses = Cl_true
        data['CO2_loading'].append(a)
        data['temperature'].append(Tl)
        for j, c in enumerate(comp):
            data[c].append(Cl_true[j])
    interp_dic = {}
    for c in comp:
        interp_dic[c] = interp1d(alpha_range, data[c])
    Cl_true_return = []
    for c in comp:
        Cl_true_return.append(float(interp_dic[c](alpha)))

    return np.array(Cl_true_return)

