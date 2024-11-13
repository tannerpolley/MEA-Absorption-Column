import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d

# From Akula Appendix of Model Development, Validation, and Part-Load Optimization of a
# MEA-Based Post-Combustion CO2 Capture Process Under SteadyState Flexible Capture Operation

def solve_ChemEQ(Fl, Tl, liquid_density):
    def solve_ChemEQ(guesses, Cl_0, Tl):

        Cl_CO2_0 = Cl_0[0]
        Cl_MEA_0 = Cl_0[1]
        Cl_H2O_0 = Cl_0[2]
        Cl_MEAH_0 = Cl_0[3]
        Cl_MEACOO_0 = Cl_0[4]
        Cl_HCO3_0 = Cl_0[5]
        Cl_CO2 = guesses[0]
        Cl_MEA = guesses[1]
        Cl_H2O = guesses[2]
        Cl_MEAH = guesses[3]
        Cl_MEACOO = guesses[4]
        Cl_HCO3 = guesses[5]

        # a1, b1, c1, d1 = 234.2, -1434.4, -36.8, -.0074
        # a2, b2, c2, d2 = 176.8, -991.2, -29.5, .0129

        a1, b1, c1, d1 = 164.039636, -707.0056712, -26.40136817, 0
        a2, b2, c2, d2 = 366.061867998774, -13326.25411, -55.68643292, 0

        # a1, b1, c1, d1 = 233.4, -899.9, -37.5, 0
        # a2, b2, c2, d2 = 176.72, -1947.9, -28.2, 0

        K1 = np.exp(a1 + b1 / Tl + c1 * np.log(Tl) + d1 * Tl)
        K2 = np.exp(a2 + b2 / Tl + c2 * np.log(Tl) + d2 * Tl)

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

    x = [Fl[i] / sum(Fl) for i in range(len(Fl))]
    MWs_l = np.array([44.01, 61.08, 18.02]) / 1000  # kg/mol

    w = [MWs_l[i] * x[i] / sum([MWs_l[j] * x[j] for j in range(len(Fl))]) for i in range(len(Fl))]

    alpha = x[0] / x[1]
    w_MEA = w[1]

    alpha_range = np.linspace(0.002, 1.2, 21)
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
        rho_mol_l, _, _ = liquid_density(Tl, x)
        Cl_0 = [x[i] * rho_mol_l for i in range(len(x))]
        Cl_0 += [0, 0, 0]
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

    x_true = [Cl_true_return[i]/sum(Cl_true_return) for i in range(len(Cl_true_return))]
    Fl_true = [x_true[i]*sum(Fl) for i in range(len(x_true))]

    return np.array(Fl_true), np.array(x_true), np.array(Cl_true_return)




if __name__ == '__main__':
    Fl = [3.5544993956757, 9.045716779, 69.6840552543039]
    Tl = 333.646741247888
    Fl_true, x_true, Cl_true = solve_ChemEQ(Fl, Tl)
    print(Fl_true)
    print(x_true)
    print(Cl_true)
