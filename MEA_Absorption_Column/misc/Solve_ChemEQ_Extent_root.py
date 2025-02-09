import numpy as np
from scipy.optimize import root


def solve_ChemEQ_3(Fl, Tl, liquid_density):
    a1, b1, c1, d1 = 164.039636, -707.0056712, -26.40136817, 0
    a2, b2, c2, d2 = 366.061867998774, -13326.25411, -55.68643292, 0

    rxn_1_st = [-1, -2, 0, 1, 1, 0]
    rxn_2_st = [-1, -1, -1, 1, 0, 1]

    log_K1 = a1 + b1 / Tl + c1 * np.log(Tl) + d1 * Tl
    log_K2 = a2 + b2 / Tl + c2 * np.log(Tl) + d2 * Tl

    x = [Fl[i] / sum(Fl) for i in range(len(Fl))]
    rho_mol_l, _, _ = liquid_density(Tl, x)

    Fl += [0, 0, 0]

    def root_solve(guesses, Fl, scale):
        ξ = guesses

        Fl_true = [Fl[i] + rxn_1_st[i] * ξ[0] + rxn_2_st[i] * ξ[1] for i in range(len(rxn_1_st))]
        x_true = [Fl_true[i] / sum(Fl_true) for i in range(len(Fl_true))]
        Cl_true = [x_true[i] * rho_mol_l for i in range(len(x_true))]

        logCl_true = [np.log(Cl_true[i]) for i in range(len(Cl_true))]
        RQ1 = sum([logCl_true[i] * rxn_1_st[i] for i in range(len(x_true))])
        RQ2 = sum([logCl_true[i] * rxn_2_st[i] for i in range(len(x_true))])

        eq1 = (log_K1 - RQ1)*scale
        eq2 = (log_K2 - RQ2)*scale

        print(ξ)
        print(Fl[0] - (ξ[0] + ξ[1]))
        print(eq1, eq2)
        # print(logCl_true)
        # print(log_K1, RQ1, eq1)
        # print(log_K2, RQ2, eq2)
        print()
        return eq1, eq2

    # guesses = np.array([Fl[0]*.2, Fl[0]*.1])
    scale = 1
    guesses = np.array([3.2848, 0.26967])*scale
    options = {
        # 'eps': 1e-30
    }
    res = root(root_solve, guesses, args=(np.array(Fl)*scale, scale), method='lm', options=options)
    ξ = res.x
    # ξ = guesses

    Fl_true = np.array([Fl[i] + rxn_1_st[i] * ξ[0] + rxn_2_st[i] * ξ[1] for i in range(len(rxn_1_st))])
    x_true = np.array([Fl_true[i] / sum(Fl_true) for i in range(len(Fl_true))])
    Cl_true = np.array([x_true[i] * rho_mol_l for i in range(len(x_true))])
    logCl_true = np.array([np.log(Cl_true[i]) for i in range(len(Cl_true))])
    RQ1 = sum([logCl_true[i] * rxn_1_st[i] for i in range(len(x_true))])
    RQ2 = sum([logCl_true[i] * rxn_2_st[i] for i in range(len(x_true))])

    eq1 = (log_K1 - RQ1)
    eq2 = (log_K2 - RQ2)

    print(eq1, eq2)

    return Fl_true, x_true, Cl_true


if __name__ == '__main__':
    from MEA_Absorption_Column.Properties.Density import liquid_density
    Fl = [3.5544993956757, 9.045716779, 69.6840552543039]
    Tl = 333.646741247888

    Fl_true, x_true, Cl_true = solve_ChemEQ_3(Fl, Tl, liquid_density)
    print(Fl_true)
    print(x_true)
    print(Cl_true)