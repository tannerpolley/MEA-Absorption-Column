import numpy as np
from gekko import GEKKO


def solve_ChemEQ(Fl, Tl):
    a1, b1, c1, d1 = 164.039636, -707.0056712, -26.40136817, 0
    a2, b2, c2, d2 = 366.061867998774, -13326.25411, -55.68643292, 0

    rxn_1_st = [-1, -2, 0, 1, 1, 0]
    rxn_2_st = [-1, -1, -1, 1, 0, 1]

    v_ij = np.array([rxn_1_st, rxn_2_st])

    Fl += [0, 0, 0]

    Fl = np.array(Fl)
    Fl_T_0 = np.sum(Fl)
    x0 = Fl / Fl_T_0

    ξg = np.array([3, .2])
    x_true_g = np.array([1.10119014e-08, 2.29594647e-02, 7.97550499e-01, 3.63162607e-02, 3.31457204e-02, 3.17054028e-03])

    species = ['CO2', 'MEA', 'H2O', 'MEAH', 'MEACOO', 'HCO3']

    m = GEKKO(remote=False)
    m._path = r'C:\Users\Tanner\Documents\git\MEA_Absorption_Column\MEA_Absorption_Column\Thermodynamics\gekko_run_contents'

    logK1 = m.Param(a1 + b1 / Tl + c1 * np.log(Tl) + d1 * Tl, name='lnK1')
    logK2 = m.Param(a2 + b2 / Tl + c2 * np.log(Tl) + d2 * Tl, name='lnK2')
    logK = [logK1, logK2]

    x_scales = np.array([1e-8, .01, 1, .05, .05, .005])

    x_scaled = [m.Var(value=x_true_g[i]/x_scales[i], lb=0, name=f'x_{i+1}') for i in range(len(x_true_g))]
    ξ = [m.Var(value=ξg[i], lb=0, name=f'eps_{i+1}') for i in range(len(ξg))]

    x = [m.Intermediate(x_scaled[i]*x_scales[i]) for i in range(len(x_true_g))]

    term1 = m.Intermediate(-(logK[0] * ξ[0]/Fl_T_0 + logK[1] * ξ[1]/Fl_T_0), name='term1')
    term2_x = [m.Intermediate(x[j] * m.log(x[j]), name=f'x_term2_{j+1}') for j in range(len(x))]
    term2 = m.Intermediate(term2_x[0] + term2_x[1] + term2_x[2] + term2_x[3] + term2_x[4] + term2_x[5], name='term2')
    sum_of_x = m.Intermediate(x[0] + x[1] + x[2] + x[3] + x[4] + x[5])
    term3 = m.Intermediate(-sum_of_x * m.log(sum_of_x), name='term3')
    obj = m.Intermediate(term1 + term2 + term3)

    m.Minimize(obj)
    m.Equation([x[j] == x0[j] + m.sum([v_ij[i, j]*ξ[i]/Fl_T_0 for i in range(len(ξ))]) for j in range(len(x))])

    m.options.IMODE = 3
    m.options.SOLVER = 3
    m.options.NODES = 4
    m.options.RTOL = 1e-3
    m.solve(disp=False)

    # print(ξ[0])
    ξ = [ξ[0].value[0], ξ[1].value[0]]
    #
    # print('obj:', obj.value[0])
    # print('Extent:', ξ)
    # print('Flow Rate:', Fl)

    Fl_true = [Fl[j] + np.sum([v_ij[i, j]*ξ[i] for i in range(len(ξ))]) for j in range(len(Fl))]
    x_true = [Fl_true[i]/np.sum(Fl_true) for i in range(len(Fl_true))]
    return Fl_true, x_true


if __name__ == '__main__':
    Fl = [3.5544993956757, 9.045716779, 69.6840552543039]
    Tl = 333.646741247888

    Fl_true, x_true = solve_ChemEQ(Fl, Tl)
    print(Fl_true)
    print(x_true)