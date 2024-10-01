from numpy import exp, log, array
from scipy.optimize import minimize, root

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




