import numpy as np
import pandas as pd
from numpy import sum
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from MEA_Absorption_Column.Transport.Enhancement_Factor import enhancement_factor_implicit, enhancement_factor_explicit
from MEA_Absorption_Column.Transport.Solve_MassTransfer import solve_masstransfer
from MEA_Absorption_Column.Thermodynamics.Solve_Driving_Force import solve_driving_force
from MEA_Absorption_Column.Thermodynamics.Solve_ChemEQ import solve_ChemEQ
from MEA_Absorption_Column.Properties.Henrys_Law import henrys_law
from MEA_Absorption_Column.Properties.Density import liquid_density, vapor_density
from MEA_Absorption_Column.Properties.Viscosity import viscosity
from MEA_Absorption_Column.Properties.Surface_Tension import surface_tension
from MEA_Absorption_Column.Properties.Diffusivity import liquid_diffusivity, vapor_diffusivity
from MEA_Absorption_Column.Properties.Heat_Capacity import heat_capacity
from MEA_Absorption_Column.Properties.Thermal_Conductivity import thermal_conductivity
from MEA_Absorption_Column.Parameters import MWs_l, packing_params, column_params
from MEA_Absorption_Column.Transport.Heat_Transfer import heat_transfer

def get_x(CO2_loading, amine_concentration):
    MW_MEA = 61.084
    MW_H2O = 18.02

    x_MEA_unloaded = amine_concentration / (MW_MEA / MW_H2O + amine_concentration * (1 - MW_MEA / MW_H2O))
    x_H2O_unloaded = 1 - x_MEA_unloaded

    n_MEA = 100 * x_MEA_unloaded
    n_H2O = 100 * x_H2O_unloaded

    n_CO2 = n_MEA * CO2_loading
    n_tot = n_MEA + n_H2O + n_CO2
    x_CO2, x_MEA, x_H2O = n_CO2 / n_tot, n_MEA / n_tot, n_H2O / n_tot

    return [x_CO2, x_MEA, x_H2O]


def get_y(x_CO2, x_H2O, CO2_interp, H2O_interp):
    alpha_O2_N2 = 0.08485753604

    y_CO2 = CO2_interp(x_CO2)
    y_H2O = H2O_interp(x_H2O)

    y_N2 = (1 - y_CO2 - y_H2O) / (1 + alpha_O2_N2)
    y_O2 = y_N2 * alpha_O2_N2
    y = [y_CO2, y_H2O, y_N2, y_O2]
    return y


a_p = packing_params['MellapakPlus252Y']['a_e']
ϵ = packing_params['MellapakPlus252Y']['eps']
Clp = packing_params['MellapakPlus252Y']['Cl']
Cvp = packing_params['MellapakPlus252Y']['Cv']
Cs = packing_params['MellapakPlus252Y']['Cs']
Cp_0 = packing_params['MellapakPlus252Y']['Cp_0']
Ch = packing_params['MellapakPlus252Y']['Ch']

packing = a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch

D = column_params['NCCC']['D']
A = np.pi * .25 * D ** 2
df2 = pd.read_csv(r'C:\Users\Tanner\Documents\git\MEA_Absorption_Column\MEA_Absorption_Column\data\Interpolations.csv')
x_CO2, x_H2O, y_CO2, y_H2O = df2['x_CO2'].to_numpy(), df2['x_H2O'].to_numpy(), df2['y_CO2'].to_numpy(), df2['y_H2O'].to_numpy()
CO2_interp = interp1d(x_CO2, y_CO2, kind='cubic')
H2O_interp = interp1d(x_H2O, y_H2O, kind='cubic')

alpha_range = np.arange(.15, .40+.01, .01)
T_range = np.arange(320, 350+10, 10)
y_CO2 = .1
w_MEA_nom = .3
P = 101980
Fl_T = 80
Fv_T = 30
T_fill = []
a_fill = []
E1_fill = []
E2_fill = []
Cl_MEA_true_fill = []

for T in T_range:
    for alpha in alpha_range:
        alpha = round(alpha, 2)
        x = get_x(alpha, w_MEA_nom)
        y = get_y(x[0], x[2], CO2_interp, H2O_interp)

        Tl, Tv = T, T

        Fl = [x[i] * Fl_T for i in range(len(x))]
        Fv = [y[i] * Fv_T for i in range(len(y))]

        Cl_true, x_true = solve_ChemEQ(Fl.copy(), Tl)

        H_CO2_mix = henrys_law(x, Tl)

        w = [MWs_l[i] * x[i] / sum([MWs_l[j] * x[j] for j in range(len(Fl))]) for i in range(len(Fl))]

        w_MEA = w[1]
        w_H2O = w[2]

        rho_mol_l, rho_mass_l, volume = liquid_density(Tl, x)
        rho_mol_v, rho_mass_v = vapor_density(Tv, P, y)

        ul = Fl_T / (A * rho_mol_l)
        uv = Fv_T / (A * rho_mol_v)

        sigma = surface_tension(Tl, x, w_MEA, w_H2O, alpha)

        muv_mix, mul_mix, mul_H2O, muv = viscosity(Tl, Tv, y, w_MEA, w_H2O, alpha)

        Dl_CO2, Dl_MEA, Dl_ion = liquid_diffusivity(Tl, rho_mol_l * x[1], mul_mix)
        Dv_CO2, Dv_H2O, Dv_N2, Dv_O2, Dv_T = vapor_diffusivity(Tv, y, P)

        ΔP_H, kl_CO2, kv_CO2, kv_H2O, kv_T, k_mxs, uv, a_e, hydr, const = solve_masstransfer(rho_mass_l, rho_mass_v,
                                                                                             mul_mix, muv_mix, mul_H2O,
                                                                                             sigma, Dl_CO2, Dv_CO2, Dv_H2O,
                                                                                             Dv_T,
                                                                                             A, Tv,
                                                                                             ul, uv, Fl_T, Fv_T,
                                                                                             packing)

        E1, _, _ = enhancement_factor_explicit(Tl, y[0], P, Cl_true, H_CO2_mix, kl_CO2, kv_CO2,
                                               Dl_CO2, Dl_MEA, Dl_ion)

        E2, _, _ = enhancement_factor_implicit(Tl, y[0], P, Cl_true, H_CO2_mix, kl_CO2, kv_CO2,
                                               Dl_CO2, Dl_MEA, Dl_ion)

        E1_fill.append(E1)
        E2_fill.append(E2)
        T_fill.append(T)
        a_fill.append(alpha)
        Cl_MEA_true_fill.append(Cl_true[1])
#%%
import pandas as pd
data = np.column_stack((T_fill, a_fill, E1_fill, E2_fill, Cl_MEA_true_fill))

df = pd.DataFrame(data, columns=['T', 'alpha', 'E1', 'E2', 'Cl_MEA'])
df.to_csv(r'C:\Users\Tanner\Documents\git\MEA_Absorption_Column\MEA_Absorption_Column\data\Results\Enhancement_Factor_Analysis.csv')

#%%

T_range = df['T'].unique()
alpha = df['alpha'].to_numpy()
E1 = df['E1'].to_numpy()
E2 = df['E2'].to_numpy()
mfc = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple']
fig, ax = plt.subplots(figsize=(14, 10))
for i, T in enumerate(T_range):

    df_data = df[(df['T'] == T)]
    alpha = df_data['alpha'].to_numpy()
    E1 = df_data['E1'].to_numpy()
    E2 = df_data['E2'].to_numpy()

    ax.plot(alpha, E1, label=f'Explicit - T={T}', color=mfc[i], linestyle='dashed')
    ax.plot(alpha, E2, label=f'Implicit - T={T}', color=mfc[i], linestyle='dotted')

ax.set_xlabel("CO$_{2}$ Loading, mol CO$_{2}$/mol MEA", fontsize=16)
ax.set_ylabel("Enhancement Factor", fontsize=16)
ax.legend(fontsize=16)


fig.tight_layout()
plt.show()




