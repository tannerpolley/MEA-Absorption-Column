import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# filename = r'C:\Users\Tanner\Documents\git\MEA_Absorption_Column\MEA_Absorption_Column\data\Results\Profiles.xlsx'
filename = r'C:\Users\Tanner\Documents\git\IDAES_MEA_Flowsheet_Tanner\Simulation_Results\Profiles_IDAES.xlsx'

# Load your datasets from Excel
data = pd.read_excel(filename, sheet_name='Fv')
x = data['Position'].to_numpy()[::-1]

names = {
    'Fl_CO2': 'Fl',
    'Fl_H2O': 'Fl',
    'Fv_CO2': 'Fv',
    'Fv_H2O': 'Fv',
    'Tl': 'T',
    'Tv': 'T',
    'P': 'transport'
         }

fitted_coeffs = {}
i = 0
degree = 10


def exp_decay(x, a, b):
    return a * np.exp(-b * x)


x_fit = np.linspace(x[0], x[-1], 200)

for k, v in names.items():
    y = pd.read_excel(filename, sheet_name=v)[k].to_numpy()[::-1]

    # Fit polynomial to normalized data
    if k == 'Fv_CO2':
        coeffs = curve_fit(exp_decay, x, y)[0]
        y_fit = exp_decay(x_fit, *coeffs)

        coeffs = list(coeffs) + [0]*(degree - len(list(coeffs)))
    else:
        coeffs = np.polyfit(x, y, deg=degree)

        y_fit = np.polyval(coeffs, x_fit)
        coeffs = coeffs[:-1]

    plt.plot(x, y)
    plt.plot(x_fit, y_fit, '--')
    plt.show()

    fitted_coeffs[k] = coeffs

print(fitted_coeffs)
df = pd.DataFrame(fitted_coeffs)
df.to_csv(r"C:\Users\Tanner\Documents\git\MEA_Absorption_Column\MEA_Absorption_Column\data\fitted_coefficients.csv")
print(df)



# After verifying similar normalized fits for each dataset, average them:
# For example, coefficients from fitting each dataset individually (coeffs1, coeffs2, coeffs3):
