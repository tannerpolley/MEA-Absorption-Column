import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = r'C:\Users\Tanner\Documents\git\MEA_Absorption_Column\MEA_Absorption_Column\data\Results\Profiles.xlsx'

# Load your datasets from Excel
data = pd.read_excel(filename, sheet_name='Fv')
x = data['Position'].to_numpy()[::-1]

names = {
    'Fl_CO2': 'Fl',
    'Fl_H2O': 'Fl',
    'Fv_CO2': 'Fv',
    'Fv_H2O': 'Fv',
    'Hlf': 'Hl',
    'Hvf': 'Hv',
    'P': 'transport'
         }

fitted_coeffs = {}
i = 0


def exp_decay(x, a, b):
    return a * np.exp(-b * x)


x_fit = np.linspace(x[0], x[-1], 200)

for k, v in names.items():
    y = pd.read_excel(filename, sheet_name=v)[k].to_numpy()[::-1]

    # Fit polynomial to normalized data
    if k == 'Fv_CO2':
        coeffs, _ = curve_fit(exp_decay, x, y)
        y_fit = exp_decay(x_fit, *coeffs)

        coeffs = list(coeffs) + [0]*(6 - len(list(coeffs)))
    else:
        coeffs = np.polyfit(x, y, deg=6)

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
