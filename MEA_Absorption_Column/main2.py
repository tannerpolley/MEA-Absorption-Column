from BVP.Run_Model import run_model
import pandas as pd
import warnings
import numpy as np
from scipy.optimize import minimize
from Parameters import n, column_params
import matplotlib.pyplot as plt

np.set_printoptions(legacy='1.25')

warnings.filterwarnings("ignore")

# LHC_design(25)
# df_SRP = pd.read_csv('data/LHC_design_w_SRP_cases.csv', index_col=0)
# df_NCCC = pd.read_csv('data/runs_file_NCCC_case_18.csv', index_col=0)
df_NCCC_1 = pd.read_csv('data/runs_file_NCCC_LHC.csv', index_col=0)
df_NCCC_2 = pd.read_csv('data/runs_file_NCCC_1_bed_cases.csv', index_col=0)
df_NCCC_full = pd.read_csv('data/NCCC_Data.csv', index_col=0)
df_NCCC_full = pd.read_csv('data/C_cases_data.csv', index_col=0)
# df = LHC_design(25)
# data_source = 'SRP'
data_type = 'mole'

def model(x):
    x1, x2 = x

    df = df_NCCC_1
    df.iloc[0, 1] = x1
    df.iloc[0, 3] = x2

    print(df.iloc[0, 1])

    print(df.iloc[0, :])


    CO2_cap = run_model(df,
                         method='collocation',
                         data_type=data_type,
                         run=0,
                         # save_run_results=True,
                         # plot_temperature=True,
                         # show_info=True
                         )

    return CO2_cap
print(model([3, .3]))
res = minimize(model, [3, .3])