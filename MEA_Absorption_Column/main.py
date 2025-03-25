from BVP.Run_Model import run_model
import pandas as pd
import warnings
import numpy as np
from Parameters import n, column_params
import matplotlib.pyplot as plt

np.set_printoptions(legacy='1.25')

warnings.filterwarnings("ignore")

CO2_cap_array = []
results_array = []
inputs_array = []

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

df = df_NCCC_full

for i in range(2, 3):
    CO2_cap = run_model(df,
                         method='collocation',
                         data_type=data_type,
                         run=i,
                         save_run_results=True,
                         plot_temperature=True,
                         show_info=True,
                         )
    # print(Tl_matrix[:, i])

# np.savetxt("data/Tl_matrix.csv", Tl_matrix, delimiter=',')
# np.savetxt("data/Tv_matrix.csv", Tv_matrix, delimiter=',')
