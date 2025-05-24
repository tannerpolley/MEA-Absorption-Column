from mea_absorption_column.Run_Model import run_model
import pandas as pd
import warnings
import numpy as np

np.set_printoptions(legacy='1.25')
warnings.filterwarnings("ignore")


df_NCCC_1 = pd.read_csv('data/runs_file_NCCC_LHC.csv', index_col=0)
df_NCCC_2 = pd.read_csv('data/runs_file_NCCC_1_bed_cases.csv', index_col=0)
df_NCCC_full = pd.read_csv('data/NCCC_Data.csv', index_col=0)
df_NCCC_full = pd.read_csv('data/C_cases_data.csv', index_col=0)

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

