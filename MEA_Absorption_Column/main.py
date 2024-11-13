from BVP.Run_Model import run_model
import pandas as pd
import warnings
import numpy as np
np.set_printoptions(legacy='1.25')
from data.create_LHC_design import LHC_design
warnings.filterwarnings("ignore")

CO2_cap_array = []
results_array = []
inputs_array = []

# LHC_design(25)
df = pd.read_csv('data/LHC_design_w_SRP_cases.csv', index_col=0)
# df = pd.read_csv('data/runs_file_NCCC_case_18.csv', index_col=0)
# df = LHC_design(25)
data_source = 'SRP'

for i in range(0, 1):
    # try:
    CO2_cap, shooter_message = run_model(df, method='single', data_source=data_source, run=i, save_run_results=True, )
    # CO2_cap, shooter_message = run_model(df, method='collocation', data_source=data_source, run=i, save_run_results=True, )
    # except TypeError:
    #     pass
    # except ValueError:
    #     print('Error: NaN detected')
    # except AssertionError:
    #     print('Temperature Out of Range')

