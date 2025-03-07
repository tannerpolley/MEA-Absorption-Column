from BVP.Run_Model import run_model
import pandas as pd
import warnings
import numpy as np
from scipy.optimize import minimize

np.set_printoptions(legacy='1.25')
from data.create_LHC_design import LHC_design

warnings.filterwarnings("ignore")

CO2_cap_array = []
results_array = []
inputs_array = []

# LHC_design(25)
# df_SRP = pd.read_csv('data/LHC_design_w_SRP_cases.csv', index_col=0)
# df_NCCC = pd.read_csv('data/runs_file_NCCC_case_18.csv', index_col=0)
df_NCCC = pd.read_csv('data/runs_file_NCCC_LHC.csv', index_col=0)
# df = LHC_design(25)
# data_source = 'SRP'
data_type = 'mole'

df = df_NCCC

for i in range(len(df)):
    # try:
    CO2_cap, shooter_message = run_model(df, method='scipy', data_type=data_type, run=i, save_run_results=True, plot_temperature=True)
    # CO2_cap, shooter_message = run_model(df, method='collocation', data_source=data_source, run=i, save_run_results=True, )
    # except TypeError:
    #     print('NAN detected in integrator')
    #     pass


# def optimize(x):
#     for i in range(0, 1):
#         # try:
#         if data_source == 'NCCC':
#             df = df_NCCC
#         elif data_source == 'SRP':
#             df = df_SRP
#         else:
#             raise ValueError('Data source must be either NCCC or SRP')
#         try:
#             CO2_cap, shooter_message = run_model(x[0], df, method='single', data_source=data_source, run=i,
#                                                  save_run_results=False, )
#             modifier = 1000
#         # CO2_cap, shooter_message = run_model(df, method='collocation', data_source=data_source, run=i, save_run_results=True, )
#         except ValueError:
#             modifier = 0
#             pass
#     obj = -(x + modifier)
#     return obj
#
#
# Tl_0_guess = 323.832401247888
# Tl_0_guess = 322.832401247888
#
# # answer = minimize(optimize, np.array([Tl_0_guess]), method='Nelder-Mead', options={'maxiter': 100})
# # print(answer.x)
#
#
# CO2_cap, shooter_message = run_model(324.02080635, df_NCCC, method='single', data_source=data_source, run=0,
#                                                  save_run_results=True, )