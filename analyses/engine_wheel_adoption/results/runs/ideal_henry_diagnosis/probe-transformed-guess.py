import importlib,json
import pandas as pd,numpy as np
run=importlib.import_module('mea_absorption_column.Run_Model')
original=run.scipy_BVP_solve

def solve(left,right,z,parameters,settings):
 settings=dict(settings)
 settings['initial_guess_scaled']=np.array([np.linspace(a,b,len(z)) for a,b in zip(left,right)])
 return original(left,right,z,parameters,settings)
run.scipy_BVP_solve=solve
data=pd.read_csv('src/mea_absorption_column/data/C_cases_data.csv',index_col=0)
result=run.run_model(data,method='scipy-bvp',run=data.index.get_loc('3C'),thermo_model='ideal_henry',return_details=True,staged_beds=False,solver_settings={'transform_mode':'positive_flow_pressure'})
print(json.dumps({k:v for k,v in result.items() if not k.startswith('_')},indent=2,default=str))
