import importlib,inspect,json
import pandas as pd
run=importlib.import_module('mea_absorption_column.Run_Model')
solver=importlib.import_module('mea_absorption_column.BVP.Methods.Scipy_BVP_Solve')
source=inspect.getsource(solver.scipy_BVP_solve)
source=source.replace('bcs_1 = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_a, Fv_H2O_a, Hlf_b, Hvf_a, P_a]) / scales','bcs_1 = np.array([Fl_CO2_b, Fl_H2O_b, Fv_CO2_a, Fv_H2O_a, Hlf_b, Hvf_a, P_a])').replace('bcs_2 = np.array([Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Hlf_b_bc, Hvf_a_bc, P_a_bc]) / scales','bcs_2 = np.array([Fl_CO2_b_bc, Fl_H2O_b_bc, Fv_CO2_a_bc, Fv_H2O_a_bc, Hlf_b_bc, Hvf_a_bc, P_a_bc])')
namespace=dict(solver.__dict__);exec(source,namespace);run.scipy_BVP_solve=namespace['scipy_BVP_solve']
data=pd.read_csv('src/mea_absorption_column/data/C_cases_data.csv',index_col=0)
result=run.run_model(data,method='scipy-bvp',run=data.index.get_loc('3C'),thermo_model='ideal_henry',return_details=True,staged_beds=False,solver_settings={'transform_mode':'positive_flow_pressure'})
print(json.dumps({k:v for k,v in result.items() if not k.startswith('_')},indent=2,default=str))
