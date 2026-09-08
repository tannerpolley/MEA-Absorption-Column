"""Measure energy conservation on actual nodes and a dense solver curve."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import PPoly
from mea_absorption_column.Properties.Thermophysical_Properties import enthalpy

run=Path(sys.argv[1])
base=Path('analyses/nccc_validation/results/runs/reactive_native_41')
fl=pd.read_csv(base/'Fl.csv'); fv=pd.read_csv(base/'Fv.csv'); t=pd.read_csv(base/'T.csv')
old=pd.read_csv(base/'solution_scaled.csv').to_numpy()
physical=np.column_stack([fl.Fl_CO2,fl.Fl_H2O,fv.Fv_CO2,fv.Fv_H2O,t.Tl,t.Tv,np.full(len(t),109180.)])
scales=physical[0]/old[0]
assert np.max(np.abs(old*scales-physical))<1e-7
saved=np.load(run/'solver_polynomial.npz')
curve=PPoly(saved['coefficients'],saved['x'])

def measure(z):
    values=curve(z)*scales
    net=[]
    for row in values:
        l=np.array([row[0],fl.Fl_MEA[0],row[1]])
        v=np.array([row[2],row[3],fv.Fv_N2[0],fv.Fv_O2[0]])
        net.append(v.sum()*enthalpy(row[5],v/v.sum(),'vapor')[1]-l.sum()*enthalpy(row[4],l/l.sum(),'liquid')[1])
    net=np.array(net)
    return dict(range_W=float(np.ptp(net)),endpoint_difference_W=float(net[-1]-net[0]),
                minimum_z=float(z[np.argmin(net)]),maximum_z=float(z[np.argmax(net)]),
                capture_pct=float(100*(1-values[-1,2]/values[0,2])),peak_liquid_K=float(values[:,4].max()))

report=dict(success=bool(saved['success']),nodes=len(saved['x']),max_rms_residual=float(max(saved['residuals'])),
            mesh=measure(saved['x']),export_grid=measure(np.linspace(0,1,101)),dense=measure(np.linspace(0,1,10001)))
(run/'energy_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
