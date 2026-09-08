"""Recover retained cubic curves from exported values; no column rerun."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline

root=Path('analyses/nccc_validation/results/runs')
for n in [21,41]:
    run=root/f'reactive_native_{n}'
    samples=pd.read_csv(run/'solution_scaled.csv').to_numpy()
    z=np.linspace(0,1,len(samples))
    fl=pd.read_csv(run/'Fl.csv'); fv=pd.read_csv(run/'Fv.csv'); t=pd.read_csv(run/'T.csv')
    hl=pd.read_csv(run/'Hl.csv'); hv=pd.read_csv(run/'Hv.csv'); co=pd.read_csv(run/'CO2.csv'); wa=pd.read_csv(run/'H2O.csv')
    scales=np.array([fl.Fl_CO2[0],fl.Fl_H2O[0],fv.Fv_CO2[0],fv.Fv_H2O[0],t.Tl[0],t.Tv[0],109180.])/samples[0]
    rhs=np.column_stack([6*(-co.Nl_CO2+1e-10),6*(-wa.Nl_H2O+1e-10),6*(co.Nv_CO2+1e-10),6*(wa.Nv_H2O+1e-10),hl.dTl_dz,hv.dTv_dz,np.zeros(len(z))])/scales
    shared=np.arange(0,len(z),5)
    target=np.vstack([samples,rhs[shared]/(n-1)])
    # One node added to the original uniform grid. Test every possible interval.
    choices=[]
    for i in range(n-1):
        knots=np.sort(np.r_[np.linspace(0,1,n),(i+.5)/(n-1)])
        m=len(knots)
        basis=CubicHermiteSpline(knots,np.c_[np.eye(m),np.zeros((m,m))],np.c_[np.zeros((m,m)),np.eye(m)])
        design=np.vstack([basis(z),basis.derivative()(z[shared])/(n-1)])
        coefficients,_,rank,_=np.linalg.lstsq(design,target,rcond=None)
        error=float(np.max(np.abs(design@coefficients-target)))
        choices.append((error,i,rank,knots,coefficients))
    valid=[r for r in choices if r[0]<1e-10 and r[2]==2*len(r[3])]
    assert len(valid)==1, [(r[0],r[1],r[2]) for r in choices]
    error,i,rank,knots,coefficients=valid[0]
    print(json.dumps(dict(mesh=n,error=error,interval=i,rank=int(rank),unknowns=2*len(knots))),flush=True)
    assert rank==2*len(knots) and error<1e-10, 'Cannot uniquely reconstruct retained spline'
    m=len(knots)
    curve=CubicHermiteSpline(knots,coefficients[:m],coefficients[m:])
    out=root/'energy_diagnostic_20260904'/f'reconstructed_{n}'
    out.mkdir(exist_ok=True)
    np.savez(out/'solver_polynomial.npz',x=knots,y=coefficients[:m].T,coefficients=curve.c,residuals=[np.nan],success=True)
    (out/'reconstruction.json').write_text(json.dumps(dict(max_scaled_fit_error=error,rank=int(rank),unknowns=2*m,inserted_node=float((i+.5)/(n-1))),indent=2)+'\n')
