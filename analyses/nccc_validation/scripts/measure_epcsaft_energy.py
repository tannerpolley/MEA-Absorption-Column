"""Evaluate net Engine enthalpy on 10,001 points of the raw SciPy polynomial."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PPoly

from mea_absorption_column.Thermodynamics.reactive_bundle import FIT_A, ReactiveLiquid, bulk_vapor_state


def measure(directory):
    l = pd.read_csv(directory/'Fl.csv', float_precision='round_trip')
    v = pd.read_csv(directory/'Fv.csv', float_precision='round_trip')
    t = pd.read_csv(directory/'T.csv', float_precision='round_trip')
    p = pd.read_csv(directory/'transport.csv', float_precision='round_trip').P
    physical = np.column_stack((l.Fl_CO2, l.Fl_H2O, v.Fv_CO2, v.Fv_H2O, t.Tl, t.Tv, p))
    scaled = pd.read_csv(directory/'solution_scaled.csv', float_precision='round_trip').to_numpy()
    scales = physical[0]/scaled[0]
    with np.load(directory/'solver_polynomial.npz') as archive:
        polynomial = PPoly(archive['coefficients'], archive['x'])
    np.testing.assert_allclose(polynomial(t.Position.to_numpy())*scales, physical, atol=1e-7, rtol=0)
    positions = np.linspace(0., 1., 10001)
    states = polynomial(positions)*scales
    runtime = ReactiveLiquid(FIT_A, loading_anchor=.25,
                             water_per_mea_anchor=l.Fl_H2O.iloc[-1]/l.Fl_MEA.iloc[-1], reuse_states=True)
    result = []
    for z, (lc, lw, vc, vw, tl, tv, p) in zip(positions, states, strict=True):
        liquid_feed = (lc, l.Fl_MEA.iloc[0], lw)
        vapor_feed = (vc, vw, v.Fv_N2.iloc[0], v.Fv_O2.iloc[0])
        liquid = runtime.solve(tl, p, liquid_feed, caloric=True)
        vapor = bulk_vapor_state(tv, p, vapor_feed, str(FIT_A), True)
        total_liquid = sum(liquid_feed)
        result.append((z, liquid['enthalpy']*total_liquid, vapor['enthalpy'],
                       liquid['enthalpy_partials'][0]*total_liquid, lc, lw, vc, vw, tl, tv))
    frame = pd.DataFrame(result, columns=('z', 'H_L_W', 'H_V_W', 'C_L_W_K', 'F_L_CO2', 'F_L_H2O',
                                         'F_V_CO2', 'F_V_H2O', 'T_L_K', 'T_V_K'))
    frame['net_enthalpy_W'] = frame.H_V_W-frame.H_L_W
    frame.to_csv(directory/'energy_dense.csv', index=False)
    net = frame.net_enthalpy_W
    summary = dict(linear_solver=liquid['evidence']['linear_solver'], points=len(frame), range_W=float(np.ptp(net)),
                   endpoint_difference_W=float(net.iloc[-1]-net.iloc[0]), min_C_L_W_K=float(frame.C_L_W_K.min()),
                   min_position=float(frame.z.iloc[net.argmin()]), max_position=float(frame.z.iloc[net.argmax()]),
                   capture_pct=float(100*(1-frame.F_V_CO2.iloc[-1]/frame.F_V_CO2.iloc[0])),
                   min_liquid_K=float(frame.T_L_K.min()), max_liquid_K=float(frame.T_L_K.max()),
                   peak_liquid_z=float(frame.z.iloc[frame.T_L_K.argmax()]),
                   max_vapor_K=float(frame.T_V_K.max()), peak_vapor_z=float(frame.z.iloc[frame.T_V_K.argmax()]),
                   co2_range_mol_s=float(np.ptp(frame.F_V_CO2-frame.F_L_CO2)),
                   water_range_mol_s=float(np.ptp(frame.F_V_H2O-frame.F_L_H2O)))
    summary['temperature_equivalent_K'] = summary['range_W']/summary['min_C_L_W_K']
    (directory/'energy_check.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    measure(parser.parse_args().directory)
