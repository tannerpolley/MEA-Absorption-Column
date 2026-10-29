"""Evaluate seven-state native nodes using the same area closure and properties."""
import json
from pathlib import Path
import numpy as np
from mea_absorption_column.config.Constants import MWs_l, g
from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_liquid_temperature
from mea_absorption_column.Properties.Thermophysical_Properties import density, surface_tension

out = Path(__file__).resolve().parents[2] / 'results/runs/tsai_area'
for name, (c, exponent) in {'chinen': (1.42,.12), 'tsai': (1.34,.116)}.items():
    a = json.loads((out / ('seven_'+name) / 'attempt.json').read_text())
    u = np.array(a['native_profile']['state_matrix_scaled']) * np.array(a['native_profile']['layout']['state_scale'])[:,None]
    params = a['resolved_inputs']['parameters']
    feed, cross_section, packing = params[0], params[6], params[8]
    temps, areas = [], []
    for v in u.T:
        fl = np.array([v[0],feed[1],v[1]])
        x = fl/fl.sum()
        t = float(get_liquid_temperature(x, v[4]/fl.sum()))
        w = MWs_l*x / (MWs_l*x).sum()
        mol_density, mass_density, _ = density(t,x,v[6],phase='liquid')
        sigma = surface_tension(t,x,w[1],w[2])
        ul = fl.sum()/(cross_section*mol_density)
        area = c*packing[0]*(mass_density/sigma*g**(1/3)*(ul*packing[1]/packing[0])**(4/3))**exponent
        temps.append(t); areas.append(float(area))
    summary = {'capture_percent': a['observables']['capture']['value'], 'peak_liquid_K': max(temps),
               'area_m2_m3_range': [min(areas), max(areas)], 'max_rms_residual': a['diagnostics']['max_rms_residual'],
               'solver_tolerance': a['result']['tol'], 'boundary_residual_percent_norm': a['observables']['boundary_residual_norm']['value'],
               'numerical_acceptance': a['numerical_acceptance']}
    (out / ('seven_'+name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(name, summary)
