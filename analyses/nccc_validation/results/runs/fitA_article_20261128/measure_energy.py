"""Measure Case 3C net enthalpy flow on the retained collocation polynomial."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PPoly

from mea_absorption_column.Properties.Thermophysical_Properties import enthalpy


def measure_run(run):
    fl = pd.read_csv(run / 'Fl.csv')
    fv = pd.read_csv(run / 'Fv.csv')
    temperature = pd.read_csv(run / 'T.csv')
    z_export = temperature.Position.to_numpy()
    scaled = pd.read_csv(run / 'solution_scaled.csv').to_numpy()[:, :6]
    physical = np.column_stack([
        fl.Fl_CO2, fl.Fl_H2O, fv.Fv_CO2, fv.Fv_H2O, temperature.Tl, temperature.Tv,
    ])
    scales = physical[0] / scaled[0]
    np.testing.assert_allclose(scaled * scales, physical, rtol=0, atol=1e-7)

    with np.load(run / 'solver_polynomial.npz') as saved:
        mesh = saved['x']
        curve = PPoly(saved['coefficients'], mesh)
        report = dict(
            success=bool(saved['success']),
            nodes=len(mesh),
            iterations=int(saved['iterations']),
            max_rms_residual=float(saved['residuals'].max()),
        )
    np.testing.assert_allclose(curve(z_export)[:, :6] * scales, physical, rtol=0, atol=1e-7)

    def measure(z):
        values = curve(z)[:, :6] * scales
        net = []
        for l_co2, l_water, v_co2, v_water, tl, tv in values:
            liquid = np.array([l_co2, fl.Fl_MEA.iloc[0], l_water])
            vapor = np.array([v_co2, v_water, fv.Fv_N2.iloc[0], fv.Fv_O2.iloc[0]])
            net.append(
                vapor.sum() * enthalpy(tv, vapor / vapor.sum(), 'vapor')[1]
                - liquid.sum() * enthalpy(tl, liquid / liquid.sum(), 'liquid')[1]
            )
        return dict(
            range_W=float(np.ptp(net)),
            endpoint_difference_W=net[-1] - net[0],
            minimum_z=float(z[np.argmin(net)]),
            maximum_z=float(z[np.argmax(net)]),
            capture_pct=float(100 * (1 - values[-1, 2] / values[0, 2])),
            peak_liquid_K=float(values[:, 4].max()),
        ), net

    report['mesh'], _ = measure(mesh)
    report['export_grid'], net_export = measure(z_export)
    report['dense'], _ = measure(np.linspace(mesh[0], mesh[-1], 10001))
    exported_net = pd.read_csv(run / 'Hv.csv').Hvf - pd.read_csv(run / 'Hl.csv').Hlf
    np.testing.assert_allclose(net_export, exported_net, rtol=0, atol=1e-7)
    (run / 'energy_check.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    for path in sys.argv[1:]:
        measure_run(Path(path))
