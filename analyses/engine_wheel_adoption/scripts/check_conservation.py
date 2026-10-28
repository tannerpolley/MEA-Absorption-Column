"""Check C/H/N/O and charge against the retained apparent liquid flows, without running the model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

run = Path(__file__).resolve().parents[1] / 'results/runs/reactive_3c_verified'
liquid, vapor = (pd.read_csv(run / f'{name}.csv', index_col=0) for name in ('Fl', 'Fv'))
elements = np.array([[1, 2, 0, 2, 3, 1, 1, 0, 0], [0, 7, 2, 8, 6, 1, 0, 3, 1],
                     [0, 1, 0, 1, 1, 0, 0, 0, 0], [2, 1, 1, 1, 3, 3, 3, 1, 1]])
true = liquid.filter(regex='_true$').to_numpy()
apparent = liquid[['Fl_CO2', 'Fl_MEA', 'Fl_H2O']].to_numpy()
balance = true @ elements.T - apparent @ elements[:, :3].T
charge = true @ np.array([0, 0, 0, 1, -1, -1, -2, 1, -1])
record = dict(points=len(liquid), maximum_element_residual_mol_per_s={
    name: float(np.max(np.abs(balance[:, i]))) for i, name in enumerate(['C', 'H', 'N', 'O'])},
    maximum_charge_residual_mol_per_s=float(np.max(np.abs(charge))),
    net_co2_flow_range_mol_per_s=float(np.ptp(liquid.Fl_CO2-vapor.Fv_CO2)),
    net_water_flow_range_mol_per_s=float(np.ptp(liquid.Fl_H2O-vapor.Fv_H2O)))
(run / 'conservation.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
