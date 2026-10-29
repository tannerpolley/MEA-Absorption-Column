"""Report countercurrent conservation from a retained seven-enthalpy profile."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
args = parser.parse_args()
attempt = json.loads((args.run/'attempt.json').read_text())
profile = attempt['native_profile']
assert profile['layout']['state_order'] == [
    'F_L_CO2', 'F_L_H2O', 'F_V_CO2', 'F_V_H2O', 'H_L', 'H_V', 'P']
state = np.asarray(profile['state_matrix_scaled'])*np.asarray(profile['layout']['state_scale'])[:, None]
# Both streams use positive flow magnitudes; liquid travels toward decreasing z.
conserved = np.array([state[0]-state[2], state[1]-state[3], state[4]-state[5], state[6]])
drift = np.max(abs(conserved-conserved[:, :1]), axis=1)
flows = attempt['resolved_inputs']['parameters']
scale = np.array([flows[1][0], flows[1][1], max(abs(state[4, -1]), abs(state[5, 0])), flows[7]])
reference = Path('analyses/nccc_validation/results/final/tables/raw_c_case_thermo_benchmark.csv')
rows = pd.read_csv(reference)
row = rows[(rows.case_id == '3C') & (rows.thermo_model == 'epcsaft_ionic')].iloc[0]
capture = attempt['observables']['capture']['value']
report = {
    'evidence_kind': 'numerical verification, not physical validation',
    'native_nodes': state.shape[1],
    'conserved_quantities': ['F_L_CO2-F_V_CO2', 'F_L_H2O-F_V_H2O', 'H_L-H_V', 'P'],
    'units': ['mol/s', 'mol/s', 'W', 'Pa'],
    'maximum_absolute_native_drift': drift.tolist(),
    'normalization_values': scale.tolist(),
    'maximum_relative_native_drift': (drift/scale).tolist(),
    'normalization_basis': ['inlet vapor CO2', 'inlet vapor water', 'maximum inlet enthalpy flow magnitude', 'specified pressure'],
    'capture_percent': capture,
    'boundary_residual_percent_norm': attempt['observables']['boundary_residual_norm']['value'],
    'max_rms_residual': attempt['diagnostics']['max_rms_residual'],
    'max_scaled_boundary_residual': attempt['diagnostics']['max_scaled_boundary_residual'],
    'reference_file': str(reference),
    'reference_row': json.loads(row.to_json()),
    'capture_difference_percentage_points': capture-float(row.capture_pct),
    'comparison_limit': 'Different executable/wheel and incompletely recorded earlier settings; not an isolated Engine or same-physics preservation comparison.',
}
(args.run/'conservation_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps(report, indent=2, allow_nan=False))
