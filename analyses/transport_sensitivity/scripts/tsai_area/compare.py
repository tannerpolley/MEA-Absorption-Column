"""3C area-correlation sensitivity: process-local expression substitution."""
import argparse, json, os
from pathlib import Path
import numpy as np
from mea_absorption_column.column import run_column, _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'analyses/transport_sensitivity/results/runs/tsai_area'
PAIRS = {'chinen': '1.42,0.12', 'tsai': '1.34,0.116'}
p = argparse.ArgumentParser()
p.add_argument('route', choices=['seven', 'twelve'])
p.add_argument('pair', choices=PAIRS)
a = p.parse_args()
assert os.environ.get('ISSUE84_PAIR') == PAIRS[a.pair]
out = OUT / (a.route + '_' + a.pair)
request = {'preset': 'seven_state_legacy' if a.route == 'seven' else 'twelve_state_conserved',
    'case': {'physical_input_file': 'analyses/bvp_solution_methods/input/case_3c.json'},
    'execution': {'wall_limit_s': 600, 'output_dir': str(out)}}
if a.route == 'seven':
    request['numerics'] = {'method': 'scipy-bvp'}
if a.route == 'twelve':
    request['numerics'] = {'method': 'upwind', 'solver_settings': {'nodes': 11, 'quadrature_points': 5, 'end_clustering': 1., 'max_iterations': 20, 'tolerance': 1e-7, 'liquid_density_source': 'solved_state'}}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / (a.route+'_'+a.pair+'_request.json')).write_text(json.dumps(request, indent=2)+'\n')
result = run_column(request)
print('status', result.get('execution'), 'failure', result.get('failure'), flush=True)
if a.route == 'twelve' and result.get('numerical_acceptance') == 'accepted':
    u = np.array(result['native_profile']['state_matrix'])
    assembly = _prepare_conserved_column_in_process(resolve_column_config(request))['assembly']
    h = np.array([np.asarray(assembly['balance'](v[:8], v[8:11])[-1]).ravel() for v in u.T])
    summary = {'capture_percent': 100*(1-u[2,-1]/u[2,0]), 'peak_liquid_K': float(u[4].max()),
       'area_m2_m3_range': [float(h[:,5].min()/.3216990877275948), float(h[:,5].max()/.3216990877275948)],
       'scaled_residual_inf': result['result']['scaled_residual_inf'], 'physical_certification': result['physical_certification']}
    (OUT / (a.route+'_'+a.pair+'_summary.json')).write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
