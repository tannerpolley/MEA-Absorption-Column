"""Issue 16: fixed-input twelve-state 3C film comparison, without physics changes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mea_absorption_column.column import run_column, _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'analyses/reactive_film_evidence/results/runs/engine100'
FILMS = ('enhancement_reference', 'equilibrium_manifold')


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def table(name, rows):
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run(film):
    request = {'preset': 'twelve_state_conserved',
        'model': {'film_model': film},
        'case': {'physical_input_file': 'analyses/bvp_solution_methods/input/case_3c.json'},
        'numerics': {'method': 'upwind', 'solver_settings': {'nodes': 11, 'quadrature_points': 5,
            'end_clustering': 1., 'max_iterations': 20, 'tolerance': 1e-7,
            'liquid_density_source': 'solved_state'}},
        'execution': {'wall_limit_s': 1450, 'output_dir': str(OUT / film)}}
    save(film + '_request.json', request)
    result = run_column(request)
    print(result['execution'], result.get('failure'), flush=True)


def summarize():
    rows, profiles = [], []
    for film in FILMS:
        attempt = json.loads((OUT / film / 'attempt.json').read_text())
        row = dict(film=film, numerical_acceptance=attempt['numerical_acceptance'],
            runtime_s=attempt['execution']['runtime_s'], capture_percent=None,
            flux_min_mol_m2_s=None, flux_max_mol_m2_s=None,
            scaled_residual_inf=attempt.get('result', {}).get('scaled_residual_inf'),
            physical_certification=attempt.get('physical_certification', {}).get('accepted'),
            failure=attempt.get('failure'))
        native = attempt.get('native_profile')
        if native:
            u = np.array(native['state_matrix'])
            assert u.shape == (12, 11) and np.all(np.isfinite(u))
            z = np.array(attempt['result']['grid'])
            row.update(capture_percent=100*(1-u[2,-1]/u[2,0]),
                flux_min_mol_m2_s=float(u[8].min()), flux_max_mol_m2_s=float(u[8].max()))
            assembly = _prepare_conserved_column_in_process(resolve_column_config(
                json.loads((OUT / (film+'_request.json')).read_text())))['assembly']
            bulk_rows = [assembly['balance'](v[:8], v[8:11]) for v in u.T]
            charge = np.array([0, 0, 0, 1, -1, -1, -2, 1, -1])
            for i, values in enumerate(bulk_rows):
                liquid = np.asarray(values[3]).ravel()
                vapor = np.asarray(values[4]).ravel()
                drive = vapor[0] - liquid[9]
                assert u[8,i]*drive >= -1e-12
                profiles.append(dict(film=film, z_m=float(z[i]), flux_mol_m2_s=float(u[8,i]),
                    bulk_fugacity_drive_Pa=float(drive), liquid_K=float(u[4,i]),
                    charge_mol_per_mol=float(charge @ liquid[:9]/sum(liquid[:9]))))
        rows.append(row)
    table('comparison.csv', rows)
    if profiles:
        table('profiles.csv', profiles)
    save('summary.json', rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=(*FILMS, 'summarize'))
    stage = parser.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    summarize() if stage == 'summarize' else run(stage)
