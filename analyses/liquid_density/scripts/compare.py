"""Issue 83: retained Amundsen density and controlled twelve-state 3C comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'analyses/liquid_density/results/runs/current_pin'
AMINE = Path('/home/tnnrpolley21/Workspaces/Engineering/Amine-Thermodynamics')
BUNDLE = AMINE / 'data/reference/MEA/film_chemistry_inputs/3/input.json'
DATA = ROOT / 'src/mea_absorption_column/data/epcsaft_datasets'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def table(name, rows):
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def properties():
    from mea_absorption_column.Properties.Thermophysical_Properties import density
    from mea_absorption_column.Thermodynamics.reactive_bundle import engine_liquid
    bundle = json.loads(BUNDLE.read_text())
    item = next(v for v in bundle['property_inputs'] if v['input_id'] == 'liquid_density_unloaded')
    source = AMINE / item['data_path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['data_sha256']
    records = [r for r in csv.DictReader(source.open()) if r['property'] == 'density']
    models, failures = {}, {}
    for name, dataset in [('current', 'MEA_greenfield_exploratory'), ('candidate', 'MEA_amine160_density_candidate')]:
        try:
            models[name] = engine_liquid(DATA / dataset, DATA / dataset / 'ideal-gas-thermochemistry.json', name,
                loading_policy={'loading_anchor': .25, 'max_log_loading_step': .1, 'max_loading_steps': 256})
        except Exception as error:
            failures[name] = f'{type(error).__name__}: {error}'
    save('parameter_load_failures.json', failures)
    assert 'current' in models, failures
    rows = []
    for r in records:
        t, w = float(r['temperature_C']) + 273.15, float(r['mea_mass_fraction'])
        alpha = float(r['co2_loading_mol_per_mol_mea'] or 0)
        feed = np.array([alpha * w / .061084, w / .061084, (1-w) / .018015])
        x = feed / feed.sum()
        values = {'correlation': float(density(t, x, 101325)[1]) / 1000}
        for name, model in models.items():
            if alpha == 0:
                state = model.model.state(t, P=101325, x=np.r_[x, np.zeros(6)], phase='liquid')
                rho, amounts = state.molar_density, np.r_[feed, np.zeros(6)]
            else:
                result = np.asarray(model._state([t, 101325, *feed])).ravel()
                amounts, rho = result[:9], result[11]
            volume = amounts.sum() / rho
            if name == 'candidate':
                correction = json.loads((DATA / 'MEA_amine160_density_candidate/parameters.json').read_text())['empirical_density_correction']
                volume += correction['c_cm3_mol'] * 1e-6 * (amounts[3] + amounts[4])
            values[name] = float(amounts @ model.molar_masses / volume / 1000)
        for name, value in values.items():
            error = value - float(r['value'])
            rows.append(dict(model=name, temperature_K=t, mea_mass_fraction=w, loading=alpha,
                observed_g_cm3=float(r['value']), predicted_g_cm3=value, error_g_cm3=error,
                error_percent=100*error/float(r['value']),
                error_over_uncertainty=error/.0005 if alpha == 0 else '',
                uncertainty_g_cm3=.0005 if alpha == 0 else '', source_table=r['source_table_or_figure']))
        print(t, w, alpha, values, flush=True)
    assert sum(r['model'] == 'correlation' and r['loading'] == 0 for r in rows) == 35
    table('density.csv', rows)
    save('inputs.json', {'bundle': str(BUNDLE), 'source': str(source), 'pressure_assumption_Pa': 101325,
        'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [BUNDLE, source, *[DATA / v / 'parameters.json' for v in ('MEA_greenfield_exploratory', 'MEA_amine160_density_candidate')]]},
        'unloaded_basis': 'exact zero CO2, neutral binary pressure root; no positive loading proxy',
        'candidate': 'native state plus candidate-owned ion-volume translation; no thermodynamic fitting'})


def column(source):
    from mea_absorption_column.column import run_column
    request = {'preset': 'twelve_state_conserved',
        'case': {'physical_input_file': 'analyses/bvp_solution_methods/input/case_3c.json'},
        'numerics': {'method': 'upwind', 'solver_settings': {'nodes': 11, 'quadrature_points': 5,
            'end_clustering': 1., 'max_iterations': 20, 'tolerance': 1e-7, 'liquid_density_source': source}},
        'execution': {'wall_limit_s': 1450, 'output_dir': str(OUT / source)}}
    save(source + '_request.json', request)
    result = run_column(request)
    print(result['execution'], result.get('failure'), flush=True)


def summarize():
    from mea_absorption_column.column import _prepare_conserved_column_in_process
    from mea_absorption_column.config.column import resolve_column_config
    summary, profiles = {}, []
    for source in ('solved_state', 'correlation'):
        a = json.loads((OUT / source / 'attempt.json').read_text())
        assert a['numerical_acceptance'] == 'accepted', a.get('failure')
        assert a['physical_certification']['accepted']
        u = np.array(a['native_profile']['state_matrix'])
        z = np.array(a['result']['grid'])
        assembly = _prepare_conserved_column_in_process(resolve_column_config(json.loads((OUT / (source+'_request.json')).read_text())))['assembly']
        h = np.array([np.asarray(assembly['balance'](v[:8], v[8:11])[-1]).ravel() for v in u.T])
        summary[source] = dict(capture_percent=100*(1-u[2,-1]/u[2,0]), peak_liquid_K=float(u[4].max()),
            liquid_flow_m3_s_range=[float(h[:,0].min()), float(h[:,0].max())],
            holdup_range=[float(u[7].min()), float(u[7].max())],
            scaled_residual_inf=a['result']['scaled_residual_inf'], runtime_s=a['execution']['runtime_s'])
        profiles += [dict(source=source, z_m=float(z[i]), liquid_K=float(u[4,i]), vapor_K=float(u[5,i]),
            liquid_flow_m3_s=float(h[i,0]), holdup=float(u[7,i])) for i in range(len(z))]
    n = len(profiles)//2
    summary['max_abs_liquid_temperature_difference_K'] = max(abs(profiles[i]['liquid_K']-profiles[i+n]['liquid_K']) for i in range(n))
    summary['max_abs_vapor_temperature_difference_K'] = max(abs(profiles[i]['vapor_K']-profiles[i+n]['vapor_K']) for i in range(n))
    save('column_summary.json', summary)
    table('profiles.csv', profiles)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['properties', 'solved_state', 'correlation', 'summarize'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'properties':
        properties()
    elif args.stage == 'summarize':
        summarize()
    else:
        column(args.stage)
