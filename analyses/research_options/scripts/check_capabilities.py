"""Resolve the option product and retain bounded Case 3C capability attempts."""
import csv
import gzip
import itertools
import json
from pathlib import Path

from mea_absorption_column.research import resolve_config, run_research

ROOT = Path('analyses/research_options/results/runs/capability_20261029')
AXES = {
    'preset': ['seven_state_legacy', 'twelve_state_conserved'],
    'thermo': ['ideal_henry', 'epcsaft_neutral', 'epcsaft_ionic', 'epcsaft_reactive_nine', 'reactive_epcsaft', 'enrtl', 'mdea'],
    'film': ['explicit', 'implicit', 'reactive_film_linearization', 'equilibrium_manifold', 'enhancement_reference', 'differential_finite_rate'],
    'energy': ['empirical_enthalpy', 'native_total_enthalpy', 'isothermal'],
    'pressure': ['constant', 'hydraulic_pressure_drop'],
    'method': ['single', 'scipy-bvp', 'finite', 'trapezoidal', 'central', 'upwind', 'shooting', 'collocation', 'casadi-ipopt'],
    'density': ['default', 'solved_state', 'correlation'],
}


def request(values):
    preset, thermo, film, energy, pressure, method, density = values
    seven = preset == 'seven_state_legacy'
    settings = {}
    if film in ('explicit', 'implicit'):
        settings['enhancement_type'] = film
        film = 'enhancement_factor'
    if density != 'default':
        settings['liquid_density_source'] = density
    if seven and method == 'scipy-bvp':
        settings.update(mesh_points=11, transform_mode='raw', jacobian_mode='numerical')
    if not seven and method in ('trapezoidal', 'central', 'upwind', 'shooting', 'collocation'):
        settings.update(nodes=11)
    return dict(preset=preset,
                case={'source': 'C_cases_data', 'id': '3C'} if seven else
                {'physical_input_file': 'analyses/bvp_solution_methods/input/case_3c.json'},
                model=dict(thermo_model=thermo, film_model=film, energy_model=energy, pressure_model=pressure),
                numerics=dict(method=method, solver_settings=settings),
                execution=dict(process_isolation=True, wall_limit_s=180 if not seven else 60))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    accepted = {}
    counts = {}
    with gzip.open(ROOT/'resolution.csv.gz', 'wt') as stream:
        writer = csv.writer(stream)
        writer.writerow([*AXES, 'resolution', 'reason', 'attempt'])
        for values in itertools.product(*AXES.values()):
            try:
                config = resolve_config(request(values))
                key = config.resolved_config_sha256
                if key not in accepted:
                    accepted[key] = (f'cell_{len(accepted):02d}', request(values))
                label = accepted[key][0]
                outcome, reason = 'resolves', ''
            except ValueError as exc:
                outcome, reason, label = 'refuses', str(exc), ''
            counts[outcome] = counts.get(outcome, 0) + 1
            writer.writerow([*values, outcome, reason, label])
    (ROOT/'resolution_summary.json').write_text(json.dumps(dict(axes=AXES, counts=counts, distinct_attempts=len(accepted)), indent=2)+'\n')
    rows = []
    for label, config in accepted.values():
        output = ROOT/label
        config['execution']['output_dir'] = str(output)
        (ROOT/f'{label}_request.json').write_text(json.dumps(config, indent=2)+'\n')
        print(label, config['model'], config['numerics'], flush=True)
        if (output/'attempt.json').exists():
            result = json.loads((output/'attempt.json').read_text())
        else:
            result = run_research(config)
        capture = result.get('observables', {}).get('capture', {}).get('value')
        profile = result.get('native_profile', {}).get('state_matrix')
        if capture is None and profile is not None and profile[2][0] > 0:
            capture = 100 * (1 - profile[2][-1] / profile[2][0])
        rows.append(dict(label=label, request=config, execution=result['execution'],
                         capture=capture,
                         numerical_acceptance=result.get('numerical_acceptance'),
                         failure=result.get('failure'), diagnostics=result.get('diagnostics')))
        (ROOT/'runs.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(rows[-1]['execution'], rows[-1]['capture'], rows[-1]['failure'], flush=True)


if __name__ == '__main__':
    main()
