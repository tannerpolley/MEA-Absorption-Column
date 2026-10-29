"""Compare retained MEA declarations on the same native states, units and directions."""
import json
from pathlib import Path

import numpy as np

from mea_absorption_column.column import _prepare_conserved_column_in_process, _source_identity
from mea_absorption_column.config.column import resolve_column_config
from mea_absorption_column.Thermodynamics.solvent_inputs import ROOT, StateDomainFailure


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def compare(attempts, assemblies, baseline, output):
    tree = _source_identity()["source_tree_sha256"]
    assert all(a["source_identity"]["source_tree_sha256"] == tree for a in attempts), "Replay source differs from retained solves"
    profiles = [np.array(a['native_profile']['state_matrix']) for a in attempts]
    reference = np.array(baseline['native_profile']['state_matrix'])
    capture = lambda p: 100 * (1 - p[2, -1] / p[2, 0])
    for profile in profiles:
        assert abs(capture(profile) - capture(reference)) <= 1e-6
        np.testing.assert_allclose(profile[:4], reference[:4], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(profile[4:6], reference[4:6], rtol=0, atol=1e-6)
        np.testing.assert_allclose(profile[6], reference[6], rtol=0, atol=1e-3)
        np.testing.assert_allclose(profile[[7, 11]], reference[[7, 11]], rtol=0, atol=1e-8)
        np.testing.assert_allclose(profile[8:11], reference[8:11], rtol=1e-7, atol=1e-10)
    differences, film_differences, raw_differences = [], [], []
    layouts = {}
    for node_index, state in enumerate(reference.T):
        diagnostics = [np.concatenate([np.asarray(v).ravel() for v in a['diagnostics'](state)]) for a in assemblies]
        np.testing.assert_allclose(*diagnostics, rtol=1e-7, atol=1e-10)
        film_differences.append(float(np.max(abs(diagnostics[1]-diagnostics[0]) / np.maximum(1., abs(diagnostics[0])))))
        for fraction in np.linspace(0, 1, 9):
            for phase in ('liquid', 'vapor'):
                phase_values = []
                for assembly, attempt in zip(assemblies, attempts):
                    owner = assembly[phase]
                    feed = attempt['resolved_inputs']['parameters'][0 if phase == 'liquid' else 1]
                    flows = owner.flow(state[0] * np.exp(fraction * state[11]), state[1], feed) if phase == 'liquid' else owner.flow(state[2], state[3], feed)
                    inputs = np.r_[state[4 if phase == 'liquid' else 5], state[6], np.asarray(flows).ravel()]
                    order = sorted(owner.component_ids)
                    q = owner._q
                    amounts = tuple(q.SolvedStateObservable(q.SolvedStateObservableKind.PhaseComponentAmount, 0,
                                    owner.component_ids.index(s)) for s in order)
                    observables = tuple(owner.log_activities[owner.component_ids.index(s)] for s in order) + amounts + (owner.state_observables[-1], owner.enthalpy)
                    axes = [0, 1, *[2 + owner.feed_index[s] for s in sorted(owner.feed_ids)]]
                    labels = ['T_K', 'P_Pa', *sorted(owner.feed_ids)]
                    layouts.setdefault(phase, {'raw_value_ids':
                        ([f'amount:{s}' for s in order] if phase == 'liquid' else []) + list(owner.output_ids[-4:]),
                        'raw_value_units': (['mol'] * len(order) if phase == 'liquid' else []) + ['Pa', 'Pa', 'mol/m3', 'J'],
                        'action_observables': [f'ln_activity:{s}' for s in order] + [f'amount:{s}' for s in order] + ['molar-density', 'molar_enthalpy'],
                        'observable_units': ['1'] * len(order) + ['mol'] * len(order) + ['mol/m3', 'J/mol'],
                        'directions': [[], *[[d] for d in labels], *[[a, d] for a in labels for d in labels]],
                        'input_direction_units': ['K', 'Pa', *['mol'] * len(owner.feed_ids)],
                        'encoding': 'raw values followed by direction-major, observable-minor native actions; derivative units divide observable units by direction units'})
                    values = [owner.actions(inputs, observables)]
                    values += [owner.actions(inputs, observables, (d,)) for d in axes]
                    values += [owner.actions(inputs, observables, (a, d)) for a in axes for d in axes]
                    raw = np.asarray(owner(inputs)).ravel()
                    if phase == 'liquid':
                        raw = np.r_[raw[[owner.component_ids.index(s) for s in order]], raw[len(order):]]
                    phase_values.append(np.r_[raw, np.concatenate(values)])
                scale = np.maximum(1., abs(phase_values[0]))
                error = np.max(abs(phase_values[1] - phase_values[0]) / scale)
                assert error <= 1e-7
                differences.append(float(error))
                raw_differences.append({"node": node_index, "film_fraction": float(fraction), "phase": phase,
                                        "permuted_minus_declared": (phase_values[1]-phase_values[0]).tolist()})
    owner = assemblies[0]['liquid']
    try:
        owner.solve((200., 110900., *attempts[0]['resolved_inputs']['parameters'][0]))
    except StateDomainFailure:
        pass
    else:
        raise AssertionError('Invalid state was accepted')
    save(output / 'comparison.json', {'capture_percent': [capture(p) for p in profiles],
         'baseline_capture_percent': capture(reference), 'source_tree_sha256': tree, 'max_native_scaled_difference': max(differences),
         'max_film_scaled_difference': max(film_differences),
         'native_state_comparisons': len(differences),
         'native_vector_layouts': layouts, 'raw_native_differences': raw_differences,
         'native_species_orders': [list(a['liquid'].component_ids) for a in assemblies],
         'native_feed_orders': [list(a['liquid'].feed_ids) for a in assemblies],
         'max_profile_absolute_difference': abs(profiles[0]-profiles[1]).max(axis=1).tolist(),
         'evidence': 'numerical reproduction and ordering invariance; no blend or physical validation claim'})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    attempts = [read(args.run_directory / (label + '_cosine') / 'attempt.json') for label in ('declared', 'permuted')]
    assert all(a['physical_certification']['accepted'] for a in attempts)
    assemblies = [_prepare_conserved_column_in_process(resolve_column_config(a['config']))['assembly'] for a in attempts]
    baseline = read(ROOT / 'analyses/bvp_solution_methods/results/runs/issue85_current_pin_20261029/cosine_repeat_1/attempt.json')
    compare(attempts, assemblies, baseline, args.run_directory)
    result = read(args.run_directory / 'comparison.json')
    print(json.dumps({k: result[k] for k in ('capture_percent', 'max_native_scaled_difference', 'max_film_scaled_difference', 'native_state_comparisons')}, indent=2))
