"""Reduce the two current-pin MEA repeats; no thermodynamic or column calculation."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / 'analyses/bvp_solution_methods/results/runs/issue85_current_pin_20261029'


def main():
    records = []
    profiles = []
    for label in ('seed_uniform_n17', 'cosine_repeat_1', 'cosine_repeat_2'):
        path = RUNS / label / 'attempt.json'
        attempt = json.loads(path.read_text())
        assert attempt['numerical_acceptance'] == 'accepted', label
        certificate = attempt['physical_certification']
        assert certificate['accepted'] is True, label
        profile = np.array(attempt['native_profile']['state_matrix'], dtype=float)
        records.append({
            'label': label, 'attempt_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'attempt_id': attempt['attempt_id'], 'source_head': attempt['source_identity']['commit'],
            'engine_commit': attempt['engine']['expected_commit'],
            'engine_wheel_sha256': attempt['engine']['actual_sha256'],
            'capture_percent': float(100 * (1 - profile[2, -1] / profile[2, 0])),
            'runtime_s': attempt['execution']['runtime_s'],
            'iterations': attempt['result']['iterations'],
            'scaled_equation_residual_inf': attempt['result']['scaled_residual_inf'],
            'scaled_certificate': certificate['scaled_residual_inf'],
            'initialization': attempt['config']['initialization'],
        })
        profiles.append(profile)
    assert records[1]['engine_commit'] == records[2]['engine_commit']
    delta = np.abs(profiles[2] - profiles[1])
    comparison = {
        'records': records,
        'cosine_profiles_bit_identical': bool(np.array_equal(profiles[1], profiles[2])),
        'capture_repeat_difference_percentage_points': abs(records[2]['capture_percent'] - records[1]['capture_percent']),
        'max_abs_repeat_difference_by_state': delta.max(axis=1).tolist(),
        'state_units': ['mol/s'] * 4 + ['K', 'K', 'Pa', '1', 'mol/(m2 s)', 'mol/(m2 s)', 'W/m2', '1'],
        'proposed_capture_tolerance_percentage_points': max(1e-6, 10 * abs(records[2]['capture_percent'] - records[1]['capture_percent'])),
        'evidence_type': 'deterministic numerical reproduction, not grid convergence or physical validation',
    }
    (RUNS / 'repeat_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
