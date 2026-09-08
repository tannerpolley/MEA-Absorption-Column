"""One damped Newton guess from retained two-node trapezoidal linearization."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('node_record', type=Path)
    parser.add_argument('recovery_record', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    node, recovery = (json.loads(p.read_text()) for p in (args.node_record, args.recovery_record))
    identity = recovery['measurements']['context']['software']['engine']
    assert node['engine_identity'] == identity and recovery['initialization']['accepted']
    np.testing.assert_array_equal(node['point'], recovery['initialization']['state'])
    jacobian = np.asarray(node['jacobian'], dtype=float)
    assert jacobian.shape == (19, 12) and np.all(np.isfinite(jacobian))
    bu, ru, au = jacobian[:7], jacobian[7:14], jacobian[14:]
    inputs, scales = recovery['problem']['physical_inputs'], recovery['settings']
    height = inputs['height_m']
    su = np.tile(scales['state_scale'], 2)
    sr = np.r_[np.array(scales['balance_scale'])*height, np.tile(scales['algebraic_scale'], 2), scales['boundary_scale']]
    boundary = np.zeros((7, 24))
    for row, (index, end) in enumerate([(0,1), (1,1), (2,0), (3,0), (4,1), (5,0), (6,0)]):
        boundary[row, index+12*end] = 1.
    matrix = np.vstack([np.hstack([-bu-height/2*ru, bu-height/2*ru]),
        np.hstack([au, np.zeros((5,12))]), np.hstack([np.zeros((5,12)), au]), boundary])*su/sr[:,None]
    source = np.array(recovery['local_reduction']['rhs'])*np.array(scales['balance_scale'])*height
    residual = np.r_[-height*source, np.tile(recovery['initialization']['algebraic_residual'], 2), np.zeros(7)]/sr
    step = np.linalg.solve(matrix, -residual)
    condition = float(np.linalg.cond(matrix))
    assert np.isfinite(condition) and condition < 1e12
    assert np.max(abs(matrix@step+residual)) < 1e-9
    initial = np.tile(node['point'], 2)
    lower, upper = (np.tile(np.asarray(scales[k], dtype=float), 2) for k in ('lower', 'upper'))
    fraction = 1.
    while np.any(initial+fraction*su*step <= lower) or np.any(initial+fraction*su*step >= upper):
        fraction *= .5
        if fraction < 1/1024:
            raise RuntimeError('No useful bounded predictor step')
    result = dict(physical_inputs=inputs, engine_identity=identity, grid=[0., height],
        profile=(initial+fraction*su*step).reshape(2,12).T.tolist(),
        initial_guess_only=True, physical_certification={'accepted': None},
        scope='A bounded outer Newton predictor, not a solved column; no native evaluation at the predicted states has been performed',
        damping_fraction=fraction, scaled_condition=condition,
        linear_residual_inf=float(np.max(abs(matrix@step+residual))),
        initial_scaled_residual=residual.tolist(), full_undamped_profile=(initial+su*step).reshape(2,12).T.tolist(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.node_record, args.recovery_record, Path(__file__))})
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('damping_fraction', 'scaled_condition', 'linear_residual_inf')}))


if __name__ == '__main__':
    main()
