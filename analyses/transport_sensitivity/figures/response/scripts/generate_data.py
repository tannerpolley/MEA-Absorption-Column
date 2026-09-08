"""Case 3C transport screen using the manuscript checkout read-only.

Each process runs one independently converged column. Experiment-local patches
scale shared numeric/CasADi expressions; they never edit the manuscript model.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def transport_scale(quantity, factor):
    """Preserve all dependencies; scale kL before Hatta/enhancement evaluation."""
    if quantity not in ('viscosity', 'diffusivity', 'kl') or not 0 < factor < float('inf'):
        raise ValueError('Expected a transport quantity and finite positive multiplier')
    rhs = importlib.import_module('mea_absorption_column.BVP.ABS_Column')
    jac = importlib.import_module('mea_absorption_column.BVP.reactive_jacobian')
    transfer = importlib.import_module('mea_absorption_column.Transport.Transfer_Coefficients')
    calls = {'numeric': 0, 'symbolic': 0}
    if quantity == 'kl':
        original = transfer.liquid_mass_transfer_expression

        def scaled(*args, **kwargs):
            value = original(*args, **kwargs)
            calls['symbolic' if type(value).__module__.startswith('casadi') else 'numeric'] += 1
            return factor * value

        targets = [(transfer, 'liquid_mass_transfer_expression'), (jac, 'liquid_mass_transfer_expression')]
    else:
        original = getattr(rhs, quantity)

        def scaled(*args, **kwargs):
            values = original(*args, **kwargs)
            phase_index = 4 if quantity == 'viscosity' else 5
            phase = kwargs.get('phase', args[phase_index] if len(args) > phase_index else 'liquid')
            if phase != 'liquid':
                return values
            calls['symbolic' if type(values[0]).__module__.startswith('casadi') else 'numeric'] += 1
            return (factor * values[0], *values[1:])

        targets = [(rhs, quantity), (jac, quantity)]
    with ExitStack() as stack:
        for module, name in targets:
            stack.enter_context(patch.object(module, name, scaled))
        yield calls


def self_check():
    """Unit parity, phase isolation and exact scaling of transport gradients."""
    import casadi as ca
    import numpy as np
    rhs = importlib.import_module('mea_absorption_column.BVP.ABS_Column')
    jac = importlib.import_module('mea_absorption_column.BVP.reactive_jacobian')
    tc = importlib.import_module('mea_absorption_column.Transport.Transfer_Coefficients')
    t = ca.SX.sym('temperature')
    x, y = [.03, .09, .88], [.08, .07, .76, .09]
    for name in ('viscosity', 'diffusivity', 'kl'):
        args = {'viscosity': (t, x, .25, .70),
                'diffusivity': (t, x, 101325., .003, 43000.),
                'kl': (t*1e-11, .003, 1030., .005, (250., .95, 1., 1., 1., 1., 1.))}[name]
        key = 'liquid_mass_transfer_expression' if name == 'kl' else name
        original = getattr(tc if name == 'kl' else rhs, key)
        ref = original(*args)
        ref = ref if name == 'kl' else ca.vertcat(*ref)
        original_f = ca.Function('reference', [t], [ref, ca.jacobian(ref, t)])
        for factor in (1., .9, 1.1):
            with transport_scale(name, factor):
                value = getattr(jac, key)(*args)
                value = value if name == 'kl' else ca.vertcat(*value)
                function = ca.Function('scaled', [t], [value, ca.jacobian(value, t)])
                weights = np.ones(ref.shape)
                weights[0] = factor
                for actual, expected in zip(function(330.), original_f(330.)):
                    np.testing.assert_allclose(actual, np.asarray(expected)*weights, rtol=2e-14, atol=0)
                numeric_args = (330., *args[1:])
                if name != 'kl':
                    numeric = getattr(rhs, key)(*numeric_args)
                    np.testing.assert_allclose(numeric, np.asarray(function(330.)[0]).ravel(), rtol=2e-14)
                    vapor_args = (330., y, .25, .70, 'vapor') if name == 'viscosity' else (330., y, 101325., .003, 43000., 'vapor')
                    actual, expected = getattr(rhs, key)(*vapor_args), original(*vapor_args)
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)
    print('Transport expression value/gradient scaling and vapor isolation passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quantity', choices=['viscosity', 'diffusivity', 'kl'], default='viscosity')
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root, reference = args.model_root.resolve(), args.reference.resolve()
    sys.path.insert(0, str(root/'src'))
    import numpy as np
    import pandas as pd
    from mea_absorption_column.Run_Model import run_model
    from mea_absorption_column.Thermodynamics.reactive_bundle import DATASET, parameter_set, reaction_system
    assert Path(importlib.import_module('mea_absorption_column').__file__).is_relative_to(root)
    self_check()
    if args.check_only:
        return
    if args.output is None:
        parser.error('--output is required for a column run')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = root/'src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv'
    prior = json.loads((reference/'identity.json').read_text())
    contract = json.loads((root/'integration/epcsaft_contract.json').read_text())['final_identity']
    assert prior['engine_identity'] == contract, 'Reference and current wheel/bundle declarations differ'
    assert sha(inputs) == prior['input_sha256'][str(inputs.relative_to(root))]
    for name, digest in contract['reactive_inputs_sha256'].items():
        assert sha(DATASET/name) == digest, f'Bundle changed: {name}'
    settings = dict(prior['settings'])
    assert not any(k.startswith('reactive_') for k in settings), 'Expected unperturbed reference'
    paths = sorted((root/'src/mea_absorption_column').rglob('*.py'))
    paths += sorted(DATASET.glob('*.json'))
    paths += [inputs, root/'uv.lock', root/'integration/epcsaft_contract.json', Path(__file__).resolve(),
              reference/'identity.json', reference/'result.json', reference/'solution_scaled.csv']
    hashes = {str(p): sha(p) for p in paths}
    changed = [str(p.relative_to(root)) for p in paths if p.is_relative_to(root)
               and str(p.relative_to(root)) in prior['input_sha256']
               and sha(p) != prior['input_sha256'][str(p.relative_to(root))]]
    identity = dict(model_root=str(root), settings=settings.copy(), engine_identity=contract,
                    quantity=args.quantity, factor=args.factor, input_sha256=hashes,
                    source_changes_since_reference=changed,
                    parameter_fingerprint=parameter_set(str(DATASET)).fingerprint,
                    command=sys.argv, scope='OAT exploratory ±10% transport correlation sensitivity, not confidence intervals')
    (output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    (output/'evaluated_parameters.json').write_text(json.dumps(parameter_set(str(DATASET)).to_mapping(), indent=2)+'\n')
    (output/'evaluated_reactions.json').write_text(json.dumps(reaction_system(str(DATASET)), indent=2)+'\n')
    seed = pd.read_csv(reference/'solution_scaled.csv').to_numpy().T
    settings.update(initial_guess_scaled=seed, jacobian_mode='native', reactive_reuse_states=True)
    started = time.perf_counter()
    with transport_scale(args.quantity, args.factor) as calls:
        data = pd.read_csv(inputs, index_col=0)
        result = run_model(data, method='scipy-bvp', data_type='mass', run=data.index.get_loc('3C'),
                           thermo_model='epcsaft_reactive_nine', solver_settings=settings,
                           return_details=True, staged_beds=False)
    for name, table in result['_profiles'].items():
        table.to_csv(output/f'{name}.csv')
    pd.DataFrame(result['_raw_solution_scaled'].T).to_csv(output/'solution_scaled.csv', index=False)
    result = {k:v for k,v in result.items() if not k.startswith('_')}
    result['total_wall_including_seed_and_outputs_s'] = time.perf_counter()-started
    result['transport_expression_calls'] = calls
    result['source_unchanged_during_run'] = all(sha(p) == hashes[str(p)] for p in paths)
    (output/'result.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps({k:result.get(k) for k in ['success', 'capture_pct', 'max_rms_residual',
         'runtime_s', 'transport_expression_calls', 'source_unchanged_during_run']}, indent=2), flush=True)
    assert result['success'] and result['source_unchanged_during_run']
    assert min(calls.values()) > 0, 'Multiplier did not reach both RHS and exact Jacobian'
    assert np.isfinite(result['capture_pct'])


if __name__ == '__main__':
    main()
