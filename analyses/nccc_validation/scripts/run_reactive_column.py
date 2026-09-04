"""Current nine-species nonisothermal Case 3C; retain seed, progress and outputs.

Run under an external timeout with stdout/stderr redirected to a retained log.
The optional CSV is an accepted same-case scaled profile, initialization only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd

from mea_absorption_column.Run_Model import run_model
from mea_absorption_column.Thermodynamics.reactive_bundle import DATASET, parameter_set, reaction_system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--initial-profile', type=Path)
    parser.add_argument('--mesh', type=int, default=21)
    parser.add_argument('--tol', type=float, default=.5)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--kij', help='Exact binary-interaction identity for a sensitivity run')
    choice.add_argument('--reaction', choices=['R4', 'R5'], help='Reaction equilibrium constant to perturb')
    parser.add_argument('--factor', type=float, help='Multiplier for the selected coefficient or K(T)')
    args = parser.parse_args()
    if (args.kij is None and args.reaction is None) != (args.factor is None):
        parser.error('--kij or --reaction requires --factor, and vice versa')
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = Path('src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv')
    data = pd.read_csv(inputs, index_col=0)
    settings = dict(mesh_points=args.mesh, tol=args.tol, bc_tol=.001, max_nodes=1000,
                    thermal_state_mode='temperature', transform_mode='raw',
                    vapor_composition_mode='dry_saturated', gas_flow_basis='reported_dry_mass',
                    return_internal_profile=True, return_profiles=True, verbose=2)
    if args.kij is not None:
        settings['reactive_kij_scale'] = (args.kij, args.factor)
    if args.reaction is not None:
        settings['reactive_reaction_scale'] = (args.reaction, args.factor)
    parameters = parameter_set(str(DATASET), settings.get('reactive_kij_scale'))
    parameter_path = args.output/'evaluated_parameters.json'
    parameter_path.write_text(json.dumps(parameters.to_mapping(), indent=2)+'\n')
    reaction_path = args.output/'evaluated_reactions.json'
    reaction_path.write_text(json.dumps(reaction_system(str(DATASET), settings.get('reactive_reaction_scale')), indent=2)+'\n')
    paths = [inputs, Path(__file__), Path('uv.lock'), Path('integration/epcsaft_contract.json'), parameter_path, reaction_path]
    paths += sorted(Path('src/mea_absorption_column').rglob('*.py'))
    paths += sorted(Path('src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle').glob('*.json'))
    if args.initial_profile:
        paths.append(args.initial_profile)
    identity = dict(command=sys.argv, settings=settings, pid=os.getpid(),
                    evaluated_parameter_fingerprint=parameters.fingerprint,
                    engine_identity=json.loads(Path('integration/epcsaft_contract.json').read_text())['final_identity'],
                    input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    started = time.perf_counter()
    if args.initial_profile:
        seed = pd.read_csv(args.initial_profile).to_numpy().T
    else:
        seed_result = run_model(data, method='scipy-bvp', data_type='mass', run=data.index.get_loc('3C'),
                                thermo_model='ideal_henry', solver_settings=settings,
                                return_details=True, staged_beds=False)
        if not seed_result['success']:
            raise RuntimeError('Henry initialization did not converge: '+seed_result['message'])
        seed = seed_result['_raw_solution_scaled']
        (args.output/'seed_result.json').write_text(json.dumps(
            {k:v for k,v in seed_result.items() if not k.startswith('_')}, indent=2, default=str)+'\n')
    pd.DataFrame(seed.T).to_csv(args.output/'initial_scaled.csv', index=False)
    print(f'accepted initialization available after {time.perf_counter()-started:.3f}s', flush=True)
    settings.update(initial_guess_scaled=seed, jacobian_mode='native', reactive_reuse_states=True)
    result = run_model(data, method='scipy-bvp', data_type='mass', run=data.index.get_loc('3C'),
                       thermo_model='epcsaft_reactive_nine', solver_settings=settings,
                       return_details=True, staged_beds=False)
    for name, table in result['_profiles'].items():
        table.to_csv(args.output/f'{name}.csv')
    pd.DataFrame(result['_raw_solution_scaled'].T).to_csv(args.output/'solution_scaled.csv', index=False)
    result = {k:v for k,v in result.items() if not k.startswith('_')}
    result['total_wall_including_seed_and_outputs_s'] = time.perf_counter()-started
    (args.output/'result.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps(result, indent=2, default=str), flush=True)
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
