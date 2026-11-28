"""Current nine-species nonisothermal NCCC case; retain seed, progress and outputs.

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
from mea_absorption_column.Thermodynamics.reactive_bundle import (
    BULK_VAPOR, CORRELATION_SOURCES, FIT_A, dataset_input_paths, parameter_document, reaction_system,
    reactive_property_sources,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', type=Path, default=FIT_A,
                        help='Legacy reactive bundle or Engine-native dataset directory')
    parser.add_argument('--inputs', type=Path, default=Path('src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv'))
    parser.add_argument('--case', default='3C')
    parser.add_argument('--initial-profile', type=Path)
    parser.add_argument('--wall-limit', type=int, default=300, help='External timeout limit in seconds')
    for name in CORRELATION_SOURCES:
        parser.add_argument('--'+name.replace('_', '-'), choices=('correlation', 'epcsaft'))
    parser.add_argument('--water-vaporization-enthalpy', type=float, help='Correlation offset in J/mol')

    parser.add_argument('--mesh', type=int, default=21)
    parser.add_argument('--tol', type=float, default=.5)
    parser.add_argument('--verbose', type=int, choices=(0, 2), default=2)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--kij', help='Exact binary-interaction identity (a native coefficient selects the whole pair law)')
    choice.add_argument('--reaction', choices=['R4', 'R5'], help='Reaction equilibrium constant to perturb')
    parser.add_argument('--factor', type=float, help='Multiplier for the selected k_ij or K(T)')
    args = parser.parse_args()
    if (args.kij is None and args.reaction is None) != (args.factor is None):
        parser.error('--kij or --reaction requires --factor, and vice versa')
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.inputs
    data = pd.read_csv(inputs, index_col=0)
    case_index = data.index.get_loc(args.case)
    settings = dict(mesh_points=args.mesh, tol=args.tol, bc_tol=.001, max_nodes=1000,
                    thermal_state_mode='temperature', transform_mode='raw',
                    vapor_composition_mode='dry_saturated', gas_flow_basis='reported_dry_mass',
                    return_internal_profile=True, return_profiles=True, verbose=args.verbose,
                    reactive_dataset=str(args.dataset),
                    **reactive_property_sources(args.dataset, {name:getattr(args, name)
                        for name in CORRELATION_SOURCES if getattr(args, name) is not None}))
    if args.water_vaporization_enthalpy is not None:
        settings['water_vaporization_enthalpy'] = args.water_vaporization_enthalpy
    if args.kij is not None:
        settings['reactive_kij_scale'] = (args.kij, args.factor)
    if args.reaction is not None:
        settings['reactive_reaction_scale'] = (args.reaction, args.factor)
    parameters = parameter_document(settings['reactive_dataset'], settings.get('reactive_kij_scale'))
    parameter_path = args.output/'evaluated_parameters.json'
    parameter_path.write_text(json.dumps(parameters, indent=2)+'\n')
    reaction_path = args.output/'evaluated_reactions.json'
    reaction_path.write_text(json.dumps(reaction_system(settings['reactive_dataset'], settings.get('reactive_reaction_scale')), indent=2)+'\n')
    paths = [inputs, Path('src/mea_absorption_column/data/fitted_coefficients.csv'), Path(__file__), Path('uv.lock'),
             Path('integration/epcsaft_contract.json'), parameter_path, reaction_path]
    paths += sorted(Path('src/mea_absorption_column').rglob('*.py'))
    paths += dataset_input_paths(args.dataset)
    if any(settings[key] == 'epcsaft' for key in CORRELATION_SOURCES):
        paths.append(BULK_VAPOR.relative_to(Path.cwd()))
    if args.initial_profile:
        paths.append(args.initial_profile)
    identity = dict(command=sys.argv, case_id=args.case, settings=settings, pid=os.getpid(),
                    execution=dict(wall_limit_s=args.wall_limit),
                    evaluated_parameter_fingerprint=hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
                    # Wheel only: the contract's reactive_inputs_sha256 names the Engine qualification inputs, not this run's dataset.
                    engine_identity={k:v for k,v in json.loads(Path('integration/epcsaft_contract.json').read_text())['final_identity'].items()
                                     if k != 'reactive_inputs_sha256'},
                    input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    started = time.perf_counter()
    if args.initial_profile:
        seed = pd.read_csv(args.initial_profile).to_numpy().T
    else:
        seed_result = run_model(data, method='scipy-bvp', data_type='mass', run=case_index,
                                thermo_model='ideal_henry', solver_settings=settings,
                                return_details=True, staged_beds=False)
        (args.output/'seed_result.json').write_text(json.dumps(
            {k:v for k,v in seed_result.items() if not k.startswith('_')}, indent=2, default=str)+'\n')
        if not seed_result['success']:
            raise RuntimeError('Henry initialization did not converge: '+seed_result['message'])
        seed = seed_result['_raw_solution_scaled']
    pd.DataFrame(seed.T).to_csv(args.output/'initial_scaled.csv', index=False)
    print(f'accepted initialization available after {time.perf_counter()-started:.3f}s', flush=True)
    settings.update(initial_guess_scaled=seed, jacobian_mode='native', reactive_reuse_states=True)
    result = run_model(data, method='scipy-bvp', data_type='mass', run=case_index,
                       thermo_model='epcsaft_reactive_nine', solver_settings=settings,
                       return_details=True, staged_beds=False)
    for name, table in result['_profiles'].items():
        table.to_csv(args.output/f'{name}.csv')
    pd.DataFrame(result['_raw_solution_scaled'].T).to_csv(args.output/'solution_scaled.csv', index=False)
    result = {k:v for k,v in result.items() if not k.startswith('_')}
    result['total_wall_including_seed_and_outputs_s'] = time.perf_counter()-started
    result['runtime_files_changed'] = [str(p) for p in paths
        if hashlib.sha256(p.read_bytes()).hexdigest() != identity['input_sha256'][str(p)]]
    (args.output/'result.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps(result, indent=2, default=str), flush=True)
    if not result['success'] or result['runtime_files_changed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
