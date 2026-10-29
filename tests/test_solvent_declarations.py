"""MEA ordering invariance and preflight refusal on the retained physical basis."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mea_absorption_column.column import _prepare_conserved_column_in_process, run_column
from mea_absorption_column.config.column import resolve_column_config
from mea_absorption_column.Thermodynamics.solvent_inputs import ROOT, SolventRefusal, StateDomainFailure, NativeEvaluationFailure

INPUT = ROOT / 'analyses/bvp_solution_methods/input/issue85'
BASELINE = ROOT / 'analyses/bvp_solution_methods/results/runs/issue85_current_pin_20261029/cosine_repeat_1/attempt.json'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')
    return {'repository_path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def permuted_inputs(directory):
    t, u = read(INPUT / 'thermodynamics.json'), read(INPUT / 'transport.json')
    species, feeds, gas = t['species_order'], t['liquid_feed_order'], t['vapor_feed_order']
    permutation = [8, 2, 6, 4, 1, 0, 7, 3, 5]
    new_species = [species[i] for i in permutation]
    new_feeds, new_gas = feeds[::-1], gas[::-1]
    parameters = read(ROOT / t['parameters']['repository_path'])
    parameters['components'] = [parameters['components'][i] for i in permutation]
    t['parameters'] = save(directory / 'parameters.json', parameters)
    reactions = read(ROOT / t['reactions']['repository_path'])
    reactions['parameter_sha256'] = t['parameters']['sha256']
    reactions['reactions'].reverse()
    for row in reactions['reactions']:
        row['stoichiometry'] = dict(reversed(list(row['stoichiometry'].items())))
    t['reactions'] = save(directory / 'reactions.json', reactions)
    caloric = read(ROOT / t['calorics']['repository_path'])
    caloric['components'] = dict(reversed(list(caloric['components'].items())))
    t['calorics'] = save(directory / 'calorics.json', caloric)
    vapor = read(ROOT / t['gas_parameters']['repository_path'])
    vapor['components'].reverse()
    t['gas_parameters'] = save(directory / 'gas_parameters.json', vapor)
    t['species_order'] = t['liquid_support'] = new_species
    t['liquid_feed_order'], t['vapor_feed_order'], t['vapor_support'] = new_feeds, new_gas, new_gas
    t['charge_numbers'] = [t['charge_numbers'][i] for i in permutation]
    t['components']['order'].reverse()
    t['components']['rows'] = [[row[i] for i in permutation] for row in t['components']['rows'][::-1]]
    t['formulas'] = dict(reversed(list(t['formulas'].items())))
    case = read(ROOT / u['case']['repository_path'])
    for phase, old, new in [('liquid', feeds, new_feeds), ('vapor', gas, new_gas)]:
        physical = case['physical_inputs']
        physical[phase + '_feed_ids'] = new
        physical[phase + '_feed_mol_s'] = [physical[phase + '_feed_mol_s'][old.index(s)] for s in new]
        key = phase + '_molar_masses_kg_mol'
        case[key] = [case[key][old.index(s)] for s in new]
    u['case'] = save(directory / 'case.json', case)
    u['species_diffusivities'] = {s: u['species_diffusivities'][s] for s in new_species}
    u['stationary_film_components'].reverse()
    thermo = directory / 'thermodynamics.json'
    u['thermodynamic_input'] = thermo.name
    u['thermodynamic_input_sha256'] = save(thermo, t)['sha256']
    save(directory / 'transport.json', u)
    return thermo, directory / 'transport.json'


def request(thermo, transport, output, seed=None):
    case = read(transport)['case']['repository_path']
    return {'preset': 'twelve_state_conserved', 'case': {'physical_input_file': case},
            'dependencies': {'thermodynamics_input': str(thermo), 'transport_input': str(transport)},
            'numerics': {'method': 'upwind', 'solver_settings': {'nodes': 17, 'quadrature_points': 9,
                          'tolerance': 1e-7, 'max_iterations': 20, **({'end_clustering': 1.} if seed else {})}},
            'initialization': {'values': {'retained_profile': str(seed)} if seed else {}},
            'execution': {'output_dir': str(output), 'wall_limit_s': 600}}


def test_mea_declared_reproduction_and_permutation(tmp_path):
    output = Path(os.environ.get('ISSUE85_RUN_ROOT', tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    directory = output / 'permuted_inputs'
    directory.mkdir()
    pairs = [(INPUT / 'thermodynamics.json', INPUT / 'transport.json'), permuted_inputs(directory)]
    attempts, assemblies = [], []
    for label, (thermo, transport) in zip(('declared', 'permuted'), pairs):
        seed = output / (label + '_seed')
        seeded = run_column(request(thermo, transport, seed))
        assert seeded['physical_acceptance'] == seeded['numerical_acceptance'] == 'accepted'
        req = request(thermo, transport, output / (label + '_cosine'), seed / 'attempt.json')
        attempt = run_column(req)
        assert attempt['physical_acceptance'] == attempt['numerical_acceptance'] == 'accepted', attempt.get('failure')
        attempts.append(attempt)
        assemblies.append(_prepare_conserved_column_in_process(resolve_column_config(req))['assembly'])
    from runpy import run_path
    compare = run_path(str(ROOT / 'analyses/bvp_solution_methods/scripts/compare_solvent_declarations.py'))['compare']
    compare(attempts, assemblies, read(BASELINE), output)


@pytest.mark.parametrize('defect, message', [
    ('missing_law', 'Incomplete property laws'), ('hash', 'SHA-256 mismatch'),
    ('policy', 'Only the retained 3C'), ('domain_field', 'unsupported fields in domain'),
    ('dependent_row', 'independent modulo total flux'), ('duplicate_id', 'Duplicate analytical-component'),
    ('quadrature', 'quadrature/density'), ('density', 'quadrature/density')])
def test_declaration_refuses_before_native_construction(tmp_path, monkeypatch, defect, message):
    t, u = permuted_inputs(tmp_path)
    implicit_case = request(t, u, tmp_path / 'never_run')
    implicit_case.pop('case')
    assert resolve_column_config(implicit_case).case.physical_input_file == str(tmp_path / 'case.json')
    thermo, transport = read(t), read(u)
    req = request(t, u, tmp_path / 'never_run')
    if defect == 'missing_law':
        del transport['property_laws']['gas_conductivity']
    elif defect == 'hash':
        thermo['parameters']['sha256'] = '0' * 64
    elif defect == 'policy':
        transport['execution_policy']['id'] = 'invented_policy'
    elif defect == 'domain_field':
        transport['domain']['invented_policy'] = True
    elif defect == 'dependent_row':
        rows, order = thermo['components']['rows'], thermo['components']['order']
        rows[order.index('water')] = rows[order.index('amine:monoethanolamine')][:]
    elif defect == 'duplicate_id':
        thermo['components']['order'].append('co2')
        thermo['components']['rows'].append(thermo['components']['rows'][-1][:])
    else:
        req['numerics']['solver_settings'].update({'quadrature_points': 5} if defect == 'quadrature'
                                                else {'liquid_density_source': 'correlation'})
    transport['thermodynamic_input_sha256'] = save(t, thermo)['sha256']
    save(u, transport)
    monkeypatch.setattr('epcsaft.Mixture', lambda *_a, **_k: pytest.fail('Native construction before preflight'))
    monkeypatch.setattr('mea_absorption_column.column._build_conserved_assembly', lambda *_a: pytest.fail('Assembly before preflight'))
    with pytest.raises(SolventRefusal, match=message):
        run_column(req)


@pytest.mark.parametrize('failure', [StateDomainFailure, NativeEvaluationFailure])
def test_terminal_interface_callback_keeps_failure_class(tmp_path, monkeypatch, failure):
    from mea_absorption_column import benchmark_worker
    from mea_absorption_column.Thermodynamics.casadi_reactive import EnginePhase
    original = EnginePhase.actions
    def forced(self, inputs, *args):
        if self.kind == 'liquid' and float(np.asarray(inputs).ravel()[self.co2_input]) > 2.6:
            raise failure('forced interface evaluation failure')
        return original(self, inputs, *args)
    monkeypatch.setattr(EnginePhase, 'actions', forced)
    directory = Path(os.environ['ISSUE85_RUN_ROOT']) / failure.kind if 'ISSUE85_RUN_ROOT' in os.environ else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    payload, output = directory / 'payload.json', directory / 'worker.json'
    save(payload, {'task': 'conserved_execution', 'output_path': str(output),
                   'config': resolve_column_config(request(INPUT / 'thermodynamics.json', INPUT / 'transport.json', tmp_path / 'unused')).as_dict()})
    assert benchmark_worker.main([str(payload)]) == 0
    result = read(output)
    assert result['failure_kind'] == failure.kind
    assert result['last_checkpoint']['stage'] == 'interface_initialization'
    assert result['last_checkpoint']['evaluation_failure']['kind'] == failure.kind
    assert 'forced interface evaluation failure' in result['message']
