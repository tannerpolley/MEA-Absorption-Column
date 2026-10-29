"""MEA ordering invariance and preflight refusal on the retained physical basis."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mea_absorption_column.column import _prepare_conserved_column_in_process, run_column
from mea_absorption_column.config.column import resolve_column_config
from mea_absorption_column.Thermodynamics.solvent_inputs import ROOT, SolventRefusal

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
        assert run_column(request(thermo, transport, seed))['physical_acceptance'] == 'accepted'
        req = request(thermo, transport, output / (label + '_cosine'), seed / 'attempt.json')
        attempt = run_column(req)
        assert attempt['physical_acceptance'] == 'accepted', attempt.get('failure')
        attempts.append(attempt)
        assemblies.append(_prepare_conserved_column_in_process(resolve_column_config(req))['assembly'])
    from runpy import run_path
    compare = run_path(str(ROOT / 'analyses/bvp_solution_methods/scripts/compare_solvent_declarations.py'))['compare']
    compare(attempts, assemblies, read(BASELINE), output)


def test_missing_law_refuses_before_native_construction(tmp_path, monkeypatch):
    t, u = permuted_inputs(tmp_path)
    implicit_case = request(t, u, tmp_path / 'never_run')
    implicit_case.pop('case')
    assert resolve_column_config(implicit_case).case.physical_input_file == str(tmp_path / 'case.json')
    record = read(u)
    del record['property_laws']['gas_conductivity']
    save(u, record)
    monkeypatch.setattr('epcsaft.Mixture', lambda *_a, **_k: pytest.fail('Native construction before preflight'))
    with pytest.raises(SolventRefusal, match='Incomplete property laws'):
        resolve_column_config(request(t, u, tmp_path / 'never_run'))
