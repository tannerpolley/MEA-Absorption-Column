"""Two-amine column admission and active equation checks, #85 stage 3."""
import json
import os
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from mea_absorption_column.column import _run_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config
from mea_absorption_column.Thermodynamics.solvent_inputs import SolventRefusal, StateDomainFailure
from mea_absorption_column.Thermodynamics.manufactured_neutral import ManufacturedColumnPhase, MathematicalActionUnavailable

SCRIPT = Path(__file__).resolve().parents[1]/'analyses/bvp_solution_methods/scripts/verify_manufactured_column.py'


def test_two_amine_preparation_and_active_equations(tmp_path):
    verify = run_path(str(SCRIPT))
    for label, permuted in [('declared',False),('permuted',True)]:
        verify['prepared_check'](tmp_path/label,permuted)


@pytest.mark.parametrize('defect',['missing_diffusivity','dependent_row','mea_law'])
def test_mathematical_input_refuses_before_assembly(tmp_path,monkeypatch,defect):
    verify = run_path(str(SCRIPT))
    _, u = verify['inputs'](tmp_path)
    if defect == 'missing_diffusivity':
        del u['species_diffusivities']['B']
    elif defect == 'dependent_row':
        u['stationary_film_components'][1] = u['stationary_film_components'][0]
    else:
        u['closures'] = {'selection':'viscosity:liquid'}
    verify['save'](tmp_path/'transport.json',u)
    monkeypatch.setattr(ManufacturedColumnPhase,'__init__',lambda *_: pytest.fail('Phase before refusal'))
    with pytest.raises(SolventRefusal):
        resolve_column_config(verify['request'](tmp_path))


@pytest.mark.parametrize('failure',[StateDomainFailure,MathematicalActionUnavailable])
def test_mathematical_action_failure_preserves_cause(tmp_path,monkeypatch,failure):
    verify = run_path(str(SCRIPT))
    verify['inputs'](tmp_path)
    config = resolve_column_config(verify['request'](tmp_path))
    original = ManufacturedColumnPhase.actions
    def fail(self,inputs,*args):
        if self.kind == 'vapor':
            raise failure('deliberately unavailable mathematical action')
        return original(self,inputs,*args)
    monkeypatch.setattr(ManufacturedColumnPhase,'actions',fail)
    checkpoint = {}
    with pytest.raises(RuntimeError):
        _run_conserved_column_in_process(config,checkpoint)
    assert checkpoint['evaluation_failure']['kind'] == failure.kind
    assert 'deliberately unavailable' in checkpoint['evaluation_failure']['message']
    assert 'last_accepted_native_state' not in checkpoint['evaluation_failure']
    assert not checkpoint['native_calls']


def test_invalid_evaluated_feed_keeps_domain_failure_in_worker(tmp_path,monkeypatch):
    from mea_absorption_column import benchmark_worker
    from mea_absorption_column.BVP.Methods import Casadi_Collocation
    verify = run_path(str(SCRIPT))
    directory = Path(os.environ.get('ISSUE85_DOMAIN_RUN_ROOT',tmp_path))
    directory.mkdir(parents=True,exist_ok=True)
    verify['inputs'](directory/'inputs')
    config = resolve_column_config(verify['request'](directory/'inputs'))
    def invalid_node(node,boundary,grid,initial,*args,**kwargs):
        state = initial[:,0].copy()
        state[0] = 0.
        return node(0.,state)
    monkeypatch.setattr(Casadi_Collocation,'solve_conservative_collocation',invalid_node)
    output, payload = directory/'worker.json',directory/'payload.json'
    verify['save'](payload,dict(task='conserved_execution',output_path=str(output),config=config.as_dict()))
    assert benchmark_worker.main([str(payload)]) == 0
    result = json.loads(output.read_text())
    assert result['failure_kind'] == 'state_domain_failure'
    cause = result['last_checkpoint']['evaluation_failure']
    assert cause['kind'] == 'state_domain_failure'
    assert cause['inputs']['feed_mol_s']['C'] == 0.
    assert cause['inputs']['temperature_k'] == 320.
    assert cause['inputs']['pressure_pa'] == 100000.
    assert 'mathematical_liquid' in cause['message'] and 'feed [mol/s]' in cause['message']
