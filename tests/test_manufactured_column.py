"""Two-amine column admission and active equation checks, #85 stage 3."""
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
