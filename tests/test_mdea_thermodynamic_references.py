"""Mixed source pressures and explicit species order through the column constructor."""
import runpy
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('permuted', [False, True])
def test_mdea_frozen_state_distinguishes_reference_pressures(tmp_path, permuted):
    source = ROOT/'analyses/bvp_solution_methods/results/runs/issue85_stage4_20261030/inputs'
    shutil.copytree(source,tmp_path/'inputs')
    verifier = runpy.run_path(str(ROOT/'analyses/bvp_solution_methods/scripts/verify_mdea_thermodynamics.py'))
    result = verifier['verify'](tmp_path,'reference','B',permuted)
    assert result['variants']['mixed']['maximum_difference'] <= 1e-9
    assert all(result['variants'][mode]['maximum_difference'] > 1e-8 for mode in ('fixed','system'))
    expected = [500000.,500000.,500000.,100000.][::(-1 if permuted else 1)]
    assert result['variants']['mixed']['reference_pressures_pa'] == expected
