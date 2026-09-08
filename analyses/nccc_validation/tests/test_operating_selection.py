"""An incomplete initial-guess retry must not erase a converged operating point."""
import importlib.util
from pathlib import Path


def test_incomplete_retry_preserves_converged_condition(tmp_path):
    script = Path(__file__).parents[1]/'figures/reactive_operating/scripts/render.py'
    spec = importlib.util.spec_from_file_location('operating_render', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUNS = tmp_path
    ordinary, retry = tmp_path/'LG_high', tmp_path/'LG_high_native_seed'
    ordinary.mkdir()
    retry.mkdir()
    (ordinary/'result.json').write_text('{"success": true}')
    (retry/'identity.json').write_text('{}')
    assert module.select_run('LG_high') == ordinary
    (retry/'result.json').write_text('{"success": false}')
    assert module.select_run('LG_high') == ordinary
    (retry/'result.json').write_text('{"success": true}')
    assert module.select_run('LG_high') == retry
