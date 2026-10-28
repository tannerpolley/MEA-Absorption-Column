"""One ordinary absorber request, with native reference and refusal retention."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("research_options", ROOT / "src/mea_absorption_column/research.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def selection(output=None):
    request = {
        "preset": "seven_state_legacy",
        "case": {"source": "C_cases_data", "id": "3C"},
        "model": {"parameters": {"staged_beds": False}},
        "numerics": {"method": "scipy-bvp", "solver_settings": {
            "mesh_points": 21, "max_nodes": 1000, "tol": .5, "bc_tol": .001,
            "thermal_state_mode": "enthalpy", "transform_mode": "raw",
            "jacobian_mode": "numerical", "return_profiles": True}},
    }
    if output is not None:
        request["execution"] = {"output_dir": str(output)}
    return request


@pytest.mark.parametrize("suffix", ["toml", "json"])
def test_runtime_free_preview_uses_resolved_request(tmp_path, suffix):
    path = tmp_path / f"config.{suffix}"
    path.write_text(json.dumps(selection()) if suffix == "json" else 'preset = "seven_state_legacy"\n[case]\nsource = "C_cases_data"\nid = "3C"\n')
    script = """import runpy, sys
script = sys.argv[2]
sys.argv = ['research.py', sys.argv[1]]
try:
    runpy.run_path(script, run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'mea_absorption_column' not in sys.modules
assert 'epcsaft' not in sys.modules
"""
    command = [sys.executable, "-c", script, str(path), str(ROOT / "src/mea_absorption_column/research.py")]
    preview = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, timeout=20).stdout)
    assert preview["selection"]["model"]["layout"] == "seven_enthalpy"
    assert preview["selection"]["resolved_config_sha256"]


def test_configured_henry_preserves_native_reference(tmp_path):
    reference = json.loads((ROOT / "analyses/research_options/results/configuration/henry_reference.json").read_text())
    python_attempt = research.run_research(selection(tmp_path / "henry"))
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(selection(tmp_path / "command")))
    command = [sys.executable, str(ROOT / "src/mea_absorption_column/research.py"), str(request_file), "--run"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    command_attempt = json.loads(completed.stdout)
    for attempt, output in ((python_attempt, "henry"), (command_attempt, "command")):
        assert attempt == json.loads(json.dumps(attempt, allow_nan=False))
        assert attempt == json.loads((tmp_path / output / "attempt.json").read_text())
        assert attempt["execution"]["status"] == "completed"
        assert attempt["failure"] is None
        assert attempt["resolved_inputs"]["physical_input_sha256"] == reference["physical_inputs"]["physical_input_sha256"]
        assert abs(attempt["observables"]["capture"]["value"] - reference["capture_pct"]) <= 1e-8
        native = attempt["native_profile"]
        assert native["available"]
        np.testing.assert_array_equal(native["grid"]["values"], reference["_native_grid"])
        np.testing.assert_allclose(native["state_matrix_scaled"], reference["_native_state_scaled"], rtol=1e-10, atol=1e-12)
        assert native["layout"] == reference["_native_state_layout"]
        assert attempt["solver"]["stages"] == reference["solver_stage_status"]
        np.testing.assert_allclose(native["sampled_profiles"]["T"]["data"], reference["sampled_temperature"]["data"], rtol=0, atol=1e-8)
    with pytest.raises(FileExistsError):
        research.run_research(selection(tmp_path / "henry"))
    refused = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert refused.returncode == 2
    error = json.loads(refused.stdout)["error"]
    assert error["kind"] == "FileExistsError"
    assert "Refusing overwrite existing output directory" in error["cause"]


def test_selected_retired_adapter_retains_refusal_before_outer_solver(tmp_path, monkeypatch):
    import mea_absorption_column.benchmark as benchmark
    monkeypatch.setattr(benchmark, "_run_one_case", lambda *args: pytest.fail("Unavailable adapter reached outer solver"))
    request = selection(tmp_path / "unavailable")
    request["model"]["thermo_model"] = "epcsaft_reactive_nine"
    attempt = research.run_research(request)
    assert attempt["failure"]["kind"] == "capability_refusal"
    assert "ModuleNotFoundError: No module named 'epcsaft.records'" in attempt["failure"]["cause"]
    assert attempt["solver"]["stages"] == {}
    assert attempt["observables"] == {}
    assert not attempt["native_profile"]["available"]
    assert attempt["result"] is None
    assert json.loads((tmp_path / "unavailable/attempt.json").read_text()) == attempt
    request["execution"]["output_dir"] = str(tmp_path / "command_unavailable")
    request_file = tmp_path / "unavailable.json"
    request_file.write_text(json.dumps(request))
    command = subprocess.run([sys.executable, str(ROOT / "src/mea_absorption_column/research.py"), str(request_file), "--run"], capture_output=True, text=True, timeout=20)
    assert command.returncode == 1
    returned = json.loads(command.stdout)
    assert returned == json.loads((tmp_path / "command_unavailable/attempt.json").read_text())
    assert returned["failure"]["kind"] == attempt["failure"]["kind"]
    assert returned["failure"]["cause"] == attempt["failure"]["cause"]
    assert returned["solver"]["stages"] == {}
    assert returned["observables"] == {}
    assert returned["result"] is None
    assert not returned["native_profile"]["available"]


def test_command_request_errors_do_not_fabricate_attempts(tmp_path):
    request_file = tmp_path / "request.json"
    malformed_method = selection()
    malformed_method["numerics"]["method"] = []
    for request, kind in (("{invalid", "JSONDecodeError"), (json.dumps({"unexpected": 1}), "ConfigurationError"), (json.dumps(selection()), "ValueError"), (json.dumps(malformed_method), "TypeError")):
        request_file.write_text(request)
        completed = subprocess.run([sys.executable, str(ROOT / "src/mea_absorption_column/research.py"), str(request_file), "--run"], capture_output=True, text=True, timeout=20)
        assert completed.returncode == 2
        returned = json.loads(completed.stdout)
        assert set(returned) == {"error"}
        assert returned["error"]["kind"] == kind
        assert returned["error"]["cause"]
        if kind == "TypeError":
            assert returned["error"]["cause"] == "unhashable type: 'list'"
        assert not (tmp_path / "attempt.json").exists()


def test_physical_choices_and_chemistry_cannot_conflict():
    request = selection()
    request["model"]["thermo_model"] = "epcsaft_reactive_nine"
    request["model"]["parameters"]["chemical_equilibrium_model"] = "legacy"
    with pytest.raises(ValueError, match="chemistry.*conflict"):
        research.resolve_config(request)
    request = selection()
    request["numerics"]["solver_settings"]["mass_transfer_factor"] = 2
    with pytest.raises(ValueError, match="Physical choices belong in model.parameters"):
        research.resolve_config(request)
    with pytest.raises(ValueError, match="Unknown column configuration"):
        research.resolve_config({"problem_factory": "custom:build"})


def test_conserved_routes_exact_kwargs(monkeypatch):
    received = []
    def collocate(*, node, scheme):
        received.append((node, scheme))
        return {"accepted": False}
    def reduced(*, model, method):
        received.append((model, method))
        return {"accepted": False}
    monkeypatch.setitem(sys.modules, "mea_absorption_column.BVP.Methods.Casadi_Collocation", SimpleNamespace(solve_conservative_collocation=collocate))
    monkeypatch.setitem(sys.modules, "mea_absorption_column.BVP.Methods.Conserved_Reduction", SimpleNamespace(solve_reduced_bvp=reduced))
    for method in ("trapezoidal", "central"):
        research.run_conserved({"node": "explicit"}, method)
    for method in ("shooting", "collocation"):
        research.run_conserved({"model": "explicit"}, method)
    assert received == [("explicit", name) for name in ("trapezoidal", "central", "shooting", "collocation")]
    with pytest.raises(TypeError):
        research.run_conserved({"typo": 1}, "central")
