"""Configuration checks never evaluate an absorber or import its runtime."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from dataclasses import make_dataclass

import pytest

spec = importlib.util.spec_from_file_location("research_options", Path(__file__).parents[1] / "src/mea_absorption_column/research.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def selection():
    return dict(formulation="seven_state", thermo_model="epcsaft_reactive_nine", film_model="enhancement_factor", method="scipy-bvp")


def test_resolve_preserves_choices_and_input():
    original = selection()
    result = research.resolve_config(original)
    assert result["solver_settings"] == {"co2_mass_transfer_model": "enhancement_factor"}
    assert original == selection()
    for patch in ({"method": "typo"}, {"formulation": "coupled"}, {"thermo_model": "enrtl"}, {"settings": {"typo": 1}}, {"cases": {"c_case_limit": -1}}, {"solver_settings": {"co2_mass_transfer_model": "reactive_film_linearization"}}):
        with pytest.raises(ValueError):
            research.resolve_config(original | patch)


def test_preview_does_not_execute(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.toml"
    path.write_text('\n'.join(f'{key} = "{value}"' for key, value in selection().items()))
    monkeypatch.setattr(research, "run_research", lambda _: pytest.fail("Preview executed"))
    research.main([str(path)])
    assert json.loads(capsys.readouterr().out)["selection"]["method"] == "scipy-bvp"


def test_execution_forwards_and_retains_settings(tmp_path, monkeypatch):
    received = []
    Settings = make_dataclass("Settings", [(key, object) for key in ("methods", "thermo_models", "output_dir", "solver_settings")])
    fake = SimpleNamespace(BenchmarkSettings=Settings, run_benchmark=lambda settings: received.append(settings))
    monkeypatch.setitem(sys.modules, "mea_absorption_column.benchmark", fake)
    output = tmp_path / "new-results"
    config = selection() | {"output_dir": str(output)}
    research.run_research(config)
    assert received[0].thermo_models == ("epcsaft_reactive_nine",)
    assert received[0].solver_settings["co2_mass_transfer_model"] == "enhancement_factor"
    record = json.loads((output / "research_config.json").read_text())
    assert record["benchmark_settings"]["methods"] == ["scipy-bvp"]
    with pytest.raises(FileExistsError):
        research.run_research(config)
    assert len(received) == 1


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


def test_conserved_failure_is_retained(tmp_path, monkeypatch):
    def factory(config):
        assert config["thermo_model"] == "custom"
        raise ValueError("invalid problem")
    monkeypatch.setitem(sys.modules, "study_builder", SimpleNamespace(build=factory))
    output = tmp_path / "failed"
    config = dict(formulation="conserved", thermo_model="custom", film_model="custom", method="central", problem_factory="study_builder:build", output_dir=str(output))
    with pytest.raises(ValueError, match="invalid problem"):
        research.run_research(config)
    assert json.loads((output / "failure.json").read_text())["error"] == "ValueError"
