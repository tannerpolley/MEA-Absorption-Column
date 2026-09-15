from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.metadata
import json
import subprocess
from pathlib import Path

import pytest

from mea_absorption_column import run_column
import mea_absorption_column.benchmark as benchmark
import mea_absorption_column.column as column_runner
import mea_absorption_column.config.column as column_config
from mea_absorption_column.config.column import CapabilityRefusal, ConfigurationError, default_engine, resolve_column_config, verify_worker_identity


def test_new_preset_is_immutable_and_does_not_mutate_request():
    request = {
        "preset": "seven_state_legacy",
        "case": {"source": "C_cases_data", "id": "3C"},
        "numerics": {"method": "scipy-bvp", "solver_settings": {"mesh_points": 21}},
    }
    config = resolve_column_config(request)
    assert request["numerics"]["solver_settings"] == {"mesh_points": 21}
    assert config.numerics.as_dict()["solver_settings"]["mesh_points"] == 21
    assert config.numerics.as_dict()["solver_settings"]["max_nodes"] == 1000
    with pytest.raises(FrozenInstanceError):
        config.preset = "changed"


def test_twelve_state_resolves_its_declared_native_dependencies():
    config = resolve_column_config({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "numerics": {"method": "trapezoidal", "nodes": 11},
    })
    assert config.model.layout == "twelve_conserved"
    assert config.dependencies.mobility_law == "harmonic_mean_onsager_v1"
    assert config.dependencies.references[-1].endswith("MEA_neutral_vapor/reference-thermochemistry.json")
    assert config.engine.required is True


def test_twelve_state_selects_only_implemented_film_closures():
    request = {
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
    }
    manifold = resolve_column_config(request)
    reference = resolve_column_config({
        **request,
        "model": {"film_model": "enhancement_reference"},
    })
    assert manifold.model.film_model == "equilibrium_manifold"
    assert manifold.dependencies.mobility_law == "harmonic_mean_onsager_v1"
    assert reference.model.film_model == "enhancement_reference"
    assert reference.dependencies.mobility_law is None
    assert reference.resolved_config_sha256 != manifold.resolved_config_sha256
    with pytest.raises(ConfigurationError, match="equilibrium_manifold.*enhancement_reference"):
        resolve_column_config({
            **request,
            "model": {"film_model": "differential_finite_rate"},
        })


@pytest.mark.parametrize("method", ["shooting", "collocation"])
def test_twelve_state_reduced_methods_remain_explicitly_unavailable(method):
    config = resolve_column_config({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "numerics": {"method": method},
    })
    with pytest.raises(CapabilityRefusal, match="reduced-method controls have not been migrated"):
        column_runner._run_conserved_column_in_process(config, {})


def test_twelve_state_run_column_retains_conserved_execution(monkeypatch):
    monkeypatch.setattr(column_runner, "execute_conserved_column", lambda config: {
        "resolved_inputs": {"case_id": "3C"}, "engine": {}, "assets": {}, "capabilities": {},
        "layout": {"states": 12, "conserved_balances": 7, "algebraic_equations": 5,
                   "boundary_equations": 7, "coordinate": "physical_height"},
        "result": {"accepted": False, "status": "maximum iterations", "profile": None,
                   "solver_statistics": {"success": True}},
        "physical_certification": {"accepted": False},
        "initialization": {"accepted": True}, "scaling": {"state_scale": [1]},
        "initial_profile": [[1]], "native_calls": {"liquid_value_A1": {"started": 1}},
    })
    result = run_column({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
    })
    assert result["execution"]["status"] == "completed"
    assert result["assembly"] == {
        "states": 12,
        "conserved_balances": 7,
        "algebraic_equations": 5,
        "boundary_equations": 7,
        "coordinate": "physical_height",
    }
    assert result["numerical_acceptance"] == "rejected"
    assert result["solver"]["stages"]["outer"]["success"] is True
    assert result["initialization"]["accepted"] is True
    assert result["native_calls"]["liquid_value_A1"]["started"] == 1


def test_conserved_preparation_rejects_wrong_preset_mapping():
    with pytest.raises(ConfigurationError, match="twelve_state_conserved"):
        column_runner.prepare_conserved_column({
            "preset": "seven_state_legacy",
            "case": {"source": "C_cases_data", "id": "3C"},
        })


def test_physical_input_only_resolves_real_source_lineage():
    config = resolve_column_config({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
    })
    expected = str((column_runner._ROOT / "analyses/bvp_solution_methods/input/case_3c.json").resolve())
    assert config.case.source == expected
    result = column_runner.prepare_conserved_column(config)
    assert result["source"] == expected
    assert result["resolved_inputs"]["source_lineage"]["source"] == expected


def test_conserved_preparation_launches_configured_interpreter(monkeypatch):
    engine = default_engine(required=True)
    called = {}

    def fake_run(command, **kwargs):
        called["command"] = command
        return column_runner.subprocess.CompletedProcess(command, 1, "", "worker unavailable")

    monkeypatch.setattr(column_runner.subprocess, "run", fake_run)
    request = {
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "execution": {"process_isolation": True},
        "engine": {
            "wheel": engine.wheel,
            "sha256": engine.sha256,
            "commit": engine.commit,
            "python": "/usr/bin/python3",
        },
    }
    with pytest.raises(CapabilityRefusal, match="worker failed"):
        column_runner.prepare_conserved_column(request)
    assert called["command"][0] == "/usr/bin/python3"


def test_conserved_case_policy_retains_case_values_and_explicit_loading_overrides():
    result = column_runner.prepare_conserved_column({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "numerics": {"solver_settings": {
            "reactive_loading_anchor": .3,
            "reactive_max_log_loading_step": .2,
            "reactive_max_loading_steps": 16,
        }},
    })
    policy = result["resolved_inputs"]["conserved_policy"]
    assert policy["loading_anchor"] == .3
    assert policy["max_log_loading_step"] == .2
    assert policy["max_loading_steps"] == 16
    assert len(policy["species_diffusivity_model"]["other_species_m2_s"]) == 8


def test_conserved_initialization_loading_overrides_reach_resolved_policy():
    result = column_runner.prepare_conserved_column({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "initialization": {"values": {
            "loading_anchor": .123,
            "max_log_loading_step": .2,
            "max_loading_steps": 12,
        }},
    })
    policy = result["resolved_inputs"]["conserved_policy"]
    assert policy["loading_anchor"] == .123
    assert policy["max_log_loading_step"] == .2
    assert policy["max_loading_steps"] == 12


def test_conserved_initialization_interface_bracket_is_refused_before_assembly():
    with pytest.raises(ConfigurationError, match="interface_bracket is not consumed"):
        column_runner.prepare_conserved_column({
            "preset": "twelve_state_conserved",
            "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
            "initialization": {"values": {"interface_bracket": [.1, .5]}},
        })


def test_conserved_preparation_timeout_retains_timed_out_execution(monkeypatch, tmp_path):
    original_run = column_runner.subprocess.run
    def timeout(*args, **kwargs):
        command = args[0] if args else kwargs.get("args", "worker")
        if isinstance(command, (list, tuple)) and command and command[0] == "git":
            return original_run(*args, **kwargs)
        payload = json.loads(Path(command[-1]).read_text())
        output = Path(payload["output_path"])
        column_runner._record(output.parent, output.name, {
            "last_checkpoint": {"stage": "global_solve", "result": {"profile": [[1.0, 2.0]]}}
        })
        (output.parent / f".{output.name}.tmp").write_text('{"incomplete":')
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(column_runner.subprocess, "run", timeout)
    result = column_runner.run_column({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "execution": {"wall_limit_s": 1, "output_dir": str(tmp_path / "attempt")},
    })
    assert result["execution"]["status"] == "timed_out"
    assert result["failure"]["kind"] == "timed_out"
    retained = json.loads((tmp_path / "attempt" / "attempt.json").read_text())
    assert retained["failure"]["phase"] == "global_solve"
    assert retained["failure"]["last_checkpoint"]["result"]["profile"] == [[1.0, 2.0]]


@pytest.mark.parametrize("case, expected", [
    ("negative", "nonzero_exit"), ("timeout_final", "timed_out"), ("checkpoint", "incomplete"),
    ("malformed", "output_invalid"), ("missing", "output_invalid"),
    ("wrong_shape", "output_invalid"), ("launch", "launch_failed"),
    ("interrupt", "interrupted"),
])
def test_conserved_termination_classifies_transport_and_output(monkeypatch, tmp_path, case, expected):
    original_run = column_runner.subprocess.run

    def fake_run(command, **kwargs):
        if isinstance(command, (list, tuple)) and command and command[0] == "git":
            return original_run(command, **kwargs)
        payload = json.loads(Path(command[-1]).read_text())
        output = Path(payload["output_path"])
        if case == "negative":
            column_runner._record(output.parent, output.name, {"last_checkpoint": {"stage": "global_solve"}})
            return subprocess.CompletedProcess(command, 7, "worker out", "worker err")
        if case == "checkpoint":
            column_runner._record(output.parent, output.name, {"last_checkpoint": {"stage": "global_solve"}})
            return subprocess.CompletedProcess(command, 0, "", "")
        if case == "malformed":
            output.write_text("{bad", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if case == "wrong_shape":
            output.write_text(json.dumps({"stage": "finished", "result": []}), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if case == "missing":
            return subprocess.CompletedProcess(command, 0, "", "")
        if case == "launch":
            raise OSError("interpreter missing")
        if case == "interrupt":
            raise KeyboardInterrupt()
        column_runner._record(output.parent, output.name, {
            "stage": "finished", "result": {"profile": [[1.0]]},
        })
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(column_runner.subprocess, "run", fake_run)
    result = column_runner.run_column({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "execution": {"output_dir": str(tmp_path / case)},
    })
    assert result["failure"]["kind"] == expected
    if case == "negative":
        assert result["failure"]["transport"]["returncode"] == 7
        assert result["failure"]["transport"]["stderr"] == "worker err"
    if case == "checkpoint":
        assert result["failure"]["last_checkpoint"]["stage"] == "global_solve"
    if case == "timeout_final":
        assert result["failure"]["completed_payload"]["result"]["profile"] == [[1.0]]
    if case == "launch":
        assert result["failure"]["transport"]["returncode"] is None
    if case == "interrupt":
        assert result["execution"]["status"] == "interrupted"


def test_conserved_worker_retains_preparation_failure_kind(tmp_path):
    payload = json.loads(Path("analyses/bvp_solution_methods/input/case_3c.json").read_text())
    payload["physical_inputs"]["liquid_branch_policy"]["loading_anchor"] = -1.0
    case = tmp_path / "invalid_case.json"
    case.write_text(json.dumps(payload))
    result = run_column({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": str(case)},
    })
    assert result["execution"]["status"] == "failed"
    assert result["failure"]["kind"] == "preparation_failed"
    assert "Conserved loading policy" in result["failure"]["message"]


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"numerics": {"solver_settings": {"typo": 1}}}, "Unknown or irrelevant"),
        ({"execution": {"process_isolation": "false"}}, "must be boolean"),
        ({"initialization": {"policy": "typo"}}, "Only legacy"),
        ({"acceptance": {"boundary_residual_max": -1}}, "nonnegative"),
    ],
)
def test_new_config_rejects_ambiguous_controls(extra, message):
    with pytest.raises(ConfigurationError, match=message):
        resolve_column_config({
            "preset": "seven_state_legacy",
            "case": {"source": "C_cases_data", "id": "3C"},
            **extra,
        })


@pytest.mark.parametrize("transform_mode", ["positive_flow_pressure", "case_bounded_flow_pressure"])
def test_new_config_rejects_unvalidated_flow_transforms(transform_mode):
    with pytest.raises(ConfigurationError, match="Unsupported seven-state transform_mode"):
        resolve_column_config({
            "preset": "seven_state_legacy",
            "case": {"source": "C_cases_data", "id": "3C"},
            "numerics": {"solver_settings": {"transform_mode": transform_mode}},
        })


def test_direct_physical_inputs_reject_fractional_intercoolers():
    config = resolve_column_config({
        "preset": "seven_state_legacy",
        "case": {"source": "C_cases_data", "id": "3C"},
    })
    with pytest.raises(ConfigurationError, match="integer intercooler count"):
        column_runner._direct_physical_inputs({
            "physical_inputs": {
                "liquid_feed_mol_s": [1.0, 2.0, 3.0],
                "vapor_feed_mol_s": [1.0, 2.0, 3.0, 4.0],
                "beds": 1,
                "intercoolers": 0.5,
            },
        }, config)


def test_worker_rejects_same_version_with_different_wheel_hash():
    engine = default_engine(required=True)
    with pytest.raises(CapabilityRefusal, match="SHA-256 mismatch"):
        verify_worker_identity({
            "wheel": engine.wheel,
            "sha256": "0" * 64,
            "commit": engine.commit,
            "python": engine.python,
        })


def test_worker_checks_all_installed_archive_hash_evidence(monkeypatch):
    engine = default_engine(required=True)
    real_distribution = importlib.metadata.distribution("epcsaft")

    class DistributionProxy:
        def __init__(self, archive_info):
            self.archive_info = archive_info

        def __getattr__(self, name):
            return getattr(real_distribution, name)

        def read_text(self, name):
            if name == "direct_url.json":
                return json.dumps({
                    "url": Path(engine.wheel).resolve().as_uri(),
                    "archive_info": self.archive_info,
                })
            return real_distribution.read_text(name)

    def check(archive_info):
        monkeypatch.setattr(
            column_config.importlib.metadata,
            "distribution",
            lambda name: DistributionProxy(archive_info),
        )
        return verify_worker_identity({
            "wheel": engine.wheel,
            "sha256": engine.sha256,
            "commit": engine.commit,
            "python": engine.python,
        })

    correct = check({"hashes": {"sha256": engine.sha256}})
    assert correct["archive_hash_status"] == "verified"
    absent = check({})
    assert absent["archive_hash_status"] == "verified_from_installed_contents"
    with pytest.raises(CapabilityRefusal, match="archive hash"):
        check({"hashes": {"sha256": "0" * 64}})
    with pytest.raises(CapabilityRefusal, match="archive hash fields conflict"):
        check({"hash": f"sha256={engine.sha256}", "hashes": {"sha256": "0" * 64}})


def test_run_column_retains_legacy_and_acceptance_outcomes(tmp_path, monkeypatch):
    def fake_run(df, run, case_source, method, thermo_model, settings):
        return {
            "case_id": str(df.index[run]),
            "success": True,
            "message": "solver terminated",
            "solver_iterations": 2,
            "final_mesh_nodes": 21,
            "capture_pct": 90.0,
            "raw_capture_pct": 90.0,
            "temperature_rmse_K": None,
            "boundary_residual_norm": 1e-8,
        }

    monkeypatch.setattr(benchmark, "_run_one_case", fake_run)
    result = run_column({
        "preset": "seven_state_legacy",
        "case": {"source": "C_cases_data", "id": "3C"},
        "execution": {"output_dir": str(tmp_path / "attempt")},
    })
    assert result["execution"]["status"] == "completed"
    assert result["legacy_outcome"] == {"success": True, "message": "solver terminated"}
    assert result["numerical_acceptance"] == "not_evaluated"
    retained = json.loads((tmp_path / "attempt" / "attempt.json").read_text())
    assert retained["resolved_inputs"]["physical_input_sha256"]
    assert retained["source_identity"]["source_tree_sha256"]
    assert retained["source_identity"]["dirty_patch_sha256"]
    assert retained["source_identity"]["dirty_patch"]


def test_legacy_success_does_not_imply_numerical_acceptance(monkeypatch):
    def fake_run(*args, **kwargs):
        return {"success": True, "message": "root failed; final IVP converged", "boundary_residual_norm": 0.2}

    monkeypatch.setattr(benchmark, "_run_one_case", fake_run)
    result = run_column({
        "preset": "seven_state_legacy",
        "case": {"source": "C_cases_data", "id": "3C"},
        "acceptance": {"boundary_residual_max": 0.01},
    })
    assert result["legacy_outcome"]["success"] is True
    assert result["numerical_acceptance"] == "rejected"
    assert result["solver"]["stages"] == {}


def test_capture_criterion_cannot_change_numerical_acceptance(monkeypatch):
    monkeypatch.setattr(benchmark, "_run_one_case", lambda *args, **kwargs: {
        "success": True,
        "message": "ok",
        "boundary_residual_norm": 0.0,
        "capture_error_pct": 50.0,
        "solver_stage_status": {"root": {"success": True}, "ivp": {"success": True}},
    })
    result = run_column({
        "preset": "seven_state_legacy",
        "case": {"source": "C_cases_data", "id": "3C"},
        "numerics": {"method": "single"},
        "acceptance": {"boundary_residual_max": 0.01, "capture_error_max_pct": 5.0},
    })
    assert result["numerical_acceptance"] == "accepted"
    assert result["observation_agreement"] == "rejected"


def test_run_column_retains_worker_capability_refusal(monkeypatch):
    monkeypatch.setattr(benchmark, "_run_one_case", lambda *args, **kwargs: {
        "success": False,
        "message": "Engine identity unavailable in worker",
        "failure_kind": "capability_refusal",
        "jacobian_status": "capability_refusal",
    })
    result = run_column({
        "preset": "seven_state_legacy",
        "case": {"source": "C_cases_data", "id": "3C"},
    })
    assert result["execution"]["status"] == "failed"
    assert result["failure"]["kind"] == "capability_refusal"
    assert result["failure"]["phase"] == "worker_preparation"


def test_timeout_row_is_retained_as_timeout(monkeypatch):
    monkeypatch.setattr(benchmark, "_run_one_case", lambda *args, **kwargs: {
        "success": False,
        "message": "Benchmark subprocess exceeded subprocess_timeout_s=1",
        "jacobian_status": "subprocess_timeout",
    })
    result = run_column({"preset": "seven_state_legacy", "case": {"source": "C_cases_data", "id": "3C"}})
    assert result["execution"]["status"] == "timed_out"
    assert result["failure"]["kind"] == "timed_out"


def test_resume_reuses_matching_retained_attempt(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(True)
        return {"success": True, "message": "solver terminated", "boundary_residual_norm": 0.1}

    monkeypatch.setattr(benchmark, "_run_one_case", fake_run)
    output = tmp_path / "resume"
    request = {
        "preset": "seven_state_legacy",
        "case": {"source": "C_cases_data", "id": "3C"},
        "execution": {"output_dir": str(output)},
    }
    first = run_column(request)
    second = run_column({**request, "execution": {"output_dir": str(output), "resume": True}})
    assert first["attempt_id"] == second["attempt_id"]
    assert len(calls) == 1


def test_resume_rejects_changed_source_with_same_dirty_paths(tmp_path, monkeypatch):
    source = {
        "repository_root": str(tmp_path),
        "commit": "a" * 40,
        "dirty_paths": ["src/mea_absorption_column/column.py"],
        "dirty": True,
        "source_tree_sha256": "b" * 64,
        "source_file_sha256": {"src/mea_absorption_column/column.py": "c" * 64},
        "dirty_patch": "patch-a",
        "dirty_patch_sha256": "d" * 64,
    }
    changed = {**source, "source_tree_sha256": "e" * 64, "dirty_patch_sha256": "f" * 64}
    monkeypatch.setattr(column_runner, "_source_identity", iter((source, changed)).__next__)
    monkeypatch.setattr(benchmark, "_run_one_case", lambda *args, **kwargs: {
        "success": True, "message": "solver terminated", "boundary_residual_norm": 0.1,
    })
    request = {
        "preset": "seven_state_legacy",
        "case": {"source": "C_cases_data", "id": "3C"},
        "execution": {"output_dir": str(tmp_path / "resume")},
    }
    run_column(request)
    with pytest.raises(ConfigurationError, match="source identity mismatch"):
        run_column({**request, "execution": {"output_dir": str(tmp_path / "resume"), "resume": True}})
