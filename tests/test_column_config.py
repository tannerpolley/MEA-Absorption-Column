from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
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


def test_source_homotopy_configuration_is_paired_and_trapezoidal():
    request = {
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "numerics": {"method": "trapezoidal", "solver_settings": {
            "source_homotopy_initial_step": .25, "source_homotopy_min_step": .03125}},
    }
    settings = resolve_column_config(request).numerics.as_dict()["solver_settings"]
    assert (settings["source_homotopy_initial_step"], settings["source_homotopy_min_step"]) == (.25, .03125)
    for numerics in (
        {"method": "trapezoidal", "solver_settings": {"source_homotopy_initial_step": .25}},
        {"method": "central", "solver_settings": {"source_homotopy_initial_step": .25,
                                                    "source_homotopy_min_step": .03125}},
        {"method": "trapezoidal", "solver_settings": {"source_homotopy_initial_step": .25,
                                                        "source_homotopy_min_step": .5}},
    ):
        with pytest.raises(ConfigurationError, match="Source homotopy"):
            resolve_column_config({**request, "numerics": numerics})


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
    assert result["physical_acceptance"] == "not_evaluated"
    assert result["physical_certification"] == {"accepted": False}
    assert result["diagnostics"]["physical_acceptance_reason"] == "candidate profile unavailable"
    assert result["solver"]["stages"]["outer"]["success"] is True
    assert result["initialization"]["accepted"] is True
    assert result["native_calls"]["liquid_value_A1"]["started"] == 1


class _ChargeState:
    def __init__(self, charge):
        self.charge = charge
        self.calls = 0
        self._reactions = {"charges": [0, 0, 0, 1, -1, -1, -2, 1, -1]}

    def solve(self, *_args, **_kwargs):
        self.calls += 1
        amounts = np.zeros(9)
        amounts[3] = self.charge
        return {"amounts_mol": amounts}


def _certificate_inputs(charge=0.0, outlet=1.0):
    profile = np.tile(np.array([1.0, 1.0, 2.0, 1.0, 300.0, 300.0, 1e5, 0.5, 0.0, 0.0, 0.0, 0.0])[:, None], (1, 2))
    profile[2, -1] = outlet
    reactive = _ChargeState(charge)

    def node(_z, _state):
        return np.array([1., 2., 3., 4., 5., 6., 7.]), np.zeros(7), np.zeros(5)

    assembly = {"node": node, "boundary": lambda *_: np.zeros(7),
                "reactive_liquid": reactive, "height_m": 1.0}
    lower = np.r_[[0.] * 4, 293.15, 293.15, 1., 1e-12, [-np.inf] * 4]
    upper = np.r_[[np.inf] * 4, 393.15, 393.15, 1e7, .97, [np.inf] * 4]
    scaling = {"balance_scale": np.ones(7), "algebraic_scale": np.ones(5),
               "boundary_scale": np.ones(7)}
    return {"grid": np.array([0., 1.]), "profile": profile}, assembly, lower, upper, scaling, reactive


def test_equilibrium_physical_certificate_accepts_complete_native_evidence():
    result, assembly, lower, upper, scaling, reactive = _certificate_inputs()
    certificate = column_runner._equilibrium_physical_certification(
        result, assembly, np.array([1., 2., 3.]), lower, upper, scaling, 1e-7, 3,
    )
    assert certificate["accepted"] is True
    assert set(certificate["original_residuals"]) == {"material", "energy", "charge", "interface", "boundary"}
    assert set(certificate["scaled_residual_inf"]) == set(certificate["original_residuals"])
    assert certificate["original_bounds"]["accepted"] is True
    assert certificate["capture"]["accepted"] is True
    assert reactive.calls == 6


def test_equilibrium_physical_certificate_rejects_charge_and_capture():
    result, assembly, lower, upper, scaling, _ = _certificate_inputs(charge=1.0, outlet=3.0)
    certificate = column_runner._equilibrium_physical_certification(
        result, assembly, np.array([1., 2., 3.]), lower, upper, scaling, 1e-7, 3,
    )
    assert certificate["accepted"] is False
    assert certificate["scaled_residual_inf"]["charge"] == 1.0
    assert certificate["capture"]["accepted"] is False


@pytest.mark.parametrize(
    "numerical, physical, expected, reason",
    [
        (True, True, "accepted", "complete physical certification accepted"),
        (False, True, "accepted", "complete physical certification accepted"),
        (True, False, "rejected", "physical check failed"),
    ],
)
def test_conserved_public_physical_acceptance_tracks_physical_certificate(
    monkeypatch, numerical, physical, expected, reason,
):
    monkeypatch.setattr(column_runner, "execute_conserved_column", lambda config: {
        "resolved_inputs": {}, "engine": {}, "assets": {}, "capabilities": {}, "layout": {},
        "result": {"accepted": numerical, "status": "finished", "profile": [[1.0]],
                   "solver_statistics": {"success": numerical}},
        "physical_certification": {"accepted": physical, "reason": "physical check failed"},
    })
    result = run_column({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
    })
    assert result["physical_acceptance"] == expected
    assert result["diagnostics"]["physical_acceptance_reason"] == reason


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


def _retained_profile_context(tmp_path, mismatch=None):
    accepted_path = Path("analyses/bvp_solution_methods/results/conserved_public_physical_20260915T204019Z/attempt.json")
    baseline = json.loads(accepted_path.read_text())
    source = json.loads(json.dumps(baseline))
    if mismatch == "status":
        source["physical_acceptance"] = "rejected"
    if mismatch == "certificate":
        source["physical_certification"]["accepted"] = False
    if mismatch == "record":
        source["engine"] = []
    if mismatch == "grid":
        source["native_profile"]["grid"][-1] = 5.0
    if mismatch == "profile":
        source["native_profile"]["state_matrix"] = source["native_profile"]["state_matrix"][:11]
    if mismatch == "identity":
        source["config"]["model"]["film_model"] = "enhancement_reference"
    if mismatch == "physics":
        source["resolved_inputs"]["physical_input_sha256"] = "wrong"
    if mismatch == "solver":
        source["config"]["numerics"]["solver_settings"]["quadrature_points"] = 4
    if mismatch == "engine":
        source["engine"]["actual_sha256"] = "wrong"
    if mismatch == "scale":
        source["scaling"]["state_scale"][0] += 1.0
    if mismatch == "bounds":
        source["native_profile"]["state_matrix"][0][0] = -1.0
    source_path = tmp_path / "attempt.json"
    source_path.write_text(json.dumps(source))
    config = resolve_column_config({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "numerics": {"method": "trapezoidal", "nodes": 3,
                     "solver_settings": {"quadrature_points": 3, "tolerance": 1e-7, "max_iterations": 20}},
        "initialization": {"values": {"retained_profile": str(source_path)}},
    })
    prepared = {"resolved_inputs": baseline["resolved_inputs"], "engine": baseline["engine"],
                "assembly": {"coordinate": np.linspace(0.0, 6.0, 3), "height_m": 6.0}}
    scaling = {name: np.asarray(baseline["scaling"][name]) for name in
               ("state_scale", "balance_scale", "algebraic_scale", "boundary_scale")}
    margin = .97 * np.finfo(float).eps
    lower = np.r_[[0.] * 4, 293.15, 293.15, 1., margin, [-np.inf] * 4]
    upper = np.r_[[np.inf] * 4, 393.15, 393.15, 1e7, .97-margin, [np.inf] * 4]
    return config, prepared, scaling, lower, upper, baseline, source_path


def test_retained_profile_interpolates_accepted_n2_seed_before_solver(tmp_path):
    config, prepared, scaling, lower, upper, source, source_path = _retained_profile_context(tmp_path)
    initial, provenance = column_runner._retained_initial_profile(
        config.initialization.as_dict()["values"]["retained_profile"],
        config, prepared, scaling, lower, upper,
    )
    profile = np.asarray(source["native_profile"]["state_matrix"])
    expected = np.array([np.interp([0., 3., 6.], [0., 6.], row) for row in profile])
    np.testing.assert_array_equal(initial, expected)
    assert config.initialization.as_dict()["values"]["retained_profile"] == str(source_path.resolve())
    assert provenance["attempt_id"] == source["attempt_id"]
    assert provenance["config_sha256"] == source["config_sha256"]
    assert provenance["source_file_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert provenance["engine_identity"]["actual_sha256"] == source["engine"]["actual_sha256"]
    assert provenance["source_grid"] == [0.0, 6.0]
    assert provenance["target_grid"] == [0.0, 3.0, 6.0]
    assert provenance["interpolation"] == "numpy.interp row-wise in physical state basis"
    assert provenance["accepted_source"] == {
        "execution": "completed", "numerical_acceptance": "accepted",
        "physical_acceptance": "accepted", "physical_certification": True,
    }
    relative = resolve_column_config({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "initialization": {"values": {"retained_profile":
            "analyses/bvp_solution_methods/results/conserved_public_physical_20260915T204019Z/attempt.json"}},
    })
    assert Path(relative.initialization.as_dict()["values"]["retained_profile"]).is_absolute()
    with pytest.raises(ConfigurationError, match="Unknown initialization values"):
        resolve_column_config({"preset": "seven_state_legacy",
                               "case": {"source": "C_cases_data", "id": "3C"},
                               "initialization": {"values": {"retained_profile": str(source_path)}}})


@pytest.mark.parametrize("mismatch", ["status", "certificate", "record", "grid", "profile", "identity", "physics", "solver", "engine", "scale", "bounds"])
def test_retained_profile_rejects_mismatch_before_solver(tmp_path, mismatch):
    config, prepared, scaling, lower, upper, _, _ = _retained_profile_context(tmp_path, mismatch)
    with pytest.raises(ConfigurationError, match="Retained profile"):
        column_runner._retained_initial_profile(
            config.initialization.as_dict()["values"]["retained_profile"],
            config, prepared, scaling, lower, upper,
        )


def test_retained_profile_reaches_solver_and_rejection_does_not(tmp_path, monkeypatch):
    config, prepared, _, _, _, source, _ = _retained_profile_context(tmp_path)
    payload = {
        "physical_inputs": {"liquid_feed_mol_s": [1., 1., 1.], "vapor_feed_mol_s": [1., 1., 1., 1.],
                            "liquid_temperature_k": 300., "vapor_temperature_k": 300.,
                            "bottom_pressure_pa": 1e5, "packing": [1., .97, 1., 1., 1., 1., 1.]},
        "initial_bulk_state": [1., 1., 1., 1., 300., 300., 1e5, .5],
        "initial_interface_bracket": [0., 1.], "physical_residual_tolerance": 1e-7,
        "liquid_molar_masses_kg_mol": [1., 1., 1.], "vapor_molar_masses_kg_mol": [1., 1., 1., 1.],
    }

    class Phase:
        molar_masses = np.ones(4)
        solve = solve_actions = _state = lambda *args: None
        def input_jacobian(self, *_args):
            return np.ones((1, 1))

    def fake_prepared(_config):
        liquid, vapor = Phase(), Phase()
        liquid.molar_masses = np.ones(3)
        def diagnostic(state):
            return None, state[-1] - .5, 1., np.ones(2), 1., None, np.ones(2), 1.
        def balance(*_args):
            return None, None, 0., np.r_[np.zeros(20), 1.], np.ones(2)
        assembly = {"node": lambda *_: (np.zeros(7), np.zeros(7), np.zeros(5)), "boundary": lambda *_: np.zeros(7),
                    "diagnostics": diagnostic, "balance": balance, "liquid": liquid, "vapor": vapor,
                    "reactive_liquid": liquid, "coordinate": np.array([0., 3., 6.]), "height_m": 6.}
        return {**prepared, "assembly": assembly, "assets": {}, "capabilities": {}, "layout": {}}

    expected = np.asarray(source["native_profile"]["state_matrix"])
    expected = np.array([np.interp([0., 3., 6.], [0., 6.], row) for row in expected])
    calls = []
    monkeypatch.setattr(column_runner, "_physical_payload", lambda _: payload)
    monkeypatch.setattr(column_runner, "_prepare_conserved_column_in_process", fake_prepared)
    monkeypatch.setattr(column_runner, "_retained_initial_profile", lambda *_: (expected, {"accepted": True}))
    monkeypatch.setattr("mea_absorption_column.BVP.Methods.Casadi_Collocation.solve_conservative_collocation",
                        lambda *args, **kwargs: calls.append(args[3]) or {"profile": None})
    column_runner._run_conserved_column_in_process(config, {})
    np.testing.assert_array_equal(calls, [expected])
    monkeypatch.setattr(column_runner, "_retained_initial_profile",
                        lambda *_: (_ for _ in ()).throw(ConfigurationError("Retained profile rejected")))
    with pytest.raises(ConfigurationError, match="Retained profile rejected"):
        column_runner._run_conserved_column_in_process(config, {})
    assert len(calls) == 1

    homotopy = resolve_column_config({
        "preset": "twelve_state_conserved",
        "case": {"physical_input_file": "analyses/bvp_solution_methods/input/case_3c.json"},
        "numerics": {"method": "trapezoidal", "nodes": 3, "solver_settings": {
            "quadrature_points": 3, "tolerance": 1e-7, "max_iterations": 2,
            "source_homotopy_initial_step": .5, "source_homotopy_min_step": .25}},
    })
    staged = []
    def staged_solver(*args, source_multiplier=None, **_kwargs):
        seed = np.asarray(args[3])
        accepted = bool(staged)
        staged.append((source_multiplier, seed.copy()))
        profile = seed + (source_multiplier if accepted else 100.)
        return {"accepted": accepted, "profile": profile, "grid": args[2],
                "status": "Solve_Succeeded" if accepted else "Maximum_Iterations_Exceeded",
                "scaled_residual_inf": 0. if accepted else 1., "scaled_bound_violation_inf": 0.}
    monkeypatch.setattr("mea_absorption_column.BVP.Methods.Casadi_Collocation.solve_conservative_collocation",
                        staged_solver)
    monkeypatch.setattr(column_runner, "_equilibrium_physical_certification",
                        lambda *_args, **_kwargs: {"accepted": True})
    outcome = column_runner._run_conserved_column_in_process(homotopy, {})
    assert [stage[0] for stage in staged] == [.5, .25, .5, .75, 1.]
    np.testing.assert_array_equal(staged[1][1], staged[0][1])
    np.testing.assert_array_equal(staged[2][1], staged[1][1] + .25)
    assert outcome["result"]["continuation"]["last_accepted_source_multiplier"] == 1.
    np.testing.assert_array_equal(outcome["initial_profile"], staged[0][1])

    def rejected(*args, source_multiplier=None, **_kwargs):
        return {"accepted": False, "profile": np.asarray(args[3]) + 1., "grid": args[2],
                "status": "Maximum_Iterations_Exceeded", "scaled_residual_inf": 1.,
                "scaled_bound_violation_inf": 0., "source_multiplier": source_multiplier}
    monkeypatch.setattr("mea_absorption_column.BVP.Methods.Casadi_Collocation.solve_conservative_collocation", rejected)
    incomplete = column_runner._run_conserved_column_in_process(homotopy, {})
    result = incomplete["result"]
    assert result["status"] == "Source_Homotopy_Incomplete" and result["profile"] is None
    assert incomplete["physical_certification"]["accepted"] is None
    assert [stage["result"]["source_multiplier"] for stage in result["continuation"]["stages"][1:]] == [.5, .25]
    np.testing.assert_array_equal(result["continuation"]["last_accepted_profile"], incomplete["initial_profile"])


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
    assert retained["physical_acceptance"] == "not_evaluated"
    assert retained["physical_certification"]["accepted"] is None


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
    if not retained["source_identity"]["dirty"]:
        assert retained["source_identity"]["dirty_patch"] == ""


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
