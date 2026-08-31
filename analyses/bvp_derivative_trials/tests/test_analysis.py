from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts/run_issue19_column_verification.py"
SPEC = spec_from_file_location("issue19_verification", SCRIPT)
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_failure_classification_and_capture_cluster_accounting():
    timeout = {"message": "Benchmark subprocess exceeded subprocess_timeout_s=10", "success": False, "certificate_pass": False, "basic_state_check_pass": False}
    assert MODULE._classify(timeout) == ("campaign_watchdog", "campaign_timeout", "not_established")
    subprocess_failure = {"message": "Benchmark subprocess completed without writing output row.", "success": False, "certificate_pass": False, "basic_state_check_pass": False}
    assert MODULE._classify(subprocess_failure) == ("subprocess", "subprocess_failure", "not_established")
    crashed_subprocess = {"message": "Benchmark subprocess failed with return code 1: traceback", "success": False, "certificate_pass": False, "basic_state_check_pass": False}
    assert MODULE._classify(crashed_subprocess) == ("subprocess", "subprocess_failure", "not_established")
    solver_failure = {"message": "The maximum number of mesh nodes is exceeded.", "success": False, "certificate_pass": False, "basic_state_check_pass": False}
    assert MODULE._classify(solver_failure) == ("solver", "numerical_convergence_failure", "boundary_at_state")
    certificate_failure = {"message": "The algorithm converged.", "success": True, "certificate_pass": False, "basic_state_check_pass": True}
    assert MODULE._classify(certificate_failure) == ("certificate_check", "certificate_failure", "boundary_at_state")
    state_failure = {"message": "The algorithm converged.", "success": True, "certificate_pass": True, "basic_state_check_pass": False}
    assert MODULE._classify(state_failure) == ("basic_state_check", "physical_invalidity", "boundary_at_state")

    rows = [
        {"case_id": "3C", "thermo_model": "epcsaft_ionic", "validation_pass": True, "capture_pct": capture}
        for capture in (88.1, 88.3, 91.0)
    ]
    MODULE._assign_capture_clusters(rows)
    assert [row["capture_cluster_id"] for row in rows] == ["capture_cluster_1", "capture_cluster_1", "capture_cluster_2"]
    assert all(row["capture_cluster_tolerance_pct"] == 0.5 for row in rows)


def test_reproduction_command_records_custom_timeout():
    assert MODULE._command(7.5).endswith("--case-timeout-s 7.5")
