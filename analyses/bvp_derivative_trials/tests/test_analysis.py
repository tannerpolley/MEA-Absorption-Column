from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts/run_issue19_column_verification.py"
SPEC = spec_from_file_location("issue19_verification", SCRIPT)
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_failure_classification_and_branch_accounting():
    timeout = {"message": "Benchmark subprocess exceeded subprocess_timeout_s=10", "success": False, "certificate_pass": False, "physical_check_pass": False}
    assert MODULE._classify(timeout) == ("campaign_watchdog", "campaign_timeout", "not_established")

    rows = [
        {"case_id": "3C", "thermo_model": "epcsaft_ionic", "validation_pass": True, "capture_pct": capture}
        for capture in (88.1, 88.3, 91.0)
    ]
    MODULE._assign_branches(rows)
    assert [row["branch_id"] for row in rows] == ["capture_branch_1", "capture_branch_1", "capture_branch_2"]
