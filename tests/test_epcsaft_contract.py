from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_epcsaft_contract_self_check_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_epcsaft_integration.py", "--mode", "stable", "--self-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
