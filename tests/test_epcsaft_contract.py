from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("mode", ["stable", "final"])
def test_archive_modes_reject_a_different_wheel(monkeypatch, capsys, mode):
    spec = importlib.util.spec_from_file_location("integration_check", ROOT / "scripts/check_epcsaft_integration.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    contract = check.load_contract()
    monkeypatch.setattr(check, "resolve_epcsaft", lambda _: {
        "module_path": "unused", "version": "0.2.0.dev0", "source_kind": "local_file",
        "source_detail": "different wheel", "wheel_path": contract["final_identity"]["wheel_filename"],
        "wheel_sha256": "b011d0f9d492e9db197f67cc0ae6781ac636fa3278805ddf1d6a05ecd167074b",
    })
    monkeypatch.setattr(check, "scan_direct_imports", lambda _: [])
    monkeypatch.setattr(check, "run_smoke", lambda _: None)
    assert check.main(["--mode", mode, "--self-only"]) == 1
    assert "does not match frozen identity" in capsys.readouterr().out


def test_research_mode_allows_a_distinct_immutable_wheel(monkeypatch):
    spec = importlib.util.spec_from_file_location("research_integration_check", ROOT / "scripts/check_epcsaft_integration.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    monkeypatch.setattr(check, "resolve_epcsaft", lambda _: {
        "module_path": "unused", "version": "0.2.0.dev0", "source_kind": "local_file",
        "source_detail": "explicit research wheel", "wheel_path": "candidate.whl",
        "wheel_sha256": "a" * 64,
    })
    monkeypatch.setattr(check, "scan_direct_imports", lambda _: [])
    assert check.main(["--mode", "dev", "--self-only"]) == 0
