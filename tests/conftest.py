from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(request):
    """Repo-local tmp_path replacement for sandboxed Windows runs."""
    root = Path(__file__).resolve().parents[1] / ".tmp_local" / "pytest_tmp"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)
    path = root / safe_name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path
