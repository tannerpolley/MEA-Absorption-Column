from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analyses/nccc_validation/scripts"))

from analyze_enhancement_consistency import explicit_enhancement  # noqa: E402


def test_explicit_enhancement_has_slow_reaction_limit() -> None:
    value = explicit_enhancement(
        1.0,
        c_co2=1.0,
        c_mea=1000.0,
        c_meah=1000.0,
        c_meacoo=1000.0,
        d_co2=2.0e-9,
        d_mea=1.0e-9,
        d_ion=4.0e-10,
    )
    assert value == pytest.approx(1.0)
