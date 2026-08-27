from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analyses/nccc_validation/scripts"))

from analyze_enhancement_consistency import explicit_enhancement  # noqa: E402


RESULT = ROOT / "analyses/nccc_validation/results/final/tables/retained_reactive_case3c_enhancement_formulations.csv"
COMPARISON = ROOT / "analyses/nccc_validation/results/final/tables/retained_reactive_case3c_enhancement_film_comparison.csv"


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


def test_retained_result_schema_and_failures_are_visible() -> None:
    table = pd.read_csv(RESULT)
    assert len(table) == 21 * 5
    assert set(table.outcome) <= {
        "evaluated",
        "numerical_convergence_failure",
        "physical_invalidity",
        "input_preflight_failure",
        "not_established",
    }
    required = {
        "Position",
        "formulation",
        "outcome",
        "diagnostic",
        "Ha",
        "E",
        "Psi_H",
        "predicted_flux_mol_s_m",
    }
    assert required <= set(table.columns)
    failed = table.outcome != "evaluated"
    if failed.any():
        assert table.loc[failed, "diagnostic"].fillna("").astype(str).str.len().gt(0).all()


def test_mechanistic_film_is_compared_without_an_invented_enhancement_factor() -> None:
    table = pd.read_csv(COMPARISON)
    film = table.loc[table.method_class.eq("mechanistic_reaction_diffusion")]

    assert len(film) == 1
    assert film.E.isna().all()
    assert film.flux_ratio_to_current.iloc[0] > 0.0
