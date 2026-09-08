from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analyses/nccc_validation/scripts"))

from analyze_issue17_enhancement_comparison import (  # noqa: E402
    FIXED_STATES,
    FORMULATIONS,
    STAGE_TABLE,
    aggregate_results,
    build_summary,
    evaluate_fixed_states,
    explicit_equation,
    ranking_sensitivity,
    scalar_reference,
)


def test_issue17_four_formulation_reference_and_gates() -> None:
    # Independently source-derived values at Ha=6, Q=3, R+=R-=4 (S_MEA=9).
    sentinels = {
        "EF-AOP-78-PUBLISHED-MEA": 0.43636363636363634,
        "EF-AOP-73-CORRECTED-MEA": 1.7142857142857142,
        "EF-CURRENT": 1.2380952380952381,
    }
    hatta, q, r_plus, r_minus = 6.0, 3.0, 4.0, 4.0
    s_mea = r_plus + r_minus + 1.0
    for formulation, expected in sentinels.items():
        assert explicit_equation(formulation, hatta, q, s_mea) == pytest.approx(expected, rel=1.0e-15)
        assert scalar_reference(formulation, hatta, q, r_plus, r_minus) == pytest.approx(
            expected, rel=1.0e-15
        )

    result = evaluate_fixed_states()
    fixed = pd.read_csv(FIXED_STATES)
    expected_positions = [index / 20.0 for index in range(21)]
    assert len(result) == 84
    assert set(result.formulation) == set(FORMULATIONS)
    assert result.Position.drop_duplicates().tolist() == pytest.approx(
        expected_positions, rel=0.0, abs=1.0e-15
    )
    assert result.groupby("Position").size().eq(4).all()
    assert result.groupby("Position").formulation.nunique().eq(4).all()
    fixed_columns = [column for column in fixed.columns if column not in {"E", "Psi"}]
    for formulation in FORMULATIONS:
        actual = result.loc[result.formulation.eq(formulation), fixed_columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            actual,
            fixed[fixed_columns],
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
    assert result.evaluation_status.eq("evaluated").all()
    assert result.finite_values_pass.all()
    assert result.positive_enhancement_pass.all()
    assert result.flux_direction_pass.all()
    assert result.reverse_check_pass.all()
    assert not result.fallback_used.any()
    assert result.current_E_relative_reproduction_error.max() <= 1.0e-12
    implicit = result.loc[result.formulation.eq("EF-GF-IMPLICIT")]
    explicit = result.loc[result.formulation.ne("EF-GF-IMPLICIT")]
    assert implicit.scaled_equation_residual.max() <= 1.0e-8
    assert implicit.initial_guess_relative_spread.max() <= 1.0e-3
    assert len(explicit) == 63
    assert explicit.scalar_reference_relative_error.between(0.0, 1.0e-12).all()
    published = result.loc[result.formulation.eq("EF-AOP-78-PUBLISHED-MEA")]
    assert published.E.lt(1.0).all()
    assert published.outcome.eq("physical_invalidity").all()
    aggregates = aggregate_results(result)
    _, generated_stages = build_summary(result, aggregates)
    for stages in (generated_stages, pd.read_csv(STAGE_TABLE)):
        blocked = stages.loc[stages.stage.isin([4, 5])]
        assert blocked.attempted.eq("no").all()
        assert blocked.stopped_by.eq("physical_check").all()
        assert blocked.outcome.eq("not_attempted").all()
    assert {
        "p05_E_difference_from_current",
        "p95_flux_difference_from_current_mol_s_m",
    }.issubset(aggregates)
    sensitivity, orders, reversals = ranking_sensitivity(result)
    assert len(sensitivity) == 9
    assert len(orders) == 1
    assert reversals == []
