from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analyses/nccc_validation/scripts"))

from analyze_issue17_enhancement_comparison import (  # noqa: E402
    FORMULATIONS,
    aggregate_results,
    evaluate_fixed_states,
    explicit_equation,
    ranking_sensitivity,
    scalar_reference,
)


def test_issue17_four_formulation_reference_and_gates() -> None:
    hatta, q, r_plus, r_minus = 20.0, 100.0, 2.0, 3.0
    s_mea = r_plus + r_minus + 1.0
    for formulation in FORMULATIONS[1:]:
        assert explicit_equation(formulation, hatta, q, s_mea) == pytest.approx(
            scalar_reference(formulation, hatta, q, r_plus, r_minus),
            rel=1.0e-15,
        )

    result = evaluate_fixed_states()
    assert len(result) == 84
    assert set(result.formulation) == set(FORMULATIONS)
    assert result.evaluation_status.eq("evaluated").all()
    assert result.finite_values_pass.all()
    assert result.positive_enhancement_pass.all()
    assert result.flux_direction_pass.all()
    assert result.reverse_check_pass.all()
    assert not result.fallback_used.any()
    assert result.current_E_relative_reproduction_error.max() <= 1.0e-12
    implicit = result.loc[result.formulation.eq("EF-GF-IMPLICIT")]
    assert implicit.scaled_equation_residual.max() <= 1.0e-8
    assert implicit.initial_guess_relative_spread.max() <= 1.0e-3
    published = result.loc[result.formulation.eq("EF-AOP-78-PUBLISHED-MEA")]
    assert published.E.lt(1.0).all()
    assert published.outcome.eq("physical_invalidity").all()
    aggregates = aggregate_results(result)
    assert {
        "p05_E_difference_from_current",
        "p95_flux_difference_from_current_mol_s_m",
    }.issubset(aggregates)
    sensitivity, orders, reversals = ranking_sensitivity(result)
    assert len(sensitivity) == 9
    assert len(orders) == 1
    assert reversals == []
