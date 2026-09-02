from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import sys


SCRIPT = Path(__file__).parents[1] / "scripts/run_chemical_potential_film.py"
SPEC = spec_from_file_location("chemical_potential_film", SCRIPT)
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_flux_expansion_closes_total_and_current():
    independent = np.asarray([0, 1, 4, 5, 6, 7, 8])
    dependent = np.asarray([2, 3])
    flux = MODULE._expand_flux(np.arange(7, dtype=float), independent, dependent)
    assert np.sum(flux) == pytest.approx(0.0)
    assert np.dot(MODULE.CHARGES, flux) == pytest.approx(0.0)


def test_rank_deficiency_is_reported_without_regularization():
    with pytest.raises(MODULE.ChemicalPotentialFilmError, match=r"rank=1, expected=2"):
        MODULE._rank_or_stop(np.ones((2, 2)), 2, "test block")
