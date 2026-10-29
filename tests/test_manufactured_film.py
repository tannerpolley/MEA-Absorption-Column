"""Six-species/four-component film versus an exact nonzero exchange solution."""
from pathlib import Path
from runpy import run_path


def test_neutral_two_amine_film_against_exact_solution(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'analyses/bvp_solution_methods/scripts/verify_manufactured_film.py'
    # The retained verifier owns the independent matrix, action, flux, permutation
    # and duplicate-row assertions; do not maintain a second set of equations.
    run_path(str(script))['run'](tmp_path)
