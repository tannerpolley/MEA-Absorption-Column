"""Reuse the manuscript directional check under all six transport perturbations.

Centered differences verify the exact assembled Jacobian here; they are never
used to build the solver Jacobian in the sensitivity runs.
"""
import argparse
from contextlib import chdir
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

from generate_data import transport_scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.model_root.resolve()
    sys.path.insert(0, str(root/'src'))
    import numpy as np
    from capture_failure import validate_profile
    np.testing.assert_array_equal(validate_profile(np.ones((7, 3))), np.ones((7, 3)))
    for invalid in (np.ones((6, 3)), np.ones((7, 1)), np.full((7, 3), np.nan)):
        try:
            validate_profile(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid explicit profile was admitted')
    source = root/'tests/test_reactive_column_jacobian.py'
    check = runpy.run_path(str(source))['test_native_column_direction_and_exact_reuse']
    rows = []
    with chdir(root):
        for quantity in ('viscosity', 'diffusivity', 'kl'):
            for factor in (.9, 1.1):
                started = time.perf_counter()
                with transport_scale(quantity, factor) as calls:
                    check(False, None, None)
                assert min(calls.values()) > 0
                rows.append(dict(quantity=quantity, factor=factor, passed=True,
                                 seconds=time.perf_counter()-started, calls=calls))
                print(rows[-1], flush=True)
    output = Path(__file__).resolve().parents[1]/'output'/'derivative_checks.json'
    output.write_text(json.dumps(dict(reference_check=str(source), reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Complete seven-state RHS directional derivative, exact-state reuse and failed-input isolation; '
              'existing check step 1e-3, rtol 2e-5, atol 2e-8 above native root noise. Lean Case 3C state.',
        profile_shape_finiteness_checks_passed=True, results=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
