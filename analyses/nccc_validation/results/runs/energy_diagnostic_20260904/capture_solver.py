"""Disposable observation of the unchanged solver; arguments go to the existing runner."""
import importlib
from pathlib import Path
import runpy
import sys
import numpy as np

module = importlib.import_module('mea_absorption_column.BVP.Methods.Scipy_BVP_Solve')
original = module.solve_bvp
output = Path(sys.argv[sys.argv.index('--output') + 1])

def capture(*args, **kwargs):
    sol = original(*args, **kwargs)
    np.savez(output / 'solver_polynomial.npz', x=sol.x, y=sol.y, coefficients=sol.sol.c,
             residuals=sol.rms_residuals, success=sol.success)
    return sol

module.solve_bvp = capture
runpy.run_path('analyses/nccc_validation/scripts/run_reactive_column.py', run_name='__main__')
