import json
import sys
from pathlib import Path
import numpy as np
from mea_absorption_column import column
from mea_absorption_column.benchmark_worker import main as worker
def sweep(balance, inlet, solver, candidate, lower, upper, p, profile):
    jacobian = solver.get_function("nlp_jac_g")
    liquid, vapor = balance._thermodynamic_callbacks
    exact = np.asarray(jacobian(candidate["x"], p)[1])
    errors = {}
    for step in json.loads(sys.argv[2]):
        liquid.fd_relative_step = vapor.fd_relative_step = step
        trial = np.asarray(jacobian(candidate["x"], p)[1])
        errors[step] = np.max(np.max(abs(trial - exact), axis=0) / np.maximum(1., np.max(abs(exact), axis=0)))
    liquid.fd_relative_step = vapor.fd_relative_step = None
    return errors
if len(sys.argv) == 4:
    sys.exit(worker(sys.argv[3:], design_check=sweep if sys.argv[2] != "[]" else lambda *args: {}))
column._run_conserved_preparation(column.resolve_column_config(json.loads(Path(sys.argv[1]).read_text())), worker_args=("-u", __file__, "--worker", sys.argv[2]))
