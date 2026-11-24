import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'analyses/reactive_film_evidence/scripts')
import check_finite_rate_film as check
import evaluate_finite_rate_cases as study
base = Path('analyses/reactive_film_evidence/results/runs')
out = base / 'finite_rate_film_cp5_2026-11-24'
case, height, basis = sys.argv[1], float(sys.argv[2]), sys.argv[3]
a = study.read(base / f'finite_rate_film_cp3_2026-11-23/{case}/n2o/try1/attempt.json')
p = a['native_profile']; i = int(np.argmin(abs(np.asarray(p['grid']) - height)))
assert abs(p['grid'][i] - height) < 5e-4
p['grid'] = [p['grid'][i]]; p['state_matrix'] = np.asarray(p['state_matrix'])[:, [i]].tolist()
original = check.solve_bvp
check.solve_bvp = lambda *args, **kw: original(*args, **(kw | {'max_nodes': 100000}))
frozen = study.frozen
study.frozen = lambda *args: (*frozen(*args)[:3], {basis: frozen(*args)[3][basis]})
study.c7(out, case, a, 'n2o')
