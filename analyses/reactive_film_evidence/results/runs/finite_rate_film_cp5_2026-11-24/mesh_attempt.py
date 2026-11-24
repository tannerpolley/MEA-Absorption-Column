import json, sys
from pathlib import Path
from mea_absorption_column.column import run_column
root=Path('analyses/reactive_film_evidence/results/runs/finite_rate_film_cp5_2026-11-24')
case=Path('analyses/nccc_validation/results/runs/joint_fit_f2_20261123/bridge/7C/17/cosine/request.json')
n=int(sys.argv[1]); seed=Path(sys.argv[2]); out=root/f'7C_n{n}'/'try2'
r=json.loads(case.read_text()); r['model']={'film_model':'finite_rate_film','henry_basis':'n2o_analogy'}
r['initialization']={'values':{'retained_profile':str(seed)}}
r['numerics']['solver_settings'].update(nodes=n,max_iterations=20)
r['execution'].update(output_dir=str(out),wall_limit_s={33:3600,65:7200}[n])
out.parent.mkdir(parents=True,exist_ok=True); (root/f'7C_n{n}_try2_request.json').write_text(json.dumps(r,indent=2)+'\n')
a=run_column(r); print(a['execution'],a['solver'],a.get('failure'),flush=True)
