"""Retain the rejected warm start and compare a fresh loading-path solve."""
import json
from pathlib import Path
import runpy
import sys
import traceback

import numpy as np

out = Path('analyses/nccc_validation/results/runs/r18_co2_water_105_probe')
sys.argv = ['run_reactive_column.py', '--initial-profile',
            'analyses/nccc_validation/results/runs/reactive_native_41/solution_scaled.csv',
            '--mesh', '41', '--tol', '.05', '--kij', 'pair/carbon-dioxide/water/k_ij',
            '--factor', '1.05', '--output', str(out)]
try:
    runpy.run_path('analyses/nccc_validation/scripts/run_reactive_column.py', run_name='__main__')
except RuntimeError as error:
    traceback.print_exc()
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_code.co_name == 'solve' and frame.f_code.co_filename.endswith('reactive_bundle.py'):
            v = frame.f_locals
            liquid = v['self']
            old = liquid._accepted
            record = dict(error=str(error), inputs=v['inputs'], start=v['start'],
                          previous={k:old[k] for k in ['feed_amounts_mol','amounts_mol','density_mol_m3','parameter_fingerprint']})
            (out/'failed_state.json').write_text(json.dumps(record,indent=2,
                default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
            from mea_absorption_column.Thermodynamics.reactive_bundle import ReactiveLiquid
            fresh = ReactiveLiquid(liquid.dataset, loading_anchor=liquid.loading_anchor,
                                   water_per_mea_anchor=liquid.water_per_mea_anchor,
                                   reuse_states=False,kij_scale=liquid.kij_scale)
            temperature,pressure,*feed = v['inputs']
            result = fresh.solve(temperature,pressure,feed,state_input_derivatives=True)
            result = {k:result[k] for k in ['density_mol_m3','amounts_mol','feed_amounts_mol',
                                           'parameter_fingerprint','fugacities_pa','evidence']}
            result['calls'] = fresh.stats
            (out/'fresh_state.json').write_text(json.dumps(result,indent=2,
                default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
            print('FRESH_LOADING_PATH_PASSED', result['density_mol_m3'], flush=True)
            break
        tb = tb.tb_next
    else:
        raise
