"""Rerun seven one-at-a-time Case 3C conditions with retained, bounded subprocesses."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

FIGURE = Path(__file__).resolve().parents[1]
ROOT = FIGURE.parents[3]
RUNS = FIGURE.parents[1] / 'results/runs/reactive_operating_rerun_20260904'
SCENARIOS = {
    'baseline_current': {}, 'LG_low': {'L': .9}, 'LG_high': {'L': 1.1},
    'loading_low': {'alpha': .225}, 'loading_high': {'alpha': .275},
    'temperature_low': {'Tl': 313.15}, 'temperature_high': {'Tl': 323.15},
}


def run(item, initial_profile=None):
    name, changes = item
    output = RUNS / (name + '_native_seed' if initial_profile else name)
    inputs = RUNS / f'{output.name}_inputs.csv'
    data = pd.read_csv(ROOT / 'src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv', index_col=0).loc[['3C']]
    for column, value in changes.items():
        data.loc['3C', column] = data.loc['3C', column] * value if column == 'L' else value
    data.to_csv(inputs)
    command = [sys.executable, str(ROOT / 'analyses/nccc_validation/scripts/run_reactive_column.py'),
               '--inputs', str(inputs), '--output', str(output)]
    if initial_profile:
        command += ['--initial-profile', str(initial_profile.resolve())]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (RUNS / f'{output.name}.log').open('w') as log:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            status = dict(returncode=result.returncode, timeout_s=900, command=command)
        except subprocess.TimeoutExpired:
            status = dict(returncode=None, status='timeout', timeout_s=900, command=command)
    output.mkdir(exist_ok=True)
    (output / 'execution.json').write_text(json.dumps(status, indent=2) + '\n')
    data.to_csv(output / 'inputs.csv')
    print(name, status.get('status', status['returncode']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-profile', type=Path)
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=bool(args.initial_profile))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda name: run((name, SCENARIOS[name]), args.initial_profile), args.scenarios))


if __name__ == '__main__':
    main()
