"""Run the bounded fit-A property-source comparisons in issue #99."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.interpolate import PPoly

from check_epcsaft_properties import ANALYSIS, ARCHIVE, FIT_A
from mea_absorption_column.Thermodynamics.reactive_bundle import CORRELATION_SOURCES

SCRIPTS = Path(__file__).parent
RESULTS = ANALYSIS/'results/runs/fitA_epcsaft_properties_20261128'
HENRY_7C = ANALYSIS/'results/runs/reactive_parallel_20260904/7C_native_seed/initial_scaled.csv'
STEPS = {
    'density_enthalpy': ('liquid-density-source', 'energy-model'),
    'water_fugacity': ('liquid-density-source', 'energy-model', 'water-transfer-source'),
    'inlet_humidity': ('liquid-density-source', 'energy-model', 'water-transfer-source', 'inlet-humidity-source'),
    'density_only': ('liquid-density-source',),
    'enthalpy_only': ('energy-model',),
    'water_enthalpy_43990': (),
}


def run(label, step, case, *, mesh=21, tol=.5, limit=300, seed=None):
    output = RESULTS/label
    runner = ARCHIVE/'capture_solver.py' if case == '3C' else SCRIPTS/'run_reactive_column.py'
    command = ['timeout', str(limit), sys.executable, str(runner), '--output', str(output),
               '--dataset', str(FIT_A.relative_to(Path.cwd())), '--case', case, '--mesh', str(mesh),
               '--tol', str(tol), '--wall-limit', str(limit)]
    for key in CORRELATION_SOURCES:
        option = key.replace('_', '-')
        command.extend(('--'+option, 'epcsaft' if option in STEPS[step] else 'correlation'))
    if seed is not None:
        command.extend(('--initial-profile', str(seed)))
    elif case == '7C':
        command.extend(('--initial-profile', str(HENRY_7C)))
    if step == 'water_enthalpy_43990':
        command.extend(('--water-vaporization-enthalpy', '43990'))
    return execute(label, command, limit)


def execute(label, command, limit):
    log = RESULTS/(label+'.log')
    started = time.perf_counter()
    with log.open('x') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    record = dict(command=command, wall_limit_s=limit, elapsed_s=time.perf_counter()-started,
                  returncode=process.returncode, log=str(log), threads=1)
    (RESULTS/(label+'.execution.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(label, process.returncode, round(record['elapsed_s'], 1), flush=True)
    return process.returncode == 0


def refine(*, start_with_21=True):
    step = 'inlet_humidity'
    previous = step+'_3C'
    if start_with_21 and not run(previous, step, '3C'):
        return
    for mesh, source_mesh, tol, limit in ((41, 21, .05, 600), (81, 41, 5e-5, 1200)):
        label = f'3C_{mesh}_from_{source_mesh}'
        seed = RESULTS/(label+'.seed.csv')
        with np.load(RESULTS/previous/'solver_polynomial.npz') as saved:
            curve = PPoly(saved['coefficients'], saved['x'])
            pd.DataFrame(curve(np.linspace(0., 1., mesh))).to_csv(seed, index=False)
        if not run(label, step, '3C', mesh=mesh, tol=tol, limit=limit, seed=seed):
            break
        previous = label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--refine-existing-3C', action='store_true')
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.refine_existing_3C:
        refine(start_with_21=False)
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(refine)]
        for case in ('3C', '1C', '7C', '2C', '4C', '5C', '6C'):
            for step in ('density_enthalpy', 'water_fugacity', 'inlet_humidity'):
                if (step, case) != ('inlet_humidity', '3C'):
                    jobs.append(pool.submit(run, f'{step}_{case}', step, case))
        for step in ('density_only', 'enthalpy_only', 'water_enthalpy_43990'):
            jobs.append(pool.submit(run, step+'_3C', step, '3C'))
        for job in jobs:
            job.result()


if __name__ == '__main__':
    main()
