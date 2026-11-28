"""Replay the retained fit-A article with issue #99 bulk properties."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

NEW = Path(__file__).parent.relative_to(Path.cwd())
OLD = NEW.with_name('fitA_article_20261128')
ISSUE = NEW.with_name('fitA_epcsaft_properties_20261128')
SEED = NEW/'7C_initial_scaled.csv'
FLAGS = ['--'+key for key in ('liquid-density-source', 'energy-model', 'water-transfer-source', 'inlet-humidity-source')]
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           EPCSAFT_HSL_LIBRARY='/home/tnnrpolley21/.cache/epcsaft-hsl/coinhsl-lib/libhsl.so')


def replay(path):
    name = 'campaign_1C' if path.parent == ISSUE else path.parent.name
    output = NEW/name if path.parent == ISSUE else NEW/path.parent.relative_to(OLD)
    if (output/'execution.json').exists():
        return
    prior = json.loads(path.read_text())
    original = prior['command']
    index = next(i for i, arg in enumerate(original) if arg.endswith(('python', 'python3')))
    args = [arg.replace(str(OLD), str(NEW)) for arg in original[index+1:]]
    args[args.index('--output')+1] = str(output)
    if output.parent.name == 'transport_reproduction' and '--initial-profile' in args:
        index = args.index('--initial-profile')+1
        args[index] = str(output.parent/'transport_baseline/solution_scaled.csv')
    limit = prior['wall_limit_s']
    command = ['nice', '-n', '10', 'timeout', str(limit)]
    if name.startswith('timing_'):
        command += ['/usr/bin/time', '-f', 'peak_rss_kib=%M']
    if args[0].endswith(('run_reactive_column.py', 'capture_solver.py')):
        for flag in FLAGS:
            if flag not in args:
                args.extend((flag, 'epcsaft'))
        if '--wall-limit' not in args:
            args.extend(('--wall-limit', str(limit)))
    if name == 'campaign_7C':
        seed_index = args.index('--initial-profile')+1
        shutil.copyfile(args[seed_index], SEED)
        args[seed_index] = str(SEED)
    command += ['uv', 'run', '--frozen', 'python', *args]
    log = output.with_name(output.name+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open('x') as stream:
        status = subprocess.run(command, env=ENV, stdout=stream, stderr=subprocess.STDOUT).returncode
    output.mkdir(exist_ok=True)
    record = dict(command=command, wall_limit_s=limit, elapsed_s=time.monotonic()-start,
                  returncode=status, log=str(log), threads=1, retained_command_source=str(path))
    (output/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    print(name, status, round(record['elapsed_s'], 1), flush=True)


def measure_energy(name):
    output = NEW/name
    limit = 600
    log = NEW/(name+'.energy.log')
    if (output/'energy_check.json').exists():
        return
    command = ['timeout', str(limit), 'uv', 'run', '--frozen', 'python',
               str(NEW/'measure_energy.py'), str(output)]
    start = time.monotonic()
    with log.open('x') as stream:
        status = subprocess.run(command, env=ENV, stdout=stream, stderr=subprocess.STDOUT).returncode
    record = dict(command=command, wall_limit_s=limit, elapsed_s=time.monotonic()-start,
                  returncode=status, log=str(log), threads=1)
    (output/'energy_execution.json').write_text(json.dumps(record, indent=2)+'\n')
    print(name, 'energy', status, round(record['elapsed_s'], 1), flush=True)


def main():
    paths = {p.parent.name:p for p in OLD.glob('*/execution.json')
             if not p.parent.name.startswith(('nce_', 'legacy_'))}
    paths['campaign_1C'] = ISSUE/'inlet_humidity_1C.execution.json'
    # The issue #99 record supplies the 1C command because the article lacks execution.json.
    rows = {name:path for name,path in paths.items() if name != 'campaign_1C'}
    initial = [path for name,path in rows.items() if name == 'coarse' or
               name.startswith('campaign_') or name in
               ('operating_LG_low','operating_LG_high','operating_loading_low',
                'operating_loading_high','operating_temperature_low','operating_temperature_high')]
    paired = [rows['refined'], rows['energy_paired_replay']] + [
        path for name,path in rows.items() if name.startswith('operating_') and name.endswith('_seeded')]
    remaining = [path for name,path in rows.items() if name.startswith(('thermo_', 'transport_', 'energy_dense_'))
                 and name != 'transport_factor_one_control']
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(replay, [*initial, paths['campaign_1C']]))
        list(pool.map(replay, paired))
        list(pool.map(replay, remaining))
    replay(rows['transport_factor_one_control'])
    for i in range(1, 4):
        replay(rows[f'timing_{i}'])
    reproduce = OLD/'transport_reproduction'
    control = reproduce/'transport_factor_one_control/execution.json'
    with ThreadPoolExecutor(max_workers=4) as pool:
        dense = [pool.submit(measure_energy, name)
                 for name in ('energy_paired_replay', 'energy_dense_41', 'energy_dense_81')]
        list(pool.map(replay, sorted(set(reproduce.glob('*/execution.json')) - {control})))
        replay(control)
        for job in dense:
            job.result()


if __name__ == '__main__':
    main()
