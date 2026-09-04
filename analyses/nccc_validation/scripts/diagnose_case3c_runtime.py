"""Controlled Case 3C timing; passive observers do not change solver decisions."""
import argparse
from contextlib import contextmanager
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import time
from zipfile import ZipFile

import epcsaft
from epcsaft import equilibrium, _core
import pandas as pd
import numpy as np
import scipy.integrate._bvp as bvp

run_module = importlib.import_module('mea_absorption_column.Run_Model')
solver_module = importlib.import_module('mea_absorption_column.BVP.Methods.Scipy_BVP_Solve')
bundle = importlib.import_module('mea_absorption_column.Thermodynamics.reactive_bundle')
ROOT = Path(__file__).resolve().parents[1]
events, stack = [], []
stage = 'setup'


@contextmanager
def measured(name, details=None):
    event = dict(name=name, stage=stage, parent=stack[-1]['name'] if stack else None,
                 depth=len(stack), child_s=0., **(details or {}))
    stack.append(event)
    event['begin_ns'] = time.perf_counter_ns()
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield event
    except BaseException as error:
        event['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        event['end_ns'] = time.perf_counter_ns()
        event.update(wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)
        event['exclusive_s'] = event['wall_s']-event.pop('child_s')
        assert stack.pop() is event
        if stack:
            stack[-1]['child_s'] += event['wall_s']
        events.append(event)


def observe(owner, attribute, name=None):
    original = getattr(owner, attribute)
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with measured(name or attribute):
            return original(*args, **kwargs)
    setattr(owner, attribute, wrapped)


def install_observers():
    original = equilibrium.solve
    @functools.wraps(original)
    def solve(*args, **kwargs):
        before = _core._instrumentation()
        with measured('equilibrium.solve') as row:
            result = original(*args, **kwargs)
            row['evidence'] = dict(result.evidence)
            row['outputs'] = len(result.rows)
            row['status'] = [result.status, result.numerical_status, result.physical_status]
            row['native'] = {k: v-before[k] for k,v in _core._instrumentation().items()}
            return result
    equilibrium.solve = solve
    observe(epcsaft.Mixture, 'state', 'Mixture.state')
    observe(bundle, '_require_liquid_pressure_root', 'liquid_root_certification')
    observe(bundle, '_solve_homogeneous_reactive_result', 'reactive_request')
    observe(bundle.ReactiveLiquid, 'solve', 'ReactiveLiquid.solve')
    observe(run_module, 'save_run_outputs', 'profile_calculation')
    observe(run_module, 'scipy_BVP_solve', 'column_solver')
    original_bvp = solver_module.solve_bvp
    def solve_bvp(fun, bc, *args, **kwargs):
        def callback(fn, name):
            @functools.wraps(fn)
            def call(*a, **kw):
                details = {'nodes': len(a[0])} if name != 'boundary' else {}
                with measured(name, details):
                    return fn(*a, **kw)
            return call
        if kwargs.get('fun_jac') is not None:
            kwargs['fun_jac'] = callback(kwargs['fun_jac'], 'rhs_jacobian')
        with measured('scipy.solve_bvp'):
            return original_bvp(callback(fun,'rhs'), callback(bc,'boundary'), *args, **kwargs)
    solver_module.solve_bvp = solve_bvp
    original_newton = bvp.solve_newton
    signature = inspect.signature(original_newton)
    def solve_newton(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        col_fun = bound.arguments['col_fun']
        def collocation(y, p):
            local = inspect.currentframe().f_back.f_locals
            details = {k: local[k] for k in ('iteration','trial','alpha','njev','recompute_jac') if k in local}
            with measured('newton_residual', details):
                return col_fun(y,p)
        bound.arguments['col_fun'] = collocation
        with measured('newton_mesh_solve', {'nodes':bound.arguments['m']}):
            return original_newton(*bound.args, **bound.kwargs)
    bvp.solve_newton = solve_newton


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def main():
    global stage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--plain', action='store_true', help='No passive call observers.')
    parser.add_argument('--native-trace', action='store_true', help='Retain every enabled native scope; no runtime plot.')
    args = parser.parse_args()
    output = ROOT/'results/runs/runtime_diagnostics_20260904'/args.label
    output.mkdir(parents=True,exist_ok=False)
    runtime = Path.cwd()
    wheel = args.wheel.resolve()
    installed = importlib.metadata.distribution('epcsaft')
    with ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('epcsaft/') and not name.endswith('/'):
                assert installed.locate_file(name).read_bytes() == archive.read(name), name
    paths = [Path(__file__), runtime/'src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv']
    paths += sorted((runtime/'src/mea_absorption_column').rglob('*.py'))
    paths += sorted(Path(bundle.DATASET).glob('*.json'))
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    settings = dict(mesh_points=21,tol=.5,bc_tol=.001,max_nodes=1000,
        thermal_state_mode='temperature',transform_mode='raw',
        vapor_composition_mode='dry_saturated',gas_flow_basis='reported_dry_mass',
        return_internal_profile=True,return_profiles=True,verbose=0)
    dump(output/'identity.json',dict(label=args.label,pid=os.getpid(),wheel=str(wheel),
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),source_sha256=hashes,
        parameters=bundle.parameter_set(str(bundle.DATASET)).fingerprint,
        settings=settings.copy(),observers=not args.plain,native_trace=args.native_trace,platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)),cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
        versions={n:importlib.metadata.version(n) for n in ('epcsaft','scipy','numpy','pandas')},
        threads={n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    if not args.plain:
        install_observers()
    data = pd.read_csv(runtime/'src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv',index_col=0)
    proc_before = Path('/proc/stat').read_text().splitlines()[0]
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    _core._instrumentation_reset()
    if args.native_trace:
        _core._trace_start()
    experiment_begin_ns = time.perf_counter_ns()
    start = time.perf_counter()
    try:
        stage = 'henry_initialization'
        with measured('initialization'):
            seed = run_module.run_model(data,method='scipy-bvp',data_type='mass',run=data.index.get_loc('3C'),
                thermo_model='ideal_henry',solver_settings=settings,return_details=True,staged_beds=False)
        assert seed['success'], seed['message']
        pd.DataFrame(seed['_raw_solution_scaled'].T).to_csv(output/'initial_scaled.csv',index=False)
        settings.update(initial_guess_scaled=seed['_raw_solution_scaled'],jacobian_mode='native',reactive_reuse_states=True)
        stage = 'reactive_column'
        print(f'{args.label}: fresh Henry seed ready; starting reactive column',flush=True)
        with measured('reactive_run_model'):
            result = run_module.run_model(data,method='scipy-bvp',data_type='mass',run=data.index.get_loc('3C'),
                thermo_model='epcsaft_reactive_nine',solver_settings=settings,return_details=True,staged_beds=False)
        stage = 'serialization'
        with measured('serialization'):
            for name, table in result['_profiles'].items():
                table.to_csv(output/f'{name}.csv')
            pd.DataFrame(result['_raw_solution_scaled'].T).to_csv(output/'solution_scaled.csv',index=False)
        result = {k:v for k,v in result.items() if not k.startswith('_')}
        result['experiment_wall_s'] = time.perf_counter()-start
        result['experiment_begin_ns'] = experiment_begin_ns
        result['experiment_end_ns'] = time.perf_counter_ns()
        result['runtime_files_changed'] = [str(p) for p in paths if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)]]
        dump(output/'result.json',result)
        print(json.dumps(result,default=str),flush=True)
        assert result['success'] and not result['runtime_files_changed'], result
    except BaseException as error:
        dump(output/'failure.json',dict(error=str(error),type=type(error).__name__,wall_s=time.perf_counter()-start))
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        proc_after = Path('/proc/stat').read_text().splitlines()[0]
        if args.native_trace:
            labels, native_events = _core._trace_stop()
            np.savez_compressed(output/'native_trace.npz', labels=np.asarray(labels), events=native_events)
        dump(output/'diagnostics.json',dict(events=events,native=_core._instrumentation(),
            proc_stat_before=proc_before,proc_stat_after=proc_after,
            usage={k:getattr(usage,k)-getattr(usage_before,k) for k in
                   ('ru_utime','ru_stime','ru_nvcsw','ru_nivcsw')},max_rss_kb=usage.ru_maxrss))
        assert all(row['exclusive_s'] >= -1e-6 for row in events)


if __name__ == '__main__':
    main()
