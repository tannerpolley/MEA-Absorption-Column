"""Retain the exact nested equilibrium inputs of a failing transport run."""
from contextlib import ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import generate_data


def validate_profile(profile):
    """Reject explicit profiles the column solver would silently ignore."""
    import numpy as np
    profile = np.asarray(profile, dtype=float)
    if profile.ndim != 2 or profile.shape[0] != 7 or profile.shape[1] < 2 or not np.isfinite(profile).all():
        raise ValueError('Expected a finite seven-state column profile with at least two positions')
    return profile


def main():
    command = sys.argv.copy()
    initial_profile = None
    anchor_start = None
    anchor_start_path = None
    if '--anchor-start' in sys.argv:
        index = sys.argv.index('--anchor-start')
        anchor_start_path = Path(sys.argv[index+1]).resolve()
        anchor_start = next(row for row in json.loads(anchor_start_path.read_text())
                            if row['label'] == 'exact_anchor_warm' and row['success'])
        del sys.argv[index:index+2]
    if '--initial-profile' in sys.argv:
        index = sys.argv.index('--initial-profile')
        initial_profile = Path(sys.argv[index+1]).resolve()
        del sys.argv[index:index+2]
    root = Path(sys.argv[sys.argv.index('--model-root')+1]).resolve()
    output = Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite retained run: {output}')
    sys.path.insert(0, str(root/'src'))
    module = importlib.import_module('mea_absorption_column.Thermodynamics.reactive_bundle')
    original = module._solve_homogeneous_reactive_result
    failures = []
    seeded = []
    run_module = importlib.import_module('mea_absorption_column.Run_Model')
    original_run = run_module.run_model
    profile_shape = None

    def initialized(*args, **kwargs):
        nonlocal profile_shape
        import pandas as pd
        profile = validate_profile(pd.read_csv(initial_profile).to_numpy().T)
        source = json.loads(initial_profile.with_name('result.json').read_text())
        if (not source['success'] or source['case_id'] != '3C'
                or source['thermal_state_mode'] != kwargs['solver_settings']['thermal_state_mode']):
            raise ValueError('Initial profile must be a converged Case 3C result with matching thermal state variables')
        profile_shape = profile.shape
        kwargs['solver_settings'] = dict(kwargs['solver_settings'], initial_guess_scaled=profile)
        return original_run(*args, **kwargs)

    def traced(dataset, temperature, pressure, apparent, **kwargs):
        try:
            # Supply a certified composition as an initial guess, never as a
            # substituted property value. The native solve/root checks still run
            # at the requested T, P and feed, with unchanged stopping tolerances.
            if anchor_start and kwargs.get('_phase_start') is None and kwargs.get('loading_anchor') is None:
                import numpy as np
                feed = np.r_[np.asarray(apparent)/sum(apparent), np.zeros(6)]
                start = module._conservative_start(np.asarray(anchor_start['amounts_mol']),
                    np.asarray(anchor_start['feed_amounts_mol']), feed,
                    kwargs['reactions'], kwargs['molar_masses'])
                kwargs['_phase_start'] = dict(amounts_mol=start.tolist(),
                    molar_volume_m3_per_mol=1/anchor_start['density_mol_m3'])
                seeded.append(dict(temperature_K=float(temperature), pressure_Pa=float(pressure),
                                   apparent_amounts_mol=list(map(float, apparent))))
            return original(dataset, temperature, pressure, apparent, **kwargs)
        except Exception:
            failures.append(dict(temperature_K=float(temperature), pressure_Pa=float(pressure),
                apparent_amounts_mol=list(map(float, apparent)),
                options={k:v for k,v in kwargs.items() if k not in ('model', 'thermochemistry', '_diagnostics')},
                traceback=traceback.format_exc()))
            raise

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, '_solve_homogeneous_reactive_result', traced))
            if initial_profile:
                stack.enter_context(patch.object(run_module, 'run_model', initialized))
            generate_data.main()
    finally:
        if output.exists():
            (output/'failure_trace.json').write_text(json.dumps(dict(
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                failures=failures), indent=2, default=str)+'\n')
            if initial_profile or anchor_start:
                (output/'initialization.json').write_text(json.dumps(dict(command=command,
                    initial_profile=None if initial_profile is None else str(initial_profile),
                    sha256=None if initial_profile is None else hashlib.sha256(initial_profile.read_bytes()).hexdigest(),
                    accepted_profile_shape=profile_shape,
                    initial_grid='uniform normalized height, bottom to top; same convention as source export',
                    anchor_start=None if anchor_start_path is None else str(anchor_start_path),
                    anchor_start_sha256=None if anchor_start_path is None else hashlib.sha256(anchor_start_path.read_bytes()).hexdigest(),
                    seeded_native_solves=seeded,
                    purpose='Only initial guesses change after the diagnosed cold-anchor failure; '
                            'physical inputs, transport multiplier, equations and tolerances are unchanged.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
