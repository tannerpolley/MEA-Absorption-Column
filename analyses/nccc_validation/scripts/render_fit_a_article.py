"""Curate the fit-A campaign, operating sweep and article verification tables."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

import numpy as np
import pandas as pd

ANALYSIS = Path('analyses/nccc_validation')
FIGURES = ANALYSIS/'figures'
LATEX = Path('docs/scientific/latex/tables')
INPUTS = Path('src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv')


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2)+'\n')


def verify_profile(run):
    from render import read_run

    frame, row, identity, sources = read_run(run)
    result = json.loads((run/'result.json').read_text())
    assert result['capture_pct'] == row['capture_pct']
    assert 0 <= row['capture_pct'] <= 100
    assert np.isclose(frame.capture_pct.iloc[-1], row['capture_pct'], rtol=0, atol=1e-8)
    feed = (frame.Fl_CO2 + frame.Fl_MEA + frame.Fl_H2O).min()
    assert row['species_balance_max_mol_s']/feed < 1e-8
    assert row['charge_max_mol_s']/feed < 1e-10
    assert not result.get('runtime_files_changed')
    dataset = Path(identity['settings']['reactive_dataset'])
    assert dataset.name == 'MEA_A_196faf11'
    parameter = dataset/'parameters.json'
    digest = hashlib.sha256(parameter.read_bytes()).hexdigest()
    assert digest == identity['input_sha256'][str(parameter)]
    row.update(success=True, status='converged', parameter_sha256=digest,
               case_id=result['case_id'], source_review_required=False)
    return frame, row, sources


def campaign(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    source = FIGURES/'reactive_parallel/inputs/morgan2020_table_c2.csv'
    observations = pd.read_csv(source).set_index('case_id')
    case_inputs = pd.read_csv(INPUTS, index_col=0)
    rows, profiles, taps, sources = [], [], [], [source, INPUTS, Path(__file__)]
    for case in (f'{i}C' for i in range(1,8)):
        run = runs/('coarse' if case == '3C' else f'campaign_{case}')
        if (run/'result.json').is_file():
            frame, row, paths = verify_profile(run)
            assert row['case_id'] == case
            sources.extend(paths)
            profiles.append(frame.assign(case_id=case, capture_axial_pct=frame.capture_pct))
        else:
            assert case == '1C'
            frame = None
            row = dict(case_id=case, success=False, status='original_cold_start_unavailable',
                       observed_capture_pct=case_inputs.loc[case, 'CO2  %'])
        rows.append(row)
        for column, temperature in observations.loc[case].items():
            source_position = float(column.removeprefix('position_').removesuffix('_C'))
            position = 1-source_position
            taps.append(dict(case_id=case, source_position_top_to_bottom=source_position,
                             Position=position, reported_temperature_C=temperature, reported_temperature_K=temperature+273.15,
                             predicted_liquid_K=None if frame is None else np.interp(position, frame.Position, frame.Tl),
                             predicted_vapor_K=None if frame is None else np.interp(position, frame.Position, frame.Tv)))
    results = pd.DataFrame(rows)
    results.to_csv(output/'summary.csv', index=False)
    pd.concat(profiles, ignore_index=True).to_csv(output/'profiles.csv', index=False)
    pd.DataFrame(taps).to_csv(output/'temperature_observations.csv', index=False)
    renderer = FIGURES/'reactive_parallel/scripts/render.py'
    runpy.run_path(str(renderer))['render_figures'](output)
    write_json(output/'provenance.json', dict(
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources, renderer]},
        cases_used=results.loc[results.success, 'case_id'].tolist(),
        observed_capture_source=str(INPUTS),
        mean_absolute_capture_error_pp=results.capture_error_pct.abs().mean()))
    return results.loc[results.success]


def operating(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    scenarios = {
        'baseline':'coarse', 'LG_low':'operating_LG_low_seeded', 'LG_high':'operating_LG_high_seeded',
        'loading_low':'operating_loading_low', 'loading_high':'operating_loading_high',
        'temperature_low':'operating_temperature_low_seeded', 'temperature_high':'operating_temperature_high',
    }
    rows, frames, sources = [], [], [Path(__file__)]
    for name, directory in scenarios.items():
        run = runs/directory
        frame, row, paths = verify_profile(run)
        identity = json.loads((run/'identity.json').read_text())
        command = identity['command']
        inputs = Path(command[command.index('--inputs')+1]) if '--inputs' in command else INPUTS
        params = pd.read_csv(inputs, index_col=0).loc['3C']
        row.update(scenario=name, mass_L_over_reported_dry_G=params.L/params.G,
                   lean_loading_mol_mol=params.alpha, liquid_inlet_temperature_K=params.Tl,
                   liquid_inlet_temperature_C=params.Tl-273.15)
        rows.append(row)
        frames.append(frame[['Position','Tl','Tv']].assign(scenario=name))
        sources.extend([*paths, inputs, run/'execution.json'])
    result = pd.DataFrame(rows)
    result['capture_change_pp'] = result.capture_pct-result.capture_pct.iloc[0]
    result['peak_liquid_temperature_change_K'] = result.peak_liquid_temperature_K-result.peak_liquid_temperature_K.iloc[0]
    result.to_csv(output/'summary.csv', index=False)
    pd.concat(frames, ignore_index=True).to_csv(output/'temperature_profiles.csv', index=False)
    renderer = FIGURES/'reactive_operating/scripts/render.py'
    runpy.run_path(str(renderer))['render_figures'](output)
    write_json(output/'provenance.json', dict(
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources, renderer]},
        scope='Fit-A Case 3C operating calculations at 21 initial nodes and tolerance 0.5.'))
    return result


def iterations(run):
    log = run.with_name(run.name+'.log')
    trace = log.read_text().split('accepted initialization available')[-1].splitlines()
    rhs, jac, state_calls, rows = 0, 0, 0, []
    for line in trace:
        if match := re.match(r'column (rhs|jacobian)_batches=(\d+) nodes=(\d+)', line):
            if match[1] == 'rhs':
                rhs = int(match[2])
            else:
                jac = int(match[2])
                state_calls += int(match[3])
        if match := re.fullmatch(r'\s*(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s*', line):
            rows.append(dict(run=run.name, mesh_iteration=int(match[1]), nodes=int(match[4]),
                             max_rms_residual_printed=match[2], max_scaled_boundary_residual_printed=match[3],
                             nodes_added=int(match[5]), rhs_batches_cumulative=rhs,
                             jacobian_batches_cumulative=jac, state_jacobian_evaluations=state_calls))
    return rows


def replace_table_rows(path, values):
    rows = path.read_text().splitlines()
    for number, line in enumerate(rows):
        key = line.split(' &', 1)[0].strip()
        if key in values:
            rows[number] = '    '+key+' & '+' & '.join(values[key])+r' \\'
    path.write_text('\n'.join(rows)+'\n')


def verification(runs, figures, summary):
    rows = [item for name in ('coarse','refined') for item in iterations(runs/name)]
    pd.DataFrame(rows).to_csv(figures/'refinement/iteration_history.csv', index=False)
    formatted = {key: [] for key in (
        'Initial / final mesh nodes', 'Collocation tolerance', 'Mesh iterations',
        r'Maximum RMS residual, iterations 1 $\to$ 2', 'Final maximum RMS residual',
        'Maximum scaled boundary residual', 'Column RHS evaluation batches', 'Jacobian evaluation batches',
        'State Jacobian evaluations', 'Liquid equilibrium solves, including profile output',
        r'\COtwo{} capture (\unit{\percent})', r'Peak liquid temperature (\unit{K})',
        r'Net-enthalpy-flow range, 101 positions (\unit{W})')}
    for row in summary.itertuples():
        hist = [x for x in rows if x['run']==row.run]
        last = hist[-1]
        assert row.solver_iterations == len(hist)
        values = (f'{row.mesh_points} / {row.final_mesh_nodes}', f'{row.tol:g}', str(row.solver_iterations),
                  '$'+r'\to'.join(f"{float(x['max_rms_residual_printed']):g}" for x in hist)+'$',
                  f'{row.max_rms_residual:.6f}', f'{row.max_scaled_boundary_residual:g}',
                  str(last['rhs_batches_cumulative']), str(last['jacobian_batches_cumulative']),
                  str(last['state_jacobian_evaluations']), str(row.reactive_native_solves),
                  f'{row.capture_pct:.5f}', f'{row.peak_liquid_temperature_K:.4f}', f'{row.net_energy_range_W:.3f}')
        for key, value in zip(formatted, values, strict=True):
            formatted[key].append(value)
    replace_table_rows(LATEX/'reactive_numerical_verification.tex', formatted)


def span(values, fmt):
    a, b = format(values.min(), fmt), format(values.max(), fmt)
    return a if a == b else a+'--'+b


def diagnostics(groups):
    path = LATEX/'run_diagnostics.tex'
    original = path.read_text().splitlines()
    data = []
    balance = []
    for name, group in groups.items():
        label = f'{name} ({len(group)})'
        fields = [
            span(group.mesh_points, '.0f'), span(group.final_mesh_nodes, '.0f'),
            span(group.tol, 'g'), span(group.solver_iterations, '.0f'),
            bound(group.max_rms_residual), bound(group.max_scaled_boundary_residual),
        ]
        data.append(label+' & '+' & '.join(fields)+r' \\')
        fields = [bound(group.species_balance_max_mol_s), bound(group.charge_max_mol_s),
                  span(group.net_energy_range_W, '.2f'), '--']
        balance.append(label+' & '+' & '.join(fields)+r' \\')
    for rows in (data, balance):
        rows.append(r'Additional Case 3C refinements & \multicolumn{'+str(6 if rows is data else 4)+r'}{l}{Original run inputs unavailable} \\')
    blocks = iter((data, balance))
    lines = []
    for line in original:
        if line.startswith(('Campaign (','Paired Case','Thermodynamic (','Transport (','Operating (','Timing (','Additional Case')):
            continue
        if line == r'\bottomrule':
            lines.extend(next(blocks))
        if line.startswith('RMS is'):
            line = ('RMS is the maximum normalized differential residual. BC is the maximum scaled boundary residual. '
                    'Discrepancies are rounded upward. Shared baselines occur in more than one group. '
                    'Thermodynamic rows exclude the reference. Transport includes the factor-one control. '
                    'Case 1C is excluded until its original equilibrium start can be reproduced. '
                    'Dense energy diagnostics require the original additional-refinement runs.')
        if line.startswith(r'\caption{'):
            line = r'\caption{Convergence and balance diagnostics for the fit-A calculations. Counts identify the calculations included in each group.}'
        lines.append(line)
    path.write_text('\n'.join(lines)+'\n')


def bound(values):
    maximum = values.max()
    if maximum == 0:
        return '0'
    exponent = int(np.floor(np.log10(maximum)))
    mantissa = np.ceil(maximum / 10.**exponent * 100)/100
    return rf'$\leq\num{{{mantissa:.2f}e{exponent:+d}}}$'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    args = parser.parse_args()
    runs = args.runs
    figures = runs/'figures'
    sys.path.insert(0, str(FIGURES/'reactive_column/scripts'))
    cases = campaign(runs, figures/'campaign')
    operating_sweep = operating(runs, figures/'operating')
    paired = pd.read_csv(figures/'refinement/summary.csv')
    thermo = pd.read_csv(figures/'thermodynamics/summary.csv')
    transport = pd.concat([pd.read_csv(figures/'transport'/f'{stem}.csv')
                           for stem in ('summary','initialization_control')], ignore_index=True)
    timing = []
    for count in range(1,4):
        run = runs/f'timing_{count}'
        _, row, _ = verify_profile(run)
        log = run.with_name(run.name+'.log')
        row['peak_rss_mib'] = int(re.search(r'^peak_rss_kib=(\d+)$', log.read_text(), re.M)[1])/1024
        timing.append(row)
    timing = pd.DataFrame(timing)
    timing.to_csv(runs/'timing.csv', index=False)
    verification(runs, figures, paired)
    diagnostics({'Campaign':cases, 'Paired Case 3C':paired, 'Thermodynamic':thermo.iloc[1:],
                 'Transport':transport, 'Operating':operating_sweep, 'Timing':timing})


if __name__ == '__main__':
    main()
