"""Read the parallel campaign; retain audited tables and multi-case figures."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE = Path(__file__).resolve().parents[1]
ANALYSIS = FIGURE.parents[1]
RUNS = ANALYSIS/'results/runs/reactive_parallel_20260904'
RUNTIME = Path('/home/tnnrpolley21/.codex/worktrees/bad5/MEA-Absorption-Column')
PAPER = Path('/home/tnnrpolley21/Zotero/storage/HX2358GV/Morgan et al. - 2020 - Development of a framework for sequential Bayesian design of experiments Application to a pilot-sca.pdf')


def main():
    output = FIGURE/'output'
    output.mkdir(parents=True, exist_ok=True)
    reaction_path = RUNTIME/'src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle/reaction-system.json'
    reactions = json.loads(reaction_path.read_text())
    balances, charges = np.asarray(reactions['balance_matrix']), np.asarray(reactions['charges'])
    taps_path = FIGURE/'inputs/morgan2020_table_c2.csv'
    reported = pd.read_csv(taps_path).set_index('case_id')
    assert reported.shape == (7,5) and np.isfinite(reported.to_numpy()).all()
    summary, profiles, sources, taps = [], [], [Path(__file__), reaction_path, taps_path, PAPER], []
    for case in (f'{i}C' for i in range(1, 8)):
        run = RUNTIME/'analyses/nccc_validation/results/runs/reactive_native_21' if case == '3C' else RUNS/case
        current_baseline=RUNS.parent/'reactive_operating_20260904/baseline_current'
        if case=='3C' and (current_baseline/'result.json').exists():
            run=current_baseline
        if case == '7C' and (RUNS/'7C_native_seed/result.json').exists():
            run = RUNS/'7C_native_seed'
        confirmed = RUNS.parent/'reactive_confirmed_20260904/retry'/case
        if (confirmed/'result.json').exists():
            run = confirmed
        declared = RUNS.parent/'reactive_confirmed_20260904/declared_start'/case
        if (declared/'result.json').exists():
            run = declared
        if case == '3C':
            run = RUNS.parent/'runtime_diagnostics_20260904/candidate_plain'
        result_path = run/'result.json'
        row = {'case_id':case, 'run_directory':str(run)}
        if not result_path.exists():
            row.update(status='no_column_result', success=False)
            summary.append(row)
            continue
        identity_path = run/'identity.json'
        result, identity = json.loads(result_path.read_text()), json.loads(identity_path.read_text())
        assert result['case_id'] == case
        sources.extend([result_path, identity_path])
        changed = result.get('runtime_files_changed', [])
        row['source_changed_on_disk_during_run'] = bool(changed)
        row['source_review_required'] = bool(changed)
        row['cold_start_seed_fraction'] = identity.get('declared_cold_start_seed_fraction', 1e-3)
        row.update({key:result.get(key) for key in ('success', 'capture_pct','capture_error_pct',
            'mesh_points','tol','final_mesh_nodes','solver_iterations','runtime_s','solver_cpu_time_s',
            'total_wall_including_seed_and_outputs_s','max_rms_residual','max_scaled_boundary_residual',
            'boundary_residual_norm','invalid_state_count','guard_penalty_count')})
        parameter_hash, = [h for p,h in identity.get('input_sha256', identity.get('source_sha256', {})).items() if p.endswith('/MEA_reactive_epcsaft_bundle/parameters.json')]
        row['wheel_sha256'] = identity.get('wheel_sha256', identity.get('engine_identity', {}).get('wheel_sha256'))
        row.update(status='converged' if result['success'] else 'solver_rejected', parameter_sha256=parameter_hash)
        if not result['success']:
            summary.append(row)
            continue
        tables = {}
        for name in ('Fl','Fv','Cl','T','CO2','Hl','Hv','enhance_factor'):
            path = run/f'{name}.csv'
            sources.append(path)
            tables[name] = pd.read_csv(path).set_index('Position')
            assert np.isfinite(tables[name].to_numpy(dtype=float)).all(), (case, name)
        frame = pd.concat([tables[name] for name in ('Fl','Fv','Cl','T','CO2')], axis=1)
        frame['capture_axial_pct'] = 100*(1-frame.Fv_CO2/frame.Fv_CO2.iloc[0])
        frame['net_energy_W'] = tables['Hv'].Hvf-tables['Hl'].Hlf
        frame['E'] = tables['enhance_factor'].E
        frame['case_id'] = case
        truth = tables['Fl'].filter(regex='_true$').to_numpy()
        apparent = tables['Fl'][['Fl_CO2','Fl_MEA','Fl_H2O']].to_numpy()
        assert truth.shape[1] == 9 and (truth > 0).all()
        assert (tables['Cl'].filter(regex='_true$') > 0).all().all()
        assert 0 <= result['capture_pct'] <= 100
        assert np.isclose(frame.capture_axial_pct.iloc[-1], result['capture_pct'], atol=1e-8, rtol=0)
        row.update(observed_capture_pct=result['capture_pct']-result['capture_error_pct'],
            peak_liquid_temperature_K=frame.Tl.max(), peak_vapor_temperature_K=frame.Tv.max(),
            min_temperature_K=frame[['Tl','Tv']].min().min(),
            co2_conservation_range_mol_s=np.ptp(frame.Fv_CO2-frame.Fl_CO2),
            water_conservation_range_mol_s=np.ptp(frame.Fv_H2O-frame.Fl_H2O),
            species_balance_max_mol_s=np.max(np.abs(truth@balances.T-apparent@balances[:,:3].T)),
            charge_max_mol_s=np.max(np.abs(truth@charges)),
            net_energy_range_W=np.ptp(frame.net_energy_W),
            enhancement_min=frame.E.min(), enhancement_max=frame.E.max())
        for column, temperature in reported.loc[case].items():
            source_position = float(column.removeprefix('position_').removesuffix('_C'))
            # Morgan2020 Appendix C p.22: source 0=top, 1=bottom.
            position = 1-source_position
            taps.append(dict(case_id=case, source_position_top_to_bottom=source_position,
                Position=position, reported_temperature_C=temperature,
                reported_temperature_K=temperature+273.15,
                predicted_liquid_K=np.interp(position, frame.index, frame.Tl),
                predicted_vapor_K=np.interp(position, frame.index, frame.Tv)))
        summary.append(row)
        profiles.append(frame.reset_index())
    summary = pd.DataFrame(summary)
    summary.to_csv(output/'summary.csv', index=False)
    if not profiles:
        raise RuntimeError('No profiles available to plot')
    pd.concat(profiles, ignore_index=True).to_csv(output/'profiles.csv', index=False)
    pd.DataFrame(taps).to_csv(output/'temperature_observations.csv', index=False)
    # Figures consume the exact retained tables, not a rerun or an opaque live object.
    summary = pd.read_csv(output/'summary.csv')
    profiles = pd.read_csv(output/'profiles.csv')
    taps = pd.read_csv(output/'temperature_observations.csv')
    accepted = summary[summary.success == True]
    assert accepted.parameter_sha256.nunique() == 1
    assert len(taps) == 5*len(accepted)
    np.testing.assert_allclose(taps.Position+taps.source_position_top_to_bottom, 1, atol=1e-15)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(8,4.3), layout='constrained')
    x = np.arange(len(accepted))
    ax.plot(x-.07, accepted.observed_capture_pct, 'kx', ms=8, label='NCCC observation')
    ax.plot(x+.07, accepted.capture_pct, 'o', color='#0072B2', ms=6, label='Nine-species ePC-SAFT')
    ax.set(xticks=x, xticklabels=accepted.case_id, xlabel='NCCC one-bed case',
           ylabel='CO₂ capture (%)', ylim=(0,103), title='Conventional enhancement-factor column')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)
    if accepted.source_review_required.any():
        fig.supxlabel('Provisional: shared source changed on disk during execution; see provenance notes.', fontsize=8)
    save(fig, output/'capture_comparison')
    fig, axes = plt.subplots(4,2, figsize=(9,10), layout='constrained', sharex=True, sharey=True)
    for ax, case in zip(axes.flat, (f'{i}C' for i in range(1,8))):
        frame = profiles[profiles.case_id == case]
        ax.set_title(case, loc='left')
        if frame.empty:
            ax.text(.5,.5,'No converged column result', transform=ax.transAxes, ha='center')
        else:
            ax.plot(frame.Position,frame.Tl,color='#0072B2',label='Predicted liquid')
            ax.plot(frame.Position,frame.Tv,'--',color='#D55E00',label='Predicted vapor')
            points = taps[taps.case_id == case]
            ax.scatter(points.Position, points.reported_temperature_K, marker='x', color='black',
                       s=28, zorder=4, label='NCCC packing temperature')
            metrics = accepted.set_index('case_id').loc[case]
            ax.text(.04,.07,f'Capture: {metrics.capture_pct:.1f}% model / {metrics.observed_capture_pct:.1f}% measured',
                    transform=ax.transAxes,fontsize=8)
        ax.set(xlim=(0,1), ylabel='Temperature (K)')
        ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes[0,0].get_legend_handles_labels(),loc='upper left',frameon=False,fontsize=9)
    axes.flat[-1].text(0,.56,'Measurements: Morgan et al. (2020), Table C2.\nSource coordinate transformed: z = 1 − x.\n\nBottom: vapor inlet (0)\nTop: liquid inlet (1)\n\nNine-species reactive ePC-SAFT;\nconventional enhancement-factor transport.',va='top',linespacing=1.5,fontsize=9,transform=axes.flat[-1].transAxes)
    axes[3,0].set_xlabel('Normalized packed height')
    axes[2,1].set_xlabel('Normalized packed height')
    axes[2,1].tick_params(labelbottom=True)
    if accepted.source_review_required.any():
        fig.supxlabel('Provisional: shared source changed during execution; see provenance notes.', fontsize=8)
    save(fig, output/'temperature_profiles')
    fig, ax = plt.subplots(figsize=(6.5,4), layout='constrained')
    frame = profiles[profiles.case_id == '3C']
    points = taps[taps.case_id == '3C']
    ax.plot(frame.Position,frame.Tl,color='#0072B2',label='Predicted liquid')
    ax.plot(frame.Position,frame.Tv,'--',color='#D55E00',label='Predicted vapor')
    ax.scatter(points.Position,points.reported_temperature_K,marker='x',color='black',s=32,
               zorder=4,label='NCCC packing temperature')
    ax.set(xlim=(0,1),xlabel='Normalized packed height, bottom to top',ylabel='Temperature (K)',
           title='Case 3C: nine-species reactive ePC-SAFT')
    ax.legend(frameon=False,fontsize=9,loc='lower center')
    ax.grid(alpha=.2)
    save(fig, output/'case_3c_temperature')
    write = {'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
             'scope':'Nine-species conventional enhancement, empirical energy; coarse campaign. '
                     'Run directories and wheel identities are in summary.csv. No seven-case mesh-convergence claim.',
             'observations':{'doi':'10.1016/j.apenergy.2020.114533', 'attachment_key':'HX2358GV',
                'table':'C2, page 27, all seven one-bed cases, Celsius',
                'coordinate_definition':'Appendix C, page 22: source x=0 top, x=1 bottom; model z=1-x',
                'phase':'Source labels absorber temperature; no phase-specific sensor claim.',
                'verification':'All 35 entries and coordinate definition visually checked against PDF; 1C-6C match repository NCCC_2017_absorber_temperature_profiles.csv.',
                'uncertainty':'Not supplied in Table C2; no invented error bars.'}}
    write['source_review_required'] = bool(accepted.source_review_required.any())
    write['source_note'] = ('Historical runs and warnings are retained unchanged. Figure inputs prefer confirmed unchanged-source '
        'reruns where available; the source_review_required flag describes only the selected runs.')
    (output/'provenance.json').write_text(json.dumps(write,indent=2)+'\n')
    print(summary.to_string(index=False))


def save(fig, stem):
    for extension in ('svg','png','pdf'):
        fig.savefig(stem.with_suffix('.'+extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
