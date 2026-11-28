"""Curate and plot accepted current-parameter column outputs; never run a model."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_run(run):
    # Carbon, nitrogen and charge are fixed by the nine-species formulation.
    # The adopted EOS parameters and reaction constants do not enter this check.
    reactions = Path('src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle/reaction-system.json')
    chemistry = json.loads(reactions.read_text())
    balances, charges = np.asarray(chemistry['balance_matrix']), np.asarray(chemistry['charges'])
    paths = [run/'identity.json', run/'result.json']
    identity, result = [json.loads(p.read_text()) for p in paths]
    if not result['success']:
        raise ValueError(f'{run}: no accepted column result')
    if result['max_rms_residual'] > result['tol'] or result['max_scaled_boundary_residual'] > result['bc_tol']:
        raise ValueError(f'{run}: residual exceeds the declared tolerance')
    if result.get('invalid_state_count') or result.get('guard_penalty_count'):
        raise ValueError(f'{run}: invalid state or guard penalty recorded')
    evaluated = run/'evaluated_parameters.json'
    if evaluated.exists():
        components = json.loads(evaluated.read_text())['components']
        if [c['component_id'] for c in components] != chemistry['species_ids']:
            raise ValueError(f'{run}: species order differs from the balance matrix')
        paths.append(evaluated)
    tables = {}
    for name in ['Fl', 'Fv', 'Cl', 'x', 'T', 'CO2', 'Hl', 'Hv', 'enhance_factor']:
        p = run/f'{name}.csv'
        paths.append(p)
        tables[name] = pd.read_csv(p).set_index('Position')
    frame = pd.concat([tables[name] for name in ['Fl', 'Fv', 'Cl', 'x', 'T', 'CO2']], axis=1)
    frame['capture_pct'] = 100*(1-frame.Fv_CO2/frame.Fv_CO2.iloc[0])
    frame['net_energy_W'] = tables['Hv'].Hvf-tables['Hl'].Hlf
    frame['E'] = tables['enhance_factor'].E
    frame['mesh'] = result['mesh_points']
    frame['tol'] = result['tol']
    frame['run'] = run.name
    flows = tables['Fl']
    truth = flows.filter(regex='_true$').to_numpy()
    app = flows[['Fl_CO2', 'Fl_MEA', 'Fl_H2O']].to_numpy()
    if not np.isfinite(frame.to_numpy(dtype=object)[:, :-1].astype(float)).all():
        raise ValueError('Nonfinite retained profile')
    if (truth <= 0).any() or (tables['Cl'].filter(regex='_true$') <= 0).any().any():
        raise ValueError('Nonpositive species in retained profile')
    row = {key:result.get(key) for key in ['mesh_points','tol','final_mesh_nodes','solver_iterations',
           'runtime_s','solver_cpu_time_s','total_wall_including_seed_and_outputs_s','max_rms_residual',
           'max_scaled_boundary_residual','boundary_residual_norm','capture_pct','capture_error_pct',
           'invalid_state_count','guard_penalty_count']}
    row.update(run=run.name, run_directory=str(run), observed_capture_pct=result['capture_pct']-result['capture_error_pct'],
               peak_liquid_temperature_K=frame.Tl.max(), peak_vapor_temperature_K=frame.Tv.max(),
               co2_conservation_range_mol_s=np.ptp(frame.Fv_CO2-frame.Fl_CO2),
               water_conservation_range_mol_s=np.ptp(frame.Fv_H2O-frame.Fl_H2O),
               species_balance_max_mol_s=np.max(np.abs(truth@balances.T-app@balances[:,:3].T)),
               charge_max_mol_s=np.max(np.abs(truth@charges)),
               net_energy_range_W=np.ptp(frame.net_energy_W),
               net_energy_relative_range=np.ptp(frame.net_energy_W)/np.max(np.abs(frame.net_energy_W)),
               min_temperature_K=frame[['Tl','Tv']].min().min(),
               min_true_concentration_mol_m3=tables['Cl'].filter(regex='_true$').min().min(),
               enhancement_min=frame.E.min(), enhancement_max=frame.E.max())
    row.update({f'reactive_{k}':v for k,v in result['reactive_evaluations'].items()})
    return frame.reset_index(), row, identity, [reactions, *paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    frames, summary, sources, identities = [], [], [Path(__file__)], {}
    for run in args.runs:
        frame, row, identity, paths = read_run(run)
        frames.append(frame)
        summary.append(row)
        identities[run.name] = identity
        sources.extend(paths)
    profiles = pd.concat(frames, ignore_index=True)
    profiles.to_csv(output/'profiles.csv', index=False)
    pd.DataFrame(summary).to_csv(output/'summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), layout='constrained')
    colors = ['#0072B2', '#D55E00']
    for i, frame in enumerate(frames):
        label = f"{summary[i]['mesh_points']} initial nodes; tol {summary[i]['tol']:g}"
        style = '-' if i == len(frames)-1 else '--'
        axes[0,0].plot(frame.Position, frame.capture_pct, style, color=colors[i%2], label=label)
        axes[0,1].plot(frame.Position, frame.Tl, style, color=colors[i%2], label=label+' (liquid)')
    last = frames[-1]
    axes[0,0].plot(1., summary[-1]['observed_capture_pct'], 'kx', ms=7, clip_on=False, label='NCCC outlet observation')
    axes[0,1].plot(last.Position, last.Tv, ':', color='#333333', label='Vapor (finest run)')
    for name,label,style,color in [('fv_CO2','Vapor','-','#0072B2'),('fl_CO2','Liquid','--','#D55E00')]:
        axes[1,0].plot(last.Position,last[name]/1000,style,color=color,label=label)
    for species,style,color in [('MEA','-','#0072B2'),('MEAH','--','#D55E00'),('MEACOO',':','#009E73'),
                                ('CO2','-.','#333333'),('HCO3','-','#CC79A7'),('CO3','--','#666666')]:
        labels = {'MEA':'MEA','MEAH':r'MEAH$^+$','MEACOO':r'MEACOO$^-$',
                  'CO2':r'CO$_2$','HCO3':r'HCO$_3^-$','CO3':r'CO$_3^{2-}$'}
        axes[1,1].semilogy(last.Position,last[f'Cl_{species}_true'],style,color=color,label=labels[species])
    for ax,ylabel,title in zip(axes.flat, ['Cumulative CO₂ removal (%)','Temperature (K)',
                              'CO₂ fugacity (kPa)','True concentration (mol m⁻³)'],
                              ['(a) Case 3C capture','(b) Nonisothermal profiles',
                               '(c) Bulk fugacity driving force','(d) Bulk reactive speciation']):
        ax.set(xlabel='Normalized height, bottom to top',ylabel=ylabel,title=title,xlim=(0,1))
        ax.grid(alpha=.18)
        ax.legend(fontsize=7,frameon=False)
    axes[1,1].legend(fontsize=7, frameon=False, loc='lower left', ncol=3,
                     columnspacing=.8, handlelength=1.5)
    for suffix in ['svg','png','pdf']:
        fig.savefig(output/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    timing = args.runs[0].parent/'reactive_jacobian_timing.json'
    provenance = dict(run_identities=identities,
                      scope='Current-parameter nine-species nonisothermal conventional-film Case 3C. '
                            'Temperature observations are not plotted without confirmed phase/coordinate mapping. '
                            'Numerical refinement is separate from experimental agreement.')
    if timing.is_file():
        sources.append(timing)
        provenance['local_jacobian_timing'] = json.loads(timing.read_text())
    provenance['input_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
