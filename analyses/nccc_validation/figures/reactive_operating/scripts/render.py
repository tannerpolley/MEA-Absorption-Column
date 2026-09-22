"""Curate the Case 3C operating sweep and plot discrete model predictions."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE = Path(__file__).resolve().parents[1]
RUNS = FIGURE.parents[1]/'results/runs/reactive_operating_rerun_20260904'
RUNTIME = FIGURE.parents[3]
BASE = RUNS/'baseline_current'
SCENARIOS = ('baseline','LG_low','LG_high','loading_low','loading_high','temperature_low','temperature_high')


def select_run(name):
    ordinary = BASE if name == 'baseline' else RUNS/name
    seeded = RUNS/(name+'_native_seed')
    for run in (seeded, ordinary):
        path = run/'result.json'
        if path.exists() and json.loads(path.read_text()).get('success'):
            return run
    return seeded if seeded.exists() else ordinary


def main():
    if not (BASE/'result.json').exists():
        raise RuntimeError('Wait for the current-source baseline before calculating operating responses')
    output=FIGURE/'output'
    output.mkdir(parents=True,exist_ok=True)
    source=RUNTIME/'src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv'
    chemistry_path=RUNTIME/'src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle/reaction-system.json'
    chemistry=json.loads(chemistry_path.read_text())
    balances,charges=np.asarray(chemistry['balance_matrix']),np.asarray(chemistry['charges'])
    base_identity=json.loads((BASE/'identity.json').read_text())
    base_sources={p.split('src/mea_absorption_column/',1)[1]:h for p,h in base_identity['input_sha256'].items()
                  if 'src/mea_absorption_column/' in p and (p.endswith('.py') or '/MEA_reactive_epcsaft_bundle/' in p)}
    rows,profiles,sources=[],[],[Path(__file__),Path(__file__).with_name('generate_data.py'),source,chemistry_path]
    sources += sorted(RUNS.glob('*.log')) + sorted(RUNS.glob('*/execution.json')) + sorted(RUNS.glob('*/identity.json'))
    for name in SCENARIOS:
        run=select_run(name)
        path=run/'result.json'
        row=dict(scenario=name,run_directory=str(run))
        sources.append(RUNS/f'{run.name}.log')
        row['newton_jacobian_policy']='SciPy default reuse after full steps'
        inputs=pd.read_csv(run/'inputs.csv',index_col=0).loc['3C']
        row.update(mass_L_over_reported_dry_G=float(inputs.L/inputs.G),lean_loading_mol_mol=float(inputs.alpha),
            liquid_inlet_temperature_K=float(inputs.Tl),liquid_inlet_temperature_C=float(inputs.Tl-273.15))
        if not path.exists():
            row.update(success=False,status='no_accepted_result')
            for retained in ('inputs.csv','identity.json','execution.json'):
                if (run/retained).exists():
                    sources.append(run/retained)
            rows.append(row)
            continue
        result=json.loads(path.read_text())
        identity_path=run/'identity.json'
        identity=json.loads(identity_path.read_text())
        for source_name, expected in identity['input_sha256'].items():
            assert hashlib.sha256(Path(source_name).read_bytes()).hexdigest() == expected, source_name
        own_sources={p.split('src/mea_absorption_column/',1)[1]:h for p,h in identity['input_sha256'].items()
                     if 'src/mea_absorption_column/' in p and (p.endswith('.py') or '/MEA_reactive_epcsaft_bundle/' in p)}
        assert own_sources==base_sources, (name,'Absorber source/input mismatch against current baseline')
        assert identity['engine_identity']==base_identity['engine_identity']
        sources.extend([path,identity_path])
        assert not result.get('runtime_files_changed')
        sources.extend([run/'inputs.csv', run/'execution.json'])
        row.update({key:result.get(key) for key in ('success','capture_pct','runtime_s',
            'solver_cpu_time_s','total_wall_including_seed_and_outputs_s','max_rms_residual',
            'max_scaled_boundary_residual','boundary_residual_norm','final_mesh_nodes',
            'solver_iterations','invalid_state_count','guard_penalty_count')})
        parameter_hash, = [h for p,h in identity['input_sha256'].items() if p.endswith('/MEA_reactive_epcsaft_bundle/parameters.json')]
        row.update(status='converged' if result['success'] else 'solver_rejected',
            mass_L_over_reported_dry_G=float(inputs.L/inputs.G),lean_loading_mol_mol=float(inputs.alpha),
            liquid_inlet_temperature_K=float(inputs.Tl),liquid_inlet_temperature_C=float(inputs.Tl-273.15),
            parameter_sha256=parameter_hash)
        if result['success']:
            assert result['max_rms_residual'] <= identity['settings']['tol']
            assert result['max_scaled_boundary_residual'] <= identity['settings']['bc_tol']
            assert result['invalid_state_count'] == 0
            tables={}
            for label in ('T','Fl','Fv','Cl','Hl','Hv'):
                p=run/f'{label}.csv'
                tables[label]=pd.read_csv(p).set_index('Position')
                sources.append(p)
                assert np.isfinite(tables[label].to_numpy(dtype=float)).all()
            t,fl,fv=tables['T'],tables['Fl'],tables['Fv']
            truth=fl.filter(regex='_true$').to_numpy()
            apparent=fl[['Fl_CO2','Fl_MEA','Fl_H2O']].to_numpy()
            assert truth.shape[1]==9 and (truth>0).all()
            # Same normalized-feed tolerances as test_epcsaft_reactive_chemistry.py.
            feed_total = apparent.sum(axis=1)
            assert np.max(np.abs(truth@balances.T-apparent@balances[:,:3].T)/feed_total[:,None]) < 1e-8
            assert np.max(np.abs(truth@charges)/feed_total) < 1e-10
            assert (tables['Cl'].filter(regex='_true$')>0).all().all()
            assert 0<=result['capture_pct']<=100
            row.update(peak_liquid_temperature_K=t.Tl.max(),peak_vapor_temperature_K=t.Tv.max(),
                min_temperature_K=t.min().min(),co2_conservation_range_mol_s=np.ptp(fv.Fv_CO2-fl.Fl_CO2),
                water_conservation_range_mol_s=np.ptp(fv.Fv_H2O-fl.Fl_H2O),
                species_balance_max_mol_s=np.max(np.abs(truth@balances.T-apparent@balances[:,:3].T)),
                charge_max_mol_s=np.max(np.abs(truth@charges)),
                net_energy_range_W=np.ptp(tables['Hv'].Hvf-tables['Hl'].Hlf))
            assert np.isfinite([row[k] for k in ('co2_conservation_range_mol_s',
                'water_conservation_range_mol_s', 'net_energy_range_W')]).all()
            t=t.copy()
            t['scenario']=name
            profiles.append(t.reset_index())
        rows.append(row)
    summary=pd.DataFrame(rows)
    base=summary[summary.scenario=='baseline'].iloc[0]
    assert base.success, 'A converged baseline is required for operating responses'
    summary['capture_change_pp']=summary.capture_pct-base.capture_pct
    summary['peak_liquid_temperature_change_K']=summary.peak_liquid_temperature_K-base.peak_liquid_temperature_K
    summary.to_csv(output/'summary.csv',index=False)
    pd.concat(profiles,ignore_index=True).to_csv(output/'temperature_profiles.csv',index=False)
    summary=pd.read_csv(output/'summary.csv')
    assert summary.loc[summary.success==True,'parameter_sha256'].nunique()==1
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained',sharey='row')
    for col,(prefix,variable,label) in enumerate((
        ('LG','mass_L_over_reported_dry_G','Liquid / dry-gas mass ratio (kg/kg)'),
        ('loading','lean_loading_mol_mol','Lean loading (mol CO₂/mol MEA)'),
        ('temperature','liquid_inlet_temperature_C','Liquid inlet temperature (°C)'))):
        points=summary[summary.scenario.isin([prefix+'_low',prefix+'_high']) & (summary.success==True)]
        for i,y in enumerate(('capture_pct','peak_liquid_temperature_K')):
            ax=axes[i,col]
            ax.plot(points[variable],points[y],'o',color='#0072B2',ms=6,label='Perturbed condition')
            ax.plot(base[variable],base[y],'kx',ms=8,label='Case 3C baseline')
            ax.set_xlabel(label)
            domain = summary.loc[summary.scenario.isin(['baseline',prefix+'_low',prefix+'_high']),variable]
            margin = .05*(domain.max()-domain.min())
            ax.set_xlim(domain.min()-margin, domain.max()+margin)
            ax.grid(alpha=.2)
    axes[0,0].set_ylabel('CO₂ capture (%)')
    axes[1,0].set_ylabel('Peak liquid temperature (K)')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Operating response: one variable changed at a time',fontsize=12)
    missing=summary.loc[summary.success!=True,'scenario'].tolist()
    if missing:
        names = {'LG_low':'lower liquid/dry-gas ratio', 'LG_high':'higher liquid/dry-gas ratio',
                 'loading_low':'lower loading', 'loading_high':'higher loading',
                 'temperature_low':'colder inlet', 'temperature_high':'warmer inlet'}
        fig.supxlabel('Not converged (omitted): '+', '.join(names.get(n,n) for n in missing)+'.',fontsize=8)
    for extension in ('svg','png','pdf'):
        fig.savefig(output/f'operating_response.{extension}',dpi=180)
    plt.close(fig)
    (output/'provenance.json').write_text(json.dumps(dict(
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
                       if p.name in ('summary.csv','temperature_profiles.csv','operating_response.pdf',
                                     'operating_response.png','operating_response.svg')},
        scope='Discrete perturbations around 3C, coarse numerics. No parameter uncertainty, '
              'experimental validation of perturbed conditions, interaction study or optimum.'),indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
