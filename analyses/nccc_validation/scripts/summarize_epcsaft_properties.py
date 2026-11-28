"""Summarize common-basis NCCC case predictions and matched temperature taps."""
import hashlib
import json

import numpy as np
import pandas as pd

from check_epcsaft_properties import ABSOLUTE_LIMITS, ANALYSIS, ARCHIVE, INPUTS
from run_epcsaft_properties import RESULTS, STEPS


TAPS = ANALYSIS/'figures/reactive_parallel/inputs/morgan2020_table_c2.csv'


def read_profiles(directory):
    names = ('Fl', 'Fv', 'T', 'Hl', 'Hv', 'CO2', 'H2O', 'transport')
    return {key: pd.read_csv(directory/(key+'.csv')) for key in names}


def summarize(directory, step, case, observed_capture, observed_taps):
    result = json.loads((directory/'result.json').read_text())
    profiles = read_profiles(directory)
    l, v, t, hl, hv, _, _, transport = profiles.values()
    liq_peak, vap_peak = t.Tl.idxmax(), t.Tv.idxmax()
    row = dict(step=step, case=case, capture_pct=result['capture_pct'], observed_capture_pct=observed_capture,
               capture_error_pp=result['capture_pct']-observed_capture,
               peak_liquid_K=t.Tl.iloc[liq_peak], peak_liquid_z=t.Position.iloc[liq_peak],
               peak_vapor_K=t.Tv.iloc[vap_peak], peak_vapor_z=t.Position.iloc[vap_peak],
               co2_range_mol_s=np.ptp(v.Fv_CO2-l.Fl_CO2), water_range_mol_s=np.ptp(v.Fv_H2O-l.Fl_H2O),
               enthalpy_range_W=np.ptp(hv.Hvf-hl.Hlf), min_liquid_K=t.Tl.min(),
               inlet_water_mol_s=v.Fv_H2O.iloc[0], inlet_water_mole_fraction=v.Fv_H2O.iloc[0]/v.Fv_T.iloc[0],
               inlet_dry_mol_s=v.Fv_T.iloc[0]-v.Fv_H2O.iloc[0], nodes=result['final_mesh_nodes'],
               max_rms_residual=result['max_rms_residual'], success=result['success'],
               linear_solver=json.loads(result['epcsaft_chemistry_last_evidence'])['linear_solver'],
               solve_s=result['runtime_s'], wall_s=result['total_wall_including_seed_and_outputs_s'])
    phi = directory/'vapor_fugacity.csv'
    if phi.is_file():
        fugacity = pd.read_csv(phi)
        for gas in ('CO2', 'H2O', 'N2', 'O2'):
            values = fugacity['f_'+gas+'_Pa']/(v['Fv_'+gas]/v.Fv_T*transport.P)
            row.update({'phi_'+gas+'_min': values.min(), 'phi_'+gas+'_max': values.max()})
    # CO2 transfer still uses the reduced vapor basis, rescaled to wet y_CO2.
    co2_phi = profiles['CO2'].fv_CO2/(v.Fv_CO2/v.Fv_T*transport.P)
    row.update(phi_CO2_reduced_min=co2_phi.min(), phi_CO2_reduced_max=co2_phi.max())
    taps = []
    for column, observation in observed_taps.items():
        source_z = float(column.removeprefix('position_').removesuffix('_C'))
        z = 1.-source_z
        taps.append(dict(step=step, case=case, source_position_top_to_bottom=source_z, z=z,
                         observed_K=observation+273.15, liquid_K=np.interp(z, t.Position, t.Tl),
                         vapor_K=np.interp(z, t.Position, t.Tv)))
    tap_frame = pd.DataFrame(taps)
    row.update(tap_count=int(tap_frame.observed_K.count()),
               liquid_tap_rmse_K=np.sqrt(((tap_frame.liquid_K-tap_frame.observed_K)**2).mean()),
               vapor_tap_rmse_K=np.sqrt(((tap_frame.vapor_K-tap_frame.observed_K)**2).mean()))
    return row, tap_frame


def main():
    observed = pd.read_csv(INPUTS, index_col=0)['CO2  %']
    taps = pd.read_csv(TAPS, index_col='case_id')
    rows, tap_frames, missing = [], [], []
    cases = [f'{i}C' for i in range(1, 8)]
    for step in ('correlation', *STEPS):
        for case in cases if step in ('correlation', *list(STEPS)[:3]) else ('3C',):
            if step == 'correlation':
                directory = ARCHIVE/('coarse' if case == '3C' else 'campaign_'+case)
            else:
                directory = RESULTS/(step+'_'+case)
            if not (directory/'result.json').is_file():
                missing.append(str(directory))
                continue
            row, frame = summarize(directory, step, case, observed[case], taps.loc[case])
            rows.append(row)
            tap_frames.append(frame)

    runs = pd.DataFrame(rows)
    temperatures = pd.concat(tap_frames, ignore_index=True)
    baseline = runs.loc[runs.step=='correlation'].set_index('case')
    compared = ('capture_pct', 'peak_liquid_K', 'peak_liquid_z', 'peak_vapor_K', 'peak_vapor_z',
                'co2_range_mol_s', 'water_range_mol_s', 'enthalpy_range_W', 'inlet_water_mol_s',
                'liquid_tap_rmse_K', 'vapor_tap_rmse_K')
    for metric in compared:
        runs[metric+'_minus_correlation'] = runs[metric]-runs.case.map(baseline[metric])
    baseline_taps = temperatures.loc[temperatures.step=='correlation', ['case', 'z', 'liquid_K', 'vapor_K']]
    temperatures = temperatures.merge(baseline_taps, on=['case', 'z'], suffixes=('', '_correlation'))
    for phase in ('liquid', 'vapor'):
        temperatures[phase+'_error_K'] = temperatures[phase+'_K']-temperatures.observed_K
        temperatures[phase+'_minus_correlation_K'] = temperatures[phase+'_K']-temperatures[phase+'_K_correlation']
    runs.to_csv(RESULTS/'case_comparison.csv', index=False)
    temperatures.to_csv(RESULTS/'temperature_taps.csv', index=False)

    overall = {}
    for step, group in runs.groupby('step', sort=False):
        temp = temperatures[(temperatures.step==step) & temperatures.observed_K.notna()]
        overall[step] = dict(cases=len(group), capture_mae_pp=group.capture_error_pp.abs().mean(),
                             matched_taps=len(temp), liquid_rmse_K=np.sqrt((temp.liquid_error_K**2).mean()),
                             vapor_rmse_K=np.sqrt((temp.vapor_error_K**2).mean()),
                             min_liquid_K=group.min_liquid_K.min(), max_liquid_K=group.peak_liquid_K.max())
        for gas in ('CO2', 'H2O', 'N2', 'O2', 'CO2_reduced'):
            lo, hi = group['phi_'+gas+'_min'], group['phi_'+gas+'_max']
            if lo.notna().any():
                overall[step]['phi_'+gas] = [lo.min(), hi.max()]
    coarse = ARCHIVE/'coarse'
    replay = RESULTS/'C1_final'
    compared_files = {p.name:p.read_bytes() == (replay/p.name).read_bytes() for p in sorted(coarse.glob('*.csv'))}
    c1 = dict(byte_identical=all(compared_files.values()), archived_seed=str(coarse/'initial_scaled.csv'),
              archived_seed_sha256=hashlib.sha256((coarse/'initial_scaled.csv').read_bytes()).hexdigest(),
              comparisons=compared_files)
    c4 = []
    for label in ('inlet_humidity_3C', '3C_41_from_21', '3C_81_from_41'):
        path = RESULTS/label/'energy_check.json'
        if path.is_file():
            c4.append(dict(label=label, **json.loads(path.read_text())))
    for previous, current in zip(c4, c4[1:]):
        current['capture_change_pp'] = current['capture_pct']-previous['capture_pct']
        current['peak_liquid_change_K'] = current['max_liquid_K']-previous['max_liquid_K']
        current['peak_vapor_change_K'] = current['max_vapor_K']-previous['max_vapor_K']
    c4_pass = (len(c4)==3 and all(b['range_W'] < a['range_W'] for a, b in zip(c4, c4[1:]))
               and c4[-1]['temperature_equivalent_K'] <= .05)
    fine = pd.read_csv(RESULTS/'3C_81_from_41'/'energy_dense.csv')
    flow = fine[['F_L_CO2', 'F_L_H2O']].to_numpy()
    transferred = np.abs(np.diff(flow, axis=0)).sum(axis=0)
    heat_capacity = fine.C_L_W_K.min()
    phase_error = ABSOLUTE_LIMITS['h_L_CO2_J_mol']
    bound = dict(case='3C', input_nodes=81, co2_transferred_mol_s=transferred[0],
                 absolute_water_transferred_mol_s=transferred[1], min_C_L_W_K=heat_capacity,
                 partial_H_tolerance_per_phase_J_mol=phase_error,
                 max_liquid_temperature_equivalent_K=2*phase_error*sum(transferred)/heat_capacity,
                 basis='Integrated absolute liquid component flow changes. Allow both phases equal-magnitude '
                       'partial-enthalpy errors of opposite sign. The resulting heat-flow bound is divided '
                       'by the minimum liquid heat-capacity rate.')
    humidity = runs.loc[runs.step=='inlet_humidity'].set_index('case')
    water = runs.loc[runs.step=='water_fugacity'].set_index('case')
    humidity_trial = dict(max_abs_y_H2O_change=(humidity.inlet_water_mole_fraction-water.inlet_water_mole_fraction).abs().max(),
                          max_abs_capture_change_pp=(humidity.capture_pct-water.capture_pct).abs().max(),
                          max_abs_dry_flow_change_mol_s=(humidity.inlet_dry_mol_s-water.inlet_dry_mol_s).abs().max())
    summary = dict(C1=c1, C2_partial_enthalpy_tolerance_basis=bound,
                   C4=dict(passed=c4_pass, limit_K=.05, runs=c4), overall=overall, missing=missing,
                   humidity_trial=humidity_trial,
                   source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (TAPS, INPUTS)})
    (RESULTS/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
