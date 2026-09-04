"""Retain and display transport sensitivities from completed column runs only."""
import argparse
from contextlib import chdir
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--diffusivity-minus-run', default='diffusivity_090_seeded',
                        help='Explicit retained completion of the diagnosed -10%% case')
    args = parser.parse_args()
    root = args.model_root.resolve()
    sys.path.insert(0, str(root/'analyses/nccc_validation/figures/reactive_column/scripts'))
    from render import read_run
    output = Path(__file__).resolve().parents[1]/'output'
    names = ['baseline'] + [f'{q}_{f}' for q in ('viscosity', 'diffusivity', 'kl') for f in ('090', '110')]
    rows, profiles, inputs = [], [], []
    reference = json.loads((output/'baseline/identity.json').read_text())
    for name in ('diffusivity_090_diagnostic', 'diffusivity_090_reinitialized'):
        folder = output/name
        records = [folder/p for p in ('identity.json', 'failure_trace.json', 'state_probe.json')]
        identity, failure, probe = [json.loads(p.read_text()) for p in records]
        assert identity['engine_identity'] == reference['engine_identity']
        assert identity['parameter_fingerprint'] == reference['parameter_fingerprint']
        assert identity['quantity'] == 'diffusivity' and identity['factor'] == .9
        assert len(failure['failures']) == 2 and len(probe) == 4
        assert [row['success'] for row in probe] == [False, True, True, True]
        assert probe[-1]['temperature_K'] == failure['failures'][-1]['temperature_K']
        assert probe[-1]['pressure_Pa'] == failure['failures'][-1]['pressure_Pa']
        assert probe[-1]['apparent_amounts_mol'] == failure['failures'][-1]['apparent_amounts_mol']
        for row in probe[2:]:
            assert row['evidence']['local_stability_status'] == 'passed'
            assert row['evidence']['sensitivity_status'] == 'available'
        inputs.extend(records)
    archived = set()
    for archive in sorted((output.parent/'input').glob('*.tar.gz')):
        inputs.append(archive)
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                path = str(root/member.name)
                if member.isfile() and path in reference['input_sha256']:
                    assert hashlib.sha256(source.extractfile(member).read()).hexdigest() == reference['input_sha256'][path]
                    archived.add(path)
    assert all(path in archived for path in reference['input_sha256'] if path.startswith(str(root/'src')))
    for name in names:
        run = output/(args.diffusivity_minus_run if name == 'diffusivity_090' else name)
        with chdir(root):
            profile, row, identity, paths = read_run(run)
            inputs.extend(p.resolve() for p in paths)
        result = json.loads((run/'result.json').read_text())
        assert result['source_unchanged_during_run'] and min(result['transport_expression_calls'].values()) > 0
        assert identity['settings'] == reference['settings']
        assert identity['engine_identity'] == reference['engine_identity']
        assert identity['parameter_fingerprint'] == reference['parameter_fingerprint']
        assert identity['factor'] == (1. if name == 'baseline' else int(name.rsplit('_', 1)[1])/100)
        if name != 'baseline':
            assert identity['quantity'] == name.rsplit('_', 1)[0]
        for path, digest in reference['input_sha256'].items():
            assert identity['input_sha256'][path] == digest, f'Inputs changed between runs: {path}'
        evidence = json.loads(result['epcsaft_chemistry_last_evidence'])
        assert evidence['parameter_fingerprint'] == reference['parameter_fingerprint']
        assert result['max_rms_residual'] < identity['settings']['tol']
        assert result['max_scaled_boundary_residual'] <= identity['settings']['bc_tol']
        assert not result['invalid_state_count'] and not result['guard_penalty_count']
        props = pd.read_csv(run/'Prop_l.csv').set_index('Position')
        enhancement = pd.read_csv(run/'enhance_factor.csv').set_index('Position')
        transport = pd.read_csv(run/'transport.csv').set_index('Position')
        inputs.extend(run/p for p in ('Prop_l.csv', 'enhance_factor.csv', 'transport.csv',
                                      'evaluated_parameters.json', 'evaluated_reactions.json'))
        if (run/'initialization.json').exists():
            inputs.extend([run/'initialization.json', run/'failure_trace.json'])
            initialization = json.loads((run/'initialization.json').read_text())
            if initialization['initial_profile']:
                assert hashlib.sha256(Path(initialization['initial_profile']).read_bytes()).hexdigest() == initialization['sha256']
                initial = pd.read_csv(initialization['initial_profile']).to_numpy().T
                assert initial.shape[0] == 7 and initial.shape[1] >= 2 and np.isfinite(initial).all()
                source = json.loads(Path(initialization['initial_profile']).with_name('result.json').read_text())
                assert source['success'] and source['case_id'] == '3C'
                assert source['thermal_state_mode'] == identity['settings']['thermal_state_mode']
            if initialization.get('anchor_start'):
                assert hashlib.sha256(Path(initialization['anchor_start']).read_bytes()).hexdigest() == initialization['anchor_start_sha256']
                assert initialization['seeded_native_solves']
            assert not json.loads((run/'failure_trace.json').read_text())['failures']
        row.update(quantity='baseline' if name == 'baseline' else identity['quantity'],
                   factor=identity['factor'], perturbation_pct=round(100*(identity['factor']-1)),
                   liquid_outlet_temperature_K=float(profile.Tl.iloc[0]),
                   vapor_outlet_temperature_K=float(profile.Tv.iloc[-1]),
                   loading_min_mol_CO2_per_mol_MEA=float((profile.Fl_CO2/profile.Fl_MEA).min()),
                   loading_max_mol_CO2_per_mol_MEA=float((profile.Fl_CO2/profile.Fl_MEA).max()))
        fields = {key:props[key] for key in ('mul_mix', 'Dl_CO2', 'Dl_MEA', 'Dl_ion')}
        fields.update({key:enhancement[key] for key in ('kl_CO2', 'Ha', 'E', 'Psi_H')})
        fields.update({key:transport[key] for key in ('kv_CO2', 'h_L', 'a_e', 'UT') if key in transport})
        for table in (props, enhancement, transport):
            np.testing.assert_array_equal(table.index, profile.Position)
        fields['Ekl_CO2_m_s'] = enhancement.E*enhancement.kl_CO2
        fields['overall_CO2_conductance_mol_m2_s_Pa'] = transport.kv_CO2*enhancement.Psi_H
        for key, values in fields.items():
            row[key+'_min'], row[key+'_max'] = values.min(), values.max()
            profile[key] = values.to_numpy()
        if profiles:
            for key in ('E', 'kl_CO2', 'Ekl_CO2_m_s', 'overall_CO2_conductance_mol_m2_s_Pa'):
                ratio = profile[key]/profiles[0][key]
                row[key+'_ratio_min'], row[key+'_ratio_median'], row[key+'_ratio_max'] = ratio.min(), ratio.median(), ratio.max()
        rows.append(row)
        profiles.append(profile)
    data = pd.DataFrame(rows)
    data['capture_change_pp'] = data.capture_pct-data.capture_pct.iloc[0]
    data['peak_temperature_change_K'] = data.peak_liquid_temperature_K-data.peak_liquid_temperature_K.iloc[0]
    refinement_path = root/'analyses/nccc_validation/figures/reactive_column/output/summary.csv'
    refinement = pd.read_csv(refinement_path).set_index('mesh_points')
    capture_refinement = abs(refinement.loc[41, 'capture_pct']-refinement.loc[21, 'capture_pct'])
    temperature_refinement = abs(refinement.loc[41, 'peak_liquid_temperature_K']-refinement.loc[21, 'peak_liquid_temperature_K'])
    with chdir(root):
        control_profile, control, control_identity, control_paths = read_run(output/'baseline_seeded')
        inputs.extend(p.resolve() for p in control_paths)
    control_result = json.loads((output/'baseline_seeded/result.json').read_text())
    assert control_identity['factor'] == 1. and control_identity['settings'] == reference['settings']
    assert control_identity['input_sha256'] == reference['input_sha256']
    assert control_result['source_unchanged_during_run']
    assert not json.loads((output/'baseline_seeded/failure_trace.json').read_text())['failures']
    control.update(capture_change_pp=control['capture_pct']-data.capture_pct.iloc[0],
                   peak_temperature_change_K=control['peak_liquid_temperature_K']-data.peak_liquid_temperature_K.iloc[0],
                   max_liquid_temperature_profile_change_K=float(np.max(np.abs(control_profile.Tl-profiles[0].Tl))),
                   max_vapor_temperature_profile_change_K=float(np.max(np.abs(control_profile.Tv-profiles[0].Tv))))
    assert abs(control['capture_change_pp']) < capture_refinement
    assert abs(control['peak_temperature_change_K']) < temperature_refinement
    pd.DataFrame([control]).to_csv(output/'initialization_control.csv', index=False)
    inputs.extend([output/'baseline_seeded/initialization.json', output/'baseline_seeded/failure_trace.json',
                   output/'derivative_checks.json', output.parent/'input/engine_check.txt'])
    data['capture_exceeds_previous_refinement_change'] = abs(data.capture_change_pp) > capture_refinement
    data['temperature_exceeds_previous_refinement_change'] = abs(data.peak_temperature_change_K) > temperature_refinement
    data.to_csv(output/'summary.csv', index=False)
    pd.concat(profiles, ignore_index=True).to_csv(output/'profiles.csv', index=False)
    inputs.extend([refinement_path, Path(__file__), root/'analyses/nccc_validation/figures/reactive_column/scripts/render.py'])
    provenance = dict(input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        reference_identity=reference, previous_refinement_capture_pp=capture_refinement,
        previous_refinement_temperature_K=temperature_refinement,
        limitation='Previous mesh/tolerance response is a resolution indicator, not a certified error bound. '
        'Three OAT ±10% correlation multipliers are not independent statistical uncertainties. '
        'Profile ratios compare corresponding axial positions after column recoupling, not fixed-state partial derivatives.')
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), layout='constrained')
    for ax, column, label, resolution in zip(axes,
        ['capture_change_pp', 'peak_temperature_change_K'],
        ['Capture change (percentage points)', 'Peak liquid-temperature change (K)'],
        [capture_refinement, temperature_refinement]):
        ax.axhspan(-resolution, resolution, color='.9', label='Prior mesh/tolerance change')
        ax.axhline(0, color='.4', lw=.7)
        for quantity, title, marker, color in [('viscosity', 'Liquid viscosity', 'o', '#0072B2'),
                 ('diffusivity', 'Liquid CO₂ diffusivity', 's', '#D55E00'),
                 ('kl', 'Liquid CO₂ transfer coefficient', '^', '#009E73')]:
            subset = data[data.quantity == quantity]
            ax.scatter(subset.perturbation_pct, subset[column], marker=marker, color=color,
                       facecolors='none' if quantity == 'kl' else color, s=65, label=title)
        ax.plot(0, 0, 'kx', ms=6, label='Baseline')
        ax.set(xlabel='One-at-a-time multiplier change (%)', ylabel=label, xticks=[-10, 0, 10], xlim=(-12, 12))
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=7, loc='best')
    for extension in ('svg', 'png', 'pdf'):
        fig.savefig(output/f'transport_sensitivity.{extension}', dpi=200)
    plt.close(fig)
    print(data[['quantity', 'factor', 'capture_pct', 'capture_change_pp',
                'peak_liquid_temperature_K', 'peak_temperature_change_K', 'max_rms_residual']].to_string(index=False))


if __name__ == '__main__':
    main()
