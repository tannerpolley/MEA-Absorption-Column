"""Plot retained one-at-a-time Case 3C thermodynamic sensitivity; never run a model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from render import read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--initialization-check', type=Path,
                        help='Retained rejected state and fresh-loading-path comparison')
    args = parser.parse_args()
    reference_fingerprint = json.loads(json.loads((args.reference/'result.json').read_text())[
        'epcsaft_chemistry_last_evidence'])['parameter_fingerprint']
    rows, profiles, identities, sources, documents = [], [], {}, [Path(__file__), Path(__file__).with_name('render.py')], []
    labels = {'pair/monoethanolamine/water/k_ij': 'MEA–water',
              'pair/carbon-dioxide/water/k_ij': 'CO₂–water',
              'R4': 'R4: carbamate hydrolysis', 'R5': 'R5: MEAH⁺ dissociation'}
    for run in [args.reference, *args.runs]:
        frame, row, identity, paths = read_run(run)
        interaction = identity['settings'].get('reactive_kij_scale')
        reaction = identity['settings'].get('reactive_reaction_scale')
        if interaction is not None and reaction is not None:
            raise ValueError('Expected a one-at-a-time perturbation')
        scale = interaction or reaction
        row.update(parameter='Reference' if scale is None else labels[scale[0]],
                   parameter_identity='' if scale is None else scale[0],
                   factor=1. if scale is None else scale[1])
        row['perturbation_pct'] = round(100*(row['factor']-1), 10)
        if interaction is not None:
            document = run/'evaluated_parameters.json'
            data = json.loads(document.read_text())
            coefficient = next(c for p in data['pairs'] for c in p['coefficients'] if c['identity'] == scale[0])
            row['coefficient_value'] = coefficient['value']['magnitude']
            row['baseline_coefficient_value'] = row['coefficient_value']/scale[1]
            documents.append((document, run.name+'-parameters.json'))
            paths.append(document)
        if reaction is not None:
            document = run/'evaluated_reactions.json'
            data = json.loads(document.read_text())
            if data['sensitivity']['reaction_id'] != scale[0] or data['sensitivity']['equilibrium_constant_multiplier'] != scale[1]:
                raise ValueError('Reaction document does not match the declared multiplier')
            documents.append((document, run.name+'-reactions.json'))
            paths.extend([document, run/'evaluated_parameters.json'])
        if scale is not None:
            native = json.loads(json.loads((run/'result.json').read_text())['epcsaft_chemistry_last_evidence'])
            if native['parameter_fingerprint'] != identity['evaluated_parameter_fingerprint']:
                raise ValueError('Final state did not use the declared perturbed parameters')
            if reaction is not None and native['parameter_fingerprint'] != reference_fingerprint:
                raise ValueError('Reaction-only sensitivity changed the EOS parameters')
        rows.append(row)
        profiles.append(frame)
        identities[run.name] = identity
        sources.extend(paths)
    data = pd.DataFrame(rows)
    for quantity, delta in [('capture_pct','capture_change_pp'),
                            ('peak_liquid_temperature_K','peak_temperature_change_K')]:
        data[delta] = data[quantity]-data[quantity].iloc[0]
    selected = set(data.parameter.iloc[1:])
    if (data.iloc[1:].duplicated(['parameter','perturbation_pct']).any()
            or set(zip(data.parameter.iloc[1:], data.perturbation_pct.iloc[1:])) != {
                (p, sign) for p in selected for sign in (-5.,5.)}):
        raise ValueError('Expected exactly both ±5% perturbations for each selected parameter')
    output = Path(__file__).resolve().parents[1]/'output'/'sensitivity'
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output/'summary.csv',index=False)
    pd.concat(profiles,ignore_index=True).to_csv(output/'profiles.csv',index=False)
    for document, name in documents:
        shutil.copyfile(document, output/name)
    if args.initialization_check:
        for name in ('failed_state.json', 'fresh_state.json'):
            source = args.initialization_check/name
            sources.append(source)
            shutil.copyfile(source, output/name)
    fig, axes = plt.subplots(1,2,figsize=(8.5,3.7),layout='constrained')
    fig.suptitle('Case 3C: capture and temperature response to thermodynamic inputs', fontsize=11)
    for ax, column, label in zip(axes, ['capture_change_pp','peak_temperature_change_K'],
                                ['Capture change (percentage points)','Peak liquid-temperature change (K)']):
        for parameter,marker,color in [('MEA–water','o','#0072B2'),('CO₂–water','s','#D55E00'),
                                       ('R4: carbamate hydrolysis','^','#009E73'),('R5: MEAH⁺ dissociation','D','#CC79A7')]:
            subset = data[data.parameter==parameter].sort_values('perturbation_pct')
            if not subset.empty:
                ax.scatter(subset.perturbation_pct,subset[column],marker=marker,color=color,label=parameter,s=42)
        ax.plot(0,0,'kx',label='Selected-parameter reference')
        ax.axhline(0,color='#777777',lw=.7)
        ax.set(xlabel='Multiplicative change in coefficient or K(T) (%)',ylabel=label,xticks=[-5,0,5],xlim=(-6,6))
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False,fontsize=8)
    for suffix in ['png','pdf']:
        fig.savefig(output/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    provenance=dict(input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    run_identities=identities,
                    scope='One-at-a-time ±5% k_ij or reaction K(T) perturbations, one nonisothermal Case 3C. '
                          'Multipliers define a local sensitivity screen, not parameter uncertainty or a refit. '
                          'Only the named input changes; other thermodynamic inputs, transport and operating conditions are fixed. '
                          'Interaction changes apply to both EOS phases; failures are not admitted.')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(data[['parameter','perturbation_pct','capture_pct','capture_change_pp',
                'peak_liquid_temperature_K','peak_temperature_change_K','max_rms_residual']].to_string(index=False))


if __name__ == '__main__':
    main()
