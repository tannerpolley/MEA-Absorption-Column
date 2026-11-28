"""Issue 83 comparison repeated with the Chapter 3 MEA set (fit A, Amine-Thermodynamics a812493cc): Amundsen density rows and the
11-node twelve-state 3C column with solved-state vs correlation hydraulic density. Writes results/runs/fit_a/. Stages as compare.py."""
import argparse
import json
import compare

compare.OUT = compare.ROOT / 'analyses/liquid_density/results/runs/fit_a'
compare.DATASETS = [('current', 'MEA_A_196faf11')]
compare.QUADRATURE = 9  # the fit-A transport declares 9
FIT_COMMIT = 'a812493cc'
INPUTS = 'analyses/nccc_validation/results/runs/chapter3_mea_20261128/inputs'
compare.DEPENDENCIES = {'thermodynamics_input': f'{INPUTS}/thermodynamics.json', 'transport_input': f'{INPUTS}/transport_3c.json'}


def column(source):
    """compare.column; the correlation run needs a transport copy declaring that hydraulic density."""
    if source == 'correlation':
        transport = json.loads((compare.ROOT / INPUTS / 'transport_3c.json').read_text())
        transport.update(hydraulic_density='correlation', thermodynamic_input=f'../../../../../{INPUTS}/thermodynamics.json')
        (compare.OUT / 'transport_3c_correlation.json').write_text(json.dumps(transport, indent=2) + '\n')
        compare.DEPENDENCIES['transport_input'] = str((compare.OUT / 'transport_3c_correlation.json').relative_to(compare.ROOT))
    compare.column(source)


def split():
    """Tag each Amundsen row as in or out of the fit-A objective (Amine-Thermodynamics a812493cc, replay rows of fit A) and summarize."""
    import csv, gzip, subprocess
    path = 'analyses/amines/results/mea/reported/fit/A.json.gz'
    blob = subprocess.run(['git', '-C', str(compare.AMINE), 'show', f'{FIT_COMMIT}:{path}'], capture_output=True, check=True).stdout
    fitted = []
    for item in json.loads(gzip.decompress(blob))['replay']['rows']:
        if item['group'] not in ('loaded density', 'CO2-free density'):
            continue
        r, f = item['row'], item['row'].get('analytical_feed')
        w = f['monoethanolamine'] * .061084 / (f['monoethanolamine'] * .061084 + f['water'] * .018015) if f else r['MEA_weight_fraction']
        fitted.append((r['T_K'], w, f['carbon-dioxide'] / f['monoethanolamine'] if f else 0.))
    groups = {}
    for r in csv.DictReader((compare.OUT / 'density.csv').open()):
        t, w, a = float(r['temperature_K']), float(r['mea_mass_fraction']), float(r['loading'])
        inside = any(abs(k[0] - t) < .02 and abs(k[1] - w) < .003 and abs(k[2] - a) < .006 for k in fitted)
        groups.setdefault(('loaded' if a > 0 else 'unloaded', 'fitted' if inside else 'held_out', r['model']), []).append(float(r['error_percent']))
    summary = [dict(group=g, rows=f, model=m, n=len(v), mean_absolute_percent=sum(map(abs, v)) / len(v),
                    bias_percent=sum(v) / len(v), max_absolute_percent=max(map(abs, v))) for (g, f, m), v in sorted(groups.items())]
    compare.save('density_fit_split.json', {'fit_record': f'{compare.AMINE.name}@{FIT_COMMIT}:{path}', 'fit_density_rows': len(fitted),
        'match': 'same T within 0.02 K, MEA mass fraction within 0.003, loading within 0.006', 'summary': summary})
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['properties', 'solved_state', 'correlation', 'summarize', 'split'])
    stage = parser.parse_args().stage
    compare.OUT.mkdir(parents=True, exist_ok=True)
    {'properties': compare.properties, 'summarize': compare.summarize, 'split': split}.get(stage, lambda: column(stage))()
