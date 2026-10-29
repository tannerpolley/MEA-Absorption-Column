"""Render retained issue 83 values; no thermodynamic or column calculations."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'results/runs/current_pin'
rows = list(csv.DictReader((OUT / 'density.csv').open()))
summary = []
for loaded in (False, True):
    for model in ('correlation', 'current'):
        group = [r for r in rows if r['model'] == model and (float(r['loading']) > 0) == loaded]
        errors = np.array([float(r['error_g_cm3']) for r in group])
        rmse = float(np.sqrt(np.mean(errors**2)))
        summary.append(dict(group='loaded' if loaded else 'unloaded', model=model, n=len(group),
            bias_g_cm3=float(errors.mean()), rmse_g_cm3=rmse,
            mean_absolute_percent=float(np.mean([abs(float(r['error_percent'])) for r in group])),
            rmse_over_uncertainty=None if loaded else rmse/.0005,
            within_uncertainty=None if loaded else int(sum(abs(errors) <= .0005))))
(OUT / 'density_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
for ax, alpha in zip(axes, (0, .3)):
    selected = [r for r in rows if float(r['mea_mass_fraction']) == .3 and float(r['loading']) == alpha]
    for model, marker, color in [('correlation', 's', '#0072B2'), ('current', '^', '#D55E00')]:
        group = sorted([r for r in selected if r['model'] == model], key=lambda r: float(r['temperature_K']))
        ax.scatter([float(r['temperature_K']) for r in group], [float(r['predicted_g_cm3']) for r in group], label=model, marker=marker, color=color)
    obs = [r for r in selected if r['model'] == 'correlation']
    ax.errorbar([float(r['temperature_K']) for r in obs], [float(r['observed_g_cm3']) for r in obs],
        yerr=.0005 if alpha == 0 else None, fmt='ko', label='Amundsen 2009', capsize=3)
    ax.set(xlabel='Temperature (K)', ylabel='Density (g/cm³)', title=f'30 wt% MEA, loading {alpha:g}')
    ax.legend(fontsize=8)
profiles = list(csv.DictReader((OUT / 'profiles.csv').open()))
for source, marker, color in [('correlation', 's', '#0072B2'), ('solved_state', '^', '#D55E00')]:
    group = [r for r in profiles if r['source'] == source]
    axes[2].scatter([float(r['z_m']) for r in group], [float(r['liquid_K']) for r in group], label=source, marker=marker, color=color)
axes[2].set(xlabel='Height above gas inlet (m)', ylabel='Liquid temperature (K)', title='Case 3C: 11 native nodes')
axes[2].legend(fontsize=8)
fig.savefig(OUT / 'comparison.svg')
fig.savefig(OUT / 'comparison.png', dpi=160)
