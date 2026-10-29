"""Render the retained discrete 3C flux predictions without model evaluation."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'results/runs/engine100'
rows = list(csv.DictReader((OUT / 'profiles.csv').open()))
fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
for film, marker, color in [('enhancement_reference', 's', '#0072B2'),
                            ('equilibrium_manifold', '^', '#D55E00')]:
    group = [r for r in rows if r['film'] == film]
    assert len(group) == 11
    ax.scatter([float(r['z_m']) for r in group],
        [float(r['flux_mol_m2_s']) for r in group], label=film, marker=marker, color=color)
ax.axhline(0, color='0.5', linewidth=.7)
ax.set(xlabel='Height above gas inlet (m)', ylabel='CO₂ flux into liquid (mol m⁻² s⁻¹)',
       title='3C: closure comparison at 11 native nodes')
ax.legend()
fig.savefig(OUT / 'flux_comparison.png', dpi=160)
