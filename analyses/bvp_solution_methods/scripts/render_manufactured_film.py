"""Render fixed-quadrature errors from retained film values; never run the model."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output = Path(sys.argv[1])
rows = json.loads((output/'manufactured_film.json').read_text())['declared']['quadratures']
plotted = [(r['points'], r['relative_flux_error'], abs(r['interface_fugacity_pa']/10000-1)) for r in rows]
with (output/'quadrature_errors.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['quadrature_points', 'relative_flux_error', 'relative_interface_fugacity_error'])
    writer.writerows(plotted)
points, flux, fugacity = zip(*plotted)
fig, ax = plt.subplots(figsize=(6.4, 4.2), layout='constrained')
ax.loglog(points, flux, 'o', color='#0072B2', label='CO₂ flux')
ax.loglog(points, fugacity, 'x', color='#D55E00', label='Interface fugacity')
ax.axhline(1e-5, color='0.3', ls='--', label='129-point allowed relative error')
ax.set(xlabel='Trapezoidal quadrature points (count)', ylabel='Relative error to exact reference (1)',
       title='Neutral two-amine film: exact exchange reference', xticks=points)
ax.set_xticklabels(points)
ax.legend(frameon=False)
ax.grid(alpha=.15, which='both')
fig.savefig(output/'quadrature_errors.png', dpi=180)
plt.close(fig)
