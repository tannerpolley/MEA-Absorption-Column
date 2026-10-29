"""Engine 1.0.0 film invariants at the declared 3C inlet state."""
import json
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.optimize import brentq

from mea_absorption_column.column import _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config
from mea_absorption_column.Transport.Enhancement_Factor import enhancement_reference_expression

OUT = Path('analyses/reactive_film_evidence/results/runs/engine100')
case = json.loads(Path('analyses/bvp_solution_methods/input/case_3c.json').read_text())
rows = []
for film in ('enhancement_reference', 'equilibrium_manifold'):
    request = json.loads((OUT / (film+'_request.json')).read_text())
    a = _prepare_conserved_column_in_process(resolve_column_config(request))['assembly']
    u = np.r_[case['initial_bulk_state'], [0., 0., 0., 0.]]
    bulk = np.asarray(a['balance'](u[:8], u[8:11])[3]).ravel()
    bulk_f = bulk[9]
    charge = np.array([0, 0, 0, 1, -1, -1, -2, 1, -1])
    for relative_drive in (0., -.1, .1):
        vapor_f = bulk_f * (1+relative_drive)
        def residual(loading):
            u[11] = loading
            diagnostic = a['diagnostics'](u)
            integral, delta, kg, interface_f = (
                float(diagnostic[1]), float(diagnostic[2]), float(diagnostic[3][0]), float(diagnostic[7]))
            return integral/delta - kg*(vapor_f-interface_f)
        endpoint = brentq(residual, -.05, .05, xtol=1e-13)
        residual_value = residual(endpoint)
        d = a['diagnostics'](u)
        flux = float(d[1])/float(d[2])
        characteristic = float(d[3][0])*bulk_f
        assert abs(residual_value) < 1e-7*characteristic
        if relative_drive == 0:
            assert abs(flux) < 1e-8*characteristic
        else:
            assert flux*relative_drive > 0
        charge_fraction = float(charge @ bulk[:9]/sum(bulk[:9]))
        assert abs(charge_fraction) < 1e-10
        rows.append(dict(film=film, relative_fugacity_drive=relative_drive,
            flux_mol_m2_s=flux, interface_residual_mol_m2_s=residual_value,
            characteristic_flux_mol_m2_s=characteristic, bulk_charge_mol_per_mol=charge_fraction))
# Mathematical no-reactant limit of the unchanged enhancement expression: Ha -> 0, E -> 1.
c = ca.MX.sym('c', 9)
e = ca.Function('enhancement_limit', [c], [enhancement_reference_expression(
    318.15, c, 1e-4, ca.DM([2e-9, 1e-9, 1e-9]))])
concentrations = np.array([1., 1e-30, 5e4, 1., 1., 1., 1., 1., 1.])
assert abs(float(e(concentrations))-1.) < 1e-12
(OUT / 'engine_limit_checks.json').write_text(json.dumps({
    'rows': rows, 'enhancement_no_reactant_limit_E': float(e(concentrations)),
    'limits': {'interface_relative': 1e-7, 'zero_drive_relative': 1e-8,
               'bulk_charge_mol_per_mol': 1e-10, 'no_reactant_E_absolute': 1e-12},
    'scope': 'Single 3C inlet state; artificial gas fugacity perturbations; no column or physics change.'}, indent=2)+'\n')
print(json.dumps(rows, indent=2))
