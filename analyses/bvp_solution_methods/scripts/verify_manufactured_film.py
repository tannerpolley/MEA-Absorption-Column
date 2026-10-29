"""Neutral two-amine film verification; equations and limits: #85 comment 6509541282."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.optimize import brentq

from mea_absorption_column.Thermodynamics.casadi_reactive import build_loading_path_function
from mea_absorption_column.Transport.Reactive_Film import (
    equilibrium_manifold_conductance_integral, onsager_mobility_expression,
    binary_diffusivities_from_species, ReactiveFilmDomainError,
)

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / 'analyses/bvp_solution_methods/input/issue85/manufactured_film.json'


from mea_absorption_column.Thermodynamics.manufactured_neutral import ManufacturedNeutralPhase


def verify(declaration):
    phase = ManufacturedNeutralPhase(declaration)
    ids, feeds = phase.component_ids, phase.feed_ids
    rows = {s: np.array([declaration['components'][s][i] for i in ids]) for s in declaration['component_order']}
    constraints = np.array([rows[s] for s in declaration['stationary_components']])
    u = np.array([declaration['temperature_k'], declaration['pressure_pa'],
                  *[declaration['feed_mol'][s] for s in feeds]])
    k, volume, d, delta, kg = (declaration[s] for s in
        ('K_per_mol', 'volume_m3', 'diffusivity_m2_s', 'thickness_m', 'gas_coefficient_mol_m2_s_pa'))
    a, b = (declaration['feed_mol'][s] for s in ('A_total', 'B_total'))
    total = a + b
    path = build_loading_path_function(phase)
    bulk, loading = ca.MX.sym('bulk', len(u)), ca.MX.sym('loading')
    values, tangent = path(bulk, loading)
    n = values[:len(ids)]
    pairs = binary_diffusivities_from_species(ca.DM.ones(len(ids))*d)
    mobility = onsager_mobility_expression(n/ca.sum1(n), values[len(ids)], pairs,
                                          additional_flux_constraints=constraints)
    g = ca.dot(ca.DM(rows['C_total']), mobility @ tangent)
    sample = ca.Function('film_state', [bulk, loading],
        [values, tangent, ca.jacobian(tangent, loading), mobility, g, ca.jacobian(g, loading)])
    atoms = np.array([[declaration['formulas'][s].get(e, 0) for s in ids] for e in declaration['atomic_masses_kg_mol']])
    masses = np.array(list(declaration['atomic_masses_kg_mol'].values())) @ atoms
    stoich = np.array([[reaction.get(s, 0) for s in ids] for reaction in declaration['reactions']]).T
    errors, states = {}, []

    def close(name, actual, expected, rtol=1e-7, atol=1e-10):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
        errors[name] = max(errors.get(name, 0.), float(np.max(abs(np.asarray(actual)-expected))))

    close('reaction_components', np.array(list(rows.values())) @ stoich, 0., 0., 1e-12)
    close('reaction_atoms', atoms @ stoich, 0., 0., 1e-12)
    close('reaction_mass_kg', masses @ stoich, 0., 0., 1e-12)
    charges = np.array([declaration['charge_numbers'][s] for s in ids])
    close('neutral_charge_numbers', charges, 0., 0., 0.)
    close('reaction_charge', charges @ stoich, 0., 0., 1e-12)
    exact_loading, exact_flux, exact_integral = np.log(21/13), 1/1200, 1/12000000
    for lam in np.linspace(0., exact_loading, 17):
        val, t, dt, matrix, conductance, slope = [np.asarray(v) for v in sample(u, lam)]
        val, t, dt = val.ravel(), t.ravel(), dt.ravel()
        f = declaration['feed_mol']['C_total'] * np.exp(lam)
        # Independent polynomial root and closed derivatives, no callback/projection.
        s = max(np.roots([k, 1+k*(total-f), -f]))
        r = 1+k*s
        fp, fpp = 1+total*k/r**2, -2*total*k*k/r**3
        v, w = f/fp, f/fp-f*f*fpp/fp**3
        nt = dict(C=s, A=a/r, B=b/r, W=declaration['feed_mol']['W_total'], CA=a*k*s/r, CB=b*k*s/r)
        tc, ta = v/s, -k*v/r
        dc, da = w/s-v*v/s**2, -k*w/r+k*k*v*v/r**2
        tref, dtref = dict(C=tc, A=ta, B=ta, W=0., CA=tc+ta, CB=tc+ta), dict(C=dc, A=da, B=da, W=0., CA=dc+da, CB=dc+da)
        close('amounts_mol', val[:len(ids)], [nt[s] for s in ids], 1e-12, 1e-12)
        close('density_mol_m3', val[len(ids)], sum(nt.values())/volume, 1e-12, 1e-12)
        for component, row in rows.items():
            close('component_mol', row @ val[:len(ids)], f if component == 'C_total' else declaration['feed_mol'][component], 1e-12, 1e-12)
        close('mass_reconstruction_kg', masses @ val[:len(ids)], masses @ np.array([nt[s] for s in ids]), 1e-12, 1e-12)
        mu = np.log(val[:len(ids)]) - [np.log(k) if s in ('CA', 'CB') else 0. for s in ids]
        close('affinities_over_rt', stoich.T @ mu, 0., 0., 1e-12)
        close('state_charge_mol', charges @ val[:len(ids)], 0., 0., 1e-12)
        close('loading_tangent', t, [tref[s] for s in ids])
        close('loading_second_derivative', dt, [dtref[s] for s in ids])
        reference = np.zeros_like(matrix)
        for free, bound in [('A', 'CA'), ('B', 'CB')]:
            indices = [ids.index(free), ids.index(bound)]
            reference[np.ix_(indices, indices)] = d/volume*nt[free]*nt[bound]/(nt[free]+nt[bound])*np.array([[1., -1.], [-1., 1.]])
        close('mobility_mol_m_s', matrix, reference, atol=1e-14)
        close('g_mol_m_s', conductance, d/volume*total*k*v/r**2, atol=1e-14)
        close('dg_mol_m_s', slope, d/volume*total*k*(w/r**2-2*k*v*v/r**3), atol=1e-14)
        assert np.min(val[:len(ids)]) > 0 and float(conductance.item()) > 0 and d > 0
        assert float(t @ matrix @ t) >= -1e-14
        states.append(dict(loading=float(lam), values=val.tolist(), tangent=t.tolist(), tangent_derivative=dt.tolist(),
                           mobility=matrix.tolist(), g=float(conductance.item()), dg=float(slope.item())))
    quadratures = []
    for count in (9, 17, 33, 65, 129):
        integral, conductances, interface = equilibrium_manifold_conductance_integral(
            path, bulk, loading, ca.DM.ones(len(ids))*d, rows['C_total'], constraints, len(ids), count)
        film = ca.Function('manufactured_film', [bulk, loading], [integral, ca.vertcat(*conductances), interface])
        def residual(lam):
            integral_value, _, at_interface = film(u, lam)
            return float(integral_value)/delta - kg*(declaration['gas_fugacity_pa']-float(at_interface[-1]))
        solved = brentq(residual, 0., np.log(2.), xtol=1e-12)
        integral_value, gs, interface = film(u, solved)
        flux = float(integral_value)/delta
        val, t, _, matrix, conductance, _ = [np.asarray(v) for v in sample(u, solved)]
        species_flux = (matrix @ t).ravel()*flux/float(conductance.item())
        close('stationary_flux_mol_m2_s', constraints @ species_flux, 0., 0., 1e-12)
        close('total_flux_mol_m2_s', species_flux.sum(), 0., 0., 1e-12)
        quadratures.append(dict(points=count, loading=solved, integral_mol_m_s=float(integral_value),
            flux_mol_m2_s=flux, interface_fugacity_pa=float(interface[-1]), conductances=np.asarray(gs).ravel().tolist(),
            species_flux_mol_m2_s=species_flux.tolist(), closure_mol_m2_s=residual(solved),
            relative_flux_error=abs(flux/exact_flux-1), relative_integral_error=abs(float(integral_value)/exact_integral-1)))
    last = quadratures[-1]
    close('flux_mol_m2_s', last['flux_mol_m2_s'], exact_flux, 1e-5, 0.)
    close('integral_mol_m_s', last['integral_mol_m_s'], exact_integral, 1e-5, 0.)
    close('interface_fugacity_pa', last['interface_fugacity_pa'], 10000., 0., .1)
    close('closure_mol_m2_s', last['closure_mol_m2_s'], 0., 0., 1e-12)
    # Species target errors inherit scalar quadrature error; invariants above do not.
    target = dict(C=0., A=-1/3000, B=-1/2000, W=0., CA=1/3000, CB=1/2000)
    close('species_flux_mol_m2_s', last['species_flux_mol_m2_s'], [target[s] for s in ids], 1e-5, 1e-12)
    try:
        equilibrium_manifold_conductance_integral(path, bulk, loading, ca.DM.ones(len(ids))*d,
            rows['C_total'], np.vstack([constraints, constraints[0]]), len(ids), 9)
    except ReactiveFilmDomainError as error:
        refusal = str(error)
        assert 'independent modulo total flux' in refusal
    else:
        raise AssertionError('Duplicate stationary row was not refused')
    return dict(input=declaration, states=states, quadratures=quadratures, maximum_absolute_errors=errors, duplicate_row_refusal=refusal)


def run(output):
    declaration = json.loads(INPUT.read_text())
    permuted = copy.deepcopy(declaration)
    permuted['species_order'] = ['CB', 'W', 'A', 'C', 'CA', 'B']
    for key in ('feed_order', 'component_order', 'stationary_components'):
        permuted[key].reverse()
    original, reordered = verify(declaration), verify(permuted)
    mapping = [permuted['species_order'].index(s) for s in declaration['species_order']]
    for left, right in zip(original['states'], reordered['states']):
        for key in ('tangent', 'tangent_derivative'):
            np.testing.assert_allclose(left[key], np.array(right[key])[mapping], rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(left['values'][:6], np.array(right['values'][:6])[mapping], rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(left['values'][6:], right['values'][6:], rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(left['mobility'], np.array(right['mobility'])[np.ix_(mapping, mapping)], rtol=1e-7, atol=1e-14)
        np.testing.assert_allclose([left['g'], left['dg']], [right['g'], right['dg']], rtol=1e-7, atol=1e-14)
    for left, right in zip(original['quadratures'], reordered['quadratures']):
        np.testing.assert_allclose(left['conductances'], right['conductances'], rtol=1e-7, atol=1e-14)
        np.testing.assert_allclose([left['integral_mol_m_s'], left['flux_mol_m2_s'], left['loading'], left['interface_fugacity_pa']],
            [right['integral_mol_m_s'], right['flux_mol_m2_s'], right['loading'], right['interface_fugacity_pa']], rtol=1e-7, atol=1e-14)
        np.testing.assert_allclose(left['species_flux_mol_m2_s'], np.array(right['species_flux_mol_m2_s'])[mapping], rtol=1e-7, atol=1e-12)
    sources = [Path(__file__), ROOT/'src/mea_absorption_column/Transport/Reactive_Film.py',
               ROOT/'src/mea_absorption_column/BVP/Coupled_Column.py',
               ROOT/'src/mea_absorption_column/Thermodynamics/casadi_reactive.py']
    result = dict(declared=original, permuted=reordered, input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        engine_identity=json.loads((ROOT/'integration/epcsaft_contract.json').read_text())['final_identity'])
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output)/'manufactured_film.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    result = run(parser.parse_args().output)
    print(json.dumps(result['declared']['quadratures'][-1], indent=2))
