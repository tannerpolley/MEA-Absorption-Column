"""Full-column mathematical verification: #85 amended design 6510746468."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from mea_absorption_column.column import (
    _prepare_conserved_column_in_process, _initialize_conserved_column, _safe, run_column,
)
from mea_absorption_column.config.column import resolve_column_config

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT/'analyses/bvp_solution_methods/input/issue85/column'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(value), indent=2)+'\n')
    return dict(repository_path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def inputs(directory, points=129, permuted=False):
    t, u, case = [json.loads((INPUT/f).read_text()) for f in ('thermodynamics.json','transport.json','case.json')]
    if permuted:
        t['species_order'] = ['CB','W','A','C','CA','B']
        for key in ('liquid_feed_order','vapor_feed_order'):
            t[key].reverse()
        t['components'] = dict(reversed(list(t['components'].items())))
        u['stationary_film_components'].reverse()
        for phase, key in [('liquid','liquid_feed_order'),('vapor','vapor_feed_order')]:
            physical = case['physical_inputs']
            old = physical[phase+'_feed_ids']
            for field in (phase+'_feed_mol_s',):
                physical[field] = [physical[field][old.index(s)] for s in t[key]]
            masses = phase+'_molar_masses_kg_mol'
            case[masses] = [case[masses][old.index(s)] for s in t[key]]
            physical[phase+'_feed_ids'] = t[key]
    u['quadrature_points'] = points
    u['case'] = save(directory/'case.json', case)
    u['thermodynamic_input_sha256'] = save(directory/'thermodynamics.json',t)['sha256']
    save(directory/'transport.json',u)
    return t, u


def request(directory, nodes=17, points=129, output=None):
    return dict(preset='twelve_state_conserved',dependencies=dict(
        thermodynamics_input=str(directory/'thermodynamics.json'),transport_input=str(directory/'transport.json')),
        numerics=dict(method='upwind',solver_settings=dict(nodes=nodes,quadrature_points=points,tolerance=1e-9,max_iterations=40)),
        execution=dict(wall_limit_s=600 if nodes==129 else 120, **(dict(output_dir=str(output)) if output else {})))


def prepared_check(directory, permuted=False):
    t, u = inputs(directory, permuted=permuted)
    config = resolve_column_config(request(directory))
    assert config.model.thermo_model == 'mathematical_neutral' and config.model.energy_model == 'mathematical_total_enthalpy'
    assert not any((config.dependencies.dataset,config.dependencies.thermal_reference,config.dependencies.references))
    prepared = _prepare_conserved_column_in_process(config)
    assembly, resolved = prepared['assembly'], prepared['resolved_inputs']
    liquid, vapor = assembly['liquid'], assembly['vapor']
    errors = {}
    def close(name, actual, expected, rtol=1e-12, atol=1e-12):
        actual, expected = np.asarray(actual,dtype=float).ravel(), np.asarray(expected,dtype=float).ravel()
        np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol,err_msg=name)
        errors[name] = dict(actual=actual.tolist(),expected=expected.tolist(),maximum_absolute_error=float(np.max(abs(actual-expected))))
    for kind, phase, expected in [('liquid',liquid,dict(C=13/60,A=.2,B=.3,W=1)),('vapor',vapor,dict(C=11/60,W=.1,I=1))]:
        close(kind+'_feeds',resolved['parameters'][0 if kind=='liquid' else 1],[expected[s] for s in phase.feed_ids])
    close('loading_reporting',resolved['raw_input'][2:4],[13/30,.5228080027336228])
    close('separate_amine_reporting',list(resolved['metadata']['unloaded_amine_mass_fractions'].values()),[.16453654233804213,.35827146039558067])
    close('total_amine',resolved['metadata']['total_amine_flow_mol_s'],.5)
    checkpoint = {}
    _initialize_conserved_column(config, prepared, u['execution_policy']['initial_bulk_state'],[0,1],1e-9,checkpoint)
    scales = dict(capacity_rates_w_k=[45,30],state_scale=[13/60,1,11/60,.1,300,300,1e5,.97,11/6000,1e-4,1000,1],
        balance_scale=[13/60,1,11/60,.1,4500,3000,1e5],algebraic_scale=[.97,11/60000000,11/6000,1e-4,1000],
        boundary_scale=[13/60,1,11/60,.1,100,100,1e5])
    for key, expected in scales.items():
        close(key,checkpoint['scaling'][key],expected)
    state = np.array([.24,1.1,.16,.12,325,330,99950,.06,.0006,.00002,25,np.log(1.2)])
    b, r, algebraic = assembly['node'](0,state)
    close('conserved',b,[.24,1.1,.16,.12,240,300,99950])
    close('sources',r,[-.06,-.002,-.06,-.002,-2500,-2500,-100])
    close('boundary',assembly['boundary'](state,state),[.24-13/60,.1,.16-11/60,.02,5,10,-50])
    # Independent scalar roots and trapezoid; no loading callback or mobility.
    def free(f):
        return max(np.roots([10,1+10*(.5-f),-f]))
    s, si = free(.24),free(.288)
    exact = 1e-6*.5*(10*si/(1+10*si)-10*s/(1+10*s))
    lambdas = np.linspace(0,np.log(1.2),129)
    ss = np.array([free(.24*np.exp(lam)) for lam in lambdas])
    gs = 1e-6*.5*10*(.24*np.exp(lambdas)/(1+5/(1+10*ss)**2))/(1+10*ss)**2
    integral = np.trapezoid(gs,lambdas)
    close('quadrature',integral,exact,1e-5,1e-14)
    close('holdup',algebraic[0],.01)
    close('liquid_interface',algebraic[1],1e-4*.0006-integral,1e-7,1e-14)
    close('gas_water_heat',algebraic[2:],[.0006-1e-7*(16000-1e5*si),.00002,-25])
    diag = assembly['diagnostics'](state)
    close('transport_diagnostics',[float(diag[2]),float(diag[3][0]),float(diag[3][1]),float(diag[4])],[1e-4,1e-7,1e-7,10])
    close('vapor_caloric',[float(diag[5]),*np.asarray(diag[6]).ravel()],[23.4375,0,0])
    for phase in (liquid,vapor):
        flows = np.asarray(phase.flow(state[0 if phase.kind=='liquid' else 2],state[1 if phase.kind=='liquid' else 3],
                                    resolved['parameters'][0 if phase.kind=='liquid' else 1])).ravel()
        args = np.r_[325 if phase.kind=='liquid' else 330,99950,flows]
        close(phase.kind+'_off_capacity',phase.enthalpy_temperature_derivative(args),48 if phase.kind=='liquid' else 30)
    # Independent vapor molar-enthalpy derivatives, including all second actions.
    args = np.r_[330,99950,[dict(C=.16,W=.12,I=1)[s] for s in vapor.feed_ids]]
    n, c, dt = 1.28,30.,10.
    ci = np.r_[0.,0.,[30. if s=='I' else 0. for s in vapor.feed_ids]]
    for i in range(len(args)):
        first = c/n if i==0 else 0. if i==1 else (ci[i]/n-c/n**2)*dt
        close('molar_h_first_'+str(i),vapor.actions(args,[vapor.enthalpy],(i,)),first)
        for j in range(len(args)):
            second = ((ci[max(i,j)]/n-c/n**2) if min(i,j)==0 and max(i,j)>1 else
                      (-(ci[i]+ci[j])/n**2+2*c/n**3)*dt if min(i,j)>1 else 0.)
            close(f'molar_h_second_{i}_{j}',vapor.actions(args,[vapor.enthalpy],(i,j)),second,1e-7,1e-10)
    result = dict(checks=errors,order=t,scaling=checkpoint['scaling'],capabilities=prepared['capabilities'],source_identity=prepared['source_identity'])
    save(directory/'prepared_check.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    for label, permuted in [('declared',False),('permuted',True)]:
        result = prepared_check(args.output/label,permuted)
        print(label,len(result['checks']),'independent prepared/equation checks passed',flush=True)
