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


def scalar_reference(grid, epsilon=1e-12):
    """S3-4/S3-5: closed integral, independent of column/film implementations."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    def free(f):
        return max(np.roots([10,6-10*f,-f]))
    def flux(f,g):
        s = free(f)
        def residual(si):
            return .005*(10*si/(1+10*si)-10*s/(1+10*s))-1e-7*(1e5*g-1e5*si)
        si = brentq(residual,s,g,xtol=1e-14)
        return 1e-7*(1e5*g-1e5*si)
    def length(f,bottom):
        return quad(lambda value: 1/(100*flux(value,value-bottom+11/60)),13/60,f,
                    epsabs=epsilon,epsrel=epsilon)[0]
    bottom = brentq(lambda value:length(value,value)-1,.217,.3,xtol=1e-13)
    liquid = np.array([bottom if z==0 else 13/60 if z==1 else
                       brentq(lambda value:length(value,bottom)-(1-z),13/60,bottom,xtol=1e-13) for z in grid])
    gas = liquid-bottom+11/60
    return dict(bottom_liquid_co2_mol_s=bottom,top_vapor_co2_mol_s=float(gas[-1]),
                capture_percent=100*(bottom-13/60)/(11/60),grid=np.asarray(grid),
                liquid=liquid,vapor=gas,flux=[flux(f,g) for f,g in zip(liquid,gas)],
                reference_integration_tolerance=epsilon)


def solve_case(root,nodes,points,permuted=False):
    label = f'n{nodes}_q{points}'+('_permuted' if permuted else '')
    directory = root/label
    inputs(directory/'inputs',points,permuted)
    req = request(directory/'inputs',nodes,points,directory/'attempt')
    result = run_column(req)
    assert result['numerical_acceptance']==result['physical_acceptance']=='accepted', result.get('failure')
    profile, grid = (np.asarray(result['native_profile'][key]) for key in ('state_matrix','grid'))
    ref = scalar_reference(grid)
    capture = 100*(1-profile[2,-1]/profile[2,0])
    metrics = dict(label=label,nodes=nodes,points=points,runtime_s=result['execution']['runtime_s'],
        capture_percent=capture,capture_error_percentage_points=abs(capture-ref['capture_percent']),
        endpoint_flow_error_mol_s=max(abs(profile[0,0]-ref['bottom_liquid_co2_mol_s']),abs(profile[2,-1]-ref['top_vapor_co2_mol_s'])),
        maximum_co2_profile_error_mol_s=float(np.max(abs(profile[[0,2]]-np.array([ref['liquid'],ref['vapor']])))),
        maximum_flux_error_mol_m2_s=float(np.max(abs(profile[8]-ref['flux']))),
        scaled_residual_inf=result['result']['scaled_residual_inf'])
    np.testing.assert_allclose(profile[0]-profile[2],profile[0,0]-profile[2,0],rtol=0,atol=1e-10)
    for rows,expected,tol in [([1,3],[[1],[.1]],1e-10),([4,5],320,1e-7),([9,10],0,1e-10),([7],.05,1e-10)]:
        np.testing.assert_allclose(profile[rows],np.broadcast_to(expected,profile[rows].shape),rtol=0,atol=tol)
    np.testing.assert_allclose(profile[6],1e5-100*grid,rtol=0,atol=1e-5)
    prepared = _prepare_conserved_column_in_process(resolve_column_config(req))
    liquid = prepared['assembly']['liquid']
    fixed = prepared['resolved_inputs']['parameters'][0]
    states, minimum_amount, component_error, affinity = [],np.inf,0.,0.
    for state in profile.T:
        for fraction in (0.,1.):
            flow = np.asarray(liquid.flow(state[0]*np.exp(fraction*state[11]),state[1],fixed)).ravel()
            u = np.r_[state[4],state[6],flow]
            values = np.asarray(liquid(u)).ravel()
            amounts = dict(zip(liquid.component_ids,values[:len(liquid.component_ids)]))
            minimum_amount = min(minimum_amount,min(amounts.values()))
            for name,row in liquid.components.items():
                expected = flow[liquid.feed_index[{'co2':'C','water':'W','amine:A':'A','amine:B':'B'}[name]]]
                component_error = max(component_error,abs(np.dot(row,values[:6])-expected))
            for amine,bound in [('A','CA'),('B','CB')]:
                affinity = max(affinity,abs(np.log(amounts[bound]/(10*amounts['C']*amounts[amine]))))
            states.append(dict(inputs=u,values=values))
    assert minimum_amount>0 and component_error<=1e-12 and affinity<=1e-12
    metrics.update(minimum_amount_mol=minimum_amount,component_error_mol_s=component_error,affinity_over_rt=affinity)
    if nodes==points==129:
        assert metrics['capture_error_percentage_points']<=.1
        assert metrics['maximum_co2_profile_error_mol_s']<=2e-4
        assert metrics['maximum_flux_error_mol_m2_s']<=1e-5
    save(directory/'comparison.json',dict(metrics=metrics,reference=ref,phase_states=states))
    print(json.dumps(metrics),flush=True)


def compare_study(root, previous):
    labels = [f'n{n}_q33' for n in (17,33,65,129)]+['n129_q65','n129_q129','n129_q129_permuted']
    paths = [(previous if label not in labels[-2:] else root)/label for label in labels]
    rows = [json.loads((path/'comparison.json').read_text())['metrics'] for path in paths]
    for key in ('capture_error_percentage_points','maximum_co2_profile_error_mol_s'):
        assert all(a[key]>b[key] for a,b in zip(rows[:3],rows[1:4]))
    assert abs(rows[4]['capture_percent']-rows[5]['capture_percent'])<=.001
    profiles = [np.array(json.loads((root/label/'attempt/attempt.json').read_text())['native_profile']['state_matrix'])
                for label in labels[-2:]]
    left,right = profiles
    reference = np.array(json.loads((previous/'n129_q129/attempt/attempt.json').read_text())
                         ['native_profile']['state_matrix'])
    for indices,rtol,atol in [([0,1,2,3],1e-8,1e-9),([4,5],0,1e-6),([6],0,1e-3),([7,11],0,1e-8),([8,9,10],1e-7,1e-10)]:
        np.testing.assert_allclose(left[indices],right[indices],rtol=rtol,atol=atol)
        np.testing.assert_allclose(left[indices],reference[indices],rtol=rtol,atol=atol)
    delta = rows[-1]['capture_percent']-rows[-2]['capture_percent']
    assert abs(delta)<=1e-6
    assert abs(rows[-2]['capture_percent']-100*(1-reference[2,-1]/reference[2,0]))<=1e-6
    refs = [scalar_reference([0,1],epsilon) for epsilon in (1e-10,1e-12)]
    assert abs(refs[0]['capture_percent']-refs[1]['capture_percent'])<=1e-7
    save(root/'study.json',dict(cases=rows,permutation_capture_difference_percentage_points=delta,references=refs,
        original_capture_difference_percentage_points=rows[-2]['capture_percent']-100*(1-reference[2,-1]/reference[2,0]),
        previous_comparison_root=str(previous)))
    print(json.dumps(dict(cases=rows,permutation_capture_difference_percentage_points=delta)),flush=True)


def render_study(root):
    """Render retained error values only; do not run the reference or column."""
    import matplotlib.pyplot as plt
    rows = json.loads((root/'study.json').read_text())['cases'][:4]
    fig,ax = plt.subplots(figsize=(6,3.5),layout='constrained')
    ax.plot([r['nodes'] for r in rows],[r['capture_error_percentage_points'] for r in rows],'o-',color='#0072B2',label='Upwind, 33 film points')
    ax.axhline(.1,color='#D55E00',linestyle='--',label='Terminal capture limit')
    ax.set(xlabel='Spatial nodes',ylabel='Capture error (percentage point)',title='Mathematical two-amine column refinement')
    ax.legend();fig.savefig(root/'refinement.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--case',nargs=2,type=int,metavar=('NODES','POINTS'))
    parser.add_argument('--permuted',action='store_true')
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--compare',action='store_true')
    parser.add_argument('--render',action='store_true')
    args = parser.parse_args()
    if args.case:
        solve_case(args.output,*args.case,args.permuted);raise SystemExit
    if args.compare:
        compare_study(args.output,args.previous);raise SystemExit
    if args.render:
        render_study(args.output);raise SystemExit
    for label, permuted in [('declared',False),('permuted',True)]:
        result = prepared_check(args.output/label,permuted)
        print(label,len(result['checks']),'independent prepared/equation checks passed',flush=True)
