"""Two-state candidate-wheel numerical verification: #85 comment 6531482275."""
import argparse
import copy
import csv
import hashlib
import json
import runpy
import time
from pathlib import Path

import casadi as ca
import numpy as np
from mea_absorption_column.Thermodynamics.reactive_bundle import engine_liquid
from mea_absorption_column.Thermodynamics.casadi_reactive import build_loading_path_function

ROOT = Path(__file__).resolve().parents[3]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda a: np.asarray(a).tolist(), allow_nan=False)+'\n')


def construct(root, permuted=False):
    source = root/'inputs'
    h = json.loads((source/'handoff.json').read_text())
    entry = h['fitted_parameter_document']['B1']
    assert hashlib.sha256((source/'parameters.json').read_bytes()).hexdigest() == entry['sha256']
    assert hashlib.sha256((source/'speciation.csv').read_bytes()).hexdigest() == '488940ecf3fc984edeee3fa145fe6f76936396fce802b5af881bcbee70f677ef'
    caloric_hash = next(r['sha256'] for r in h['files'] if r['path']=='inputs/caloric.json')
    assert hashlib.sha256((source/'caloric.json').read_bytes()).hexdigest()==caloric_hash
    ids, feeds = entry['components'][::(-1 if permuted else 1)], ['carbon-dioxide','water','methyldiethanolamine']
    rows = list(dict.fromkeys(k for s in h['species'].values() for k in s['analytical_components']))
    if permuted:
        feeds.reverse()
        rows.reverse()
    ref = {k: v for k, v in h['neutral_references']['aqueous'].items() if k != 'effective_species_reference'}
    declaration = dict(species_order=ids, liquid_feed_order=feeds,
        liquid_support=h['problem_templates']['loaded-liquid']['phases'][0]['support'][::(-1 if permuted else 1)],
        roles=dict(co2='carbon-dioxide',water='water',amines=['methyldiethanolamine'],inert_gases=[]),
        atomic_masses_kg_mol=dict(C=.012011,H=.001008,N=.014007,O=.015999),
        formulas={s: {k:v for k,v in h['species'][s]['element_formula'].items() if k!='charge'} for s in ids},
        components=dict(order=rows,rows=[[h['species'][s]['analytical_components'].get(k,0) for s in ids] for k in rows]))
    reaction = dict(parameter_sha256=entry['sha256'],neutral_reference=ref,
                    reactions=h['reactions']['aqueous'][::(-1 if permuted else 1)])
    directory = root/('permuted' if permuted else 'declared')
    save(directory/'reactions.json',reaction)
    save(directory/'declaration.json',declaration)
    phase = engine_liquid(source, source/'caloric.json', 'mdea_reordered' if permuted else 'mdea',
        declarations=declaration,reaction_path=directory/'reactions.json',parameter_path=source/'parameters.json',row_tolerance=1e-10)
    assert list(phase.component_ids)==ids and list(phase.feed_ids)==feeds
    return h, declaration, phase


def affinity(phase, compiled, point):
    chemical = {r.index: v for r,v in zip(compiled.rows,compiled.evaluate(point).values) if r.kind.name=='Chemical'}
    return np.array([-sum(row.stoichiometry.get(s,0)*chemical[i] for i,s in enumerate(phase.component_ids)) for row in phase.reactions])


def verify(root, group, fixture, permuted=False):
    started = time.monotonic()
    h, declaration, phase = construct(root,permuted)
    q, ids, feeds = phase._q, list(phase.component_ids), list(phase.feed_ids)
    rows = {r['identity']:r for r in csv.DictReader((root/'inputs/speciation.csv').open())}
    row = rows['comparison:229' if fixture=='B' else 'comparison:233']
    feed = dict(zip(['carbon-dioxide','water','methyldiethanolamine'],[float(row[k]) for k in ['alpha','water','amine']]))
    u = np.array([float(row['T']),float(row['P']),*[feed[s] for s in feeds]])
    scales = np.r_[1.,1e4,.1*u[2:]]
    controls = ['T','P',*feeds]
    labels = [*phase.output_ids,*('log_activity:'+s for s in ids),'molar_enthalpy']
    output_scale = np.r_[np.ones(8),1e5,1e5,5e4,1e5,np.ones(8),1e5]
    symbol = ca.MX.sym('u',5)
    expression = ca.vertcat(phase(symbol),phase._molar_enthalpy(symbol))
    # All log activities are native values/actions, independent of the two fugacity outputs.
    def values(v):
        return np.r_[np.asarray(phase(v)).ravel(),phase.actions(v,phase.log_activities),phase.actions(v,(phase.enthalpy,))]
    physical_jac = ca.Function('physical_jac',[symbol],[ca.jacobian(expression,symbol)])
    def first(v):
        physical = np.array(physical_jac(v))
        logs = np.column_stack([phase.actions(v,phase.log_activities,(j,)) for j in range(5)])
        return np.vstack([physical[:-1],logs,physical[-1:]])
    checks = {}
    def close(name,a,b,rtol=1e-5,atol=1e-8):
        a,b = np.asarray(a),np.asarray(b)
        ratio = abs(a-b)/(atol+rtol*np.maximum(abs(a),abs(b)))
        checks[name] = dict(maximum_ratio=float(ratio.max()),maximum_absolute_error=float(abs(a-b).max()),worst_index=list(np.unravel_index(ratio.argmax(),ratio.shape)))
        assert ratio.max()<=1, (name,checks[name])
    compiled, state = phase.solve(tuple(u))
    runtime = runpy.run_path(str(ROOT/'scripts/check_epcsaft_integration.py'))
    engine = runtime['resolve_epcsaft'](runtime['load_contract']())
    result = dict(engine=engine,fixture=row,inputs=u,controls=controls,species=ids,reactions=[r.name for r in phase.reactions],labels=labels,scales=scales,output_scale=output_scale,
                  execution=dict(wall_limit_s=30),solver=state.solver_status,linear_solver=state.linear_solver,
                  requested_tolerance_met=state.requested_tolerance_met,checks=checks)
    assert state.requested_tolerance_met and min(np.asarray(phase(u)).ravel()[:8])>0
    template = copy.deepcopy(h['problem_templates']['loaded-liquid'])
    template.update(T=float(u[0]),P=float(u[1]),feed=dict(basis='amounts',values=feed))
    template.pop('reaction_set')
    template['neutral_reference'] = {k:v for k,v in h['neutral_references']['aqueous'].items() if k!='effective_species_reference'}
    template['reactions'] = [dict(r,correlation=q.ReactionLogPolynomial(**r['correlation'])) for r in h['reactions']['aqueous']]
    if group=='values':
        reference = q.solve_equilibrium(phase.model,q.problem_from_mapping(template))
        assert reference.requested_tolerance_met
        direct = q.compile_problem(phase.model,q.problem_from_mapping(template))
        native = [r for obs in [phase.state_observables,phase.log_activities,(phase.enthalpy,)]
                  for r in q.solved_state_actions(direct,reference,q.SolvedStateActionRequest(list(obs),[phase._direction(None)])).results]
        assert all(r.status.name=='Available' for r in native)
        raw = np.array([r.value for r in native])
        expected = np.r_[raw[:8],np.exp(raw[8:10])*phase._gas_constant*u[0],raw[10],raw[:8].sum()*raw[-1],raw[11:]]
        y,jac = values(u),first(u)*scales/output_scale[:,None]
        close('direct_native_values',y/output_scale,expected/output_scale,1e-8,1e-10)
        atoms = np.array([[declaration['formulas'][s].get(e,0) for s in ids] for e in ['C','H','N','O']])
        analytical = np.array([feed.get(s,0) for s in ids])
        close('element_mol',atoms@y[:8],atoms@analytical,0,1e-10)
        close('component_mol',np.array(declaration['components']['rows'])@(y[:8]-analytical),0,0,1e-10)
        close('charge_mol',np.array(phase.charges)@y[:8],0,0,1e-10)
        for step in [1e-3,5e-4]:
            difference = np.column_stack([(values(u+step*scales[j]*np.eye(5)[j])-values(u-step*scales[j]*np.eye(5)[j]))/(2*step*output_scale) for j in range(5)])
            close('first_'+str(step),jac,difference)
        result.update(values=y,first=jac,affinity_over_rt=affinity(phase,compiled,state.coordinates))
    elif group=='actions':
        c = phase.co2_input
        second = np.column_stack([phase.actions(u,phase.log_activities,(c,j)) for j in range(5)])*scales[c]*scales
        reverse = np.column_stack([phase.actions(u,phase.log_activities,(j,c)) for j in range(5)])*scales[c]*scales
        close('second_symmetry',second,reverse)
        bulk = u.copy()
        bulk[c] = .128
        lam = 0. if fixture=='B' else np.log(.258/.128)
        path = build_loading_path_function(phase)
        z = ca.MX.sym('path_inputs',6)
        outputs = ca.vertcat(*path(z[:5],z[5]))
        graph = ca.Function('path_jac',[z],[outputs,ca.jacobian(outputs,z)])
        p = np.r_[bulk,lam]
        ps, pq = np.r_[1.,1e4,.1*bulk[2:],1.],np.r_[output_scale[:12],np.ones(8)]
        path_value,path_jac = [np.asarray(a) for a in graph(p)]
        path_jac = path_jac*ps/pq[:,None]
        dt = second/scales[c]/scales*u[c]
        dt[:,c] += phase.actions(u,phase.log_activities,(c,))
        manual = np.column_stack([dt,dt[:,c]*u[c]])
        manual[:,c] *= np.exp(lam)
        close('path_chain_rule',path_jac[12:],manual*ps)
        close('path_tangent',path_value.ravel()[12:],u[c]*phase.actions(u,phase.log_activities,(c,)))
        for step in [1e-3,5e-4]:
            difference = np.column_stack([(phase.actions(u+step*scales[j]*np.eye(5)[j],phase.log_activities,(c,))-phase.actions(u-step*scales[j]*np.eye(5)[j],phase.log_activities,(c,)))*scales[c]/(2*step) for j in range(5)])
            close('second_'+str(step),second,difference)
            difference = np.column_stack([(np.asarray(graph(p+step*ps[j]*np.eye(6)[j])[0]).ravel()-np.asarray(graph(p-step*ps[j]*np.eye(6)[j])[0]).ravel())/(2*step*pq) for j in range(6)])
            close('path_'+str(step),path_jac,difference)
        result.update(second=second,path_value=path_value.ravel()/pq,path_jac=path_jac)
    elif group=='reference':
        base = q.problem_from_mapping(template)
        frozen = q.solve_equilibrium(phase.model,base)
        assert frozen.requested_tolerance_met
        reference_compiled = q.compile_problem(phase.model,base)
        assert len(frozen.coordinates)==compiled.primal_size+compiled.rank+compiled.charge_row_count
        target = affinity(phase,reference_compiled,frozen.coordinates)
        variants = {}
        for mode in ['mixed','fixed','system']:
            reactions = [q.Reaction(r.stoichiometry,name=r.name,correlation=r.correlation,
                reference=q.ReactionReference(r.reference.source_basis,reference_pressure_pa=r.reference.reference_pressure_pa if mode=='mixed' else (1e5 if mode=='fixed' else None))) for r in phase.reactions]
            problem = q.Problem(phases=base.phases,T=base.T,P=base.P,feed=base.feed,reactions=reactions,neutral_reference=base.neutral_reference,row_tolerance=1e-10)
            other = q.compile_problem(phase.model,problem)
            assert [(r.kind.name,r.phase,r.index) for r in compiled.rows]==[(r.kind.name,r.phase,r.index) for r in other.rows]==[(r.kind.name,r.phase,r.index) for r in reference_compiled.rows]
            a = affinity(phase,other,frozen.coordinates)
            difference = float(abs(a-target).max())
            assert difference<=1e-9 if mode=='mixed' else difference>1e-8
            variants[mode] = dict(affinity_over_rt=a,affinity_j_mol=a*phase._gas_constant*u[0],maximum_difference=difference,
                reference_pressures_pa=[r.reference.reference_pressure_pa or float(u[1]) for r in reactions])
        result.update(frozen_coordinates=frozen.coordinates,composition=frozen.mole_fractions,density=frozen.molar_densities,variants=variants)
    result.update(seconds=time.monotonic()-started,native_solves=phase.stats["native_solves"])
    assert result["seconds"]<=30, result["seconds"]
    save(root/f'{group}_{fixture}_{"permuted" if permuted else "declared"}.json',result)
    return result


def permutation(root):
    errors = {}
    for fixture in ['B','I']:
        left,right = [json.loads((root/f'values_{fixture}_{order}.json').read_text()) for order in ['declared','permuted']]
        rm,cm = [right['labels'].index(s) for s in left['labels']],[right['controls'].index(s) for s in left['controls']]
        pairs = [('values',np.array(left['values'])/left['output_scale'],(np.array(right['values'])/right['output_scale'])[rm]),('first',left['first'],np.array(right['first'])[np.ix_(rm,cm)])]
        reactions = [right['reactions'].index(s) for s in left['reactions']]
        pairs.append(('affinity_over_rt',left['affinity_over_rt'],np.array(right['affinity_over_rt'])[reactions]))
        a,b = [json.loads((root/f'actions_{fixture}_{order}.json').read_text()) for order in ['declared','permuted']]
        species = [b['species'].index(s) for s in a['species']]
        pm = rm[:12]+[12+i for i in species]
        pairs += [('second',a['second'],np.array(b['second'])[np.ix_(species,cm)]),('path_value',a['path_value'],np.array(b['path_value'])[pm]),('path_jac',a['path_jac'],np.array(b['path_jac'])[np.ix_(pm,cm+[5])])]
        for name,x,y in pairs:
            ratio = abs(np.asarray(x)-y)/(1e-9+1e-7*np.maximum(abs(np.asarray(x)),abs(y)))
            errors[fixture+'_'+name] = float(ratio.max())
            assert ratio.max()<=1,errors
    save(root/'permutation.json',errors)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('group',choices=['values','actions','reference','permutation'])
    parser.add_argument('--fixture',choices=['B','I'],default='B')
    parser.add_argument('--permuted',action='store_true')
    args = parser.parse_args()
    permutation(args.root) if args.group=='permutation' else verify(args.root,args.group,args.fixture,args.permuted)
