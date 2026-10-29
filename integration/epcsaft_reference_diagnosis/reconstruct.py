"""Reconstruct the finite-dilution test reference using the selected wheel, without modifying model inputs.

Engine 9e1bef97: electrolyte_reference.hpp:14-94 and eos_source_reference.cpp:630-1250.
Current reference changes: bd7b6e388 (#353) and 3b893fc2c (#823).
Run from the repository root with uv run --frozen python and a 60 s wall limit.
"""
import json,math
from pathlib import Path
import numpy as np
import epcsaft
from epcsaft import equilibrium as q
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', choices=['reference', 'loading_jump'], default='reference')
args=parser.parse_args()
temperature, pressure = (318.15, 110900.) if args.case=='reference' else (326.2714274736848, 109500.)
apparent = ([5.277293318431409,9.729282110481522,76.97519250729157] if args.case=='reference'
            else [3.979265359040124,9.729875260332735,75.6233916590782])
expected_density = 53024.00899976211 if args.case=='reference' else 50951.5005892302
expected_fugacity = 8710.019674586 if args.case=='reference' else None
root=Path('src/mea_absorption_column/data/epcsaft_datasets')
d=json.loads((root/'MEA_reactive_epcsaft_bundle/parameters.json').read_text())
r=json.loads((root/'MEA_reactive_epcsaft_bundle/reaction-system.json').read_text())
for f in d['model_families']:
 if f['kind']=='electrolyte': f.update(ionic_dispersion='exclude-same-sign-ion-pairs',c_shell=1.,c_dielectric=1.)
for fitted in d['reaction_correlations']:
 row=next(v for v in r['reactions'] if v['reaction_id']==fitted['reaction_id'])
 row['coefficients']={v['name']:v['value']['magnitude'] for v in fitted['coefficients']};row['standard_state_offset']=0.
charges={c['component_id']:c['fixed']['charge_number']['value']['magnitude'] for c in d['components']}
d['pairs']=[p for p in d['pairs'] if charges[p['component_id_a']]*charges[p['component_id_b']]<=0]
if args.case=='loading_jump':
 for pair in d['pairs']:
  for c in pair['coefficients']:
   if c['identity']=='pair/carbon-dioxide/water/k_ij': c['value']['magnitude']*=1.05
model=epcsaft.Mixture(epcsaft.Parameters.from_mapping(d));ids=model.component_ids
reactions=[]
for row in r['reactions']:
 c=row['coefficients'];a,b,c0,d0=(c.get(n,0.) for n in ['a','b_k','c','d_per_k'])
 if row['ln_k_form'].startswith('-ln(10)'):a,b,c0,d0=-math.log(10)*c['b'],-math.log(10)*c['a_k'],0.,-math.log(10)*c['c_per_k']
 else:a+=row.get('standard_state_offset',0.)
 a+=sum(row['stoichiometry'][i]*r['source_standard_state']['log_activity_scale_factors'][i] for i in range(9))
 lo,hi=row['temperature_domain_k']
 # Correlation uses ln(T/Tref), whereas this input uses ln(T).
 correlation=q.ReactionLogPolynomial(a+c0*math.log(313.15),b,c0,d0,313.15,lo,hi,'aqueous-mole-fraction-infinite-dilution-water-v1')
 reactions.append(q.Reaction(dict(zip(ids,row['stoichiometry'])),name=row['reaction_id'],correlation=correlation,reference=q.ReactionReference(q.ReferenceSourceBasis.RawMoleFractionInfiniteDilution,reference_pressure_pa=r["source_standard_state"]["reference_pressure_pa"])))
n=json.loads((root/'MEA_reactive_nine_species/engine-reactions.json').read_text())['neutral_reference']
feed=np.array(apparent);feed/=feed.sum()
problem=q.Problem(phases=[q.Phase('liquid',kind='liquid')],T=temperature,P=pressure,feed=q.Amounts(dict(zip(ids[:3],feed))),reactions=reactions,neutral_reference=q.NeutralReference(**n))
result=q.solve_equilibrium(model,problem)
print(result.success,result.message,result.molar_densities,result.mole_fractions,flush=True)
if result.success:
 state=model.state(temperature,rho=result.molar_densities[0],x=result.mole_fractions)
 f=np.exp(state.log_fugacity_coefficient)*np.array(result.mole_fractions)*pressure
 print('CO2 fugacity Pa',repr(float(f[0])))
 print('density delta',result.molar_densities[0]-expected_density, 'fugacity delta',None if expected_fugacity is None else f[0]-expected_fugacity)
 print('residuals',result.residuals)
print('reference diagnostics',[(x.reaction_index,x.terminal_contraction,x.reference_pressure_pa) for x in result.neutral_reference_diagnostics])
# Reconstruct the source-reference value algorithm recorded by Engine 9e1bef97.
nu=np.array([row['stoichiometry'] for row in r['reactions']]);standard=r['source_standard_state']
neutral_trace=np.array([1.,1.,0.,0.,0.,0.,0.,0.,0.]);solvent=np.eye(9)[2]/model.parameters.molar_masses[2]
B=list(np.eye(9)[:3]);transfers=[];m=1e-12
ref=model.state(temperature,P=100000.,x=(solvent+m*neutral_trace)/(solvent+m*neutral_trace).sum(),phase='liquid')
transfers.extend(ref.log_fugacity_coefficient[:3])
for pair,coeff in standard['ion_pairs']:
 b=np.zeros(9)
 for name,value in zip(pair,coeff):b[ids.index(name)]=value
 trace=neutral_trace+b
 point=model.state(temperature,P=100000.,x=(solvent+m*trace)/(solvent+m*trace).sum(),phase='liquid')
 B.append(b);transfers.append(np.dot(b,point.log_fugacity_coefficient))
contractions=np.linalg.lstsq(np.array(B).T,nu.T,rcond=None)[0].T @ transfers
print('9e1bef97 finite-reference lnphi contractions',contractions)
print('new reference contractions', [v.terminal_contraction for v in result.neutral_reference_diagnostics])
old_c=contractions+nu.sum(axis=1)*math.log(100000./(epcsaft.GAS_CONSTANT_J_PER_MOL_K*temperature))
print('old finite minus current reference contractions',old_c-np.array([v.terminal_contraction for v in result.neutral_reference_diagnostics]))
old_reactions=[]
for row,rx,c in zip(r['reactions'],reactions,old_c):
 cp=rx.correlation;t=temperature
 ln_k=cp.a+cp.b/t+cp.c*math.log(t/cp.reference_temperature)+cp.d*t+c
 old_reactions.append(q.Reaction(dict(zip(ids,row['stoichiometry'])),name=row['reaction_id'],correlation=q.ReactionLogPolynomial(ln_k,0.,0.,0.,t,273.15,498.15,q.EOS_STANDARD_STATE_ID)))
old_problem=q.Problem(phases=[q.Phase('liquid',kind='liquid')],T=temperature,P=pressure,feed=q.Amounts(dict(zip(ids[:3],feed))),reactions=old_reactions)
old_result=q.solve_equilibrium(model,old_problem)
print('finite-reference solve',old_result.success,old_result.message)
s0=model.state(temperature,rho=old_result.molar_densities[0],x=old_result.mole_fractions)
f0=float(math.exp(s0.log_fugacity_coefficient[0])*old_result.mole_fractions[0]*pressure)
print('finite-reference density,fugacity',old_result.molar_densities[0],f0)
print('finite-reference vs test',old_result.molar_densities[0]-expected_density,None if expected_fugacity is None else f0-expected_fugacity)
import hashlib
terms=['hard_chain','dispersion','association','debye_huckel','born']
record=dict(basis=dict(temperature_k=temperature,pressure_pa=pressure,source_reference_pressure_pa=100000.,apparent_amounts=apparent,feed_basis='normalized to one mole'),engine_wheel=epcsaft.provenance()[1],source_parameter_sha256=hashlib.sha256((root/'MEA_reactive_epcsaft_bundle/parameters.json').read_bytes()).hexdigest(),test_reference=dict(density_mol_m3=expected_density,co2_fugacity_pa=expected_fugacity),current_reference=dict(density_mol_m3=result.molar_densities[0],co2_fugacity_pa=float(f[0]),mole_fractions=result.mole_fractions,log_fugacity_coefficients=state.log_fugacity_coefficient,helmholtz_terms={k:getattr(state,k) for k in terms}),reconstructed_finite_reference=dict(density_mol_m3=old_result.molar_densities[0],co2_fugacity_pa=f0,mole_fractions=old_result.mole_fractions,log_fugacity_coefficients=s0.log_fugacity_coefficient,helmholtz_terms={k:getattr(s0,k) for k in terms}),finite_minus_current_reference_contractions=(old_c-np.array([v.terminal_contraction for v in result.neutral_reference_diagnostics])).tolist(),method='Engine 9e1bef97 electrolyte_reference.hpp terminal 1e-12 mol/kg; eos_source_reference.cpp neutral/ion-pair contractions; applied only as a diagnostic to unchanged 1.0.0 EOS',claim='Numerical reconstruction isolates the test-reference discrepancy; not physical validation.')
record['case']=args.case
print(json.dumps(record, indent=2))
print('stability diagnostics', result.reduced_hessian_minimum, result.requested_tolerance_met, result.solver_status)
