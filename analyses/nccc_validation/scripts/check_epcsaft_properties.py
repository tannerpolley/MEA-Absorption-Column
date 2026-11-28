"""Numerical verification of the nine-species bulk-phase property closures (issue #99)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mea_absorption_column.BVP.ABS_Column import abs_column
from mea_absorption_column.BVP.reactive_jacobian import ReactiveColumnJacobian
from mea_absorption_column.Thermodynamics.casadi_reactive import build_caloric_flow_function
from mea_absorption_column.Thermodynamics.reactive_bundle import (
    BULK_VAPOR, FIT_A, ReactiveLiquid, bulk_vapor_state, engine_liquid, engine_vapor,
)
from mea_absorption_column.misc.Convert_Data import convert_data


ANALYSIS = Path('analyses/nccc_validation')
ARCHIVE = ANALYSIS/'results/runs/fitA_article_20261128'
INPUTS = Path('src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv')
THERMOCHEMISTRY = FIT_A/'ideal-gas-thermochemistry.json'
# 10 W is below 0.05 K times either phase's heat-capacity rate.
ABSOLUTE_LIMITS = {'H_L_W': 10., 'H_V_W': 10., 'h_L_CO2_J_mol': 10., 'h_L_water_J_mol': 10.,
                   'h_V_CO2_J_mol': 10., 'h_V_water_J_mol': 10.}
RELATIVE_LIMITS = {'C_L_W_K': 1e-4, 'V_L_m3_s': 1e-4, 'cp_V_J_mol_K': 1e-4, 'rho_V_mol_m3': 1e-4}
RTOL, ATOL = 2e-5, 2e-8


def reference_phases():
    liquid = engine_liquid(FIT_A, THERMOCHEMISTRY, loading_policy=dict(
        loading_anchor=.25, max_log_loading_step=.1, max_loading_steps=32))
    vapor = engine_vapor(BULK_VAPOR, THERMOCHEMISTRY)
    return liquid, vapor, build_caloric_flow_function(vapor)


def profile_states(case):
    directory = ARCHIVE/('coarse' if case == '3C' else f'campaign_{case}')
    liquid = pd.read_csv(directory/'Fl.csv')[['Fl_CO2', 'Fl_MEA', 'Fl_H2O']].to_numpy()
    vapor = pd.read_csv(directory/'Fv.csv')[['Fv_CO2', 'Fv_H2O', 'Fv_N2', 'Fv_O2']].to_numpy()
    temperatures = pd.read_csv(directory/'T.csv')
    pressure = pd.read_csv(directory/'transport.csv').P.to_numpy()
    return [(z, (tl, p, *fl), (tv, p, *fv)) for (z, tl, tv), p, fl, fv in
            zip(temperatures.to_numpy(), pressure, liquid, vapor, strict=True)]


def caloric_comparison(runtime, references, liquid_input, vapor_input):
    liquid, vapor, caloric = references
    n, m = sum(liquid_input[2:]), sum(vapor_input[2:])
    l = runtime.solve(liquid_input[0], liquid_input[1], liquid_input[2:], caloric=True)
    v = bulk_vapor_state(vapor_input[0], vapor_input[1], vapor_input[2:], str(FIT_A), True)
    l_out = np.asarray(liquid(liquid_input)).ravel()
    l_amount = sum(l_out[:9])
    h_molar = l_out[-1]/l_amount
    d_h = []
    observables = (*liquid.state_observables[:9], liquid.enthalpy)
    for i in (0, 2, 4):
        action = liquid.actions(liquid_input, observables, (i,))
        d_h.append(h_molar*sum(action[:9]) + l_amount*action[-1])
    H, cp, hparts = caloric(vapor_input)
    vapor_density = float(vapor(vapor_input)[-2])
    actual = [l['enthalpy']*n, v['enthalpy'], *l['enthalpy_partials'][1:],
              *v['enthalpy_partials'][1:], l['enthalpy_partials'][0]*n,
              n*sum(l['amounts_mol'])/l['density_mol_m3'], v['enthalpy_partials'][0]/m, v['density']]
    expected = [l_out[-1], float(H), *d_h[1:], *np.asarray(hparts).ravel(), d_h[0],
                l_amount/l_out[-2], float(cp), vapor_density]
    return {name: (float(a), float(e)) for name, a, e in
            zip(ABSOLUTE_LIMITS | RELATIVE_LIMITS, actual, expected, strict=True)}


def column_state(liquid_input, vapor_input):
    data = pd.read_csv(INPUTS, index_col=0)
    inputs, _ = convert_data(data, run=data.index.get_loc('3C'), type='mass',
                             vapor_composition_mode='dry_saturated', gas_flow_basis='reported_dry_mass')
    _, _, _, _, _, height, area, _, packing = inputs
    tl, p, lc, lm, lw = liquid_input
    tv, _, vc, vw, vn, vo = vapor_input
    runtime = ReactiveLiquid(FIT_A, loading_anchor=.25, reuse_states=True)
    options = dict(thermo_model='epcsaft_reactive_nine', chemical_equilibrium_model='epcsaft_reactive_nine',
                   thermal_state_mode='temperature', reactive_liquid=runtime,
                   liquid_density_source='epcsaft', energy_model='epcsaft', water_transfer_source='epcsaft')
    scales = np.array([5., 100., 2., 1., 320., 320., 1e5])
    return np.array([lc, lw, vc, vw, tl, tv, p])/scales, (scales, scales, (lm, vn, vo), height, area, packing, options)


def interface_comparison(references, liquid_input, vapor_input):
    state, parameters = column_state(liquid_input, vapor_input)
    exports, _ = abs_column(0., state, parameters, run_type='saving')
    liquid, vapor, caloric = references
    liquid_f = float(liquid(liquid_input)[-3])
    vapor_f = float(vapor(vapor_input)[-3])
    water, _, kw, area, *_ = exports['H2O']
    expected_water = kw*area*(vapor_f-liquid_f)
    _, _, h_vapor = caloric(vapor_input)
    expected_h_co2 = exports['CO2'][0]*float(h_vapor[0])
    expected_h_water = expected_water*float(h_vapor[1])
    # Conductance uses independent phase partials and the exported transport coefficient.
    q = exports['transport'][8]*area*(vapor_input[0]-liquid_input[0])
    return dict(liquid_water_fugacity_Pa=liquid_f, vapor_water_fugacity_Pa=vapor_f,
                water_flux=water, expected_water_flux=expected_water,
                phase_energy=[exports['Hl'][11], -exports['Hv'][13]],
                expected_phase_energy=q+expected_h_co2+expected_h_water,
                liquid_components=exports['Hl'][7:9], expected_liquid_components=[expected_h_co2, expected_h_water])


def directional_comparison(liquid_input, vapor_input):
    state, parameters = column_state(liquid_input, vapor_input)
    jac = ReactiveColumnJacobian(parameters, 'raw')(0., state)
    # First direction retains the legacy check. Second resolves water and both temperature rows.
    for physical in ([.03, .02, -.01, .002, .2, -.1, 20.],
                     [.04, -.05, .015, .2, -.15, .28, -40.]):
        direction = np.asarray(physical)/parameters[0]
        step = 1e-3
        centered = (abs_column(0., state+step*direction, parameters)
                    - abs_column(0., state-step*direction, parameters))/(2*step)
        yield jac@direction, centered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    references = reference_phases()
    runtime = ReactiveLiquid(FIT_A, loading_anchor=.25, reuse_states=True)
    rows, jacobian = [], []
    for case in ('1C', '3C', '7C'):
        states = profile_states(case)
        for position, l, v in states:
            record = dict(case=case, position=position)
            for name, (actual, expected) in caloric_comparison(runtime, references, l, v).items():
                record.update({name: actual, name+'_reference': expected})
            rows.append(record)
        for position, l, v in (states[0], states[len(states)//2], states[-1]):
            for direction, (a, e) in enumerate(directional_comparison(l, v)):
                for i in range(len(a)):
                    jacobian.append(dict(case=case, position=position, direction=direction,
                                         row=i, jacobian=a[i], centered=e[i], error=abs(a[i]-e[i]),
                                         tolerance=ATOL+RTOL*abs(e[i])))
    frame, derivatives = pd.DataFrame(rows), pd.DataFrame(jacobian)
    frame.to_csv(args.output/'C2_caloric.csv', index=False)
    derivatives.to_csv(args.output/'C3_jacobian.csv', index=False)
    _, l, v = profile_states('3C')[0]
    transfer = interface_comparison(references, l, v)
    probe = interface_comparison(references, (318.15, 109500., 2.5, 10., 79.),
                                 (316.75, 109500., 2., 1.67, 20., 2.))
    errors = {}
    for name, limit in (ABSOLUTE_LIMITS | RELATIVE_LIMITS).items():
        expected = frame[name+'_reference']
        error = (frame[name]-expected).abs()
        if name in RELATIVE_LIMITS:
            error /= expected.abs()
        errors[name] = dict(max_error=error.max(), limit=limit)
    flux = abs(probe['water_flux']-probe['expected_water_flux'])
    transfer_error = max(np.max(np.abs(np.asarray(transfer['phase_energy'])-transfer['expected_phase_energy'])),
                         np.max(np.abs(np.asarray(transfer['liquid_components'])-transfer['expected_liquid_components'])))
    liquid = runtime.solve(l[0], l[1], l[2:], caloric=True)
    vapor = bulk_vapor_state(v[0], v[1], v[2:], str(FIT_A), True)
    record = dict(linear_solver={'liquid': liquid['evidence']['linear_solver'], 'vapor': vapor['linear_solver']},
                  C2=errors, A1=dict(transfer, max_error_W_m=transfer_error, limit_W_m=10.),
                  A2=dict(probe, abs_flux_error_mol_s_m=flux, limit_mol_s_m=1e-6),
                  C3=dict(max_error_ratio=(derivatives.error/derivatives.tolerance).max(), rtol=RTOL, atol=ATOL,
                          rows=len(derivatives)))
    (args.output/'C2_C3.json').write_text(json.dumps(record, indent=2)+'\n')
    assert all(x['max_error'] <= x['limit'] for x in errors.values()), errors
    assert transfer_error <= 10.
    assert probe['water_flux'] < 0 and flux <= 1e-6
    assert record['C3']['max_error_ratio'] <= 1.


if __name__ == '__main__':
    main()
