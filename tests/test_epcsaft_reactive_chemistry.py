import pandas as pd
import pytest
import numpy as np

from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium_with_model
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    ensure_epcsaft_importable,
)

def _requires_reactive_epcsaft_dataset():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    assert MEA_THERMODYNAMICS_EPCSAFT_DATASET.exists()


def _case_3c_liquid_state():
    df = pd.read_csv("src/mea_absorption_column/data/C_cases_data.csv", index_col=0)
    inputs, _, _ = convert_data(df, run=df.index.get_loc("3C"), type="mole", return_metadata=True)
    Fl, _, Tl, _, _, _, _, P, _ = inputs
    return list(Fl), float(Tl), float(P)


@pytest.mark.parametrize(
    "model",
    (
        "epcsaft_reactive_six_concentration",
        "epcsaft_reactive_six_activity",
        "epcsaft_reactive_six_activity_converted",
        "epcsaft_reactive_nine_activity_rebased",
    ),
)
def test_legacy_reactive_modes_fail_closed_until_constants_meet_v02_contract(model):
    _requires_reactive_epcsaft_dataset()
    Fl, Tl, P = _case_3c_liquid_state()

    with pytest.raises(RuntimeError, match="independently sourced.*standard-state conversion"):
        chemical_equilibrium_with_model(Fl, Tl, model=model, P=P, diagnostics={})


@pytest.mark.parametrize('Tl,P,Fl,reference_density,reference_fugacity', [
    (318.15, 110900., [5.277293318431409, 9.729282110481522, 76.97519250729157],
     53024.00899976211, 8710.019674586),
    (325., 109500., [4.087096190146402, 9.729875260332735, 75.2853477794609],
     None, None),
])
def test_coupled_nine_species_conserves_elements_and_charge(Tl, P, Fl, reference_density, reference_fugacity):
    from mea_absorption_column.Thermodynamics.reactive_bundle import (
        reactive_liquid,
    )

    result = reactive_liquid().solve(Tl, P, Fl, state_input_derivatives=True)
    x, amounts = result['composition'], result['amounts_mol']
    # Independent C/H/N/O counts, in the documented nine-species order.
    elements = np.array([[1, 2, 0, 2, 3, 1, 1, 0, 0],
                         [0, 7, 2, 8, 6, 1, 0, 3, 1],
                         [0, 1, 0, 1, 1, 0, 0, 0, 0],
                         [2, 1, 1, 1, 3, 3, 3, 1, 1]])
    feed = np.asarray(Fl) / sum(Fl)
    # Absolute mole tolerance tests the normalized conserved feed, not
    # percent-level drift from apparent to true-species amount normalization.
    np.testing.assert_allclose(elements @ amounts, elements[:, :3] @ feed,
                               rtol=0, atol=1e-8)
    assert abs(np.dot([0, 0, 0, 1, -1, -1, -2, 1, -1], amounts)) < 1e-10
    assert np.all(x > 0) and x.sum() == pytest.approx(1, abs=1e-12)
    assert result['density_mol_m3'] > 0
    assert result['evidence']['equilibrium_value_valid'] == 1
    assert result['evidence']['local_stability_status'] == 'passed'
    assert result['evidence']['reaction_affinity_inf_norm'] < 1e-8
    if reference_density is not None:
        # Independent Orchestrator handoff on the identical wheel and input set;
        # tolerance admits floating-point replay, not a physical-data fit.
        assert result['density_mol_m3'] == pytest.approx(reference_density, rel=1e-8)
        assert result['fugacities_pa'][0] == pytest.approx(reference_fugacity, rel=1e-8)


def test_loading_path_budget_fails_without_changing_anchor():
    from mea_absorption_column.Thermodynamics.reactive_bundle import DATASET, ReactiveLiquid
    liquid = ReactiveLiquid(DATASET, loading_anchor=.25, max_loading_steps=1)
    with pytest.raises(RuntimeError, match='declared budget is 1'):
        liquid.solve(325., 109500., [0.42, 1., 8.])


def test_coupled_column_retains_nine_species_and_legacy_flux():
    from mea_absorption_column.BVP.ABS_Column import abs_column
    from mea_absorption_column.Thermodynamics.reactive_bundle import MODEL

    df = pd.read_csv('src/mea_absorption_column/data/C_cases_data.csv', index_col=0)
    inputs, _, _ = convert_data(df, run=df.index.get_loc('3C'), type='mole', return_metadata=True)
    Fl, Fv, Tl, Tv, _, H, A, P, packing = inputs
    options = {'thermo_model': MODEL, 'chemical_equilibrium_model': MODEL,
               'thermal_state_mode': 'temperature', 'solver_diagnostics': {}}
    parameters = (np.ones(7), np.ones(7), (Fl[1], Fv[2], Fv[3]), H, A, packing, options)
    y = [Fl[0], Fl[2], Fv[0], Fv[1], Tl, Tv, P]
    out, labels = abs_column(0, y, parameters, run_type='saving', column_names=True)
    fields = {key: dict(zip(labels[key], out[key], strict=True))
              for key in ['Fl', 'Cl', 'x', 'enhance_factor', 'reactive_density']}
    assert len(fields['x']) == 12
    assert sum(out['x'][3:]) == pytest.approx(1)
    assert sum(out['Cl'][3:]) == pytest.approx(fields['reactive_density']['rho_true_mol_m3'])
    assert sum(fields['Fl'][f'Fl_{s}_true'] for s in ['MEA', 'MEAH', 'MEACOO']) == pytest.approx(Fl[1])
    assert fields['enhance_factor']['Cl_MEA_true'] == fields['Cl']['Cl_MEA_true']
    Nl, Nv, kv, area, driving_force, *_ = out['CO2']
    assert Nv == pytest.approx(-kv * area * driving_force * fields['enhance_factor']['Psi_H'])
    assert Nl == -Nv
    rhs = abs_column(0, y, parameters)
    assert np.all(np.isfinite(rhs))
    assert rhs[0] == pytest.approx((-Nl + 1e-10) * H)
    diagnostics = options['solver_diagnostics']
    assert diagnostics['epcsaft_chemistry_last_native_success'] is True
    assert diagnostics['epcsaft_chemistry_last_evidence']['equilibrium_value_valid'] == 1
    assert diagnostics['epcsaft_chemistry_last_evidence']['parameter_fingerprint']
    # A domain error must escape the shared RHS guard, never become a zero RHS.
    from mea_absorption_column.BVP.robust_core import guard_column_rhs
    y[4] = 400.0
    with pytest.raises(ValueError, match='outside'):
        guard_column_rhs(0, y, parameters, evaluator=abs_column)
    options['thermo_model'] = 'ideal_henry'
    with pytest.raises(ValueError, match='must be selected together'):
        guard_column_rhs(0, y, parameters, evaluator=abs_column)


def test_coupled_fugacity_rejects_invalid_species_without_henry_fallback():
    from mea_absorption_column.Thermodynamics.thermo_models import guarded_compute_fugacity
    with pytest.raises(ValueError, match='nine positive finite species'):
        guarded_compute_fugacity('epcsaft_reactive_nine', [0.1, 0.1, 0.7, 0.1],
                                 [0.1] * 6, [1000.] * 6, 313.15, 313.15,
                                 1000., 109500., 7000.)


@pytest.mark.parametrize('composition', [[0.1] * 9, [0.2] + [0.1] * 8])
def test_coupled_fugacity_rejects_inconsistent_composition(composition):
    from mea_absorption_column.Thermodynamics.reactive_bundle import reactive_fugacity
    with pytest.raises(ValueError, match='normalized composition matching concentrations'):
        reactive_fugacity([0.1, 0.1, 0.7, 0.1], composition, [1000.] * 9,
                          313.15, 313.15, 109500., 7000.)


@pytest.mark.parametrize('mode', ['epcsaft_reactive_nine', 'EPCSAFT_REACTIVE_NINE'])
def test_coupled_run_reports_selected_dataset_and_unavailable_legacy_metrics(monkeypatch, mode):
    import importlib
    from mea_absorption_column.Thermodynamics.reactive_bundle import DATASET
    driver = importlib.import_module('mea_absorption_column.Run_Model')

    # Stop at the driver/output boundary; this is not a converged column.
    def stopped_solve(left, right, z, parameters, settings):
        profile = np.array([np.linspace(a, b, len(z)) for a, b in zip(left, right)])
        return profile, z, 'stopped test solve', False, 'not solved'

    monkeypatch.setattr(driver, 'scipy_BVP_solve', stopped_solve)
    df = pd.read_csv('src/mea_absorption_column/data/C_cases_data.csv', index_col=0)
    result = driver.run_model(df, method='scipy-bvp', run=df.index.get_loc('3C'),
                              thermo_model=mode, return_details=True, staged_beds=False,
                              show_info=False, save_run_results=False, plot_temperature=False,
                              solver_settings={'return_internal_profile': True})
    assert result['success'] is False
    assert result['thermo_model'] == mode.lower()
    assert result['epcsaft_dataset'] == str(DATASET)
    for quantity in ['mass', 'reaction', 'charge']:
        assert result[f'epcsaft_chemistry_max_{quantity}_residual'] is None
    assert result['epcsaft_chemistry_last_evidence'] == '{}'
