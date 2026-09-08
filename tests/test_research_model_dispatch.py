"""Local arithmetic and dispatch checks; no retained column runs."""
import casadi as ca
import numpy as np
import pytest

from mea_absorption_column.BVP.ABS_Column import _reactive_film_linearized_fluxes
from mea_absorption_column.BVP.reactive_jacobian import ReactiveColumnJacobian
from mea_absorption_column.BVP.Methods.Conserved_Reduction import ConservedReduction
from mea_absorption_column.BVP.Methods.Casadi_Collocation import solve_conservative_collocation


def test_film_profile_conservation_and_validation():
    profile = ([0., 1.], [2., 4.], [1., 3.])
    assert _reactive_film_linearized_fluxes(.5, 5., 2., profile) == (-18., 18.)
    assert _reactive_film_linearized_fluxes(.5, 1., 2., profile) == (6., -6.)
    with pytest.raises(ValueError, match='valid equal-length'):
        _reactive_film_linearized_fluxes(.5, 5., 2., ([0., 1.], [0., 1.], [1., 1.]))


@pytest.mark.parametrize('options', [
    {'co2_mass_transfer_model': 'reactive_film_linearization'}, {'enhancement_type': 'implicit'},
])
def test_native_jacobian_rejects_unimplemented_film_derivatives(options):
    parameters = (np.ones(7), np.ones(7), (1., 1., 1.), 1., 1., [],
                  {'thermo_model': 'epcsaft_reactive_nine', 'thermal_state_mode': 'temperature', **options})
    with pytest.raises(ValueError, match='only explicit enhancement'):
        ReactiveColumnJacobian(parameters, 'raw')


def test_additive_conserved_solver_local_recovery():
    z, u = ca.MX.sym('z'), ca.MX.sym('u', 2)
    node = ca.Function('toy', [z, u], [u[0]**2, u[1], u[1]-2*u[0]])
    a, b = ca.MX.sym('a', 2), ca.MX.sym('b', 2)
    boundary = ca.Function('inlet', [a, b], [a[0]-2])
    model = ConservedReduction(node, boundary, initial_state=[2., 4.], lower=[.1, .1], upper=[10., 20.],
        state_scale=[1., 1.], conserved_scale=[1.], algebraic_scale=[1.], boundary_scale=[1.],
        tolerance=1e-9, solver_tolerance=1e-12, max_evaluations=30, max_condition=100.)
    result = model.evaluate(0., [9.])
    np.testing.assert_allclose(result['state'], [3., 6.], atol=1e-9)
    np.testing.assert_allclose(result['rhs_jacobian'], [[1/3]], atol=1e-9)
    with pytest.raises(ValueError, match='strictly increasing'):
        solve_conservative_collocation(node, boundary, [0., 0.], np.ones((2, 2)), [.1, .1], [10., 20.],
            state_scale=[1., 1.], balance_scale=[1.], algebraic_scale=[1.], boundary_scale=[1.])


def test_staged_film_rejected_before_solver(monkeypatch):
    import importlib
    module = importlib.import_module('mea_absorption_column.Run_Model')
    monkeypatch.setattr(module, 'convert_data', lambda *a, **k: (None, np.ones(9), {'beds': 2, 'intercoolers': 0}))
    with pytest.raises(ValueError, match='per-bed profile coordinates'):
        module.run_model(None, method='scipy-bvp', solver_settings={'co2_mass_transfer_model': 'reactive_film_linearization'})


def test_shooting_seed_uses_supported_derivative_mode(monkeypatch):
    import importlib
    module = importlib.import_module('mea_absorption_column.Run_Model')
    run = module.run_model
    monkeypatch.setattr(module, 'convert_data', lambda *a, **k: (None, np.ones(9), {'beds': 1, 'intercoolers': 0}))
    class SeedInspected(Exception):
        pass
    def seed(*args, **kwargs):
        assert kwargs['method'] == 'single'
        assert kwargs['solver_settings']['jacobian_mode'] == 'numerical'
        assert kwargs['solver_settings']['seed_from_shooting'] is False
        raise SeedInspected
    monkeypatch.setattr(module, 'run_model', seed)
    settings = {'jacobian_mode': 'native', 'seed_from_shooting': True}
    with pytest.raises(SeedInspected):
        run(None, method='scipy-bvp', staged_beds=False, solver_settings=settings)
    assert settings['jacobian_mode'] == 'native'
