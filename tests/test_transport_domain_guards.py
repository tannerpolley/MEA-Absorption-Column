import math

import numpy as np
import pytest

from mea_absorption_column.BVP.robust_core import make_solver_diagnostics
from mea_absorption_column.Transport.Enhancement_Factor import enhancement_factor
from mea_absorption_column.Transport.Hydraulic_Variables_Correlations import (
    holdup,
    interfacial_area,
)
from mea_absorption_column.Transport.Pressure_Drop import pressure_drop
from mea_absorption_column.Transport.Transfer_Coefficients import (
    heat_transfer_coeff,
    mass_transfer_coeff,
)
from mea_absorption_column.Transport.domain_guards import DomainGuardError
from mea_absorption_column.config.Constants import packing_params


PACKING = (
    packing_params["MellapakPlus252Y"]["a_p"],
    packing_params["MellapakPlus252Y"]["eps"],
    packing_params["MellapakPlus252Y"]["Cl"],
    packing_params["MellapakPlus252Y"]["Cv"],
    packing_params["MellapakPlus252Y"]["Cs"],
    packing_params["MellapakPlus252Y"]["Cp_0"],
    packing_params["MellapakPlus252Y"]["Ch"],
)


def test_pressure_drop_returns_finite_positive_for_valid_domain():
    diagnostics = make_solver_diagnostics()

    value = pressure_drop(
        h_L=0.05,
        rho_mass_l=1050.0,
        rho_mass_v=1.2,
        mul_mix=0.003,
        muv_mix=1.8e-5,
        A=0.32,
        ul=0.002,
        uv=1.5,
        packing=PACKING,
        diagnostics=diagnostics,
    )

    assert math.isfinite(value)
    assert value > 0.0
    assert diagnostics["domain_guard_counts"].get("pressure_drop", 0) == 0


def test_pressure_drop_rejects_liquid_holdup_above_void_fraction():
    diagnostics = make_solver_diagnostics()

    with pytest.raises(DomainGuardError):
        pressure_drop(
            h_L=0.98,
            rho_mass_l=1050.0,
            rho_mass_v=1.2,
            mul_mix=0.003,
            muv_mix=1.8e-5,
            A=0.32,
            ul=0.002,
            uv=1.5,
            packing=PACKING,
            diagnostics=diagnostics,
        )

    assert diagnostics["domain_guard_counts"]["pressure_drop"] == 1
    assert diagnostics["first_failed_domain"] == "pressure_drop"


def test_hydraulic_guards_reject_nonpositive_inputs():
    diagnostics = make_solver_diagnostics()

    with pytest.raises(DomainGuardError):
        interfacial_area(
            rho_mass_l=-1.0,
            sigma=0.07,
            ul=0.002,
            A=0.32,
            packing=PACKING,
            diagnostics=diagnostics,
        )

    assert diagnostics["domain_guard_counts"]["hydraulics"] == 1


def test_holdup_clips_flooded_state_with_diagnostic():
    diagnostics = make_solver_diagnostics()

    h_L, h_V = holdup(ul=10.0, mul_mix=0.1, rho_mass_l=10.0, packing=PACKING, diagnostics=diagnostics)

    assert diagnostics["domain_guard_counts"]["hydraulics"] == 1
    assert 0.0 < h_L < PACKING[1]
    assert h_V > 0.0


def test_transfer_guards_reject_nonpositive_vapor_holdup():
    diagnostics = make_solver_diagnostics()

    with pytest.raises(DomainGuardError):
        mass_transfer_coeff(
            h_L=0.05,
            h_V=-0.1,
            rho_mass_v=1.2,
            muv_mix=1.8e-5,
            Dl_CO2=1e-9,
            Dv_CO2=1e-5,
            Dv_H2O=1e-5,
            Dv_T=1e-5,
            A=0.32,
            Tv=320.0,
            ul=0.002,
            uv=1.5,
            packing=PACKING,
            diagnostics=diagnostics,
        )

    assert diagnostics["domain_guard_counts"]["mass_transfer"] == 1


def test_heat_transfer_guards_reject_nonpositive_coefficients():
    diagnostics = make_solver_diagnostics()

    with pytest.raises(DomainGuardError):
        heat_transfer_coeff(
            P=101325.0,
            kv_CO2=0.0,
            kt_vap=0.02,
            Cpv_T=35.0,
            rho_mol_v=40.0,
            Dv_CO2=1e-5,
            a_eA=50.0,
            diagnostics=diagnostics,
        )

    assert diagnostics["domain_guard_counts"]["heat_transfer"] == 1


def test_enhancement_factor_is_finite_for_valid_inputs():
    diagnostics = make_solver_diagnostics()

    E, Psi, Psi_H, payload = enhancement_factor(
        Tl=323.15,
        Cl_true=[900.0, 9000.0, 30000.0, 1800.0, 1500.0, 200.0],
        y_CO2=0.08,
        P=109500.0,
        H_CO2_mix=2.5e6,
        kl_CO2=2e-4,
        kv_CO2=0.01,
        Dl_CO2=1e-9,
        Dl_MEA=8e-10,
        Dl_ion=8e-10,
        E_type="implicit",
        diagnostics=diagnostics,
    )

    assert E >= 1.0
    assert Psi > 0.0
    assert Psi_H > 0.0
    assert all(np.isfinite(payload))


def test_enhancement_factor_falls_back_to_explicit_when_implicit_subsolve_fails(monkeypatch):
    diagnostics = make_solver_diagnostics()

    class FailedSolve:
        success = False
        message = "synthetic failure"
        x = np.array([np.nan, np.nan])

    monkeypatch.setattr(
        "mea_absorption_column.Transport.Enhancement_Factor.least_squares",
        lambda *args, **kwargs: FailedSolve(),
    )

    E, Psi, Psi_H, payload = enhancement_factor(
        Tl=323.15,
        Cl_true=[900.0, 9000.0, 30000.0, 1800.0, 1500.0, 200.0],
        y_CO2=0.08,
        P=109500.0,
        H_CO2_mix=2.5e6,
        kl_CO2=2e-4,
        kv_CO2=0.01,
        Dl_CO2=1e-9,
        Dl_MEA=8e-10,
        Dl_ion=8e-10,
        E_type="implicit",
        diagnostics=diagnostics,
    )

    assert E >= 1.0
    assert Psi > 0.0
    assert Psi_H > 0.0
    assert all(np.isfinite(payload))
    assert diagnostics["domain_guard_counts"]["enhancement_factor"] == 1


def test_enhancement_factor_rejects_invalid_domain():
    diagnostics = make_solver_diagnostics()

    with pytest.raises(DomainGuardError):
        enhancement_factor(
            Tl=323.15,
            Cl_true=[900.0, 9000.0, 30000.0, 1800.0, 1500.0, 200.0],
            y_CO2=0.08,
            P=109500.0,
            H_CO2_mix=2.5e6,
            kl_CO2=0.0,
            kv_CO2=0.01,
            Dl_CO2=1e-9,
            Dl_MEA=8e-10,
            Dl_ion=8e-10,
            E_type="implicit",
            diagnostics=diagnostics,
        )

    assert diagnostics["domain_guard_counts"]["enhancement_factor"] == 1
