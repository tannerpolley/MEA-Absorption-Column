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


def test_pressure_drop_uses_grouped_packing_area_exponent():
    diagnostics = make_solver_diagnostics()
    h_L = 0.05
    rho_mass_l = 1050.0
    rho_mass_v = 1.2
    mul_mix = 0.003
    muv_mix = 1.8e-5
    A = 0.32
    ul = 0.002
    uv = 1.5

    value = pressure_drop(
        h_L=h_L,
        rho_mass_l=rho_mass_l,
        rho_mass_v=rho_mass_v,
        mul_mix=mul_mix,
        muv_mix=muv_mix,
        A=A,
        ul=ul,
        uv=uv,
        packing=PACKING,
        diagnostics=diagnostics,
    )

    a_p, eps, *_rest, cp_0, ch = PACKING
    diameter = (A * 4 / np.pi) ** 0.5
    reynolds_liquid = ul * rho_mass_l / (a_p * mul_mix)
    wetted_area_ratio = ch * reynolds_liquid ** 0.15 * (uv ** 2 * a_p / 9.80665) ** 0.1
    static_holdup = (12 / 9.80665 * mul_mix / rho_mass_l * ul * a_p ** 2) ** (1 / 3) * wetted_area_ratio ** (2 / 3)
    vapor_kinematic_viscosity = muv_mix / rho_mass_v
    dry_factor = uv * rho_mass_v ** 0.5
    packing_diameter = 6 * (1 - eps) / a_p
    wall_factor = (1 + 2 / 3 * (1 / (1 - eps)) * packing_diameter / diameter) ** -1
    reynolds_vapor = uv * packing_diameter / ((1 - eps) * vapor_kinematic_viscosity) * wall_factor
    c1 = 13300 / (a_p ** (3 / 2))
    froude_liquid = ul ** 2 * a_p / 9.80665
    expected_psi = (
        cp_0
        * (64 / reynolds_vapor + 1.8 / reynolds_vapor ** 0.08)
        * ((eps - h_L) / eps) ** 1.5
        * (h_L / static_holdup) ** 0.3
        * np.exp(c1 * np.sqrt(froude_liquid))
    )
    expected = expected_psi * a_p / (eps - h_L) ** 3 * dry_factor ** 2 / (2 * wall_factor)

    assert value == pytest.approx(expected)


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


def test_domain_guards_can_record_without_raising_for_legacy_timing_probe():
    diagnostics = make_solver_diagnostics()
    diagnostics["_strict_domain_guards"] = False

    interfacial_area(
        rho_mass_l=-1.0,
        sigma=0.07,
        ul=0.002,
        A=0.32,
        packing=PACKING,
        diagnostics=diagnostics,
    )

    assert diagnostics["domain_guard_counts"]["hydraulics"] >= 1


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
            rho_mass_l=1000.0,
            rho_mass_v=1.2,
            mul_mix=1.0e-3,
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
    assert payload[-1] == pytest.approx(1.0)


def test_enhancement_factor_eta_psi_scales_driving_factor():
    diagnostics = make_solver_diagnostics()
    common = dict(
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
        E_type="explicit",
        diagnostics=diagnostics,
    )

    _, _, baseline, baseline_payload = enhancement_factor(**common, eta_psi=0.3)
    _, _, unscaled, unscaled_payload = enhancement_factor(**common, eta_psi=1.0)

    assert unscaled / baseline == pytest.approx(1.0 / 0.3)
    assert baseline_payload[-1] == pytest.approx(0.3)
    assert unscaled_payload[-1] == pytest.approx(1.0)


def test_enhancement_factor_rejects_nonpositive_eta_psi():
    diagnostics = make_solver_diagnostics()

    with pytest.raises(DomainGuardError):
        enhancement_factor(
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
            E_type="explicit",
            diagnostics=diagnostics,
            eta_psi=0.0,
        )

    assert diagnostics["domain_guard_counts"]["enhancement_factor"] == 1


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
