from numpy import log, exp
from ..config.Constants import R, g
from .domain_guards import require_positive


def liquid_mass_transfer_expression(diffusivity, viscosity, density, velocity, packing):
    """Liquid coefficient [m/s]; positive physical inputs, numeric or CasADi."""
    a_p, eps, Clp, *_ = packing
    return (Clp * (g * density / viscosity) ** (1 / 6)
            * (a_p / (4 * eps)) ** .5 * (velocity / a_p) ** (1 / 3)
            * diffusivity ** .5)


def gas_mass_transfer_expression(diffusivity, viscosity, density, velocity, holdup, temperature, packing):
    """Gas coefficient [mol/(m² s Pa)]; positive numeric or CasADi inputs."""
    a_p, eps, _, Cvp, *_ = packing
    return (Cvp / (R * temperature) * (a_p ** 2 / (4 * eps * holdup)) ** .5
            * diffusivity ** (2 / 3) * (viscosity / density) ** (1 / 3)
            * (velocity * density / (a_p * viscosity)) ** .75)


def heat_transfer_expression(pressure, gas_coefficient, conductivity, heat_capacity, molar_density, diffusivity):
    """Heat coefficient [W/(m² K)]; positive numeric or CasADi inputs."""
    return exp(log(gas_coefficient) + log(pressure) + (
        2 * log(conductivity) + log(heat_capacity)
        - 2 * log(molar_density) - 2 * log(diffusivity)
    ) / 3)


def mass_transfer_coeff(
    h_L,
    h_V,
    rho_mass_l,
    rho_mass_v,
    mul_mix,
    muv_mix,
    Dl_CO2,
    Dv_CO2,
    Dv_H2O,
    Dv_T,
    A,
    Tv,
    ul,
    uv,
    packing,
    diagnostics=None,
):

    a_p, ϵ, Clp, Cvp, Cs, Cp_0, Ch = packing
    require_positive(
        "mass_transfer",
        diagnostics,
        h_L=h_L,
        h_V=h_V,
        rho_mass_l=rho_mass_l,
        rho_mass_v=rho_mass_v,
        mul_mix=mul_mix,
        muv_mix=muv_mix,
        Dl_CO2=Dl_CO2,
        Dv_CO2=Dv_CO2,
        Dv_H2O=Dv_H2O,
        Dv_T=Dv_T,
        A=A,
        Tv=Tv,
        ul=ul,
        uv=uv,
        a_p=a_p,
        eps=ϵ,
        Clp=Clp,
        Cvp=Cvp,
    )

    d_h = 4 * ϵ / a_p
    Lp = A * a_p / ϵ

    def f_kv(Dv):
        return gas_mass_transfer_expression(Dv, muv_mix, rho_mass_v, uv, h_V, Tv, packing)

    kl_CO2 = liquid_mass_transfer_expression(Dl_CO2, mul_mix, rho_mass_l, ul, packing)
    kv_CO2 = f_kv(Dv_CO2)
    kv_H2O = f_kv(Dv_H2O)
    kv_T = f_kv(Dv_T) * (R * Tv)
    require_positive("mass_transfer", diagnostics, kl_CO2=kl_CO2, kv_CO2=kv_CO2, kv_H2O=kv_H2O, kv_T=kv_T)

    return kl_CO2, kv_CO2, kv_H2O, kv_T, [Clp, Cvp, ϵ, a_p, A, Lp, d_h]


def heat_transfer_coeff(P, kv_CO2, kt_vap, Cpv_T, rho_mol_v, Dv_CO2, a_eA, diagnostics=None):
    require_positive(
        "heat_transfer",
        diagnostics,
        P=P,
        kv_CO2=kv_CO2,
        kt_vap=kt_vap,
        Cpv_T=Cpv_T,
        rho_mol_v=rho_mol_v,
        Dv_CO2=Dv_CO2,
        a_eA=a_eA,
    )

    UT = heat_transfer_expression(P, kv_CO2, kt_vap, Cpv_T, rho_mol_v, Dv_CO2)
    require_positive("heat_transfer", diagnostics, UT=UT)
    return UT
