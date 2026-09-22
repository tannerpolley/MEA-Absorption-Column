from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config.Constants import MWs_l
from .Thermophysical_Properties import density, enthalpy, heat_capacity, henrys_law, surface_tension
from .Transport_Properties import diffusivity, viscosity


@dataclass(frozen=True)
class AmineProperties:
    """Physical-property inputs for an apparent CO2/amine/H2O liquid phase.

    Correlations receive SI temperatures and pressures and the apparent liquid
    mole fractions in ``(CO2, amine, H2O)`` order. Their return shapes retain
    the column's existing property contracts.
    """

    amine_id: str
    amine_molar_mass_kg_per_mol: float
    henry_co2_correlation: Callable
    density_correlation: Callable
    surface_tension_correlation: Callable
    heat_capacity_correlation: Callable
    enthalpy_correlation: Callable
    viscosity_correlation: Callable
    diffusivity_correlation: Callable
    enthalpy_temperature_derivative_correlation: Callable
    chemical_equilibrium_correlation: Callable | None = None
    enhancement_factor_correlation: Callable | None = None
    missing_column_inputs: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.amine_id.strip():
            raise ValueError("amine_id must not be empty")
        if not np.isfinite(self.amine_molar_mass_kg_per_mol) or self.amine_molar_mass_kg_per_mol <= 0:
            raise ValueError("amine_molar_mass_kg_per_mol must be finite and positive")
        for field_name in (
            "henry_co2_correlation",
            "density_correlation",
            "surface_tension_correlation",
            "heat_capacity_correlation",
            "enthalpy_correlation",
            "viscosity_correlation",
            "diffusivity_correlation",
            "enthalpy_temperature_derivative_correlation",
        ):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"{field_name} must be callable")
        for field_name in ("chemical_equilibrium_correlation", "enhancement_factor_correlation"):
            correlation = getattr(self, field_name)
            if correlation is not None and not callable(correlation):
                raise TypeError(f"{field_name} must be callable or None")

    @property
    def liquid_molar_masses_kg_per_mol(self):
        return np.array([MWs_l[0], self.amine_molar_mass_kg_per_mol, MWs_l[2]])

    def mass_fractions(self, liquid_mole_fractions):
        x = np.asarray(liquid_mole_fractions, dtype=float)
        if x.shape != (3,):
            raise ValueError("liquid composition must contain CO2, amine, and H2O")
        masses = x * self.liquid_molar_masses_kg_per_mol
        return masses / masses.sum()

    def henry_co2(self, temperature_K, liquid_mole_fractions):
        return self.henry_co2_correlation(temperature_K, liquid_mole_fractions)

    def density(self, temperature_K, liquid_mole_fractions, pressure_Pa):
        return self.density_correlation(temperature_K, liquid_mole_fractions, pressure_Pa, phase="liquid")

    def surface_tension(
        self, temperature_K, liquid_mole_fractions, amine_mass_fraction, water_mass_fraction
    ):
        return self.surface_tension_correlation(
            temperature_K, liquid_mole_fractions, amine_mass_fraction, water_mass_fraction
        )

    def heat_capacity(self, temperature_K, liquid_mole_fractions):
        return self.heat_capacity_correlation(temperature_K, liquid_mole_fractions, phase="liquid")

    def enthalpy(self, temperature_K, liquid_mole_fractions):
        return self.enthalpy_correlation(temperature_K, liquid_mole_fractions, phase="liquid")

    def viscosity(
        self, temperature_K, liquid_mole_fractions, amine_mass_fraction, water_mass_fraction
    ):
        return self.viscosity_correlation(
            temperature_K,
            liquid_mole_fractions,
            amine_mass_fraction,
            water_mass_fraction,
            phase="liquid",
        )

    def diffusivity(
        self,
        temperature_K,
        liquid_mole_fractions,
        pressure_Pa,
        viscosity_Pa_s,
        molar_density_mol_m3,
    ):
        return self.diffusivity_correlation(
            temperature_K,
            liquid_mole_fractions,
            pressure_Pa,
            viscosity_Pa_s,
            molar_density_mol_m3,
            phase="liquid",
        )

    def enthalpy_temperature_derivative(self, temperature_K, liquid_mole_fractions):
        return self.enthalpy_temperature_derivative_correlation(temperature_K, liquid_mole_fractions)

    def chemical_equilibrium(self, flows, temperature_K, **kwargs):
        return self.chemical_equilibrium_correlation(flows, temperature_K, **kwargs)

    def enhancement_factor(self, *args, **kwargs):
        return self.enhancement_factor_correlation(*args, **kwargs)

    def require_column_ready(self):
        missing = list(self.missing_column_inputs)
        if self.chemical_equilibrium_correlation is None:
            missing.append("a solvent-specific chemical-equilibrium callable")
        if self.enhancement_factor_correlation is None:
            missing.append("a solvent-specific enhancement-factor callable")
        if missing:
            raise RuntimeError(
                f"{self.amine_id} column model is incomplete; missing required inputs: "
                + "; ".join(missing)
            )


def _mea_enthalpy_temperature_derivative(temperature_K, liquid_mole_fractions):
    from ..misc.special_functions import f_dHl_dT

    return f_dHl_dT(temperature_K, liquid_mole_fractions)


def _missing_mdea_henry_co2(temperature_K, liquid_mole_fractions):
    raise RuntimeError("the aqueous-MDEA CO2 fugacity/Henry closure has not been admitted")


def _mdea_density(temperature_K, liquid_mole_fractions, pressure_Pa, phase="liquid"):
    """Weiland et al. (1998), Eqs. 1--4 and Table 6."""
    T = float(np.asarray(temperature_K).reshape(-1)[0])
    x_co2, x_mdea, x_water = np.asarray(liquid_mole_fractions, dtype=float)
    a, b, c = -4.86099e-7, -4.24935e-4, 1.20528
    v_mdea = 119.17 / (a * T**2 + b * T + c)
    rho_water = -3.2484e-6 * T**2 + 0.00165 * T + 0.793
    v_water = 18.01528 / rho_water
    v_co2 = -2.8558
    v = (
        x_mdea * v_mdea
        + x_water * v_water
        + x_co2 * v_co2
        - 6.65 * x_mdea * x_water
        + x_mdea * x_co2 * (12.983 + 397.72 * x_mdea)
    )
    v_m3_per_mol = v * 1.0e-6
    molar_mass = float(
        np.dot((x_co2, x_mdea, x_water), (MWs_l[0], 0.119163, MWs_l[2]))
    )
    return 1.0 / v_m3_per_mol, molar_mass / v_m3_per_mol, [
        v_m3_per_mol,
        v_co2 * 1.0e-6,
        v_mdea * 1.0e-6,
        v_water * 1.0e-6,
    ]


def _mdea_surface_tension(
    temperature_K, liquid_mole_fractions, amine_mass_fraction, water_mass_fraction
):
    """Fu et al. (2013) pure-component endpoints, ideal mass mixing."""
    T = float(temperature_K)
    solvent_amine_fraction = amine_mass_fraction / max(
        amine_mass_fraction + water_mass_fraction, 1.0e-30
    )
    sigma_mdea = (38.6 - 0.08 * (T - 298.2)) * 1.0e-3
    sigma_water = (72.0 - 0.15 * (T - 298.2)) * 1.0e-3
    return (1.0 - solvent_amine_fraction) * sigma_water + solvent_amine_fraction * sigma_mdea


_MDEA_CP_MASS_FRACTIONS = np.array([0.30, 0.40, 0.50, 0.60])
_MDEA_CP_UNLOADED_J_KG_K = np.array([3787.0, 3585.0, 3407.0, 3174.0])
_MDEA_CP_LOADING_SLOPES_J_KG_K = np.array([-702.0, -690.0, -626.0, -635.0])


def _mdea_cp_mass(temperature_K, liquid_mole_fractions):
    x_co2, x_mdea, x_water = np.asarray(liquid_mole_fractions, dtype=float)
    solvent_mass = x_mdea * 0.119163 + x_water * MWs_l[2]
    w_mdea = x_mdea * 0.119163 / max(solvent_mass, 1.0e-30)
    loading = x_co2 / max(x_mdea, 1.0e-30)
    cp0 = np.interp(w_mdea, _MDEA_CP_MASS_FRACTIONS, _MDEA_CP_UNLOADED_J_KG_K)
    slope = np.interp(w_mdea, _MDEA_CP_MASS_FRACTIONS, _MDEA_CP_LOADING_SLOPES_J_KG_K)
    return float(max(cp0 + slope * np.clip(loading, 0.0, 0.5), 1.0))


def _mdea_heat_capacity(temperature_K, liquid_mole_fractions, phase="liquid"):
    x = np.asarray(liquid_mole_fractions, dtype=float)
    mixture_molar_mass = float(np.dot(x, (MWs_l[0], 0.119163, MWs_l[2])))
    cp_molar = _mdea_cp_mass(temperature_K, x) * mixture_molar_mass
    return np.full(3, cp_molar), cp_molar


def _mdea_enthalpy(temperature_K, liquid_mole_fractions, phase="liquid"):
    T = float(np.asarray(temperature_K).reshape(-1)[0])
    x = np.asarray(liquid_mole_fractions, dtype=float)
    cp_molar = _mdea_heat_capacity(T, x)[1]
    solvent_mass = x[1] * 0.119163 + x[2] * MWs_l[2]
    w_mdea_percent = 100.0 * x[1] * 0.119163 / max(solvent_mass, 1.0e-30)
    # Merkley et al. (1987), Eq. 1; valid below saturation for 20--60 wt%.
    heat_absorption = 1000.0 * (-0.101 * w_mdea_percent - 0.126 * T - 8.60)
    sensible = cp_molar * (T - 298.15)
    values = np.array([heat_absorption + sensible, sensible, sensible])
    return values, float(np.dot(x, values))


def _mdea_viscosity(
    temperature_K, liquid_mole_fractions, amine_mass_fraction, water_mass_fraction, phase="liquid"
):
    """Weiland et al. (1998), Eq. 5 and Table 7."""
    T = float(temperature_K)
    x = np.asarray(liquid_mole_fractions, dtype=float)
    omega = 100.0 * amine_mass_fraction / max(
        amine_mass_fraction + water_mass_fraction, 1.0e-30
    )
    loading = x[0] / max(x[1], 1.0e-30)
    water = 1.002e-3 * 10 ** (
        1.3272 * (293.15 - T - 0.001053 * (T - 293.15) ** 2) / (T - 168.15)
    )
    a, b, c, d, e, f, g = -0.1944, 0.4315, 80.684, 2889.1, 0.0106, 0.0, -0.2141
    exponent = omega * ((a * omega + b) * T + c * omega + d) * (
        loading * (e * omega + f * T + g) + 1.0
    ) / T**2
    return float(water * np.exp(np.clip(exponent, -50.0, 50.0))), float(water)


def _mdea_diffusivity(
    temperature_K,
    liquid_mole_fractions,
    pressure_Pa,
    viscosity_Pa_s,
    molar_density_mol_m3,
    phase="liquid",
):
    T = float(temperature_K)
    x = np.asarray(liquid_mole_fractions, dtype=float)
    c_mdea = float(molar_density_mol_m3) * x[1]
    water_viscosity = _mdea_viscosity(T, x, 0.0, 1.0)[0]
    d_co2_water = 2.35e-6 * np.exp(-2119.0 / T)
    d_co2 = d_co2_water * (water_viscosity / max(float(viscosity_Pa_s), 1.0e-30)) ** 0.8
    # Snijder et al. (1993), Eq. 12; 298--348 K and 8--4010 mol/m3.
    d_mdea = np.exp(-13.088 - 2360.7 / T - 24.727e-5 * c_mdea)
    d_ion = np.exp(-22.64 - 1000.0 / T - 0.7 * np.log(max(viscosity_Pa_s, 1.0e-30)))
    return float(d_co2), float(d_mdea), float(d_ion)


def _mdea_enthalpy_temperature_derivative(temperature_K, liquid_mole_fractions):
    x = np.asarray(liquid_mole_fractions, dtype=float)
    step = 1.0e-3
    return (
        _mdea_enthalpy(float(temperature_K) + step, x)[1]
        - _mdea_enthalpy(float(temperature_K) - step, x)[1]
    ) / (2.0 * step)


def _chemical_equilibrium(
    flows,
    temperature_K,
    *,
    amine_id,
    model,
    pressure_Pa,
    diagnostics,
    liquid_molar_density,
):
    from ..Thermodynamics.Chemical_Equilibrium import chemical_equilibrium_with_model

    return chemical_equilibrium_with_model(
        flows,
        temperature_K,
        model=model,
        P=pressure_Pa,
        diagnostics=diagnostics,
        amine_id=amine_id,
        liquid_molar_density=liquid_molar_density,
    )


def _enhancement_factor(*args, amine_id, **kwargs):
    from ..Transport.Enhancement_Factor import enhancement_factor

    return enhancement_factor(*args, amine_id=amine_id, **kwargs)


def _mea_chemical_equilibrium(*args, **kwargs):
    return _chemical_equilibrium(*args, amine_id="MEA", **kwargs)


def _mdea_chemical_equilibrium(*args, **kwargs):
    return _chemical_equilibrium(*args, amine_id="MDEA", **kwargs)


def _mea_enhancement_factor(*args, **kwargs):
    return _enhancement_factor(*args, amine_id="MEA", **kwargs)


def _mdea_enhancement_factor(*args, **kwargs):
    return _enhancement_factor(*args, amine_id="MDEA", **kwargs)


MEA_PROPERTIES = AmineProperties(
    amine_id="MEA",
    amine_molar_mass_kg_per_mol=float(MWs_l[1]),
    henry_co2_correlation=henrys_law,
    density_correlation=density,
    surface_tension_correlation=surface_tension,
    heat_capacity_correlation=heat_capacity,
    enthalpy_correlation=enthalpy,
    viscosity_correlation=viscosity,
    diffusivity_correlation=diffusivity,
    enthalpy_temperature_derivative_correlation=_mea_enthalpy_temperature_derivative,
    chemical_equilibrium_correlation=_mea_chemical_equilibrium,
    enhancement_factor_correlation=_mea_enhancement_factor,
)


MDEA_PROPERTIES = AmineProperties(
    amine_id="MDEA",
    amine_molar_mass_kg_per_mol=0.119163,
    henry_co2_correlation=_missing_mdea_henry_co2,
    density_correlation=_mdea_density,
    surface_tension_correlation=_mdea_surface_tension,
    heat_capacity_correlation=_mdea_heat_capacity,
    enthalpy_correlation=_mdea_enthalpy,
    viscosity_correlation=_mdea_viscosity,
    diffusivity_correlation=_mdea_diffusivity,
    enthalpy_temperature_derivative_correlation=_mdea_enthalpy_temperature_derivative,
    chemical_equilibrium_correlation=_mdea_chemical_equilibrium,
    enhancement_factor_correlation=_mdea_enhancement_factor,
    missing_column_inputs=(
        "an admitted reactive ePC-SAFT MDEA parameter artifact",
        "an aqueous-MDEA CO2 fugacity/Henry closure",
        "standard-state reaction inputs on the adopted activity basis",
        "an admitted packed-column kinetic/enhancement correlation",
        "mapped MDEA column feed and validation cases",
    ),
)


def resolve_amine_properties(properties=None, *, require_column_ready=False):
    if properties is None:
        properties = MEA_PROPERTIES
    elif not isinstance(properties, AmineProperties):
        raise TypeError("amine_properties must be an AmineProperties instance")
    if require_column_ready:
        properties.require_column_ready()
    return properties
