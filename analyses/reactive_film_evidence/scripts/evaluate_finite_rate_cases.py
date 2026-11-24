"""Check the resolved film at column nodes and reconstruct Dugas's wetted-wall comparison."""
import csv
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from check_finite_rate_film import (
    C6_LIMIT, REFINEMENTS, REFERENCE_LIMIT, binary_diffusivities_from_species,
    build_loading_path_function, compare, constrained_onsager_mobility, diffusivity,
    hatta_expression, henrys_law, record, viscosity,
)
from run_finite_rate_cases import FILMS, bridge_request, read
from mea_absorption_column.column import _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config
INPUTS = Path(__file__).resolve().parents[1] / "inputs"

def prepared(case, basis="n2o"):
    request = read(bridge_request(case)) | {"model": FILMS[basis]}
    return _prepare_conserved_column_in_process(resolve_column_config(request))

def frozen(assembly, path, temperature, pressure, flow, density):
    liquid = assembly["liquid"]
    mass = flow * np.asarray(liquid.molar_masses)[:3]
    x = flow / flow.sum()
    mu = viscosity(temperature, x, *(mass[1:] / mass.sum()))[0]
    d, mea, ion = diffusivity(temperature, x, pressure, mu, density)
    water = assembly["diffusion_model"]["water"]["constant_m2_s"]
    pairs = binary_diffusivities_from_species(np.r_[d, mea, water, np.full(6, ion)])
    bulk = np.r_[temperature, pressure, flow]
    co2, rho = (liquid.output_index[k] for k in ("fugacity:carbon-dioxide", "molar-density"))

    @lru_cache(maxsize=None)
    def at(xi):
        values, tangent = [np.asarray(v).ravel() for v in path(bulk, xi)]
        concentration = values[:9] / values[:9].sum() * values[rho]
        mobility = constrained_onsager_mobility(concentration / values[rho], values[rho], pairs,
                                               additional_flux_constraints=liquid.film_constraints)
        _, ha = hatta_expression(temperature, *concentration[1:3], d, 1.)
        return values[co2], tangent[0], liquid.components["co2"] @ mobility @ tangent, ha**2 / d

    values = np.asarray(liquid(bulk)).ravel()
    return at, d, mu, dict(n2o=henrys_law(temperature, x),
                           epcsaft=values[co2] * values[:9].sum() / (values[rho] * values[0]))

def c7(output, case, attempt, basis):
    assembly = prepared(case, basis)["assembly"]
    liquid = assembly["liquid"]
    path = build_loading_path_function(liquid)
    feed = attempt["resolved_inputs"]["parameters"][0]
    profile = attempt["native_profile"]
    for z, state in zip(profile["grid"], np.asarray(profile["state_matrix"]).T, strict=True):
        flow = np.asarray(liquid.flow(state[0], state[1], feed)).ravel()
        volume = float(assembly["balance"](state[:8], state[8:11])[5][0])
        at, d, _, henry = frozen(assembly, path, state[4], state[6], flow, flow.sum() / volume)
        diagnostics = assembly["diagnostics"](state)
        f_i, kl = float(diagnostics[assembly["diagnostics"].index_out("interface_fugacity")]), d / float(diagnostics[2])
        fb, xi = at(0.)[0], state[11]
        sign = np.sign(f_i - fb)
        if sign == 0:
            record(output, "C7", case=case, profile_basis=basis, z_m=z, status="zero drive; relative error undefined")
            continue
        upper = 1.01 * abs(xi)
        while sign * (at(sign * upper)[0] - f_i) <= 0.:
            upper *= 1.25
        upper = brentq(lambda q: at(sign * q)[0] - f_i, 0., upper)
        q = np.linspace(0., upper, 33)
        table = np.array([at(sign * point) for point in q])
        fugacity = table[:, 0].copy()
        table[:, 0] = fb + sign * (fugacity - fb)
        table[:, 1] *= fugacity / table[:, 0]
        for check_basis, h in henry.items():
            fine = compare(q, table, REFINEMENTS[1], h, d, kl, table[-1, 0])
            coarse = compare(q[::2], table[::2], REFINEMENTS[0], h, d, kl, table[-1, 0])
            uncertainty = error = None
            if "failure" not in fine and "failure" not in coarse:
                uncertainty = abs(fine["reference_flux"] / coarse["reference_flux"] - 1)
                if uncertainty <= REFERENCE_LIMIT and abs(fine["C6_error"]) <= C6_LIMIT:
                    error = fine["reduced_flux"] / fine["reference_flux"] - 1
            record(output, "C7", case=case, profile_basis=basis, basis=check_basis, z_m=z,
                   f_i_Pa=f_i, f_b_Pa=fb, H_Pa_m3_mol=h, direction=int(sign),
                   fine=fine, coarse=coarse, reference_uncertainty=uncertainty, C7_error=error)
            if error is not None and abs(error) >= .02:
                raise SystemExit(f"{case} {basis} z={z:g} m: C7 error {error:.3%}")

def wetted_wall_flux(at, h, d, kl, kg, drive):
    nodes, weights = np.polynomial.legendre.leggauss(9)
    fb = at(0.)[0]
    delta = d / kl

    def interface(xi):
        f, tangent, g, rate = at(xi)
        bound = g - d * f * tangent / h
        if bound <= 0. or rate <= 0.:
            raise ValueError(f"RF-2 inadmissible at xi={xi:g}: G-Dm={bound:g}, k1={rate:g}")
        ha = delta * np.sqrt(rate / d * g / bound)
        theta = np.tanh(ha) / ha
        conductance = [at(q)[2] for q in .5 * xi * (1 + nodes)]
        flux = .5 * xi * (weights @ conductance) / (delta * (1 - theta))
        fi = f + flux * theta * h / kl
        return flux, fi

    def mismatch(xi):
        flux, fi = interface(xi)
        return fi - fb + flux / kg - drive
    edge = np.copysign(.025, drive)
    while mismatch(0.) * mismatch(edge) > 0.:
        edge *= 1.5
    flux, fi = interface(brentq(mismatch, *sorted((0., edge))))
    if fi <= 0. or (drive and (flux * drive <= 0. or (fi - fb) / flux <= 0.)):
        raise ValueError(f"Inadmissible flux or liquid resistance at drive={drive:g} Pa, fi={fi:g} Pa")
    return dict(flux=flux, interface_fugacity_pa=fi)

def dugas(output):
    assembly = prepared("3C")["assembly"]
    liquid = assembly["liquid"]
    path = build_loading_path_function(liquid)
    settings = list(csv.DictReader((INPUTS / "dugas2009_7m_mea_settings.csv").read_text().splitlines()))
    rows = [row for row in csv.DictReader((INPUTS / "dugas2011_table1_mea.csv").read_text().splitlines())
            if row["mea_molal"] == "7"]
    height, perimeter, area, atm = .091, np.pi * .0126, .003852, 101325.
    for row in rows:
        t_c, alpha, p_obs, observed, kl_obs = (float(row[key]) for key in
            ("T_C", "co2_loading_mol_per_mol_mea", "pco2_star_Pa", "kg_prime_mol_per_Pa_m2", "k_l0_m_per_s"))
        source = [s for s in settings if float(s["loading"]) == alpha and float(s["T_C"]) == t_c]
        assert len(source) == 6 and float(source[0]["equilibrium_Pa"]) == p_obs
        a, b = np.array([[float(s[f"{side}_wet_Pa"]) - p_obs for s in source]
                         for side in ("inlet", "outlet")])
        drive = np.array([np.nan if x*y <= 0. else x if x == y else (x - y) / np.log(x / y)
                          for x, y in zip(a, b, strict=True)])
        temperature, pressure = t_c + 273.15, atm + float(source[0]["pressure_psig"]) * 6894.757293168
        kg, q = float(source[0]["kg"]), float(source[0]["liquid_ml_s"]) * 1e-6
        feed = np.r_[7 * alpha, 7., 1. / liquid.molar_masses[2]]
        bulk = np.asarray(liquid(np.r_[temperature, pressure, feed])).ravel()
        volume = bulk[:9].sum() / bulk[liquid.output_index["molar-density"]]
        at, d, mu, henry = frozen(assembly, path, temperature, pressure, feed, feed.sum() / volume)
        rho = (feed @ np.array(liquid.molar_masses[:3])) / volume
        kl = 3**(1/3) * np.sqrt(2/np.pi) * q**(1/3) * np.sqrt(height*d) * perimeter**(2/3) / area * (9.80665*rho/mu)**(1/6)
        concentration = feed[1] / volume
        flags = dict(Snijder_T=not 297.9 <= temperature <= 333., Snijder_C=not 43 <= concentration <= 5016,
                     Luo_T=temperature > 343., Luo_loading=alpha > .4, R5_T=temperature > 323.15)
        for basis, h in henry.items():
            for thickness_case, coefficient in (("Eq. 8", kl), ("Table 1 -10%", .9*kl_obs), ("Table 1 +10%", 1.1*kl_obs)):
                trials = []
                for force in np.r_[0., drive]:
                    try:
                        if np.isnan(force): raise ValueError("Wet endpoint drives straddle measured equilibrium")
                        trials.append(wetted_wall_flux(at, h, d, coefficient, kg, force))
                    except (RuntimeError, ValueError) as exc:
                        trials.append({"failure": str(exc)})
                predicted = None
                if all("flux" in trial for trial in trials):
                    flux = np.array([trial["flux"] for trial in trials])
                    overall = drive @ flux[1:] / (drive @ drive)
                    if flux[0] == 0. and np.min(flux) < 0. < np.max(flux) and 0 < overall < kg:
                        predicted = 1 / (1 / overall - 1 / kg)
                record(output, "dugas", row=row["row_id"], T_K=temperature, alpha=alpha,
                       source_table=source[0]["table"], source_page=source[0]["page"], basis=basis, thickness_case=thickness_case,
                       kg_prime_predicted=predicted, kg_prime_observed=observed,
                       signed_error_percent=None if predicted is None else 100*(predicted/observed - 1),
                       drives_pa=[float(v) if np.isfinite(v) else None for v in drive], trials=trials, admissible=predicted is not None,
                       pressure_pa=pressure, atmosphere_pa=atm, height_m=height, perimeter_m=perimeter, area_m2=area,
                       flow_m3_s=q, density_kg_m3=rho, viscosity_Pa_s=mu, D_m2_s=d, kl_m_s=coefficient,
                       delta_m=d/coefficient, kg=kg, H_Pa_m3_mol=h, bulk_fugacity_pa=at(0.)[0], observed_P_star_pa=p_obs,
                       bulk_pressure_proxy_error_pa=at(0.)[0]-p_obs, C_MEA_mol_m3=concentration, domain_flags=flags)

if __name__ == "__main__":
    output = Path(sys.argv[1])
    if sys.argv[2] == "dugas": dugas(output)
    else:
        case, basis, attempt = sys.argv[2:]
        c7(output, case, read(Path(attempt)), basis)
