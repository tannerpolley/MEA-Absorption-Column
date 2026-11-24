"""Issue 16 checkpoint 1 at fixed interfacial fugacity. Run from repository root."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_bvp
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PPoly
from scipy.optimize import brentq

from mea_absorption_column.Properties.Thermophysical_Properties import henrys_law
from mea_absorption_column.Properties.Transport_Properties import diffusivity, viscosity
from mea_absorption_column.Thermodynamics.casadi_reactive import build_loading_path_function
from mea_absorption_column.Transport.Enhancement_Factor import (
    CO2_CONCENTRATION_DIVISOR, explicit_enhancement_expression, hatta_expression,
)
from mea_absorption_column.Transport.Reactive_Film import (
    binary_diffusivities_from_species, constrained_onsager_mobility,
)
from mea_absorption_column.column import _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config

SOURCE = Path("analyses/reactive_film_evidence/results/runs/engine100")
FILMS = ("enhancement_reference", "equilibrium_manifold")
REFINEMENTS = (2e-6, 2e-8)
SWEEP = (3., 10., 35., 75., 165., 350., 700.)
REFERENCE_LIMIT = 2e-4  # 1/100 of the owner-set flux-error limit
C6_LIMIT = 2e-7  # ten times the refined collocation tolerance
MESH = np.r_[0., np.geomspace(1e-5, 1., 100)]


def record(output, table, **row):
    with (output / f"{table}.jsonl").open("a") as stream:
        print(json.dumps(row, allow_nan=False), file=stream)


def constant_film(output):
    # D = delta = m = 1, G = 2. Independent free and bound concentration BVP.
    rate, q = 2.25, np.linspace(0., 1., 33)
    def rhs(x, y):
        reaction = rate * (y[0] - y[2])
        return np.vstack((y[1], reaction, y[3], -reaction))
    def bc(a, b):
        return a[0] - 1, a[3], b[0], b[2]
    sol = solve_bvp(rhs, bc, MESH, np.zeros((4, MESH.size)), tol=2e-9, max_nodes=20000)
    assert sol.success, sol.message
    table = np.column_stack((1 + q, 1 / (1 + q), np.full_like(q, 2.), np.full_like(q, rate)))
    film = compare(q, table, 2e-9, 1., 1., 1., 2.)
    error = film["reduced_flux"] / -sol.y[1, 0] - 1
    record(output, "C5", independent_flux=-sol.y[1, 0], rf2_error=error, limit=1e-5, **film)
    assert "failure" not in film and abs(error) <= 1e-5, f"C5 failed: {film}, RF-2 error {error:g}"


def gaspar(output, z, temperature, c, d, f_bulk, f_interface):
    # Gaspar and Fosbøl (2015), Eqs. 10, 12, 13, 17 and 22 with m=n=1.
    co2, mea, water, meah, carbamate = c[:5]
    da, db, di = d
    ci = co2 * f_interface / f_bulk
    drive = ci - co2
    depletion = db * mea / (2 * da)
    def equilibrium(y_mea):
        product_gain = da * depletion * (1 - y_mea) / di
        return co2 * (1 + product_gain / meah) * (1 + product_gain / carbamate) / y_mea**2
    y_infinity = brentq(lambda y: equilibrium(y) - ci, 1e-6, 1.)
    enhancement = lambda y: 1 + depletion * (1 - y) / drive
    q = np.linspace(0., 1 - y_infinity, 33)
    product_gain = da * depletion * q / di
    tangent = 2 / (1 - q) + da * depletion / di * (1 / (meah + product_gain) + 1 / (carbamate + product_gain))
    equilibrium_co2 = equilibrium(1 - q)
    _, ha_unit_kl = hatta_expression(temperature, mea * (1 - q), water, da, 1.)
    table = np.column_stack((equilibrium_co2, tangent, da * (depletion + equilibrium_co2 * tangent), ha_unit_kl**2 / da))
    for ha in (35., 75., 110., 165.):
        y = brentq(lambda y: enhancement(y) - ha * np.sqrt(y) * (ci - equilibrium(y)) / drive,
                   y_infinity, 1.)
        explicit = explicit_enhancement_expression(ha, db, mea, di, meah, di, carbamate,
                                                   da, co2 / CO2_CONCENTRATION_DIVISOR)
        ideal_film = compare(q, table, REFINEMENTS[1], 1., da, ha_unit_kl[0] / ha, ci)
        record(output, "gaspar", film=FILMS[0], z_m=z, Ha=ha,
               Ha_over_Einf=ha / enhancement(y_infinity), E_general=enhancement(y),
               E_explicit=explicit, explicit_error=explicit / enhancement(y) - 1,
               E_RF2=ideal_film["reduced_flux"] / (ideal_film["k_L_m_s"] * drive), ideal_film=ideal_film)


def compare(q, table, tolerance, henry, d, kl, f_i, target=None, sweep=None, rate_scale=1.):
    f, tangent, g, k = table.T
    c_star = CubicHermiteSpline(q, (f - f[0]) / henry, f * tangent / henry)
    conductance, k1 = CubicSpline(q, g), CubicSpline(q, k)
    m = c_star.derivative()
    bound = PPoly(conductance.c - d * np.vstack((np.zeros(q.size - 1), m.c)), q)
    extrema = bound.derivative().roots(extrapolate=False)
    minimum = float(np.min(bound(np.r_[q, extrema[np.isfinite(extrema)]])))
    if minimum <= 0:
        raise ValueError(f"G - D m = {minimum:.9g} mol/(m s) on the tabulated interval")
    delta, drive = d / kl, (f_i - f[0]) / henry
    upper, integral = q[-1], conductance.antiderivative()
    def rf2(xi):
        ha = delta * np.sqrt(rate_scale * k1(xi) / d * conductance(xi) / bound(xi))
        return (drive - c_star(xi)) / drive * ha / np.tanh(ha), ha
    if target:
        theta = np.tanh(target) / target
        xi = brentq(lambda a: (drive - c_star(a)) / theta - drive + c_star(a)
                    - (integral(a) - integral(0)) / d, 0., upper)
        ratio = target / rf2(xi)[1]
        delta, rate_scale = (delta * ratio, 1.) if sweep == "k_L" else (delta, ratio**2)
    xi = brentq(lambda a: rf2(a)[0] - 1 + c_star(a) / drive
                - (integral(a) - integral(0)) / (d * drive), 0., upper)
    reduced, ha = rf2(xi)
    # x/delta, xi/upper, Delta/drive, and J_b and N scaled by D drive/delta.
    def rhs(x, y):
        loading, excess, jb, total = y
        loading_prime = -d * drive * jb / bound(upper * loading)
        return np.vstack((loading_prime / upper, jb - total - m(upper * loading) * loading_prime / drive,
                          delta**2 / d * rate_scale * k1(upper * loading) * excess, np.zeros_like(x)))
    def bc(a, b):
        return c_star(upper * a[0]) / drive + a[1] - 1, a[2], b[0], b[1]
    layer = np.exp(-ha * MESH)
    guess = np.vstack((xi / upper * (1 - MESH), (1 - c_star(xi) / drive) * layer,
                       (reduced - 1) * (1 - layer), np.full_like(MESH, reduced)))
    sol = solve_bvp(rhs, bc, MESH, guess, tol=tolerance, max_nodes=20000)
    values = dict(Ha=ha, Ha_over_Einf=ha * d * drive / (integral(upper) - integral(0)),
                  k_L_m_s=d / delta, rate_multiplier=rate_scale, G_minus_Dm_min=minimum,
                  nodes=sol.x.size, max_rms_residual=float(np.max(sol.rms_residuals)),
                  reduced_flux=reduced * d * drive / delta)
    if not sol.success:
        return dict(values, failure=sol.message)
    loading_profile = PPoly(sol.sol.c[:, :, 0], sol.x)
    extrema = loading_profile.derivative().roots(extrapolate=False)
    path = loading_profile(np.r_[sol.x, extrema[np.isfinite(extrema)]])
    if path.min() < -1e-10 or path.max() > 1 + 1e-10:
        return dict(values, failure="Resolved loading leaves the tabulated interval")
    xi_i, excess_i, _, flux = sol.y[:, 0] * [upper, 1., 1., 1.]
    independent_I, _ = quad(conductance, 0., xi_i, points=q[(q > 0) & (q < xi_i)])
    c6 = (flux - excess_i - independent_I / (d * drive)) / flux
    return dict(values, reference_flux=flux * d * drive / delta, C6_error=c6, resolved_loading=xi_i)


def main(output):
    constant_film(output)
    for film in FILMS:
        request_path, attempt_path = SOURCE / f"{film}_request.json", SOURCE / film / "attempt.json"
        request, attempt = [json.loads(p.read_text()) for p in (request_path, attempt_path)]
        request["execution"]["output_dir"] = str(output)
        prepared = _prepare_conserved_column_in_process(resolve_column_config(request))
        a, feed = prepared["assembly"], prepared["resolved_inputs"]["parameters"][0]
        liquid = a["liquid"]
        path = build_loading_path_function(liquid)
        record(output, "inputs", film=film, source_hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (request_path, attempt_path)},
            engine=prepared["engine"], assets=prepared["assets"], species=liquid.component_ids,
            refinements=REFINEMENTS, sweep_targets=SWEEP)
        for z, state in zip(attempt["result"]["grid"], np.array(attempt["native_profile"]["state_matrix"]).T, strict=True):
            T, P = state[[4, 6]]
            flow = np.asarray(liquid.flow(state[0], state[1], feed)).ravel()
            inputs = np.r_[T, P, flow]
            l, _, h = [np.asarray(x).ravel() for x in a["balance"](state[:8], state[8:11])[3:]]
            diagnostics = a["diagnostics"](state)
            thickness, f_i = float(diagnostics[2]), float(diagnostics[7])
            x = flow / flow.sum()
            mass = flow * np.asarray(liquid.molar_masses)[:3]
            mu = viscosity(T, x, mass[1] / mass.sum(), mass[2] / mass.sum())[0]
            d = diffusivity(T, x, P, mu, flow.sum() / h[0])
            ds = np.r_[d[:2], a["diffusion_model"]["other_species_m2_s"][1], np.full(6, d[2])]
            pairs = binary_diffusivities_from_species(ds)
            co2, rho = liquid.output_index["fugacity:carbon-dioxide"], liquid.output_index["molar-density"]
            c = l[rho] * l[:9] / l[:9].sum()
            if film == FILMS[0]:
                gaspar(output, z, T, c, d, l[co2], f_i)
            record(output, "nodes", film=film, z_m=z, state=state.tolist(), f_i_Pa=f_i,
                   f_bulk_Pa=l[co2], d_m2_s=ds.tolist(), kl_m_s=d[0] / thickness)
            q = np.linspace(0., state[11], 33)
            table = np.empty((q.size, 4))
            for i, xi in enumerate(q):
                values, tangent = [np.asarray(y).ravel() for y in path(inputs, xi)]
                composition = values[:9] / values[:9].sum()
                mobility = constrained_onsager_mobility(composition, values[rho], pairs,
                                                       additional_flux_constraints=liquid.film_constraints)
                _, ha = hatta_expression(T, *(values[rho] * composition[1:3]), d[0], 1.)
                table[i] = values[co2], tangent[0], liquid.components["co2"] @ mobility @ tangent, ha**2 / d[0]
            record(output, "manifold", film=film, z_m=z, q=q.tolist(), f_t_G_k1=table.tolist())
            bases = (("N2O analogy", henrys_law(T, x)), ("ePC-SAFT free CO2", l[co2] / c[0]))
            for sweep, target in [("column", None), *((s, t) for s in ("k_L", "rate") for t in SWEEP)]:
                for basis, henry in bases:
                    print(f"{film} z={z:g} {basis} {sweep} {target} H={henry:g}", flush=True)
                    fine = compare(q, table, REFINEMENTS[1], henry, d[0], d[0] / thickness, f_i, target, sweep)
                    coarse = compare(q[::2], table[::2], REFINEMENTS[0], henry, d[0],
                                     fine["k_L_m_s"], f_i, rate_scale=fine["rate_multiplier"])
                    error, uncertainty = None, None
                    if "failure" not in coarse and "failure" not in fine:
                        uncertainty = abs(fine["reference_flux"] / coarse["reference_flux"] - 1)
                        if uncertainty <= REFERENCE_LIMIT and abs(fine["C6_error"]) <= C6_LIMIT:
                            error = fine["reduced_flux"] / fine["reference_flux"] - 1
                    record(output, "C6_C7", film=film, z_m=z, basis=basis, H_Pa_m3_mol=henry,
                           sweep=sweep, target=target, coarse=coarse, fine=fine,
                           reference_uncertainty=uncertainty, C7_error=error)
                    if error is not None and abs(error) >= .02:
                        raise SystemExit("C7 reached or exceeded 2 %.")

if __name__ == "__main__":
    main(Path(sys.argv[1]))
