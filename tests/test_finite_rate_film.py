"""Issue 16 C1–C4 and native node Jacobian on frozen Case 3C states."""
import json
from pathlib import Path

import casadi as ca
import numpy as np
import pytest
from scipy.optimize import brentq

from mea_absorption_column.column import _build_conserved_assembly, _prepare_conserved_column_in_process
from mea_absorption_column.config.column import resolve_column_config


@pytest.mark.parametrize("basis", ["n2o_analogy", "epcsaft_free_co2"])
def test_finite_rate_limits_and_node_jacobian(basis):
    directory = Path("analyses/reactive_film_evidence/results/runs")
    request = json.loads((directory / "engine100/enhancement_reference_request.json").read_text())
    request["model"] = {"film_model": "finite_rate_film", "henry_basis": basis}
    request["execution"]["output_dir"] = ".tmp_local/finite_rate_film_test"
    config = resolve_column_config(request)
    prepared = _prepare_conserved_column_in_process(config)
    normal = prepared["assembly"]
    slow, fast = (_build_conserved_assembly(config, prepared["resolved_inputs"], finite_rate_multiplier=m)
                  for m in (1e-12, 1e16))

    def flux(assembly, state, fg, bulk_gas, bracket, manifold=False):
        diag = assembly["diagnostics"]
        def rows(loading):
            d = diag(np.r_[state[:8], 0., state[9:11], loading])
            gas_r, kinetic_r, fraction = map(float, (d[10], d[11], d[13]))
            n = (fg - float(d[7])) / (gas_r if manifold else gas_r + kinetic_r)
            return float(d[1]) / float(d[2]) - (1. if manifold else fraction) * n, n
        loading = brentq(lambda xi: rows(xi)[0], *bracket, xtol=5e-15)
        point = np.r_[state[:8], rows(loading)[1], state[9:11], loading]
        if not manifold:
            d = diag(point)
            _, _, residual = assembly["node"](0., point)
            assert abs(float(residual[1]) / float(d[2])) < 1e-12
            assert abs(float(residual[2]) + (bulk_gas - fg) / float(d[10])) < 1e-12
        return point[8]

    retained = directory / "finite_rate_film_cp1_2026-11-23_rf2_corrected/nodes.jsonl"
    for index, line in enumerate(retained.read_text().splitlines()):
        row = json.loads(line)
        state = np.array(row["state"])
        fg = float(normal["balance"](state[:8], state[8:11])[4][0])
        fb = row["f_bulk_Pa"]
        if index == 0:
            zero = flux(normal, state, fb, fg, (-.05, .05))
            reversals = [flux(normal, state, fb * f, fg, (-.05, .05)) for f in (.9, 1.1)]
            d = normal["diagnostics"](np.r_[state[:8], 0., state[9:11], 0.])
            kg = float(d[3][0])
            assert abs(zero) <= 1e-8 * kg * fb and np.isnan(float(d[12]))
            assert reversals[0] < 0 < reversals[1]
            evidence = directory / "finite_rate_film_cp1_2026-11-23_rf2_corrected/C6_C7.jsonl"
            basis_name = {"n2o_analogy": "N2O analogy", "epcsaft_free_co2": "ePC-SAFT free CO2"}[basis]
            retained = next(r for line in evidence.read_text().splitlines() if (r := json.loads(line))
                            and r["film"] == row["film"] and r["z_m"] == row["z_m"]
                            and r["basis"] == basis_name and r["sweep"] == "column")
            physical = (fg - fb) / (1. / kg + retained["H_Pa_m3_mol"] / retained["fine"]["k_L_m_s"])
            slow_error = flux(slow, state, fg, fg, (0., .6)) / physical - 1.
            print(basis, "C1 flux/limit", zero, 1e-8 * kg * fb, "C2 fluxes", reversals, "C3 error", slow_error)
            assert abs(slow_error) <= 1e-6
            x = ca.MX.sym("state", 12)
            node = normal["node"]
            jac = ca.Function("finite_rate_node_jacobian", [x], [ca.jacobian(ca.vertcat(*node.call([ca.MX(0.), x], True, False)), x)])
            direction = np.array([.1, 1., .2, .1, 1., 1., 1e4, .01, 1e-4, 0., 0., .05])
            plus, minus = (np.asarray(ca.vertcat(*node(0., state + sign * 1e-3 * direction))).ravel()
                           for sign in (1, -1))
            difference = (plus - minus) / 2e-3
            exact = np.asarray(jac(state)) @ direction
            jac_error = np.max(abs(exact - difference) / np.maximum(1e-12, np.maximum(abs(exact), abs(difference))))
            print(basis, "node Jacobian relative error", jac_error)
            assert jac_error < 1e-5
        baseline = flux(fast, state, fg, fg, (0., .6), manifold=True)
        reactive = flux(fast, state, fg, fg, (0., .6))
        relative = abs(reactive / baseline - 1.)
        print(basis, "C4", row["film"], row["z_m"], baseline, reactive, relative, "margin", 1e-6 - relative, flush=True)
        assert relative <= 1e-6
