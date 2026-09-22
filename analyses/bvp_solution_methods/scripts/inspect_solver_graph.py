"""Inspect native callback graphs and analytic IPOPT startup without evaluating physics."""
import json
import hashlib
from collections import Counter
from pathlib import Path

import casadi as ca

from mea_absorption_column.Thermodynamics.casadi_reactive import ReactiveLiquidCallback, FixedCompositionVaporCallback
from mea_absorption_column.Thermodynamics.reactive_bundle import ReactiveLiquid, load_reference_thermochemistry
from mea_absorption_column.Thermodynamics.thermo_models import MEA_THERMODYNAMICS_EPCSAFT_DATASET as dataset, ensure_epcsaft_importable
from mea_absorption_column.BVP.Coupled_Column import build_coupled_column_functions


def leaves(function):
    try:
        count = function.n_instructions()
    except RuntimeError:
        return Counter({function.name(): 1})
    result = Counter()
    for i in range(count):
        if function.instruction_id(i) == ca.OP_CALL:
            result.update(leaves(function.instruction_MX(i).which_function()))
    return result


def main():
    analysis = Path(__file__).resolve().parents[1]
    p = json.loads((analysis / "input/case_3c.json").read_text())["physical_inputs"]
    reference = load_reference_thermochemistry(dataset / "anchored-reference-thermochemistry.json")
    liquid = ReactiveLiquidCallback("inspect_liquid", ReactiveLiquid(dataset, thermochemistry=reference, **p["liquid_branch_policy"]))
    vapor_path = dataset.parent / "MEA_neutral_vapor"
    vapor = FixedCompositionVaporCallback("inspect_vapor", ensure_epcsaft_importable().Parameters.from_json(vapor_path / "parameters.json"),
        load_reference_thermochemistry(vapor_path / "reference-thermochemistry.json", liquid_reference=reference), packing_interval=(1e-6, .1))
    d = p["species_diffusivity_model"]
    diffusion = lambda t: ca.vertcat(d["co2_prefactor_m2_s"] * ca.exp(-d["co2_activation_j_mol"] / (d["gas_constant_j_mol_k"] * t)), *d["other_species_m2_s"])
    node, boundary, _ = build_coupled_column_functions(liquid, vapor, species_diffusivities=diffusion, quadrature_points=3,
        **{k: p[k] for k in ("liquid_feed_mol_s", "vapor_feed_mol_s", "liquid_temperature_k", "vapor_temperature_k", "bottom_pressure_pa", "area_m2", "packing")})
    x, parameter = ca.MX.sym("x", 12, 2), ca.MX.sym("p", 0)
    results = {}
    for label, inline in (("default", False), ("inline_cse_jacobian", True)):
        evaluated = [node.call([ca.MX(k * 6.), x[:, k]], inline, False) for k in range(2)]
        g = ca.vertcat(evaluated[1][0] - evaluated[0][0] - 3 * (evaluated[0][1] + evaluated[1][1]),
                      evaluated[0][2], evaluated[1][2], boundary(x[:, 0], x[:, 1]))
        options = {"print_time": False, "ipopt.print_level": 0, "ipopt.hessian_approximation": "limited-memory"}
        if inline:
            options["jac_g"] = ca.Function("inline_jac_g", [ca.vec(x), parameter], [g, ca.jacobian(g, ca.vec(x))], {"cse": True})
        solver = ca.nlpsol("inspect_" + label, "ipopt", {"x": ca.vec(x), "f": 0., "g": g}, options)
        results[label] = {name: dict(leaves(solver.get_function(name))) for name in ("nlp_g", "nlp_jac_g")}
    assert results["default"]["nlp_g"] == results["inline_cse_jacobian"]["nlp_g"]
    old, new = (results[k]["nlp_jac_g"] for k in ("default", "inline_cse_jacobian"))
    assert all(new[k] == v for k, v in old.items() if k.startswith("jac_"))
    assert sum(new.values()) < sum(old.values())
    startup = []
    scalar = ca.MX.sym("scalar")
    for scaling in ("gradient-based", "none"):
        solver = ca.nlpsol("startup", "ipopt", {"x": scalar, "f": 0., "g": scalar**2-1},
            {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.max_iter": 0,
             "ipopt.hessian_approximation": "limited-memory", "ipopt.nlp_scaling_method": scaling})
        solver(x0=.5, lbx=0., ubx=10., lbg=0., ubg=0.)
        stats = solver.stats()
        startup.append(dict(scaling=scaling, jacobian_calls=stats["n_call_nlp_jac_g"],
                            constraint_calls=stats["n_call_nlp_g"], iterations=stats["iter_count"], status=stats["return_status"]))
    assert [r["jacobian_calls"] for r in startup] == [2, 1]
    root = analysis.parents[1]
    sources = [Path(__file__).resolve(), analysis / "input/case_3c.json",
               root / "src/mea_absorption_column/BVP/Coupled_Column.py",
               root / "src/mea_absorption_column/Thermodynamics/casadi_reactive.py"]
    print(json.dumps(dict(scope="Static physical callback leaves and scalar analytic startup; no thermodynamic evaluation or column solve. No measured native speedup claim.",
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, graphs=results,
        startup_problem="x^2-1=0, x0=.5, bounds [0,10], constant zero objective, max_iter=0",
        startup=startup, ipopt_option_source="https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_nlp_scaling_method"), indent=2))


if __name__ == "__main__":
    main()
