"""One bounded corrected-3C calculation; each invocation retains its own attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import runpy
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analyses/bvp_solution_methods"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    """Preserve non-finite failure evidence as explicit strings, never zeros."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity():
    snapshot = json.loads((ANALYSIS / "results/candidate_snapshot.json").read_text())
    expected = {r["path"]: r["sha256"] for r in snapshot["files"]}
    adoption = json.loads((ANALYSIS / "results/runtime_adoption.json").read_text())
    expected.update(adoption["file_sha256"])
    evaluator = json.loads((ANALYSIS / "results/evaluator_adoption.json").read_text())
    for p, h in evaluator["previous_file_sha256"].items():
        if expected[p] != h:
            raise ValueError(f"Evaluator adoption does not follow frozen source: {p}")
    expected.update(evaluator["file_sha256"])
    precision = json.loads((ANALYSIS / "results/precision_runtime_adoption.json").read_text())
    for p, h in precision["previous_file_sha256"].items():
        if expected[p] != h:
            raise ValueError(f"Precision adoption does not follow frozen source: {p}")
    expected.update(precision["file_sha256"])
    shared = json.loads((ANALYSIS / "results/shared_evaluation_adoption.json").read_text())
    for p, h in shared["previous_file_sha256"].items():
        if p not in shared["newly_frozen_paths"] and expected[p] != h:
            raise ValueError(f"Shared evaluation adoption does not follow frozen source: {p}")
    expected.update(shared["file_sha256"])
    # Own numerical extension is recorded separately from frozen physical code.
    expected.pop("src/mea_absorption_column/BVP/Methods/Casadi_Collocation.py")
    changed = [p for p, h in expected.items() if digest(ROOT / p) != h]
    if changed:
        raise ValueError(f"Reconcile the frozen physical/runtime delivery before running: {changed}")
    return {p: h for p, h in expected.items() if p.startswith("src/")}


def verified_runtime(contract):
    integration = runpy.run_path(str(ROOT / "scripts/check_epcsaft_integration.py"))
    resolved = integration["resolve_epcsaft"](contract)
    identity = contract["final_identity"]
    if (resolved["source_kind"] != "local_file"
            or Path(resolved.get("wheel_path", "")).name != identity["wheel_filename"]
            or any(resolved.get(k) != identity[k] for k in ("wheel_sha256", "core_sha256"))
            or Path(resolved["module_path"]).parent != Path(resolved["core_path"]).parent):
        raise RuntimeError(f"Installed Engine differs from the declared immutable wheel: {resolved}")
    return resolved


def worker(args, record, path):
    import casadi as ca
    import numpy as np
    import scipy
    from scipy.optimize import brentq
    from compare import STATE_LAYOUT
    from mea_absorption_column.BVP.Coupled_Column import build_column_balance_functions, build_coupled_column_functions
    from mea_absorption_column.BVP.Methods.Casadi_Collocation import solve_conservative_collocation
    from mea_absorption_column.BVP.Methods.Conserved_Reduction import ConservedReduction, solve_reduced_bvp
    from mea_absorption_column.Thermodynamics.casadi_reactive import ReactiveLiquidCallback, FixedCompositionVaporCallback
    from mea_absorption_column.Thermodynamics.reactive_bundle import ReactiveLiquid, load_reference_thermochemistry
    from mea_absorption_column.Thermodynamics.thermo_models import MEA_THERMODYNAMICS_EPCSAFT_DATASET, ensure_epcsaft_importable

    started = time.perf_counter()
    case = json.loads((ANALYSIS / "input/case_3c.json").read_text())
    physical = case["physical_inputs"]
    physical_tolerance = case["physical_residual_tolerance"]
    model_hashes = source_identity()
    record["settings"]["runner_source_sha256"] = digest(Path(__file__))
    dataset = MEA_THERMODYNAMICS_EPCSAFT_DATASET
    contract = json.loads((ROOT / "integration/epcsaft_contract.json").read_text())
    identity = contract["final_identity"]
    record["problem"] = dict(physical_inputs=physical, state_layout=STATE_LAYOUT,
        coordinate="height_m_from_gas_inlet", model_source_sha256=model_hashes,
        engine_commit=identity["engine_commit"], wheel_sha256=identity["wheel_sha256"],
        parameters_sha256=digest(dataset / "parameters.json"),
        reference_sha256=digest(dataset / "anchored-reference-thermochemistry.json"))
    record["input_record"] = case
    record["resolved_runtime"] = verified_runtime(contract)
    record["runtime_identity_verified"] = True
    record["measurements"]["context"]["software"] = dict(python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, casadi=ca.__version__, engine=identity)
    methods = ROOT / "src/mea_absorption_column/BVP/Methods"
    method_path = methods / ("Casadi_Collocation.py" if args.method in ("central", "trapezoidal") else "Conserved_Reduction.py")
    record["settings"].update(method_source_sha256=digest(method_path), case_sha256=digest(ANALYSIS / "input/case_3c.json"))
    engine = ensure_epcsaft_importable()
    reference = load_reference_thermochemistry(dataset / "anchored-reference-thermochemistry.json")
    liquid = ReactiveLiquidCallback("comparison_liquid", ReactiveLiquid(dataset, thermochemistry=reference, **physical["liquid_branch_policy"]))
    vapor_data = dataset.parent / "MEA_neutral_vapor"
    vapor = FixedCompositionVaporCallback("comparison_vapor", engine.Parameters.from_json(vapor_data / "parameters.json"),
        load_reference_thermochemistry(vapor_data / "reference-thermochemistry.json", liquid_reference=reference),
        packing_interval=(1e-6, .1))
    np.testing.assert_array_equal(liquid.molar_masses[:3], case["liquid_molar_masses_kg_mol"])
    np.testing.assert_array_equal(vapor.molar_masses, case["vapor_molar_masses_kg_mol"])
    record["native_calls"] = {}

    def instrument(owner, name, label):
        function = getattr(owner, name)
        counts = record["native_calls"][label] = dict(started=0, returned=0, failed=0, wall_s=0.)

        def measured(*a, **kw):
            counts["started"] += 1
            tick = time.perf_counter()
            try:
                result = function(*a, **kw)
                counts["returned"] += 1
                return result
            except Exception:
                counts["failed"] += 1
                raise
            finally:
                counts["wall_s"] += time.perf_counter() - tick

        setattr(owner, name, measured)

    instrument(liquid.liquid, "solve", "liquid_value_A1")
    instrument(liquid.liquid, "solve_actions", "liquid_A2")
    instrument(vapor, "_state", "vapor_value_A1_H2")
    record["native_counts_scope"] = "Native calls through the two shared callbacks; cumulative through verification. Timeout counts are the last saved snapshot."
    if args.stage == "central":
        point = case["central_readiness_point"]
        record["stage"] = "central_A1_readiness"
        save(path, record)
        state = liquid.liquid.solve(*point[:2], point[2:], state_input_derivatives=True)
        derivative = np.asarray(state["state_input_derivatives"].jacobian)
        record["central_readiness"] = dict(inputs=point, accepted=bool(np.all(np.isfinite(derivative))),
            amounts_mol=state["amounts_mol"], density_mol_m3=state["density_mol_m3"],
            fugacities_pa=state["fugacities_pa"], total_enthalpy_j=state["total_enthalpy_j"],
            native_evidence=state["evidence"], A1_finite=bool(np.all(np.isfinite(derivative))),
            A2_checked=False, full_interface_checked=False)
        record.update(execution_status="completed", stage="central_A1_finished")
        save(path, record)
        return
    inputs = {k: physical[k] for k in ("liquid_feed_mol_s", "vapor_feed_mol_s", "liquid_temperature_k",
        "vapor_temperature_k", "bottom_pressure_pa", "area_m2", "packing")}
    fl, fv = (np.asarray(physical[k]) for k in ("liquid_feed_mol_s", "vapor_feed_mol_s"))

    def species_diffusivities(temperature):
        diffusion = physical["species_diffusivity_model"]
        co2 = diffusion["co2_prefactor_m2_s"] * ca.exp(-diffusion["co2_activation_j_mol"] /
            (diffusion["gas_constant_j_mol_k"] * temperature))
        return ca.vertcat(co2, *diffusion["other_species_m2_s"])

    balance, _ = build_column_balance_functions(liquid, vapor, **inputs)
    node, boundary, diagnostics = build_coupled_column_functions(liquid, vapor,
        species_diffusivities=species_diffusivities, quadrature_points=args.film_points,
        co2_model="reactive_film", film_thickness_multiplier=1., **inputs)
    bulk = np.asarray(case["initial_bulk_state"], dtype=float)
    bulk[7] -= float(balance(bulk, [0., 0., 0.])[2])
    native = balance(bulk, [0., 0., 0.])
    gas = np.asarray(native[4]).ravel()
    record.update(stage="interface_initialization", initialization_evaluations=[])
    save(path, record)

    def residual(loading):
        d = diagnostics(np.r_[bulk, 0., 0., 0., loading])
        value = float(d[1] / d[2] - d[3][0] * (gas[0] - d[7]))
        record["initialization_evaluations"].append(dict(loading=float(loading), original_co2_residual=value))
        save(path, record)
        return value

    initialization_ok = True
    if args.initial_record:
        imported = json.loads(args.initial_record.read_text())
        # A retained state is only a guess; reevaluate every original equation
        # under the current wheel and retain the source identity separately.
        record["initial_guess_engine_identity"] = imported["engine_identity"]
        for name in ("height_m", "area_m2", "packing"):
            if imported[name] != physical[name]:
                raise ValueError(f"Initial-state case mismatch: {name}")
        np.testing.assert_array_equal(imported["liquid_flows_mol_s"], fl)
        np.testing.assert_array_equal(imported["vapor_flows_mol_s"], fv)
        point = np.asarray(imported["initialization"]["state"], dtype=float)
        if point.shape != (12,) or np.any(~np.isfinite(point)):
            raise ValueError("Initial record needs a finite twelve-state interface candidate")
        np.testing.assert_array_equal(point[:7], bulk[:7])
        record["settings"]["initial_record_sha256"] = digest(args.initial_record)
        record["imported_initialization"] = imported["initialization"]
        d = diagnostics(point)
    else:
        loading, root = brentq(residual, *case["initial_interface_bracket"], full_output=True, disp=False)
        initialization_ok = root.converged
        d = diagnostics(np.r_[bulk, 0., 0., 0., loading])
        point = np.r_[bulk, float(d[1] / d[2]), float(d[3][1] * (gas[1] - native[3][20])), 0., loading]
        point[10] = float(d[4]) * (bulk[5] - bulk[4]) + point[8:10] @ np.asarray(d[6]).ravel()
    height, span = physical["height_m"], 393.15 - 293.15
    capacity = np.array([fl.sum() * liquid.input_jacobian(np.r_[bulk[4], bulk[6], fl])[-1, 0],
                         fv.sum() * vapor.input_jacobian(np.r_[bulk[5], bulk[6], fv])[-1, 0]])
    if np.any(~np.isfinite(capacity)) or np.any(capacity <= 0):
        raise RuntimeError("Native capacity scales must be finite positive")
    flux_scale = np.asarray(d[3]).ravel() * gas[:2]
    heat_scale = float(d[4]) * span + np.abs(np.asarray(d[6]).ravel()) @ flux_scale
    su = np.r_[bulk[:4], 300., 300., bulk[6], .97, flux_scale, heat_scale, 1.]
    sb = np.r_[bulk[:4] / height, capacity * span / height, bulk[6] / height]
    sa = np.r_[.97, flux_scale[0] * float(d[2]), flux_scale, heat_scale]
    sc = np.r_[bulk[:4], span, span, bulk[6]]
    margin = .97 * np.finfo(float).eps
    lower = np.r_[[0.] * 4, 293.15, 293.15, 1., margin, [-np.inf] * 4]
    upper = np.r_[[np.inf] * 4, 393.15, 393.15, 1e7, .97-margin, [np.inf] * 4]
    algebraic = np.asarray(node(0., point)[2]).ravel()
    record["initialization"] = dict(state=point, algebraic_residual=algebraic,
        accepted=bool(initialization_ok and np.max(abs(algebraic/sa)) <= physical_tolerance))
    record["settings"].update(state_scale=su, balance_scale=sb, algebraic_scale=sa, boundary_scale=sc,
        lower=lower, upper=upper, initial_state=point, initialization="supplied interface state, original equations rechecked" if args.initial_record else "same feed-based bracketed full-film equations")
    save(path, record)
    if not record["initialization"]["accepted"]:
        raise RuntimeError("Full original algebraic initialization check failed")
    if args.stage == "initialize":
        record.update(execution_status="completed", stage="consistent_initialization_finished")
        save(path, record)
        return
    def reduced_model():
        model = ConservedReduction(node, boundary, initial_state=point, lower=lower, upper=upper,
            state_scale=su, conserved_scale=sb*height, algebraic_scale=sa, boundary_scale=sc,
            tolerance=args.local_tolerance, solver_tolerance=args.local_solver_tolerance,
            max_evaluations=args.local_evaluations, max_condition=args.max_condition)
        original = model.evaluate
        saved_count = 0

        def observed(z, w):
            nonlocal saved_count
            try:
                return original(z, w)
            finally:
                if model.counts["local_solves"] != saved_count:
                    record.update(last_local=model.last_local, local_counts=model.counts.copy())
                    save(path, record)
                    saved_count = model.counts["local_solves"]

        model.evaluate = observed
        return model

    if args.stage == "reduction":
        record["stage"] = "consistent_native_reduction"
        record["derivative_claim_limit"] = "Conditioning and local equation recovery do not certify native A2 accuracy; the coupled owner supplies the independent node-Jacobian verification."
        save(path, record)
        model = reduced_model()
        try:
            q = np.asarray(node(0., point)[0]).ravel()
            record["local_reduction"] = model.evaluate(0., q/model.conserved_scale)
            record["local_reduction"]["accepted"] = True
        finally:
            record.update(last_local=model.last_local, local_counts=model.counts)
            save(path, record)
        record.update(execution_status="completed", stage="consistent_native_reduction_finished")
        save(path, record)
        return
    grid = np.linspace(0., height, args.nodes)
    initial = np.tile(point[:, None], (1, args.nodes))
    if args.initial_profile:
        supplied = json.loads(args.initial_profile.read_text())
        if supplied["physical_inputs"] != physical:
            raise ValueError("Initial-profile physical inputs differ")
        old_grid, profile = np.asarray(supplied["grid"]), np.asarray(supplied["profile"])
        if (old_grid.ndim != 1 or len(old_grid) < 2 or profile.shape != (12, len(old_grid))
                or np.any(~np.isfinite(old_grid)) or np.any(np.diff(old_grid) <= 0)
                or old_grid[0] != 0 or old_grid[-1] != height or np.any(~np.isfinite(profile))):
            raise ValueError("Initial profile needs a finite twelve-state profile over the physical height")
        initial = np.array([np.interp(grid, old_grid, row) for row in profile])
        if np.any(initial < lower[:, None]) or np.any(initial > upper[:, None]):
            raise ValueError("Initial profile violates original physical bounds")
        record["settings"].update(initial_profile_sha256=digest(args.initial_profile),
            initialization="Retained profile interpolated as an outer numerical guess; full original equations apply")
        record["supplied_initial_profile"] = supplied
    record["initial_profile"] = initial
    record["measurements"]["setup_wall_s"] = time.perf_counter() - started
    record.update(stage="global_solve", iteration_observations=[])
    save(path, record)
    solve_started = time.perf_counter()
    if args.method in ("trapezoidal", "central"):
        class Observer(ca.Callback):
            def __init__(self):
                super().__init__()
                self.construct("comparison_iterations", {"enable_fd": False})

            def get_n_in(self): return ca.nlpsol_n_out()
            def get_n_out(self): return 1
            def get_name_in(self, i): return ca.nlpsol_out(i)
            def get_sparsity_in(self, i):
                name = self.get_name_in(i)
                return ca.Sparsity.dense(0 if name == "lam_p" else 1 if name == "f" else 12*args.nodes, 1)
            def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)
            def eval(self, values):
                record["iteration_observations"].append(dict(
                    elapsed_s=time.perf_counter()-solve_started,
                    profile=np.asarray(values[0]).reshape((12, args.nodes), order="F") * su[:, None],
                    scaled_residual_inf=float(np.max(abs(np.asarray(values[2]))))))
                save(path, record)
                return [0]

        observer = Observer()
        result = solve_conservative_collocation(node, boundary, grid, initial, lower, upper,
            state_scale=su, balance_scale=sb, algebraic_scale=sa, boundary_scale=sc,
            tolerance=args.tolerance, max_iterations=args.max_iterations, iteration_callback=observer,
            scheme=args.method, boundary_slots=[(0,-1),(1,-1),(2,0),(3,0),(4,-1),(5,0),(6,0)] if args.method == "central" else None)
    else:
        model = reduced_model()
        result = solve_reduced_bvp(model, grid, initial, method=args.method, tolerance=args.tolerance,
            boundary_tolerance=args.tolerance, max_nodes=args.max_nodes, max_evaluations=args.max_iterations,
            ivp_rtol=args.ivp_rtol, ivp_atol=args.ivp_atol)
    record["measurements"]["global_wall_s"] = time.perf_counter() - solve_started
    record.update(result=result, stage="physical_verification")
    save(path, record)
    if result["profile"] is not None:
        profiles = result["profile"]
        evaluated = [tuple(np.asarray(v).ravel() for v in node(z, u)) for z, u in zip(result["grid"], profiles.T)]
        conserved = np.column_stack([v[0] for v in evaluated])
        algebraic = np.column_stack([v[2] for v in evaluated])
        invariants = conserved[[2, 3, 5]] - conserved[[0, 1, 4]]
        drift = invariants - invariants[:, :1]
        charges = []
        for u in profiles.T:
            for fraction in np.linspace(0., 1., args.film_points):
                amounts = np.array([u[0]*np.exp(fraction*u[11]), fl[1], u[1]])
                state = liquid.liquid.solve(u[4], u[6], amounts, state_input_derivatives=False)
                charges.append(np.asarray(state["amounts_mol"]) @ np.array([0,0,0,1,-1,-1,-2,1,-1]))
        residuals = dict(material=drift[:2], energy=drift[2], charge=charges,
                         interface=algebraic, boundary=np.asarray(boundary(profiles[:, 0], profiles[:, -1])).ravel())
        scaled = dict(material=float(np.max(abs(drift[:2]/sc[:2, None]))),
            energy=float(np.max(abs(drift[2]/(sb[4]*height)))), charge=float(np.max(abs(np.asarray(charges)))),
            interface=float(np.max(abs(algebraic/sa[:, None]))),
            boundary=float(np.max(abs(residuals["boundary"]/sc))))
        physical_ok = bool(all(np.isfinite(v) and v <= physical_tolerance for v in scaled.values())
            and np.all(np.isfinite(profiles)) and np.all(profiles >= lower[:, None])
            and np.all(profiles <= upper[:, None]) and 0 <= profiles[2,-1] <= profiles[2,0])
        record["physical_certification"] = dict(accepted=physical_ok, original_residuals=residuals,
            scaled_residual_inf=scaled, criteria={k: physical_tolerance for k in scaled},
            scope="Native grid material/energy invariants, bulk/film-quadrature charge, interface, boundary and bounds; between-node accuracy still requires refinement",
            residual_units=dict(material="mol/s", energy="W", charge="elementary charge mol per apparent feed mol", interface="holdup fraction; mol/m/s; mol/m2/s; mol/m2/s; W/m2", boundary="four mol/s; two K; Pa"),
            conserved=conserved, sources=np.column_stack([v[1] for v in evaluated]))
    record.update(stage="finished", execution_status="completed", termination=result["status"], failure=result.get("failure"))
    save(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("central", "initialize", "reduction", "column"), default="column")
    parser.add_argument("--method", choices=["trapezoidal", "central", "shooting", "collocation"], required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--film-points", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-record", type=Path, help="Retained owner interface state; matching case and original equations are rechecked")
    parser.add_argument("--initial-profile", type=Path, help="Retained outer profile guess with matching physical_inputs, grid and twelve-state profile")
    parser.add_argument("--wall-limit", type=float, default=600.)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=100)
    parser.add_argument("--local-tolerance", type=float, default=1e-9)
    parser.add_argument("--local-solver-tolerance", type=float, default=1e-11)
    parser.add_argument("--local-evaluations", type=int, default=30)
    parser.add_argument("--max-condition", type=float, default=1e12)
    parser.add_argument("--ivp-rtol", type=float, default=1e-7)
    parser.add_argument("--ivp-atol", type=float, default=1e-9)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.initial_record:
        args.initial_record = args.initial_record.resolve()
    if args.initial_profile:
        args.initial_profile = args.initial_profile.resolve()
    if args.nodes < (3 if args.method == "central" else 2) or args.film_points < 2 or args.wall_limit <= 0:
        parser.error("Invalid grid, quadrature or wall limit")
    path = args.output / "run.json"
    if args.worker:
        record = json.loads(path.read_text())
        try:
            worker(args, record, path)
        except Exception as error:
            record.update(execution_status="failed", failure=str(error), failure_type=type(error).__name__)
            save(path, record)
            raise
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), platform.machine())
    settings = {k: v for k, v in vars(args).items() if k not in ("worker", "output", "wall_limit", "method")}
    record = dict(run_id=args.output.name, method=args.method, settings=settings, execution_status="not_run",
        stage="starting", result=None, physical_certification={"accepted": None}, failure=None, limit_seconds=args.wall_limit,
        measurements={"context": {"hardware": {"platform": platform.platform(), "cpu_model": cpu_model, "logical_cpus": os.cpu_count()},
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "software": None, "scope": "One fresh worker including import/setup/initialization/solve/verification; global_wall_s includes iteration checkpoint I/O"}})
    save(path, record)
    started = time.perf_counter()
    child_args = ["--worker", "--output", str(args.output)]
    if args.initial_record:
        child_args += ["--initial-record", str(args.initial_record)]
    if args.initial_profile:
        child_args += ["--initial-profile", str(args.initial_profile)]
    with (args.output / "process.txt").open("w") as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], *child_args],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.wall_limit)
            status = None
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            status = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = process.returncode
    record = json.loads(path.read_text())
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record["measurements"].update(wall_s=time.perf_counter()-started, cpu_s=usage.ru_utime+usage.ru_stime,
                                  peak_rss_bytes=usage.ru_maxrss*1024)
    if status:
        record.update(execution_status=status, termination=status, failure=f"Worker {status} at stage {record['stage']}")
    elif code != 0:
        record.update(execution_status="failed", termination=f"worker_exit_{code}")
    save(path, record)
    print(json.dumps({k: record.get(k) for k in ("run_id", "execution_status", "stage", "failure")}))
    accepted = ((record.get("central_readiness") or {}).get("accepted") if args.stage == "central"
        else (record.get("initialization") or {}).get("accepted") if args.stage == "initialize"
        else (record.get("local_reduction") or {}).get("accepted") if args.stage == "reduction"
        else record["physical_certification"].get("accepted") and (record.get("result") or {}).get("accepted"))
    if record["execution_status"] != "completed" or not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
