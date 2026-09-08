"""One native derivative-connection check; not a local-root or column solve."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time

import numpy as np

from mea_absorption_column.BVP.Coupled_Column import build_coupled_column_functions
from mea_absorption_column.BVP.Methods.Conserved_Reduction import ConservedReduction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[3]
    snapshot = json.loads((root/"analyses/bvp_solution_methods/results/candidate_snapshot.json").read_text())
    expected = {r["path"]: r["sha256"] for r in snapshot["files"]}
    adoption_path = root/"analyses/bvp_solution_methods/results/runtime_adoption.json"
    adoption = json.loads(adoption_path.read_text())
    for path, digest in adoption["previous_file_sha256"].items():
        if expected[path] != digest:
            raise ValueError(f"Runtime adoption does not match original snapshot: {path}")
    expected.update(adoption["file_sha256"])
    evaluator_path = root/"analyses/bvp_solution_methods/results/evaluator_adoption.json"
    evaluator = json.loads(evaluator_path.read_text())
    expected.update(evaluator["file_sha256"])
    precision_path = root/"analyses/bvp_solution_methods/results/precision_runtime_adoption.json"
    precision = json.loads(precision_path.read_text())
    for path, digest in precision["previous_file_sha256"].items():
        if expected[path] != digest:
            raise ValueError(f"Precision adoption does not follow frozen source: {path}")
    expected.update(precision["file_sha256"])
    shared_path = root/"analyses/bvp_solution_methods/results/shared_evaluation_adoption.json"
    shared = json.loads(shared_path.read_text())
    for path, digest in shared["previous_file_sha256"].items():
        if path not in shared["newly_frozen_paths"] and expected[path] != digest:
            raise ValueError(f"Shared evaluation adoption does not follow frozen source: {path}")
    expected.update(shared["file_sha256"])
    source_paths = [*expected, str(adoption_path.relative_to(root)), str(evaluator_path.relative_to(root)), str(precision_path.relative_to(root)), str(shared_path.relative_to(root)), "src/mea_absorption_column/BVP/Methods/Conserved_Reduction.py",
                    str(Path(__file__).resolve().relative_to(root))]
    current = {path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in source_paths}
    # The centered scheme is this task's declared extension to the imported solver.
    modified = "src/mea_absorption_column/BVP/Methods/Casadi_Collocation.py"
    mismatches = [path for path, digest in expected.items() if path != modified and current[path] != digest]
    if mismatches:
        raise ValueError(f"Frozen candidate source mismatch: {mismatches}")
    test_path = "tests/test_coupled_column.py"
    start = time.perf_counter()
    report = dict(loading_anchor=None, scope_limit="Historical verification case without the corrected3C loading path; use run_case.py --stage reduction for that path", source_sha256=current, frozen_candidate_verified=True,
                  runtime_adoption=adoption, declared_candidate_extensions=[modified], accepted_column=False, local_root_attempted=False, failure=None,
                  scope="Known-valid thermodynamic point, not an algebraically consistent interface or published 3C",
                  engine_identity=json.loads((root/"integration/epcsaft_contract.json").read_text())["final_identity"])
    try:
        # Reuse the exact candidate's verification case and transport estimates.
        case = runpy.run_path(str(root/test_path))
        liquid, vapor, _, _, bulk, fl, fv = case["_column"]()
        node, boundary, _ = build_coupled_column_functions(liquid, vapor,
            species_diffusivities=case["_species_diffusivities"], quadrature_points=3,
            liquid_feed_mol_s=fl, vapor_feed_mol_s=fv, liquid_temperature_k=313.15,
            vapor_temperature_k=353.15, bottom_pressure_pa=101325., area_m2=.32,
            packing=[250., .97, .203, .35, .017, .292, .119])
        point = np.r_[bulk, 1e-4, 0., 0., .05]
        su = np.array([.2, 8., 3., 2., 300., 300., 1e5, .1, .001, .001, 1000., 1.])
        sq = np.array([.2, 8., 3., 2., 100., 100., 1e5])
        sa = np.array([.1, 1e-7, .001, .001, 1000.])
        model = ConservedReduction(node, boundary, initial_state=point,
            lower=[0., 0., 0., 0., 293.15, 293.15, 1., .97*np.finfo(float).eps, -np.inf, -np.inf, -np.inf, -np.inf],
            upper=[np.inf, np.inf, np.inf, np.inf, 393.15, 393.15, 1e7, .97*(1-np.finfo(float).eps), np.inf, np.inf, np.inf, np.inf],
            state_scale=su, conserved_scale=sq, algebraic_scale=sa, boundary_scale=[.2, 8., 3., 2., 40., 40., 100.],
            tolerance=1e-7, solver_tolerance=1e-10, max_evaluations=1, max_condition=1e12)
        bu, ru, au = (np.asarray(v) for v in model.jacobian(0., point))
        b, r, a = (np.asarray(v).ravel() for v in node(0., point))
        matrix = np.vstack((bu/sq[:, None], au/sa[:, None]))*su
        rhs = np.vstack((np.eye(7), np.zeros((5, 7))))
        action = np.linalg.solve(matrix, rhs)
        defect = float(np.max(abs(matrix@action-rhs)))
        report.update(state=point.tolist(), conserved=b.tolist(), sources=r.tolist(),
                      original_algebraic_residual=a.tolist(), state_scale=su.tolist(),
                      conserved_scale=sq.tolist(), algebraic_scale=sa.tolist(),
                      scaled_matrix_condition=float(np.linalg.cond(matrix)),
                      implicit_linear_solve_residual_inf=defect,
                      node_jacobian_shape=[bu.shape[0]+ru.shape[0]+au.shape[0], bu.shape[1]],
                      derivatives_finite=bool(all(np.all(np.isfinite(v)) for v in (bu, ru, au, action))),
                      algebraically_consistent=bool(np.max(abs(a/sa)) <= model.tolerance))
        assert report["derivatives_finite"] and defect < 1e-9
    except Exception as error:
        report.update(failure=str(error), failure_type=type(error).__name__)
        raise
    finally:
        report["diagnostic_wall_s"] = time.perf_counter()-start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("derivatives_finite", "scaled_matrix_condition",
                     "implicit_linear_solve_residual_inf", "algebraically_consistent", "diagnostic_wall_s")}))


if __name__ == "__main__":
    main()
