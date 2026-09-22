"""Analyze a retained native [B; R; a] Jacobian at the verified interface state."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jacobian_record", type=Path)
    parser.add_argument("state_record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    jacobian_bytes, state_bytes = args.jacobian_record.read_bytes(), args.state_record.read_bytes()
    source, state = json.loads(jacobian_bytes), json.loads(state_bytes)
    assert source["engine_identity"] == state["measurements"]["context"]["software"]["engine"]
    np.testing.assert_array_equal(source["point"], state["initialization"]["state"])
    assert state["initialization"]["accepted"]
    jacobian = np.asarray(source["jacobian"])
    assert jacobian.shape == (19, 12) and np.all(np.isfinite(jacobian))
    scales = state["settings"]
    su = np.asarray(scales["state_scale"])
    sq = np.asarray(scales["balance_scale"])*state["problem"]["physical_inputs"]["height_m"]
    sa = np.asarray(scales["algebraic_scale"])
    matrix = np.vstack((jacobian[:7]/sq[:, None], jacobian[14:]/sa[:, None]))*su
    rhs = np.vstack((np.eye(7), np.zeros((5, 7))))
    action = np.linalg.solve(matrix, rhs)
    state_action = su[:, None]*action
    report = dict(scope="Scaled local conserved-state chart at one algebraically consistent3C interface; no column or method ranking",
        source_record=dict(path=str(args.jacobian_record), sha256=hashlib.sha256(jacobian_bytes).hexdigest()),
        state_record=dict(path=str(args.state_record), sha256=hashlib.sha256(state_bytes).hexdigest()),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        engine_identity=source["engine_identity"], point=source["point"],
        source_derivative_check_passed=source["passed"],
        source_directional_max_error=source.get("max_error"),
        source_directional_errors=source.get("errors"),
        derivative_claim_limit="Conditioning does not establish A2 accuracy; source verification may still be in progress",
        row_order="B[0:7], R[7:14], a[14:19]", original_node_jacobian=jacobian.tolist(),
        state_scale=su.tolist(), conserved_scale=sq.tolist(), algebraic_scale=sa.tolist(),
        scaled_matrix=matrix.tolist(), singular_values=np.linalg.svd(matrix, compute_uv=False).tolist(),
        scaled_condition=float(np.linalg.cond(matrix)), linear_action_residual_inf=float(np.max(abs(matrix@action-rhs))),
        state_w_jacobian=state_action.tolist(), rhs_w_jacobian=((jacobian[7:14]@state_action)/sq[:, None]).tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("scaled_condition", "linear_action_residual_inf", "source_derivative_check_passed")}))


if __name__ == "__main__":
    main()
