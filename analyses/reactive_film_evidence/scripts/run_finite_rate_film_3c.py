"""Run issue 16 checkpoint 2 on the retained Case 3C input, one solve at a time."""
import json
import sys
from pathlib import Path

import numpy as np

from mea_absorption_column.column import run_column


def main(output):
    retained = Path(__file__).resolve().parents[1] / "results/runs/engine100"
    rows, accepted = [], {}
    for nodes, quadrature in ((11, 5), (21, 9)):
        for basis in ("n2o_analogy", "epcsaft_free_co2"):
            directory = output / f"{basis}_{nodes}"
            request = json.loads((retained / "equilibrium_manifold_request.json").read_text())
            request["model"] = {"film_model": "finite_rate_film", "henry_basis": basis}
            request["numerics"]["solver_settings"].update(nodes=nodes, quadrature_points=quadrature)
            request["execution"].update(wall_limit_s=600, output_dir=str(directory))
            (output / f"{directory.name}_request.json").write_text(json.dumps(request, indent=2) + "\n")
            print(f"Starting {directory}", flush=True)
            attempt = run_column(request)
            result = attempt.get("result") or {}
            info = dict(basis=basis, nodes=nodes, quadrature_points=quadrature,
                        status=result.get("status", attempt["execution"]["status"]),
                        runtime_s=attempt["execution"].get("runtime_s"), iterations=result.get("iterations"),
                        numerical_acceptance=attempt["numerical_acceptance"],
                        physical_acceptance=attempt["physical_acceptance"],
                        scaled_residual_inf=result.get("scaled_residual_inf"),
                        scaled_invariant_drifts=(attempt.get("physical_certification") or {}).get("scaled_residual_inf"),
                        failure=(attempt["failure"] or result.get("failure") or {}))
            valid = attempt["numerical_acceptance"] == attempt["physical_acceptance"] == "accepted"
            if valid:
                accepted[basis, nodes] = attempt
                state = np.asarray(attempt["native_profile"]["state_matrix"])
                info["capture_percent"] = 100 * (1 - state[2, -1] / state[2, 0])
                assert len(attempt["diagnostics"]["finite_rate_film"]) == nodes
                if nodes == 21:
                    coarse = accepted[basis, 11]
                    previous = np.asarray(coarse["native_profile"]["state_matrix"])
                    np.testing.assert_allclose(result["grid"][::2], coarse["result"]["grid"], atol=1e-12, rtol=0)
                    scale = np.asarray(coarse["scaling"]["state_scale"])
                    info["C8_scaled_state_drift_inf"] = float(np.max(abs(state[:, ::2] - previous) / scale[:, None]))
                    info["C8_capture_change_points"] = info["capture_percent"] - 100 * (1 - previous[2, -1] / previous[2, 0])
            (directory / "run.log").write_text(json.dumps(info, indent=2) + "\n")
            rows.append(info)
            (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(info), flush=True)
            if not valid:
                raise SystemExit(f"3C attempt did not meet its numerical and physical checks: {directory}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
