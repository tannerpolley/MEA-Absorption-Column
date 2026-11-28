"""Fit-A twelve-state capture by film and mesh against Morgan et al. (2020) and the article's earlier-set summary. Run from the repository root.
Capture = 1 - outlet/inlet vapor CO2 of the profile endpoints, as in finite_rate_film_cp7_2026-11-24/summary.json."""
import json, math
from pathlib import Path
import numpy as np
T = Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128/twelve_state")
ARTICLE = json.loads(Path("analyses/reactive_film_evidence/results/runs/finite_rate_film_cp7_2026-11-24/summary.json").read_text())
FILMS = {"finite_rate": "finite_rate_n2o", "manifold": "manifold", "enhancement": "enhancement"}
NODES = (17, 33, 65, 129)
def attempt(film, case, nodes):
    if film == "finite_rate" and case in ("6C", "7C"):  # fresh [0, 1] bracket start failed; see bracket_probe/probe.py
        names = [f"bracket_probe/finite_rate_{case}_n{nodes}_b0.5"]
    else:
        names = [f"{film}_{case}_n{nodes}", f"{film}_{case}_n{nodes}_fresh"]
    for name in names:
        path = T / name / "attempt.json"
        if path.exists():
            a = json.loads(path.read_text())
            if (a.get("result") or {}).get("status") == "Solve_Succeeded":
                p = np.asarray(a["result"]["profile"])
                return dict(path=str(path), nodes=nodes, iterations=a["result"]["iterations"], seconds=a["execution"]["runtime_s"],
                            scaled_residual_inf=a["result"]["scaled_residual_inf"], inlet_vapor_co2_mol_s=p[2, 0], outlet_vapor_co2_mol_s=p[2, -1],
                            capture_pct=100 * (1 - p[2, -1] / p[2, 0]))
    return None
out = {"axial_nodes": NODES, "measurement_source": ARTICLE["measurement_source"], "earlier_set_summary": "analyses/reactive_film_evidence/results/runs/finite_rate_film_cp7_2026-11-24/summary.json", "films": {}}
for film, key in FILMS.items():
    rows = []
    for ref in ARTICLE["films"][key]:
        runs = [attempt(film, ref["case"], n) for n in NODES]
        cap = [r and r["capture_pct"] for r in runs]
        row = dict(case=ref["case"], measured_capture_pct=ref["measured_capture_pct"], capture_pct=cap,
                   signed_error_pp=[c and c - ref["measured_capture_pct"] for c in cap], earlier_set_capture_pct=ref["capture_pct"], runs=runs)
        if all(cap):
            d = np.diff(cap)
            row["signed_mesh_changes_pp"] = d.tolist()
            if d[-1] * d[-2] > 0 and abs(d[-1]) < abs(d[-2]):
                order = math.log2(d[-2] / d[-1])
                row.update(observed_order=order, conditional_richardson_capture_pct=cap[-1] + d[-1] / (2**order - 1))
        rows.append(row)
    out["films"][film] = rows
    for label, cases in (("2C_to_7C", rows[1:]), ("1C_to_7C", rows)):
        out.setdefault(f"mean_absolute_error_{label}_pp", {})[film] = [
            float(np.mean([abs(r["signed_error_pp"][i]) for r in cases])) if all(r["capture_pct"][i] for r in cases) else None for i in range(4)]
(T / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
for film, rows in out["films"].items():
    print(film, "MAE 2C-7C", out["mean_absolute_error_2C_to_7C_pp"][film], "1C-7C", out["mean_absolute_error_1C_to_7C_pp"][film])
    for r in rows:
        print("  ", r["case"], r["measured_capture_pct"], ["%.2f" % c if c else "-" for c in r["capture_pct"]], "earlier", ["%.2f" % c if c else "-" for c in r["earlier_set_capture_pct"]], "order", r.get("observed_order"))
