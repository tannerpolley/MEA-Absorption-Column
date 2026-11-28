"""Fit-A twelve-state liquid/vapor temperatures at the NCCC packing taps, 129 nodes, every film, against the reported tap temperatures
and the fit-A nine-species correlation-enthalpy runs (fitA_article_20261128/temperature_tap_comparison.csv). Position = z/H from the
bottom, the same coordinate as the nine-species profiles (both have the vapor inlet at Position 0). Run from the repository root."""
import csv, json, math
from pathlib import Path
import numpy as np
T = Path("analyses/nccc_validation/results/runs/chapter3_mea_20261128/twelve_state")
TAPS = Path("analyses/nccc_validation/results/runs/fitA_article_20261128/temperature_tap_comparison.csv")
summary = json.loads((T / "summary.json").read_text())
taps = list(csv.DictReader(TAPS.open()))
rows = []
for tap in taps:
    row = dict(case_id=tap["case_id"], Position=float(tap["Position"]), reported_K=float(tap["reported_temperature_K"]),
               nine_species_liquid_K=float(tap["predicted_liquid_K_fit_a"]) if tap["predicted_liquid_K_fit_a"] else None)
    for film, cases in summary["films"].items():
        run = next(r for r in cases if r["case"] == tap["case_id"])["runs"][-1]
        assert run["nodes"] == 129
        a = json.loads(Path(run["path"]).read_text())["result"]
        z, p = np.asarray(a["grid"]), np.asarray(a["profile"])
        row[f"{film}_liquid_K"] = float(np.interp(row["Position"] * z[-1], z, p[4]))
        row[f"{film}_vapor_K"] = float(np.interp(row["Position"] * z[-1], z, p[5]))
    rows.append(row)
with (T / "tap_temperatures.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
def stats(key, subset):
    e = [r[key] - r["reported_K"] for r in subset if r[key] is not None]
    return dict(taps=len(e), rmse_K=math.sqrt(sum(x * x for x in e) / len(e)), bias_K=sum(e) / len(e), max_abs_K=max(map(abs, e)))
keys = ["nine_species_liquid_K"] + [f"{f}_{s}_K" for f in summary["films"] for s in ("liquid", "vapor")]
matched = [r for r in rows if r["nine_species_liquid_K"] is not None]
out = {"source_taps": str(TAPS), "nodes": 129, "all_taps": {k: stats(k, rows) for k in keys}, "taps_with_nine_species_prediction": {k: stats(k, matched) for k in keys},
       "taps_with_nine_species_prediction_without_7C": {k: stats(k, [r for r in matched if r["case_id"] != "7C"]) for k in keys}}  # 7C misses by 16-24 K in every model
(T / "tap_temperatures.json").write_text(json.dumps(out, indent=1) + "\n")
for group in [g for g in out if g.startswith(("all", "taps"))]:
    print(group); [print("  %-36s n=%2d RMSE %.2f bias %+.2f max %.2f" % (k, v["taps"], v["rmse_K"], v["bias_K"], v["max_abs_K"])) for k, v in out[group].items()]
