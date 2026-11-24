"""Render the NCCC capture parity and 3C temperature figures from retained values."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

data = Path(__file__).resolve().parents[1] / "results/runs/joint_fit_f2_20261123/comparison"
capture = pd.read_csv(data / "capture.csv")
colors = {"published": "#60636b", "bridge": "#126493", "f2": "#a84519"}
labels = {"published": "Published", "bridge": "Bridge", "f2": "F2"}

fig, ax = plt.subplots(figsize=(5.3, 5.0), layout="constrained")
ax.plot([57, 102], [57, 102], color="#9b9da5", linewidth=1, linestyle=":", label="Parity")
for arm, marker in zip(colors, ["s", "o", "^"], strict=True):
    ax.errorbar(capture.measured_pct, capture[f"{arm}_pct"], xerr=capture.sd_pp, fmt=marker,
                color=colors[arm], markerfacecolor="white", capsize=2, label=labels[arm])
ax.set(xlim=(57, 102), ylim=(57, 102), xlabel="Measured CO₂ capture (%)", ylabel="Predicted CO₂ capture (%)", aspect="equal")
ax.legend(loc="upper left", frameon=False)
fig.savefig(data / "capture_parity.png", dpi=240)

fig, ax = plt.subplots(figsize=(6.1, 4.1), layout="constrained")
profiles = pd.read_csv(data / "temperature_profiles.csv").query("case_id == '3C'")
for arm, profile in profiles.groupby("arm"):
    phases = (("liquid", "-."),) if arm == "published" else (("liquid", "-"), ("vapor", "--"))
    for phase, style in phases:
        ax.plot(profile.z_m, profile[f"{phase}_K"], linestyle=style, color=colors[arm], label=f"{labels[arm]} {phase}")
taps = pd.read_csv(data / "temperature_taps.csv").query("case_id == '3C' and arm == 'bridge'")
ax.scatter(taps.z_m, taps.measured_K, c="#222222", marker="x", label="Table C2 packing taps", zorder=4, clip_on=False)
ax.set(xlim=(0, 6), xlabel="Height above gas inlet (m)", ylabel="Temperature (K)")
ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.savefig(data / "3c_temperature_profile.png", dpi=240)
