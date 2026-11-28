"""Render the twelve-state capture and liquid temperatures from retained runs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


FIGURE = Path(__file__).resolve().parents[1]
REPO = FIGURE.parents[3]
RUNS = FIGURE.parents[1] / "results/runs/chapter3_mea_20261128/twelve_state"
OUTPUT = FIGURE / "output"
STEM = "twelve-state-capture-temperatures"
NODES = 129
CASES = tuple(f"{i}C" for i in range(1, 8))
TEMPERATURE_CASES = ("3C", "6C", "7C")
FILMS = {
    "finite_rate_epcsaft_henry": ("Finite rate, ePC-SAFT Henry", "#0072B2", "o", "-"),
    "finite_rate": (r"Finite rate, N$_2$O-analogy Henry", "#D55E00", "s", "--"),
    "manifold": ("Equilibrium manifold", "#009E73", "^", "None"),
    "enhancement": ("Enhancement factor", "#CC79A7", "x", "None"),
}
TEMPERATURE_FILMS = ("finite_rate_epcsaft_henry", "finite_rate")
FIELDS = (
    "case_id", "film", "axial_nodes", "observed_capture_pct", "predicted_capture_pct",
    "z_over_H", "liquid_temperature_K", "reported_temperature_K", "source",
)


def capture_and_temperature(summary, mesh_index):
    summary_source = (RUNS / "summary.json").relative_to(REPO)
    tap_path = RUNS / "tap_temperatures.csv"
    with tap_path.open(newline="") as stream:
        taps = [row for row in csv.DictReader(stream) if row["case_id"] in TEMPERATURE_CASES]

    for film in FILMS:
        cases = {row["case"]: row for row in summary["films"][film]}
        assert set(cases) == set(CASES), film
        for case in CASES:
            row = cases[case]
            run = row["runs"][mesh_index]
            capture = row["capture_pct"][mesh_index]
            assert run["nodes"] == NODES and run["capture_pct"] == capture, (film, case)
            yield dict(case_id=case, film=film, axial_nodes=NODES,
                       observed_capture_pct=row["measured_capture_pct"],
                       predicted_capture_pct=capture, source=summary_source)

            if case not in TEMPERATURE_CASES or film not in TEMPERATURE_FILMS:
                continue
            # Use the attempt chosen by summarize.py, as recorded in summary.json.
            attempt = json.loads((REPO / run["path"]).read_text())
            result = attempt["result"]
            z = np.asarray(result["grid"])
            liquid_K = np.asarray(result["profile"][4])  # Liquid temperature state.
            assert z.shape == liquid_K.shape == (NODES,), (film, case)
            assert z[0] == 0 and np.all(np.diff(z) > 0), (film, case)
            assert np.isfinite(liquid_K).all(), (film, case)
            case_taps = [tap for tap in taps if tap["case_id"] == case]
            assert len(case_taps) == 5, case
            tap_z = np.array([float(tap["Position"]) for tap in case_taps])
            tap_liquid_K = [float(tap[f"{film}_liquid_K"]) for tap in case_taps]
            # Verify the selected curve at the source's tap positions and orientation.
            np.testing.assert_array_equal(np.interp(tap_z * z[-1], z, liquid_K), tap_liquid_K)
            for height, temperature in zip(z / z[-1], liquid_K):
                yield dict(case_id=case, film=film, axial_nodes=NODES, z_over_H=height,
                           liquid_temperature_K=temperature, source=run["path"])

    for tap in taps:
        yield dict(case_id=tap["case_id"], z_over_H=tap["Position"],
                   reported_temperature_K=tap["reported_K"], source=tap_path.relative_to(REPO))


def render(rows, summary, mesh_index):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.35, 6.3), layout="constrained")
    capture = axes[0, 0]
    capture.plot([58, 102], [58, 102], color="0.4", linestyle=":", linewidth=1, zorder=1)
    capture.set(xlim=(58, 102), ylim=(58, 102), xticks=range(60, 101, 10),
                yticks=range(60, 101, 10), xlabel="Observed CO₂ capture (%)",
                ylabel="Predicted CO₂ capture (%)")
    capture.set_aspect("equal", adjustable="box")
    capture.set_title("(a) Capture", loc="left")

    handles = []
    for order, (film, (label, color, marker, line)) in enumerate(FILMS.items()):
        points = [row for row in rows if row["film"] == film and row["predicted_capture_pct"]]
        # Keep the enhancement crosses visible inside the open markers near parity.
        zorder = 6 if film == "enhancement" else 5 - order
        (artist,) = capture.plot(
            [float(row["observed_capture_pct"]) for row in points],
            [float(row["predicted_capture_pct"]) for row in points],
            linestyle="None", marker=marker, color=color, markerfacecolor="none",
            markersize=5.5, markeredgewidth=1.1, zorder=zorder,
        )
        expected = {row["case"]: row for row in summary["films"][film]}
        assert [row["case_id"] for row in points] == list(CASES), film
        assert [int(row["axial_nodes"]) for row in points] == [NODES] * len(CASES), film
        np.testing.assert_array_equal(artist.get_xdata(),
                                      [expected[case]["measured_capture_pct"] for case in CASES])
        np.testing.assert_array_equal(artist.get_ydata(),
                                      [expected[case]["capture_pct"][mesh_index] for case in CASES])
        handles.append(Line2D([], [], color=color, linestyle=line, marker=marker,
                              markerfacecolor="none", markersize=5.5, label=label))

    capture_points = {row["case_id"]: row for row in rows
                      if row["film"] == "finite_rate" and row["predicted_capture_pct"]}
    for case in TEMPERATURE_CASES:
        point = capture_points[case]
        label = "7C*" if case == "7C" else case
        capture.annotate(
            label, (float(point["observed_capture_pct"]), float(point["predicted_capture_pct"])),
            xytext=(5, -8), textcoords="offset points", fontsize=8, va="top",
        )

    for panel, ax, case in zip("bcd", list(axes.flat)[1:], TEMPERATURE_CASES):
        case_rows = [row for row in rows if row["case_id"] == case and row["z_over_H"]]
        for film in TEMPERATURE_FILMS:
            _, color, _, line = FILMS[film]
            profile = [row for row in case_rows if row["film"] == film]
            ax.plot([float(row["z_over_H"]) for row in profile],
                    [float(row["liquid_temperature_K"]) for row in profile],
                    color=color, linestyle=line, linewidth=1.6)
        taps = [row for row in case_rows if row["reported_temperature_K"]]
        ax.plot([float(row["z_over_H"]) for row in taps],
                [float(row["reported_temperature_K"]) for row in taps],
                linestyle="None", marker="x", color="black", markersize=5.5, zorder=4,
                clip_on=False)
        ax.set(xlim=(0, 1), ylim=(316, 352), yticks=[320, 330, 340, 350],
               xticks=np.linspace(0, 1, 6), xlabel=r"Normalized packed height, $z/H$",
               ylabel="Liquid temperature (K)")
        ax.set_title(f"({panel}) Case {case}", loc="left")

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    measured = Line2D([], [], linestyle="None", marker="x", color="black",
                      markersize=5.5, label="Measured packing temperature")
    parity = Line2D([], [], color="0.4", linestyle=":", linewidth=1, label="Parity")
    fig.legend(handles=[handles[0], handles[1], measured, handles[2], handles[3], parity],
               loc="outside upper center", ncols=2, frameon=False, fontsize=9)
    fig.savefig(OUTPUT / f"{STEM}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def main():
    summary = json.loads((RUNS / "summary.json").read_text())
    mesh_index = summary["axial_nodes"].index(NODES)
    rows = list(capture_and_temperature(summary, mesh_index))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUTPUT / f"{STEM}.csv"
    with snapshot.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with snapshot.open(newline="") as stream:
        render(list(csv.DictReader(stream)), summary, mesh_index)


if __name__ == "__main__":
    main()
