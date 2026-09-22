"""Manufactured export checks only: these are not absorber solutions."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("compare", Path(__file__).parents[1] / "scripts/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def record(run_id="reference", grid=None):
    grid = [0, 2] if grid is None else grid
    profile = [[1.] * len(grid) for _ in compare.STATE_LAYOUT]
    profile[2] = [10 - 4 * z for z in grid]
    profile[4] = [300 + 2 * z for z in grid]
    profile[5] = [310.] * len(grid)
    return dict(run_id=run_id, method="reference", execution_status="completed",
                settings={"film_points": 3, "nodes": len(grid), "method_source_sha256": "f"*64},
                problem=dict(engine_commit="a" * 40, wheel_sha256="b" * 64,
                             parameters_sha256="c" * 64, reference_sha256="d" * 64,
                             model_source_sha256={"manufactured": "e" * 64},
                             state_layout=compare.STATE_LAYOUT.copy(), coordinate="height_m_from_gas_inlet",
                             physical_inputs=dict(height_m=2, case_id="manufactured export only",
                                 liquid_feed_mol_s=[1, 1, 1], vapor_feed_mol_s=[1, 1, 1, 1],
                                 liquid_temperature_k=300, vapor_temperature_k=310, bottom_pressure_pa=1e5,
                                 area_m2=1, packing=[250, .97, .2, .3, .01, .2, .1],
                                 gas_mass_flow_basis="manufactured", liquid_mass_flow_basis="manufactured",
                                 humidity_assumption="manufactured")),
                result=dict(accepted=True, grid=grid, profile=profile,
                            defects=[[0]], algebraic_residual=[[0]], boundary_residual=[0]),
                physical_certification={"accepted": True,
                    "original_residuals": {k: [0] for k in compare.RESIDUAL_GROUPS},
                    "criteria": {k: "manufactured export only" for k in compare.RESIDUAL_GROUPS}},
                measurements={"context": {"hardware": "test", "threads": {"OMP": 1},
                                           "software": "test", "scope": "test wall time"}})


class ComparisonChecks(unittest.TestCase):
    def test_unequal_grids_capture_peaks_and_failed_repeat_preservation(self):
        reference = record()
        candidate = record("candidate", [0, 1, 2])
        candidate["method"] = "candidate"
        candidate["result"]["profile"][2][-1] = 1
        candidate["result"]["profile"][4] = [300, 305, 304]
        candidate["result"]["profile"][5] = [310, 311, 310]
        runs = [reference]
        for i, duration in enumerate([1., 3., 9.]):
            repeated = deepcopy(candidate)
            repeated["run_id"] = f"repeat-{i}"
            repeated["measurements"]["wall_s"] = duration
            runs.append(repeated)
        failed = deepcopy(candidate)
        failed.update(run_id="timeout", execution_status="timeout", result=None,
                      failure="wall limit", termination="timeout", limit_seconds=600)
        failed["physical_certification"] = {"accepted": False, "reason": "no candidate"}
        failed["measurements"]["wall_s"] = 601.
        runs.append(failed)
        original = deepcopy(runs)
        report = compare.reduce_records(runs, "reference")
        row = report["rows"][1]
        self.assertAlmostEqual(row["capture_pct"], 90.)
        self.assertAlmostEqual(row["capture_delta_pp"], 10.)
        self.assertEqual(row["Tl_difference_inf_K"], 3.)
        self.assertEqual(row["Tv_difference_inf_K"], 1.)
        self.assertEqual(row["Tl_peak_first_height_m"], 1)
        self.assertEqual(report["rows"][0]["Tv_peak_first_height_m"], 0)
        self.assertEqual(report["rows"][0]["Tv_peak_last_height_m"], 2)
        timing = report["timing_groups"][1]
        self.assertEqual(timing["wall_s"], {"samples": 3, "median": 3., "min": 1., "max": 9.})
        self.assertEqual((timing["attempts"], timing["ineligible_attempts"]), (4, 1))
        self.assertIsNone(timing["cpu_s"]["median"])
        self.assertIsNone(report["rows"][-1]["capture_pct"])
        self.assertEqual(report["records"], original)
        self.assertEqual(runs, original)
        self.assertEqual({r["height_m"] for r in report["profiles"]}, {0, 1, 2})

    def test_mismatched_inputs_invalid_coordinates_and_uncertified_reference_refused(self):
        reference = record()
        for mutation in (
            lambda r: r["problem"]["physical_inputs"].update(case="different"),
            lambda r: r["problem"].update(coordinate="normalized_height"),
            lambda r: r["result"].update(grid=[0, 3]),
            lambda r: r["result"].update(grid=[2, 0]),
            lambda r: r["result"]["profile"][4].__setitem__(0, float("nan")),
            lambda r: r["physical_certification"].update(original_residuals={}),
            lambda r: r["physical_certification"]["original_residuals"].pop("charge"),
            lambda r: r["settings"].pop("method_source_sha256"),
            lambda r: r["measurements"].update(wall_s=-1),
        ):
            with self.subTest(mutation=mutation):
                bad = record("bad")
                mutation(bad)
                with self.assertRaises(ValueError):
                    compare.reduce_records([reference, bad], "reference")
        uncertified = record()
        uncertified["physical_certification"] = {"accepted": None, "reason": "unknown"}
        with self.assertRaises(ValueError):
            compare.reduce_records([uncertified], "reference")
        with self.assertRaises(ValueError):
            compare.reduce_records([reference, deepcopy(reference)], "reference")
        incomplete = [record("one"), record("two")]
        for r in incomplete:
            r["problem"]["physical_inputs"].pop("liquid_feed_mol_s")
        with self.assertRaises(ValueError):
            compare.reduce_records(incomplete, "one")

    def test_timing_context_and_settings_prevent_false_repeats(self):
        runs = [record(str(i)) for i in range(4)]
        for r in runs:
            r["measurements"]["wall_s"] = 1.
        runs[1]["settings"]["film_points"] = 5
        runs[2]["measurements"]["context"]["threads"] = {"OMP": 2}
        runs[3]["measurements"]["context"] = {}
        groups = compare.reduce_records(runs, "0")["timing_groups"]
        self.assertEqual(len(groups), 4)
        self.assertIsNone(groups[-1]["wall_s"]["median"])
        self.assertEqual(groups[-1]["wall_s"]["samples"], 0)
        timed_out = record("late")
        timed_out.update(termination="timeout", limit_seconds=600)
        timed_out["measurements"]["wall_s"] = 600.
        report = compare.reduce_records([record("reference"), timed_out], "reference")
        self.assertFalse(report["rows"][-1]["eligible"])
        self.assertEqual(report["timing_groups"][0]["wall_s"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
