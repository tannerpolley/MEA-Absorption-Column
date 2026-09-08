"""Cheap retained-record/figure checks; no physical absorber claims or solves."""
import importlib.util
import csv
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


ANALYSIS = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class PipelineCheck(unittest.TestCase):
    def test_runtime_mismatch_and_early_failure_remain_visible(self):
        fixture = runpy.run_path(str(ANALYSIS / "tests/test_compare.py"))
        runner = module("case_runtime", ANALYSIS / "scripts/run_case.py")
        identity = dict(wheel_filename="engine.whl", wheel_sha256="a"*64, core_sha256="b"*64)
        resolved = dict(source_kind="local_file", wheel_path="/engine.whl", wheel_sha256="a"*64,
                        core_sha256="b"*64, module_path="/pkg/__init__.py", core_path="/pkg/core.so")
        with patch.object(runner.runpy, "run_path", return_value={"resolve_epcsaft": lambda _: resolved}):
            self.assertEqual(runner.verified_runtime({"final_identity": identity}), resolved)
            resolved["core_sha256"] = "c"*64
            with self.assertRaisesRegex(RuntimeError, "Installed Engine differs"):
                runner.verified_runtime({"final_identity": identity})
        good, failed = fixture["record"]("good"), fixture["record"]("setup_failure")
        failed.pop("problem")
        failed["settings"].pop("method_source_sha256")
        failed.update(execution_status="failed", result=None, failure="Import failed")
        failed["physical_certification"] = {"accepted": None}
        report = fixture["compare"].reduce_records([good, failed], "good")
        self.assertFalse(report["rows"][1]["problem_identity_available"])
        self.assertFalse(report["rows"][1]["eligible"])
        self.assertEqual(report["records"][1], failed)

    def test_retained_failure_and_rendered_refinement(self):
        fixture = runpy.run_path(str(ANALYSIS / "tests/test_compare.py"))
        compare = fixture["compare"]
        runner = module("case_runner", ANALYSIS / "scripts/run_case.py")
        renderer = module("comparison_renderer", ANALYSIS / "figures/method_comparison/scripts/render_comparison.py")
        records = []
        for method in renderer.COLORS:
            for count in (2, 3):
                record = fixture["record"](f"{method}-{count}", grid=[0, 2] if count == 2 else [0, 1, 2])
                record["method"] = method
                record["settings"].update(nodes=count, film_points=3)
                record["measurements"]["wall_s"] = count
                records.append(record)
        failed = fixture["record"]("failed")
        failed.update(method="shooting", execution_status="timeout", termination="timeout", result=None)
        failed["physical_certification"] = {"accepted": False}
        failed["measurements"]["wall_s"] = 7.
        failed["settings"].update(nodes=2, film_points=3)
        records.append(failed)
        report = compare.reduce_records(records, "trapezoidal-3")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner.save(output / "failure.json", {"accepted": False, "residual": [float("nan"), float("inf"), -2.]})
            preserved = json.loads((output / "failure.json").read_text())
            self.assertEqual(preserved["residual"], ["nan", "inf", -2.])
            files = renderer.render(report, output/"figures", refinement="axial", capture_limit=.1, temperature_limit=.1)
            self.assertEqual(len(files), 9)
            self.assertTrue(all((output/"figures"/name).stat().st_size > 1000 for name in files))
            self.assertNotIn("shooting", (output/"figures/axial_refinement.csv").read_text())
            self.assertNotIn("collocation", (output/"figures/axial_refinement.csv").read_text())
            self.assertIn("timeout", (output/"figures/attempt_costs.csv").read_text())
            self.assertIn("failed", (output/"figures/attempt_summary.csv").read_text())
            for mode, method, key in (("ode", "shooting", "ivp_rtol"), ("bvp", "collocation", "tolerance")):
                pair = [fixture["record"]("coarse"), fixture["record"]("fine")]
                for run, tolerance in zip(pair, (1e-5, 1e-7)):
                    run["method"] = method
                    run["settings"][key] = tolerance
                    if mode == "ode":
                        run["settings"]["ivp_atol"] = {1e-5: 1e-7, 1e-7: 1e-9}[tolerance]
                reduced = compare.reduce_records(pair, "fine")
                renderer.render(reduced, output/mode, refinement=mode)
                with (output/mode/f"{mode}_refinement.csv").open() as stream:
                    self.assertEqual({row["reference_id"] for row in csv.DictReader(stream)}, {"fine"})
            records[-1]["measurements"]["context"] = {}
            with self.assertRaisesRegex(ValueError, "measurement contexts"):
                renderer.render(report, output/"bad", capture_limit=.1, temperature_limit=.1)


if __name__ == "__main__":
    unittest.main()
