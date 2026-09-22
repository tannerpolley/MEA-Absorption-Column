"""Run directly: python3 docs/latex/scripts/test_manuscript_checklist.py."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from manuscript_checklist import REVIEWER_FILES, reviewer_snapshot, inside, snapshot, html_page


class ChecklistTest(unittest.TestCase):
    def test_file_changes_drive_checklist_without_touching_the_paper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "figures").mkdir()
            (root / "main.tex").write_text("\\input{body}")
            text = {"id": "claim", "kind": "text", "source": "body.tex", "title": "Finding", "needs": "Exact finding", "group": "Results"}
            figure = {"id": "plot", "kind": "figure", "source": "body.tex", "title": "Plot", "needs": "Plot", "group": "Figures", "label": "fig:plot"}
            spec = {"instructions": "Keep markers", "items": [text, figure]}
            body = root / "body.tex"
            def write(content, graphics):
                body.write_text("% CHECKLIST-BEGIN: claim\n" + content + "\n% CHECKLIST-END: claim\n"
                                "\\begin{figure}\n" + graphics + "\n\\caption{Capture by case}\\label{fig:plot}\\end{figure}")
            write("% INSERT finding", "\\fbox{Final plot goes here}")
            self.assertEqual(snapshot(root, spec)["counts"], {"missing": 2})
            (root / "figures/plot.pdf").write_bytes(b"%PDF-1.4 test file")
            self.assertEqual(snapshot(root, spec)["counts"], {"missing": 2}, "Copying a file alone must not count")
            write("Capture increases by 2 percentage points.", "\\includegraphics{figures/plot}")
            before = body.read_bytes()
            self.assertEqual(snapshot(root, spec)["counts"], {"inserted": 2})
            self.assertEqual(body.read_bytes(), before, "Scanner must be read-only")
            write("A supported method definition.\n% CHECKLIST-REMAINING: Final run settings.", "\\includegraphics{figures/plot}")
            self.assertEqual(snapshot(root, spec)["items"][0]["state"], "partial")
            write("Capture increases by 2 percentage points.", "\\includegraphics{figures/plot}")
            (root / "figures/plot.pdf").unlink()
            self.assertEqual(snapshot(root, spec)["counts"], {"inserted": 1, "issue": 1})
            write("\\input{finding}", "\\includegraphics{figures/plot}")
            (root / "finding.tex").write_text("% just a comment")
            self.assertEqual(snapshot(root, spec)["items"][0]["state"], "missing")
            (root / "finding.tex").write_text("\\textit{[Quantitative comparison: supply values]}")
            self.assertEqual(snapshot(root, spec)["items"][0]["state"], "missing")
            (root / "finding.tex").write_text("The measured capture is 90 percent.")
            self.assertEqual(snapshot(root, spec)["items"][0]["state"], "inserted")
            body.write_text(body.read_text().replace("% CHECKLIST-END: claim", ""))
            self.assertEqual(snapshot(root, spec)["items"][0]["state"], "issue")
            (root / "main.tex").write_text("No body included")
            self.assertEqual(snapshot(root, spec)["counts"], {"issue": 2})

    def test_numbered_markers_do_not_match_longer_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "main.tex").write_text(
                "% CHECKLIST-BEGIN: finding-1\nWritten.\n% CHECKLIST-END: finding-1\n"
                "% CHECKLIST-BEGIN: finding-10\n% INSERT result\n% CHECKLIST-END: finding-10")
            items = [dict(id=id, kind="text", source="main.tex", title=id, needs=id, group="Results")
                     for id in ("finding-1", "finding-10")]
            self.assertEqual([x["state"] for x in snapshot(root, dict(items=items, instructions=""))["items"]],
                             ["inserted", "missing"])

    def test_path_and_html_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                inside(root, "../outside.txt")
            (root / "escape").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                inside(root, "escape/outside.txt")
            data = {"text": "</script><script>alert(1)</script>"}
            page = html_page(data, False)
            self.assertNotIn(data["text"], page)
            self.assertIn("\\u003c/script>", page)
            self.assertIn("Reviewer feedback", page)
            self.assertEqual(set(REVIEWER_FILES), {"/reviewer-scorecard", "/reviewer-notes", "/reviewer-comments", "/reviewer-original-assessment"})
            self.assertTrue(all(path.is_file() for path in REVIEWER_FILES.values()))

    def test_reviewer_score_updates_and_evidence_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            root = repo / "docs/latex"
            root.mkdir(parents=True)
            evidence = repo / "finding.txt"
            evidence.write_text("Reviewed finding")
            row = dict(id="R1.1", score=10, previous_score=2.5, checks=["figure"],
                       evidence={"finding.txt": hashlib.sha256(evidence.read_bytes()).hexdigest()})
            spec = dict(items=[row], assessed_at="2026-09-03", assessment_basis="Test", rubric={})
            manuscript = dict(items=[dict(id="figure", title="Figure", state="missing")])
            self.assertEqual(reviewer_snapshot(root, spec, manuscript)["complete"], 0)
            manuscript["items"][0]["state"] = "inserted"
            initial = reviewer_snapshot(root, spec, manuscript)
            self.assertEqual(initial["complete"], 1)
            evidence.write_text("Changed finding")
            changed = reviewer_snapshot(root, spec, manuscript)
            self.assertEqual(changed["complete"], 0)
            self.assertEqual(changed["needs_review"], 1)
            self.assertEqual(changed["items"][0]["score"], 10, "Never silently regrade changed evidence")
            row["score"] = 6.5
            row["evidence"]["finding.txt"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
            manuscript["items"][0]["state"] = "inserted"
            reviewed = reviewer_snapshot(root, spec, manuscript)
            self.assertEqual(reviewed["average"], 6.5)
            self.assertEqual(reviewed["needs_review"], 0)
            self.assertEqual(reviewed["items"][0]["checks"][0]["state"], "inserted")
            row["score"] = 11
            with self.assertRaises(ValueError):
                reviewer_snapshot(root, spec, manuscript)
        current = reviewer_snapshot()
        self.assertEqual(current["total"], 29)
        self.assertEqual(sum(row["group"].startswith("Reviewer") for row in current["items"]), 20)
        self.assertEqual(sum(row["group"] == "Additional revision work" for row in current["items"]), 9)
        self.assertEqual(current["needs_review"], 0)
        reviewer_rows = [row for row in current["items"] if row["group"].startswith("Reviewer")]
        self.assertTrue(all(row["original_feedback"] and row["previous_score"] is not None for row in reviewer_rows))
        self.assertTrue(all(row["previous_score"] is None for row in current["items"] if row["group"] == "Additional revision work"))

    def test_current_inventory_is_complete_and_consistent(self):
        data = snapshot()
        self.assertEqual(data["total"], 35)
        self.assertFalse(data["problems"])
        self.assertNotIn("issue", data["counts"])
        self.assertEqual(sum(item["kind"] == "figure" for item in data["items"]), 11)
        self.assertEqual(sum(item["kind"] == "table" for item in data["items"]), 6)
        self.assertEqual(sum(item["id"].startswith("method-") for item in data["items"]), 19)
        self.assertEqual(sum(len(item["related"]) for item in data["items"]), 14)
        json.dumps(data)

    def test_progress_and_related_links_do_not_change_completion(self):
        manuscript = snapshot()
        spec = json.loads((Path(__file__).parent / "reviewer_checklist.json").read_text())
        rows = {r["id"]: r for r in reviewer_snapshot(spec=spec, manuscript=manuscript)["items"]}
        self.assertEqual(rows["R1.8"]["status"], "in_progress")
        self.assertEqual(rows["R1.10"]["status"], "in_progress")
        self.assertEqual(rows["R2.5"]["status"], "in_progress")
        self.assertEqual(rows["A9"]["status"], "deferred")
        self.assertFalse(rows["A9"]["complete"])
        self.assertEqual(rows["R1.9"]["status"], "completed")
        self.assertTrue(any(c["state"] != "inserted" for c in rows["R1.9"]["manuscript_links"]),
                        "Unfilled supporting links must not reopen completed feedback")
        for item in manuscript["items"]:
            for reviewer in item["reviewers"]:
                self.assertIn(item["id"], [c["id"] for c in rows[reviewer["id"]]["manuscript_links"]])
        spec["items"][0]["progress"] = "unknown"
        with self.assertRaises(ValueError):
            reviewer_snapshot(spec=spec, manuscript=manuscript)

    def test_table_and_linked_text_must_both_be_present(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            body = root / "main.tex"
            child = dict(id="finding", kind="text", source="main.tex", title="Result", needs="Result", group="Results")
            item = dict(id="table", kind="table", source="main.tex", title="Table", needs="Values", group="Results", label="tab:result", related=[child], review="Check citation")
            spec = dict(instructions="Keep markers", items=[item])
            def write(table, finding):
                body.write_text("% CHECKLIST-BEGIN: table\n" + table + "\n% CHECKLIST-END: table\n% CHECKLIST-BEGIN: finding\n" + finding + "\n% CHECKLIST-END: finding")
            write("Narrative alone", "A result.")
            self.assertEqual(snapshot(root, spec)["counts"], {"missing": 1})
            table = r"\begin{table}\caption{Result}\label{tab:result}\begin{tabular}{l}90\end{tabular}\end{table}"
            write(table, "% INSERT value-dependent statement")
            self.assertEqual(snapshot(root, spec)["counts"], {"partial": 1})
            write(table, "Capture is 90 percent.")
            self.assertEqual(snapshot(root, spec)["counts"], {"inserted": 1}, "Final review does not block written content")
            write(table.replace("90", r"\textit{[value]}"), "Capture is 90 percent.")
            self.assertEqual(snapshot(root, spec)["counts"], {"missing": 1})


if __name__ == "__main__":
    unittest.main()
