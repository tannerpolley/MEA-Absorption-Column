"""Run directly: python3 docs/latex/scripts/test_reviewer_checklist.py."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile

from reviewer_checklist import ROOT, html_page, snapshot


data = snapshot()
reviewers = [row for row in data["items"] if row["group"].startswith("Reviewer")]
assert len(reviewers) == 20 and len({row["id"] for row in reviewers}) == 20
assert sum(row["complete"] for row in reviewers) == 19
assert data["needs_review"] == 0
assert all(row["original_feedback"] in (ROOT / "docs/reviewer_comments.txt").read_text() for row in reviewers)
assert "Not submission-ready" in html_page(data)
malicious = deepcopy(data)
malicious["assessment_basis"] = "</script><script>alert(1)</script>"
assert "</script><script>alert(1)" not in html_page(malicious)

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    evidence = root / "main.tex"
    evidence.write_text("Reviewed content")
    row = deepcopy(reviewers[1])
    row["evidence"] = {"main.tex": hashlib.sha256(evidence.read_bytes()).hexdigest()}
    spec = dict(items=[row], assessed_at=data["assessed_at"], assessment_basis="Test", rubric={})
    assert snapshot(root, spec)["items"][0]["complete"]
    evidence.write_text("Changed content")
    assert snapshot(root, spec)["items"][0]["status"] == "reassess"
    assert not snapshot(root, spec)["items"][0]["complete"]
    row["evidence"] = {"../outside.tex": "bad"}
    try:
        snapshot(root, spec)
    except ValueError:
        pass
    else:
        raise AssertionError("Evidence traversal was not rejected")

print("Fallback reviewer notebook: exact comments, scoring, source changes, and escaping passed.")
