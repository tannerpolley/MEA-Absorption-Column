#!/usr/bin/env python3
"""Read-only fallback reviewer notebook; reuses the main manuscript's HTML view."""

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
FILES = {
    "/manuscript": "docs/latex/builds/main.pdf",
    "/qa": "docs/latex/QA_REPORT.md",
    "/reviewer-notes": "docs/fallback_reviewer_response.md",
    "/reviewer-comments": "docs/reviewer_comments.txt",
    "/reviewer-original-assessment": "docs/reviewer_assessment_original.md",
}


def snapshot(root=ROOT, spec=None):
    if spec is None:
        spec = json.loads((root / "docs/latex/scripts/reviewer_checklist.json").read_text())
    rows, ids = [], set()
    for item in spec["items"]:
        scores = [item["score"]]
        if item.get("previous_score") is not None:
            scores.append(item["previous_score"])
        if item["id"] in ids or any(type(n) not in (int, float) or not 0 <= n <= 10 for n in scores):
            raise ValueError("Invalid or duplicate reviewer assessment")
        ids.add(item["id"])
        if not item["evidence"]:
            raise ValueError("Reviewer assessment requires manuscript evidence")
        changed = []
        for name, expected in item["evidence"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Evidence path is outside this worktree")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                changed.append(name)
        progress = item.get("progress", "planned")
        if progress not in ("planned", "in_progress", "deferred"):
            raise ValueError("Invalid reviewer progress")
        complete = item["score"] == 10 and not changed
        rows.append(dict(item, changed=changed, complete=complete,
                         status="completed" if complete else "reassess" if changed else progress,
                         manuscript_links=[]))
    return dict(items=rows, total=len(rows), complete=sum(r["complete"] for r in rows),
                needs_review=sum(bool(r["changed"]) for r in rows),
                assessed_at=spec["assessed_at"], assessment_basis=spec["assessment_basis"],
                rubric=spec["rubric"], checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def html_page(data):
    boot = json.dumps(dict(data=data, live=True)).replace("<", "\\u003c")
    return (SCRIPT_DIR / "reviewer_checklist.html").read_text().replace("/*CHECKLIST_BOOT*/null", boot)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        route = urlsplit(self.path).path
        try:
            if route in ("/", "/reviewers"):
                body, mime = html_page(snapshot()).encode(), "text/html; charset=utf-8"
            elif route == "/api/reviewers":
                body, mime = json.dumps(snapshot()).encode(), "application/json"
            elif route in FILES:
                path = (ROOT / FILES[route]).resolve()
                if not path.is_relative_to(ROOT):
                    raise ValueError("File is outside this worktree")
                body = path.read_bytes()
                mime = "application/pdf" if path.suffix == ".pdf" else "text/plain; charset=utf-8"
            else:
                self.send_error(404)
                return
        except (OSError, ValueError, KeyError, TypeError):
            self.send_error(503, "Fallback source is missing or being updated; retry after checking the record")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=0, help="Loopback port; 0 selects an available port")
    args = parser.parse_args()
    if not args.serve:
        print(json.dumps(snapshot(), indent=2))
        return
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Fallback reviewers: http://127.0.0.1:{server.server_port}/reviewers", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
