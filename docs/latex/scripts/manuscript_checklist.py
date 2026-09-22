#!/usr/bin/env python3
"""Read-only manuscript checklist: --serve for live HTML, --json for agents."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit


SCRIPT_DIR = Path(__file__).resolve().parent
LATEX_ROOT = SCRIPT_DIR.parent
REVIEWER_FILES = {
    "/reviewer-scorecard": LATEX_ROOT.parents[1] / "output/pdf/reviewer_comment_progress_summary_updated.pdf",
    "/reviewer-notes": LATEX_ROOT.parent / "reviewer_response.md",
    "/reviewer-comments": LATEX_ROOT.parent / "reviewer_comments.txt",
    "/reviewer-original-assessment": LATEX_ROOT.parent / "reviewer_assessment_original.md",
}
INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics\*?(?:\[[^\]]*\])?\{([^}]+)\}")
PLACEHOLDER = re.compile(r"\\textit\{\[|\b(?:TODO|TBD|INSERT|PLACEHOLDER)\b", re.I)


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path is outside the manuscript")
    return path


def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def expand(root, text, seen=None):
    """Expand literal manuscript inputs; report unsupported/missing inputs visibly."""
    seen = set() if seen is None else set(seen)

    def replace(match):
        name = match[1]
        if not Path(name).suffix:
            name += ".tex"
        path = inside(root, name)
        if path in seen:
            raise ValueError(f"Repeated recursive input: {name}")
        return expand(root, path.read_text(), seen | {path})

    return INPUT.sub(replace, uncomment(text))


def active_sources(root):
    todo, found = [root / "main.tex"], set()
    while todo:
        path = todo.pop().resolve()
        if path in found:
            continue
        found.add(path)
        for name in INPUT.findall(uncomment(path.read_text())):
            todo.append(inside(root, name if Path(name).suffix else name + ".tex"))
    return found


def check_content(root, item, active):
    result = dict(item, state="missing", detail="", line=1, assets=[])
    path = inside(root, item["source"])
    if path not in active:
        return dict(result, state="issue", detail="Source is not included by main.tex")
    text = path.read_text()
    if item["kind"] == "figure":
        matches = [m for m in re.finditer(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", uncomment(text), re.S)
                   if "\\label{" + item["label"] + "}" in m[0]]
        if len(matches) != 1:
            return dict(result, state="issue", detail="Expected one figure with this label")
        block = matches[0][0]
        result["line"] = text[:text.index("\\label{" + item["label"] + "}")].count("\n") + 1
        content = expand(root, block)
        graphics = GRAPHIC.findall(content)
        if "\\fbox" in content or not graphics:
            return dict(result, detail="Replace the framed panel with the final figure include")
        for name in graphics:
            asset = inside(root, name)
            choices = [asset] if asset.suffix else [asset.with_suffix(s) for s in (".pdf", ".png", ".jpg", ".jpeg", ".eps")]
            existing = next((p for p in choices if p.is_file() and p.stat().st_size), None)
            if existing is None:
                return dict(result, state="issue", detail=f"Included figure file is missing or empty: {name}")
            result["assets"].append(existing.relative_to(root).as_posix())
        if "\\caption{" not in content or PLACEHOLDER.search(content):
            return dict(result, detail="Figure is included; finish its caption and remove placeholder text")
        return dict(result, state="inserted", detail="Figure is included and its file exists; review values and caption")

    markers = [list(re.finditer(r"^% CHECKLIST-" + side + ": " + re.escape(item["id"]) + r"[ \t]*$", text, re.M))
               for side in ("BEGIN", "END")]
    if any(len(matches) != 1 for matches in markers):
        return dict(result, state="issue", detail="Insertion markers are missing, duplicated, or reversed")
    start, end = (matches[0] for matches in markers)
    if end.start() < start.end():
        return dict(result, state="issue", detail="Insertion markers are missing, duplicated, or reversed")
    block = text[start.end():end.start()]
    result["line"] = text[:start.start()].count("\n") + 1
    content = expand(root, block).strip()
    if item["kind"] == "table" and (not re.search(r"\\begin\{table\*?\}", content)
                                      or "\\label{" + item["label"] + "}" not in content):
        return dict(result, detail="Insert the named table inside this block")
    if not content or PLACEHOLDER.search(content):
        return dict(result, detail="Add the required content between the insertion markers")
    remaining = re.search(r"^% CHECKLIST-REMAINING:\s*(.+)$", block, re.M)
    if remaining:
        return dict(result, state="partial", detail=remaining[1].strip())
    # Source presence is deliberately distinct from scientific review.
    return dict(result, state="inserted", detail="Content is present in the active manuscript; review completeness and evidence")


def check_item(root, item, active):
    result = check_content(root, item, active)
    related = [check_content(root, child, active) for child in item.get("related", [])]
    result["related"] = related
    issues = [child for child in related if child["state"] == "issue"]
    unfinished = [child for child in related if child["state"] != "inserted"]
    if issues and result["state"] != "issue":
        result.update(state="issue", detail="; ".join(child["title"] + ": " + child["detail"] for child in issues))
    elif unfinished and result["state"] == "inserted":
        result.update(state="partial", detail="Finish accompanying text: " + "; ".join(child["title"] for child in unfinished))
    return result


def snapshot(root=LATEX_ROOT, spec=None):
    spec = spec or json.loads((SCRIPT_DIR / "manuscript_checklist.json").read_text())
    ids = [item["id"] for item in spec["items"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate checklist IDs")
    problems = []
    try:
        active = active_sources(root)
    except (OSError, ValueError) as error:
        active = set()
        problems.append(f"Manuscript input error: {error}")
    items = []
    for item in spec["items"]:
        try:
            items.append(check_item(root, item, active))
        except (OSError, ValueError) as error:
            items.append(dict(item, state="issue", detail=str(error), line=1, assets=[]))
    reviewer_file = root / "scripts/reviewer_checklist.json"
    reviewers = json.loads(reviewer_file.read_text())["items"] if reviewer_file.exists() else []
    for item in items:
        item["reviewers"] = [dict(id=r["id"], title=r["title"]) for r in reviewers
                             if item["id"] in r["checks"] + r.get("related_checks", [])]
    pdf = root / "builds/main.pdf"
    inputs = active | set(root.glob("*.bib"))
    inputs |= {inside(root, asset) for item in items for asset in item["assets"]}
    pdf_current = bool(active) and not problems and pdf.exists() and all(
        p.exists() and p.stat().st_mtime_ns <= pdf.stat().st_mtime_ns for p in inputs)
    payload = dict(items=items, counts=dict(Counter(i["state"] for i in items)), total=len(items),
                   instructions=spec["instructions"], problems=problems,
                   manuscript=str(root / "main.tex"), pdf_exists=pdf.exists(), pdf_current=pdf_current)
    payload["revision"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return payload


def reviewer_snapshot(root=LATEX_ROOT, spec=None, manuscript=None):
    spec = spec if spec is not None else json.loads((SCRIPT_DIR / "reviewer_checklist.json").read_text())
    manuscript = manuscript if manuscript is not None else snapshot(root)
    content = {item["id"]: item for item in manuscript["items"]}
    rows, ids = [], set()
    for item in spec["items"]:
        if item["id"] in ids or any(type(value) not in (int, float) or not 0 <= value <= 10
                                   for value in [item["score"]] + ([item["previous_score"]] if item.get("previous_score") is not None else [])):
            raise ValueError("Invalid or duplicate reviewer assessment")
        ids.add(item["id"])
        changed = []
        for name, expected in item["evidence"].items():
            path = inside(root.parents[1], name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                changed.append(name)
        checks = [content[key] for key in item["checks"]]
        complete = item["score"] == 10 and not changed and all(c["state"] == "inserted" for c in checks)
        progress = item.get("progress", "planned")
        if progress not in ("planned", "in_progress", "deferred"):
            raise ValueError("Invalid reviewer progress")
        links = [content[key] for key in dict.fromkeys(item["checks"] + item.get("related_checks", []))]
        rows.append(dict(item, changed=changed, complete=complete,
                         status="completed" if complete else "reassess" if changed else progress,
                         manuscript_links=[dict(id=c["id"], title=c["title"], state=c["state"],
                                                subsection=c.get("subsection", "")) for c in links],
                         checks=[dict(id=c["id"], title=c["title"], state=c["state"]) for c in checks]))
    return dict(items=rows, total=len(rows), complete=sum(r["complete"] for r in rows),
                needs_review=sum(bool(r["changed"]) for r in rows),
                average=round(sum(r["score"] for r in rows) / len(rows), 2) if rows else 0,
                assessed_at=spec["assessed_at"], assessment_basis=spec["assessment_basis"], rubric=spec["rubric"],
                checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def html_page(data, live, template="manuscript_checklist.html"):
    boot = json.dumps(dict(data=data, live=live)).replace("<", "\\u003c")
    return (SCRIPT_DIR / template).read_text().replace("/*CHECKLIST_BOOT*/null", boot)


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/":
                    body, mime = html_page(snapshot(), True).encode(), "text/html; charset=utf-8"
                elif parsed.path == "/api/checklist":
                    body, mime = json.dumps(snapshot()).encode(), "application/json"
                elif parsed.path == "/reviewers":
                    body, mime = html_page(reviewer_snapshot(), True, "reviewer_checklist.html").encode(), "text/html; charset=utf-8"
                elif parsed.path == "/api/reviewers":
                    body, mime = json.dumps(reviewer_snapshot()).encode(), "application/json"
                elif parsed.path in REVIEWER_FILES:
                    path = REVIEWER_FILES[parsed.path]
                    body = path.read_bytes()
                    mime = "application/pdf" if path.suffix == ".pdf" else "text/plain; charset=utf-8"
                elif parsed.path == "/file":
                    name = parse_qs(parsed.query).get("path", [""])[0]
                    path = inside(LATEX_ROOT, name)
                    types = {".tex": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
                             ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
                    if path.suffix not in types or not path.is_file():
                        raise FileNotFoundError(name)
                    body, mime = path.read_bytes(), types[path.suffix]
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (FileNotFoundError, ValueError) as error:
                self.send_error(404, str(error))
            except OSError:
                self.send_error(503, "A manuscript file is being updated; retry shortly")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Manuscript checklist: http://127.0.0.1:{server.server_port}", flush=True)
    print("Read-only; refreshes from manuscript files. Ctrl-C stops this server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--json", action="store_true")
    mode.add_argument("--html", action="store_true", help="Write an offline snapshot under builds/")
    parser.add_argument("--port", type=int, default=8766, help="Loopback port; 0 selects an available port")
    args = parser.parse_args()
    if args.serve:
        serve(args.port)
        return
    data = snapshot()
    if args.html:
        output = LATEX_ROOT / "builds/manuscript-checklist.html"
        output.parent.mkdir(exist_ok=True)
        output.write_text(html_page(data, False))
        print(output)
        reviewer_output = LATEX_ROOT / "builds/reviewer-checklist.html"
        reviewer_output.write_text(html_page(reviewer_snapshot(manuscript=data), False, "reviewer_checklist.html"))
        print(reviewer_output)
    elif args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"{data['counts'].get('inserted', 0)}/{data['total']} insertions present")
        for item in data["items"]:
            print(f"[{item['state']}] {item['id']}: {item['title']} — {item['source']}:{item['line']}")


if __name__ == "__main__":
    main()
