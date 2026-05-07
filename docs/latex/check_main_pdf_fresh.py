from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path


LATEX_DIR = Path(__file__).resolve().parent
ROOT = LATEX_DIR.parents[1]


def main() -> int:
    args = _parse_args()
    tex = LATEX_DIR / args.tex
    pdf = LATEX_DIR / args.pdf
    if not pdf.exists():
        print(f"Missing {pdf}. Run docs\\latex\\build_main.ps1.")
        return 1

    sources = _latex_sources(tex)
    stale_sources = [path for path in sources if path.exists() and path.stat().st_mtime > pdf.stat().st_mtime]
    if stale_sources:
        print(f"{pdf} is stale. Newer inputs:")
        for path in stale_sources:
            print(f"  {path.relative_to(ROOT)}")
        print(f"Run docs\\latex\\build_main.ps1 to refresh {pdf.name}.")
        return 1

    print(f"{pdf} is current.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a manuscript PDF is newer than its LaTeX inputs.")
    parser.add_argument("--tex", default="main.tex")
    parser.add_argument("--pdf", default="main.pdf")
    return parser.parse_args()


def _latex_sources(root_tex: Path) -> set[Path]:
    sources: set[Path] = set()
    for pattern in ("*.bib", "*.bst", "*.cls", "*.sty"):
        sources.update(path.resolve() for path in LATEX_DIR.glob(pattern))

    for tex_path in _tex_dependency_closure(root_tex):
        sources.add(tex_path.resolve())
        text = tex_path.read_text(encoding="utf-8")
        sources.update(_graphics_sources(text))

    return sources


def _tex_dependency_closure(root_tex: Path) -> set[Path]:
    pending = [root_tex.resolve()]
    seen: set[Path] = set()
    while pending:
        tex_path = pending.pop()
        if tex_path in seen or not tex_path.exists():
            continue
        seen.add(tex_path)
        text = tex_path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\(?:input|include)\{([^}]+)\}", text):
            target = match.group(1)
            candidate = (LATEX_DIR / target).resolve()
            if candidate.suffix != ".tex":
                candidate = candidate.with_suffix(".tex")
            pending.append(candidate)
    return seen


def _graphics_sources(text: str) -> set[Path]:
    paths: set[Path] = set()
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        target = match.group(1)
        candidates = [
            LATEX_DIR / target,
            LATEX_DIR / ".." / target,
            LATEX_DIR / ".." / ".." / target,
            ROOT / target,
        ]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists():
                paths.add(resolved)
                break
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
