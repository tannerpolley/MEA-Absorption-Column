#!/usr/bin/env python3
"""Linux-native LaTeX build, figure, mirror, and submission workflows."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path


FIGURE_COPIES = (
    (
        Path("analyses/nccc_validation/results/final/figures/nccc_one_bed_thermo_benchmark.pdf"),
        Path("nccc-one-bed-thermo-benchmark.pdf"),
    ),
    (
        Path("analyses/nccc_validation/results/final/figures/nccc_2017_epcsaft_temperature_overlays/3C_temperature_overlay.png"),
        Path("case-3c-temperature-validation.png"),
    ),
    (
        Path("analyses/nccc_validation/results/final/figures/nccc_2017_epcsaft_temperature_overlays/nccc_2017_epcsaft_temperature_overlay_contact_sheet.png"),
        Path("case-c-temperature-overlay.png"),
    ),
    (
        Path("analyses/nccc_validation/results/final/figures/method_case_solver_contrast.pdf"),
        Path("method-case-solver-contrast.pdf"),
    ),
)

DEFAULT_MIRROR = Path(
    "/home/tnnrpolley21/Workspaces/Engineering/LaTeX-Projects/MEA-Absorption-Column-LaTeX"
)
DEFAULT_BIBLIOGRAPHY_SOURCE = Path.home() / "Documents" / "Papers" / "references.bib"
PROJECTION_EXCLUDES = frozenset({"scripts", "builds"})
BUILD_SUFFIXES = {
    ".abs",
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".xdv",
}
FIGURE_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def _tex_files(root: Path) -> list[Path]:
    files = [root / "main.tex"]
    for folder in ("sections", "appendices"):
        directory = root / folder
        if directory.exists():
            files.extend(sorted(directory.glob("*.tex")))
    return [path for path in files if path.exists()]


def _figure_references(root: Path) -> set[Path]:
    references: set[Path] = set()
    for tex_file in _tex_files(root):
        for match in FIGURE_PATTERN.finditer(tex_file.read_text(encoding="utf-8")):
            value = Path(match.group(1))
            if value.parts and value.parts[0] == "figures":
                references.add(value)
    return references


def _validate_figure_references(root: Path) -> None:
    missing = sorted(str(path) for path in _figure_references(root) if not (root / path).is_file())
    if missing:
        raise RuntimeError(f"missing LaTeX figure references: {', '.join(missing)}")


def sync_figures(repo_root: Path, latex_root: Path, *, dry_run: bool = False) -> list[Path]:
    repo_root = repo_root.resolve()
    latex_root = latex_root.resolve()
    destination_root = latex_root / "figures"
    copied: list[Path] = []
    for source_relative, destination_relative in FIGURE_COPIES:
        source = repo_root / source_relative
        destination = destination_root / destination_relative
        if not source.is_file():
            raise RuntimeError(f"required figure source is missing: {source}")
        copied.append(destination)
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    if not dry_run:
        _validate_figure_references(latex_root)
    return copied


def _projected_entries(source_root: Path) -> list[Path]:
    return sorted(
        (entry for entry in source_root.iterdir() if entry.name not in PROJECTION_EXCLUDES),
        key=lambda path: path.name,
    )


def sync_projection(source_root: Path, mirror_root: Path, *, dry_run: bool = False) -> None:
    source_root = source_root.resolve()
    mirror_root = mirror_root.resolve()
    if not (mirror_root / ".git").exists():
        raise RuntimeError(f"mirror root is not a Git checkout: {mirror_root}")
    if dry_run:
        return
    for entry in mirror_root.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    for source in _projected_entries(source_root):
        destination = mirror_root / source.name
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns(
                    "*.abs",
                    "*.aux",
                    "*.bbl",
                    "*.blg",
                    "*.fdb_latexmk",
                    "*.fls",
                    "*.log",
                    "*.out",
                    "*.synctex.gz",
                    "*.xdv",
                ),
            )
        else:
            shutil.copy2(source, destination)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_path_status(path: Path) -> str:
    root = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if root.returncode != 0:
        return ""
    repo_root = Path(root.stdout.strip()).resolve()
    relative = path.resolve(strict=False).relative_to(repo_root)
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            str(relative),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sync_bibliography(
    latex_root: Path,
    source: Path = DEFAULT_BIBLIOGRAPHY_SOURCE,
    *,
    check_only: bool = False,
    replace_reviewed_dirty: bool = False,
) -> Path:
    """Project the Zotero-owned Better BibTeX export into the repository."""
    latex_root = latex_root.resolve()
    source = source.expanduser()
    target = latex_root / "references.bib"
    if not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
        raise RuntimeError(
            f"canonical Better BibTeX export must be a nonempty regular file: {source}"
        )
    if target.exists() and (not target.is_file() or target.is_symlink()):
        raise RuntimeError(f"repository bibliography must be a regular non-symlink file: {target}")

    matches = target.is_file() and _file_hash(source) == _file_hash(target)
    if matches:
        return target
    if check_only:
        raise RuntimeError(f"repository bibliography snapshot is stale: {target}")

    status = _git_path_status(target)
    if status and not replace_reviewed_dirty:
        raise RuntimeError(
            "refusing to replace a differing dirty bibliography snapshot; "
            f"review first: {status}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if target.is_symlink() or target.stat().st_size == 0 or _file_hash(source) != _file_hash(target):
        raise RuntimeError(f"bibliography projection verification failed: {target}")
    return target


def _file_map(root: Path, excluded_roots: set[str] | frozenset[str]) -> dict[Path, str]:
    result: dict[Path, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in excluded_roots:
            continue
        result[relative] = _file_hash(path)
    return result


def audit_projection(source_root: Path, mirror_root: Path) -> None:
    source_root = source_root.resolve()
    mirror_root = mirror_root.resolve()
    if not (mirror_root / ".git").exists():
        raise RuntimeError(f"mirror root is not a Git checkout: {mirror_root}")
    source = _file_map(source_root, PROJECTION_EXCLUDES)
    mirror = _file_map(mirror_root, frozenset({".git"}))
    missing = sorted(str(path) for path in source.keys() - mirror.keys())
    extra = sorted(str(path) for path in mirror.keys() - source.keys())
    changed = sorted(
        str(path) for path in source.keys() & mirror.keys() if source[path] != mirror[path]
    )
    problems: list[str] = []
    if missing:
        problems.append(f"missing paths: {', '.join(missing)}")
    if extra:
        problems.append(f"extra paths: {', '.join(extra)}")
    if changed:
        problems.append(f"hash mismatch: {', '.join(changed)}")
    if problems:
        raise RuntimeError("mirror projection mismatch; " + "; ".join(problems))
    _validate_figure_references(source_root)
    _validate_figure_references(mirror_root)


def _flatten_text(text: str) -> str:
    for prefix in ("figures/", "tables/", "sections/", "appendices/"):
        text = text.replace(prefix, "")
    return text


def prepare_submission(latex_root: Path, output_root: Path, *, zip_output: bool = False) -> Path:
    latex_root = latex_root.resolve()
    output_root = output_root.resolve()
    builds_root = (latex_root / "builds").resolve()
    if not output_root.is_relative_to(builds_root):
        raise RuntimeError(f"output must be inside {builds_root}: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    tex_files = [latex_root / "main.tex"]
    for folder in ("sections", "appendices", "tables"):
        directory = latex_root / folder
        if directory.exists():
            tex_files.extend(sorted(directory.glob("*.tex")))
    for source in tex_files:
        if not source.is_file():
            raise RuntimeError(f"required TeX source is missing: {source}")
        (output_root / source.name).write_text(
            _flatten_text(source.read_text(encoding="utf-8")),
            encoding="utf-8",
        )

    for relative in (Path("references.bib"), Path("builds/main.pdf")):
        source = latex_root / relative
        if not source.is_file():
            raise RuntimeError(f"required submission file is missing: {source}")
        shutil.copy2(source, output_root / source.name)
    optional_bbl = latex_root / "builds" / "main.bbl"
    if optional_bbl.is_file():
        shutil.copy2(optional_bbl, output_root / optional_bbl.name)

    source_text = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)
    for figure in sorted({Path(match.group(1)) for match in FIGURE_PATTERN.finditer(source_text)}):
        source = latex_root / figure
        if not source.is_file():
            raise RuntimeError(f"referenced figure does not exist: {figure}")
        shutil.copy2(source, output_root / source.name)

    if zip_output:
        shutil.make_archive(str(output_root), "zip", root_dir=output_root)
    return output_root


def build_pdf(latex_root: Path, *, clean: bool = False, open_pdf: bool = False) -> Path:
    latex_root = latex_root.resolve()
    sync_bibliography(latex_root)
    output = latex_root / "builds" / "main.pdf"
    (latex_root / "builds").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-outdir=builds", "main.tex"],
        cwd=latex_root,
        check=True,
    )
    if not output.is_file():
        raise RuntimeError(f"expected built PDF not found: {output}")
    subprocess.run(
        [sys.executable, "scripts/check_main_pdf_fresh.py"],
        cwd=latex_root,
        check=True,
    )
    if clean:
        subprocess.run(["latexmk", "-c", "-outdir=builds", "main.tex"], cwd=latex_root, check=True)
    if open_pdf:
        subprocess.run(["xdg-open", str(output)], check=True)
    return output


def clean_mirror_build_files(mirror_root: Path) -> None:
    for path in mirror_root.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith(".synctex.gz") or path.suffix in BUILD_SUFFIXES:
            path.unlink()


def _git_check(mirror_root: Path, *, verify_remote: bool, remote: str, branch: str) -> None:
    status = subprocess.run(
        ["git", "-C", str(mirror_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("mirror Git checkout is not clean")
    if verify_remote:
        subprocess.run(["git", "-C", str(mirror_root), "fetch", remote], check=True)
        head = subprocess.run(
            ["git", "-C", str(mirror_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        remote_head = subprocess.run(
            ["git", "-C", str(mirror_root), "rev-parse", f"{remote}/{branch}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != remote_head:
            raise RuntimeError(f"mirror HEAD does not match {remote}/{branch}")


def _parser() -> argparse.ArgumentParser:
    script = Path(__file__).resolve()
    latex_root = script.parent.parent
    repo_root = latex_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--clean", action="store_true")
    build.add_argument("--open", action="store_true")

    figures = subparsers.add_parser("sync-figures")
    figures.add_argument("--dry-run", action="store_true")

    bibliography = subparsers.add_parser("sync-bibliography")
    bibliography.add_argument("--source", type=Path, default=DEFAULT_BIBLIOGRAPHY_SOURCE)
    bibliography.add_argument("--check", action="store_true")
    bibliography.add_argument("--replace-reviewed-dirty", action="store_true")

    sync = subparsers.add_parser("sync-overleaf")
    sync.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR)
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--clean-build-files", action="store_true")

    audit = subparsers.add_parser("audit-overleaf")
    audit.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR)
    audit.add_argument("--require-clean-git", action="store_true")
    audit.add_argument("--verify-remote", action="store_true")
    audit.add_argument("--remote", default="origin")
    audit.add_argument("--branch", default="master")

    submission = subparsers.add_parser("prepare-submission")
    submission.add_argument("--output-root", type=Path, default=latex_root / "builds" / "elsevier_submission_flat")
    submission.add_argument("--zip", action="store_true")

    parser.set_defaults(repo_root=repo_root, latex_root=latex_root)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        print(build_pdf(args.latex_root, clean=args.clean, open_pdf=args.open))
    elif args.command == "sync-bibliography":
        print(
            sync_bibliography(
                args.latex_root,
                args.source,
                check_only=args.check,
                replace_reviewed_dirty=args.replace_reviewed_dirty,
            )
        )
    elif args.command == "sync-figures":
        for path in sync_figures(args.repo_root, args.latex_root, dry_run=args.dry_run):
            print(path)
    elif args.command == "sync-overleaf":
        sync_bibliography(args.latex_root, check_only=args.dry_run)
        sync_figures(args.repo_root, args.latex_root, dry_run=args.dry_run)
        sync_projection(args.latex_root, args.mirror_root, dry_run=args.dry_run)
        if not args.dry_run:
            if args.clean_build_files:
                clean_mirror_build_files(args.mirror_root)
            audit_projection(args.latex_root, args.mirror_root)
        print(args.mirror_root.resolve())
    elif args.command == "audit-overleaf":
        sync_bibliography(args.latex_root, check_only=True)
        audit_projection(args.latex_root, args.mirror_root)
        if args.require_clean_git or args.verify_remote:
            _git_check(
                args.mirror_root,
                verify_remote=args.verify_remote,
                remote=args.remote,
                branch=args.branch,
            )
        print("Overleaf sync audit passed.")
    elif args.command == "prepare-submission":
        sync_bibliography(args.latex_root)
        print(prepare_submission(args.latex_root, args.output_root, zip_output=args.zip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
