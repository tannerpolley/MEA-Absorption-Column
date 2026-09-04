from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "docs" / "latex" / "scripts" / "latex_workflows.py"
SPEC = importlib.util.spec_from_file_location("latex_workflows", SCRIPT)
assert SPEC and SPEC.loader
latex_workflows = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latex_workflows)


def _write(path: Path, text: str = "content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_sync_figures_copies_required_outputs_and_validates_references(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    latex = repo / "docs" / "latex"
    for source, _destination in latex_workflows.FIGURE_COPIES:
        _write(repo / source, str(source))
    _write(
        latex / "main.tex",
        r"\includegraphics{figures/nccc-one-bed-thermo-benchmark.pdf}",
    )

    copied = latex_workflows.sync_figures(repo, latex)

    assert len(copied) == len(latex_workflows.FIGURE_COPIES)
    for source, destination in latex_workflows.FIGURE_COPIES:
        assert (latex / "figures" / destination).read_bytes() == (repo / source).read_bytes()


def test_sync_projection_replaces_mirror_and_preserves_git(tmp_path: Path) -> None:
    source = tmp_path / "latex"
    mirror = tmp_path / "mirror"
    _write(source / "main.tex", "new")
    _write(source / "sections" / "body.tex", "body")
    _write(source / "scripts" / "ignored.py", "ignored")
    _write(source / "builds" / "ignored.pdf", "ignored")
    _write(mirror / ".git" / "HEAD", "ref: refs/heads/main")
    _write(mirror / "stale.txt", "stale")

    latex_workflows.sync_projection(source, mirror)

    assert (mirror / ".git" / "HEAD").exists()
    assert (mirror / "main.tex").read_text(encoding="utf-8") == "new"
    assert (mirror / "sections" / "body.tex").exists()
    assert not (mirror / "stale.txt").exists()
    assert not (mirror / "scripts").exists()
    assert not (mirror / "builds").exists()
    latex_workflows.audit_projection(source, mirror)


def test_audit_projection_rejects_changed_content(tmp_path: Path) -> None:
    source = tmp_path / "latex"
    mirror = tmp_path / "mirror"
    _write(source / "main.tex", "source")
    _write(mirror / ".git" / "HEAD", "ref: refs/heads/main")
    _write(mirror / "main.tex", "changed")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        latex_workflows.audit_projection(source, mirror)


def test_bibliography_sync_updates_clean_snapshot_and_protects_dirty_edits(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    latex = repo / "docs" / "latex"
    central = tmp_path / "Papers" / "references.bib"
    target = latex / "references.bib"
    _write(central, "@article{current}\n")
    _write(target, "@article{old}\n")
    repo.mkdir(exist_ok=True)
    for command in (
        ["git", "init"],
        ["git", "add", "docs/latex/references.bib"],
        [
            "git",
            "-c",
            "user.name=Codex Test",
            "-c",
            "user.email=codex-test@example.invalid",
            "commit",
            "-m",
            "baseline",
        ],
    ):
        latex_workflows.subprocess.run(command, cwd=repo, check=True, capture_output=True)

    latex_workflows.sync_bibliography(latex, central)
    assert target.read_bytes() == central.read_bytes()

    _write(target, "@article{manual-edit}\n")
    with pytest.raises(RuntimeError, match="dirty bibliography snapshot"):
        latex_workflows.sync_bibliography(latex, central)

    latex_workflows.sync_bibliography(
        latex,
        central,
        replace_reviewed_dirty=True,
    )
    assert latex_workflows.sync_bibliography(latex, central) == target


def test_prepare_submission_flattens_tex_paths_and_copies_referenced_figure(
    tmp_path: Path,
) -> None:
    latex = tmp_path / "latex"
    output = latex / "builds" / "elsevier_submission_flat"
    _write(
        latex / "main.tex",
        r"\input{sections/body}\includegraphics{figures/result.pdf}",
    )
    _write(latex / "sections" / "body.tex", r"\input{tables/results}")
    _write(latex / "tables" / "results.tex", "table")
    _write(latex / "figures" / "result.pdf", "figure")
    _write(latex / "references.bib", "@article{x}")
    _write(latex / "builds" / "main.pdf", "pdf")

    latex_workflows.prepare_submission(latex, output)

    assert "sections/" not in (output / "main.tex").read_text(encoding="utf-8")
    assert "tables/" not in (output / "body.tex").read_text(encoding="utf-8")
    assert (output / "result.pdf").exists()
    assert (output / "references.bib").exists()
    assert (output / "main.pdf").exists()
