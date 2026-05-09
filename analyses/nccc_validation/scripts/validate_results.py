from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
FINAL = ANALYSIS / "results" / "final"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"
PROFILES = FINAL / "profiles"
DOCS_LATEX = ROOT / "docs" / "latex"


def main() -> int:
    checks = [
        _check_required_files,
        _check_c_case_benchmark,
        _check_staged_kcase_benchmark,
        _check_diagnostic_tables,
        _check_accuracy_credibility_artifacts,
        _check_appendix_c_temperature_profiles,
        _check_profile_index,
        _check_referenced_profile_csv_dirs,
        _check_final_tables_do_not_point_to_removed_docs_paths,
        _check_latex_paths,
        _check_latex_pdf_is_current,
        _check_old_docs_benchmark_removed,
    ]
    for check in checks:
        check()
    print("NCCC validation analysis artifacts are internally consistent.")
    return 0


def _check_required_files() -> None:
    required = [
        TABLES / "verified_c_case_thermo_benchmark.csv",
        TABLES / "verified_staged_kcase_benchmark.csv",
        TABLES / "staged_epcsaft_smoke.csv",
        TABLES / "staged_epcsaft_recovery_probe.csv",
        TABLES / "staged_epcsaft_k2_blend_probe.csv",
        TABLES / "kcase_sensitivity_recoveries.csv",
        TABLES / "kcase_unresolved_diagnostics.csv",
        TABLES / "validation_evidence_registry.csv",
        TABLES / "primary_validation_gate_summary.csv",
        TABLES / "calibration_holdout_metrics.csv",
        TABLES / "error_regime_capture_data.csv",
        TABLES / "staged_epcsaft_reliability_summary.csv",
        TABLES / "intercooled_temperature_profile_k3.csv",
        TABLES / "morgan2020_case1a_measured_intercooled_profile.csv",
        TABLES / "appendix_c_temperature_profile_index.csv",
        TABLES / "nccc_legacy_k_case_name_crosswalk.csv",
        FIGURES / "c_case_thermo_benchmark.pdf",
        FIGURES / "staged_kcase_capture_error.pdf",
        FIGURES / "staged_epcsaft_smoke_capture_error.pdf",
        FIGURES / "calibration_uncertainty_band.pdf",
        FIGURES / "error_regime_capture_error.pdf",
        FIGURES / "staged_epcsaft_reliability.pdf",
        FIGURES / "intercooled_temperature_profile_k3.pdf",
        FIGURES / "morgan2020_case1a_measured_intercooled_profile.pdf",
        FIGURES / "intercooled_temperature_profile_comparison.pdf",
        FIGURES / "appendix_c_temperature_profiles.pdf",
        ANALYSIS / "data" / "input" / "morgan2020_appendix_c_case1a_absorber_temperature_profile.csv",
        ANALYSIS / "data" / "input" / "morgan2020_appendix_c_absorber_temperature_profiles.csv",
        FINAL / "reports" / "validation_summary.md",
        ANALYSIS / "scripts" / "run_case_profile.py",
        ANALYSIS / "scripts" / "generate_clean_profile_csvs.py",
        ANALYSIS / "scripts" / "generate_accuracy_credibility_artifacts.py",
        ANALYSIS / "scripts" / "render_appendix_c_temperature_profiles.py",
    ]
    _require_existing(required)


def _check_c_case_benchmark() -> None:
    data = pd.read_csv(TABLES / "verified_c_case_thermo_benchmark.csv")
    _require_columns(data, ["case_id", "thermo_model", "success", "capture_error_pct", "temperature_rmse_K"])
    if data["case_id"].nunique() != 7:
        raise AssertionError("Expected 7 one-bed C cases.")
    counts = data.groupby("thermo_model")["case_id"].nunique().to_dict()
    if counts.get("ideal_henry") != 7 or counts.get("epcsaft_neutral") != 7:
        raise AssertionError(f"Expected both thermo lanes to cover 7 C cases, got {counts!r}.")
    if not data["success"].astype(str).str.lower().eq("true").all():
        raise AssertionError("All verified C-case rows must be successful.")


def _check_staged_kcase_benchmark() -> None:
    data = pd.read_csv(TABLES / "verified_staged_kcase_benchmark.csv")
    _require_columns(
        data,
        [
            "case_id",
            "thermo_model",
            "success",
            "capture_error_pct",
            "boundary_residual_norm",
            "beds",
            "intercoolers",
            "staged_beds",
            "intercooler_assumption",
            "message",
        ],
    )
    if len(data) != 19 or data["case_id"].nunique() != 19:
        raise AssertionError("Expected 19 primary staged Henry K-case rows.")
    if set(data["thermo_model"]) != {"ideal_henry"}:
        raise AssertionError("Primary staged K-case benchmark should contain the Henry lane only.")
    if not data["success"].astype(str).str.lower().eq("true").all():
        raise AssertionError("Primary staged K-case rows must be accepted rows.")
    if data["capture_error_pct"].abs().max() > 8:
        raise AssertionError("Primary staged K-case rows must remain inside the 8 percentage-point capture gate.")


def _check_diagnostic_tables() -> None:
    recovery = pd.read_csv(TABLES / "kcase_sensitivity_recoveries.csv")
    unresolved = pd.read_csv(TABLES / "kcase_unresolved_diagnostics.csv")
    epcsaft = pd.read_csv(TABLES / "staged_epcsaft_smoke.csv")
    if recovery.empty or unresolved.empty or epcsaft.empty:
        raise AssertionError("Diagnostic/recovery tables must be present and non-empty.")
    primary_cases = set(pd.read_csv(TABLES / "verified_staged_kcase_benchmark.csv")["case_id"])
    unresolved_cases = set(unresolved["case_id"])
    if unresolved_cases & primary_cases:
        raise AssertionError("Unresolved diagnostic rows must not be mixed into the primary accepted K-case table.")


def _check_accuracy_credibility_artifacts() -> None:
    registry = pd.read_csv(TABLES / "validation_evidence_registry.csv")
    _require_columns(
        registry,
        [
            "evidence_group",
            "evidence_class",
            "primary_validation",
            "no_case_specific_tuning",
            "rows",
            "accepted_rows",
        ],
    )
    if not {"primary", "recovery", "diagnostic"}.issubset(set(registry["evidence_class"])):
        raise AssertionError("Validation registry must distinguish primary, recovery, and diagnostic evidence.")

    gate = pd.read_csv(TABLES / "primary_validation_gate_summary.csv")
    _require_columns(gate, ["evidence_group", "thermo_model", "gate_pass"])
    if not gate["gate_pass"].astype(str).str.lower().eq("true").any():
        raise AssertionError("No-case-specific-tuning gate must identify at least one passing evidence group.")
    if "staged/intercooled K cases" not in set(gate["evidence_group"]):
        raise AssertionError("No-case-specific-tuning gate must report the staged K-case evidence group.")

    metrics = pd.read_csv(TABLES / "calibration_holdout_metrics.csv")
    _require_columns(metrics, ["split", "raw_capture_mae_pct", "calibrated_capture_mae_pct"])
    if set(metrics["split"]) != {"train", "holdout"}:
        raise AssertionError("Calibration screen must report train and holdout rows.")
    if not (metrics["calibrated_capture_mae_pct"] <= metrics["raw_capture_mae_pct"]).all():
        raise AssertionError("Calibration screen should not degrade the reported low-hanging MAE metric.")

    reliability = pd.read_csv(TABLES / "staged_epcsaft_reliability_summary.csv")
    _require_columns(reliability, ["source", "outcome", "rows", "fraction"])
    if "accepted" not in set(reliability["outcome"]):
        raise AssertionError("Staged ePC-SAFT reliability summary must include accepted rows.")


def _check_appendix_c_temperature_profiles() -> None:
    measured = pd.read_csv(ANALYSIS / "data" / "input" / "morgan2020_appendix_c_absorber_temperature_profiles.csv")
    _require_columns(
        measured,
        [
            "case_id",
            "relative_position_top_to_bottom",
            "temperature_C",
            "beds",
            "intercoolers",
            "source",
        ],
    )
    expected_cases = {f"{i}A" for i in range(1, 16)} | {"1B", "2B", "3B"} | {f"{i}C" for i in range(1, 8)} | {
        f"{i}D" for i in range(1, 5)
    }
    if set(measured["case_id"]) != expected_cases:
        raise AssertionError("Appendix C absorber profile table must contain the expected NCCC case labels.")

    index = pd.read_csv(TABLES / "appendix_c_temperature_profile_index.csv")
    _require_columns(
        index,
        ["case_id", "beds", "intercoolers", "model_status", "model_quality", "model_source", "source_row", "case_plot_png", "message"],
    )
    if set(index["case_id"]) != expected_cases:
        raise AssertionError("Appendix C profile index must cover every absorber temperature-profile case.")
    if len(index) != 29:
        raise AssertionError("Appendix C profile index must contain exactly 29 absorber profile plots.")
    c_cases = {f"{i}C" for i in range(1, 8)}
    c_overlays = index[index["case_id"].isin(c_cases)]
    if not c_overlays["model_status"].astype(str).eq("profile_exported").all():
        raise AssertionError("All one-bed C cases should use exported true model profiles in the Appendix C review PDF.")
    if index.astype(str).apply(lambda col: col.str.contains("measured_data_surrogate|interpolated_model_placeholder", case=False, regex=True)).any().any():
        raise AssertionError("Appendix C profile index must not use generated placeholder curves as model evidence.")
    for raw_path in index["case_plot_png"]:
        if not (ROOT / raw_path).exists():
            raise AssertionError(f"Appendix C case plot does not exist: {raw_path}")
    page_dir = FIGURES / "appendix_c_temperature_profile_pages"
    for page_number in range(1, 9):
        if not (page_dir / f"page_{page_number:02d}.png").exists():
            raise AssertionError(f"Missing Appendix C profile page preview {page_number}.")


def _check_profile_index() -> None:
    index_path = TABLES / "clean_temperature_profile_index.csv"
    if not index_path.exists():
        raise AssertionError("Missing clean profile index. Run collect_clean_profiles.py --collect-existing.")
    data = pd.read_csv(index_path)
    _require_columns(data, ["case_id", "thermo_model", "profile_png", "clean_profile", "caveat"])
    if data.empty:
        raise AssertionError("Clean profile index is empty.")
    for raw_path in data["profile_png"]:
        path = ROOT / raw_path
        if not path.exists():
            raise AssertionError(f"Profile PNG listed in index does not exist: {raw_path}")


def _check_referenced_profile_csv_dirs() -> None:
    for table_path in sorted(TABLES.glob("*.csv")):
        data = pd.read_csv(table_path)
        if "profile_csv_dir" not in data.columns:
            continue
        for raw_path in data["profile_csv_dir"].dropna():
            raw_path = str(raw_path)
            if not raw_path:
                continue
            path = ROOT / raw_path
            if not path.exists():
                raise AssertionError(f"Profile CSV directory listed in {table_path.name} does not exist: {raw_path}")
            if not (path / "profile_manifest.json").exists():
                raise AssertionError(f"Profile CSV directory lacks manifest: {raw_path}")


def _check_latex_paths() -> None:
    for tex_name in (
        "main.tex",
        "revised_benchmark_results.tex",
        "benchmark_results_section.tex",
    ):
        text = (DOCS_LATEX / tex_name).read_text(encoding="utf-8")
        if "docs/benchmark_figures" in text or "benchmark_figures/" in text:
            raise AssertionError(f"{tex_name} still references docs benchmark figure paths.")
    for tex_name in ("main.tex",):
        main = (DOCS_LATEX / tex_name).read_text(encoding="utf-8")
        for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main):
            target = match.group(1)
            if target.startswith("Figures/") or target.startswith("figs/"):
                continue
            if not any(path.exists() for path in _latex_graphic_candidates(target)):
                raise AssertionError(f"LaTeX figure path does not resolve from {tex_name}: {target}")


def _check_latex_pdf_is_current() -> None:
    _check_one_latex_pdf_is_current("main.tex", "main.pdf", "docs\\latex\\build_main.ps1")


def _check_one_latex_pdf_is_current(tex_name: str, pdf_name: str, build_command: str) -> None:
    root_tex = DOCS_LATEX / tex_name
    pdf = DOCS_LATEX / pdf_name
    if not pdf.exists():
        raise AssertionError(f"Missing docs/latex/{pdf_name}. Run {build_command}.")
    sources = set()
    for pattern in ("*.bst", "*.cls", "*.sty"):
        sources.update(DOCS_LATEX.glob(pattern))
    for tex_path in _tex_dependency_closure(root_tex):
        sources.add(tex_path)
        text = tex_path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\bibliography\{([^}]+)\}", text):
            for bib_name in match.group(1).split(","):
                bib_path = (DOCS_LATEX / bib_name.strip()).with_suffix(".bib")
                if bib_path.exists():
                    sources.add(bib_path)
        for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            for candidate in _latex_graphic_candidates(match.group(1)):
                if candidate.exists():
                    sources.add(candidate)
                    break

    newer = [path for path in sources if path.exists() and path.stat().st_mtime > pdf.stat().st_mtime]
    if newer:
        names = "\n".join(str(path.relative_to(ROOT)) for path in sorted(newer))
        raise AssertionError(
            f"docs/latex/{pdf_name} is older than manuscript inputs. Run {build_command}.\n{names}"
        )


def _latex_graphic_candidates(target: str) -> list[Path]:
    return [
        (DOCS_LATEX / target).resolve(),
        (DOCS_LATEX / ".." / target).resolve(),
        (DOCS_LATEX / ".." / ".." / target).resolve(),
        (ROOT / target).resolve(),
    ]


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
            candidate = (DOCS_LATEX / target).resolve()
            if candidate.suffix != ".tex":
                candidate = candidate.with_suffix(".tex")
            pending.append(candidate)
    return seen


def _check_final_tables_do_not_point_to_removed_docs_paths() -> None:
    for path in sorted(TABLES.glob("*.csv")):
        text = path.read_text(encoding="utf-8")
        if "docs/benchmark_figures" in text or "benchmark_figures/" in text:
            raise AssertionError(f"Final table still points to removed docs benchmark paths: {path.name}")


def _check_old_docs_benchmark_removed() -> None:
    if (ROOT / "docs" / "benchmark_figures").exists():
        raise AssertionError("Old docs/benchmark_figures directory should not exist after migration.")


def _require_existing(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise AssertionError("Missing required analysis artifacts:\n" + "\n".join(missing))


def _require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise AssertionError(f"Missing required columns: {missing!r}")


if __name__ == "__main__":
    raise SystemExit(main())
