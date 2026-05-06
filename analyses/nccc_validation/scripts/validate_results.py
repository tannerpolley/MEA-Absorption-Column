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
        _check_profile_index,
        _check_final_tables_do_not_point_to_removed_docs_paths,
        _check_latex_paths,
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
        FIGURES / "c_case_thermo_benchmark.pdf",
        FIGURES / "staged_kcase_capture_error.pdf",
        FIGURES / "staged_epcsaft_smoke_capture_error.pdf",
        FINAL / "reports" / "validation_summary.md",
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


def _check_latex_paths() -> None:
    for tex_name in ("main.tex", "revised_benchmark_results.tex"):
        text = (DOCS_LATEX / tex_name).read_text(encoding="utf-8")
        if "docs/benchmark_figures" in text or "benchmark_figures/" in text:
            raise AssertionError(f"{tex_name} still references docs benchmark figure paths.")
    main = (DOCS_LATEX / "main.tex").read_text(encoding="utf-8")
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main):
        target = match.group(1)
        if target.startswith("Figures/") or target.startswith("figs/"):
            continue
        candidates = [
            (DOCS_LATEX / target).resolve(),
            (DOCS_LATEX / ".." / target).resolve(),
            (DOCS_LATEX / ".." / ".." / target).resolve(),
        ]
        if not any(path.exists() for path in candidates):
            raise AssertionError(f"LaTeX figure path does not resolve from docs/latex: {target}")


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
