from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from analyze_issue17_enhancement_comparison import (
    RESULT_SCIENTIFIC_HASH_EXCLUDED_COLUMNS,
    stable_csv_sha256,
)


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
FINAL = ANALYSIS / "results" / "final"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"
PROFILES = FINAL / "profiles"
DOCS_LATEX = ROOT / "docs" / "latex"
ISSUE17_INPUTS = ANALYSIS / "inputs" / "issue17_enhancement_comparison"
ISSUE17_TABLES = [
    TABLES / "issue17_fugacity_only_enhancement_formulations.csv",
    TABLES / "issue17_fugacity_only_enhancement_aggregates.csv",
    TABLES / "issue17_enhancement_stage_outcomes.csv",
    TABLES / "issue17_fugacity_only_enhancement_summary.json",
]
ISSUE17_FIGURES = [
    FIGURES / "issue17_axial_enhancement.pdf",
    FIGURES / "issue17_axial_flux.pdf",
    FIGURES / "issue17_parity_to_gaspar_implicit.pdf",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue17-only", action="store_true")
    args = parser.parse_args()
    if args.issue17_only:
        _check_issue17_enhancement_comparison()
        print("Issue 17 retained outputs are internally consistent.")
        return 0

    checks = [
        _check_issue17_enhancement_comparison,
        _check_required_files,
        _check_c_case_benchmark,
        _check_full_species_ionic_sweep,
        _check_accuracy_credibility_tables,
        _check_method_contrast,
        _check_epcsaft_v02_validation,
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


def _check_issue17_enhancement_comparison() -> None:
    _require_existing(ISSUE17_TABLES + ISSUE17_FIGURES)
    result = pd.read_csv(ISSUE17_TABLES[0])
    aggregates = pd.read_csv(ISSUE17_TABLES[1])
    stages = pd.read_csv(ISSUE17_TABLES[2])
    summary = json.loads(ISSUE17_TABLES[3].read_text(encoding="utf-8"))
    expected_positions = [index / 20.0 for index in range(21)]
    expected_formulations = {
        "EF-GF-IMPLICIT",
        "EF-AOP-78-PUBLISHED-MEA",
        "EF-AOP-73-CORRECTED-MEA",
        "EF-CURRENT",
    }
    positions = result["Position"].drop_duplicates().tolist()
    if len(result) != 84 or len(positions) != 21 or any(
        abs(actual - expected) > 1.0e-15
        for actual, expected in zip(positions, expected_positions, strict=True)
    ):
        raise AssertionError("Issue 17 must retain 84 rows at positions 0.00, 0.05, ..., 1.00.")
    if set(result["formulation"]) != expected_formulations:
        raise AssertionError("Issue 17 retained table does not contain exactly four formulations.")
    if not result.groupby("Position").size().eq(4).all():
        raise AssertionError("Each Issue 17 position must contain exactly four formulation rows.")
    explicit = result.loc[result["formulation"].ne("EF-GF-IMPLICIT")]
    if len(explicit) != 63 or not explicit["scalar_reference_relative_error"].between(
        0.0, 1.0e-12
    ).all():
        raise AssertionError("All 63 explicit Issue 17 rows must agree with scalar transcription.")
    if not (
        summary["state_count"] == 21
        and summary["formulation_count"] == 4
        and summary["admitted_row_count"] == 84
        and summary["numerical_gate_pass"] is True
        and summary["physical_gate_pass"] is False
        and summary["stage4_allowed"] is False
    ):
        raise AssertionError("Issue 17 summary gates do not match the retained negative result.")
    try:
        pd.testing.assert_frame_equal(
            aggregates,
            pd.DataFrame(summary["aggregates"]),
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    except AssertionError as error:
        raise AssertionError("Issue 17 aggregate table is stale relative to the summary.") from error
    blocked = stages.loc[stages["stage"].isin([4, 5])]
    if len(blocked) != 2 or not (
        blocked["attempted"].eq("no").all()
        and blocked["stopped_by"].eq("physical_check").all()
        and blocked["outcome"].eq("not_attempted").all()
    ):
        raise AssertionError("Issue 17 Stages 4 and 5 must stop at the Stage 2 physical check.")

    scientific_hash = stable_csv_sha256(
        ISSUE17_TABLES[0], RESULT_SCIENTIFIC_HASH_EXCLUDED_COLUMNS
    )
    if summary.get("result_table_scientific_sha256") != scientific_hash:
        raise AssertionError("Issue 17 summary is stale relative to the retained scientific values.")
    figure_marker = f"issue17_result_scientific_sha256={scientific_hash}".encode()
    for figure in ISSUE17_FIGURES:
        if figure_marker not in figure.read_bytes():
            raise AssertionError(f"Issue 17 figure is stale: {figure.name}")
    for path in list(ISSUE17_INPUTS.glob("*")) + ISSUE17_TABLES + ISSUE17_FIGURES:
        content = path.read_bytes()
        if b"/home/" in content or b".codex/worktrees" in content:
            raise AssertionError(f"Issue 17 retained file contains a machine-local path: {path.name}")


def _check_required_files() -> None:
    required = [
        TABLES / "nccc_one_bed_accepted_results.csv",
        TABLES / "nccc_one_bed_accepted_summary.csv",
        TABLES / "nccc_one_bed_all_attempted_results.csv",
        TABLES / "nccc_one_bed_case_scope.csv",
        TABLES / "nccc_2017_epcsaft_temperature_overlay_metrics.csv",
        TABLES / "nccc_2017_epcsaft_temperature_profile_index.csv",
        TABLES / "validation_evidence_registry.csv",
        TABLES / "primary_validation_gate.csv",
        TABLES / "primary_validation_gate_summary.csv",
        TABLES / "method_case_contrast.csv",
        TABLES / "full_species_ionic_2017_c_case_sweep.csv",
        TABLES / "epcsaft_v02_contribution_table.csv",
        TABLES / "epcsaft_v02_column_row.csv",
        FIGURES / "nccc_one_bed_thermo_benchmark.pdf",
        FIGURES
        / "nccc_2017_epcsaft_temperature_overlays"
        / "nccc_2017_epcsaft_temperature_overlay_contact_sheet.png",
        FIGURES / "method_case_solver_contrast.pdf",
        FINAL / "reports" / "validation_summary.md",
        ANALYSIS / "scripts" / "run_case_profile.py",
        ANALYSIS / "scripts" / "generate_clean_profile_csvs.py",
        ANALYSIS / "scripts" / "generate_accuracy_credibility_artifacts.py",
    ]
    _require_existing(required)


def _check_epcsaft_v02_validation() -> None:
    contributions = pd.read_csv(TABLES / "epcsaft_v02_contribution_table.csv")
    _require_columns(
        contributions,
        [
            "mixture_kind",
            "co2_fugacity_Pa",
            "a_ion",
            "a_born",
            "contribution_check_pass",
            "dataset_content_sha256",
            "parameter_document_content_sha256",
            "provider_parameter_fingerprint_scope",
            "engine_wheel_sha256",
            "repository_base_commit",
            "reproduction_command",
        ],
    )
    if set(contributions["mixture_kind"]) != {"neutral", "ionic"}:
        raise AssertionError("Current ePC-SAFT contribution evidence must contain neutral and ionic rows only.")
    neutral = contributions.loc[contributions["mixture_kind"] == "neutral"].iloc[0]
    ionic = contributions.loc[contributions["mixture_kind"] == "ionic"].iloc[0]
    if abs(float(neutral["a_ion"])) > 1e-12 or abs(float(neutral["a_born"])) > 1e-12:
        raise AssertionError("Neutral ePC-SAFT state has nonzero electrolyte-only contributions.")
    if abs(float(ionic["a_ion"])) <= 1e-8 or abs(float(ionic["a_born"])) <= 1e-8:
        raise AssertionError("Ionic ePC-SAFT state does not activate both ion and Born contributions.")
    if float(ionic["co2_fugacity_Pa"]) <= 0.0:
        raise AssertionError("Ionic ePC-SAFT state has nonpositive CO2 fugacity.")
    if not contributions["contribution_check_pass"].astype(str).str.lower().eq("true").all():
        raise AssertionError("Current ePC-SAFT contribution check contains a failed row.")
    if set(contributions["provider_parameter_fingerprint_scope"]) != {
        "checkout-path-local; not portable provenance"
    }:
        raise AssertionError("Provider parameter fingerprints must be labeled checkout-path-local.")

    column = pd.read_csv(TABLES / "epcsaft_v02_column_row.csv")
    _require_columns(column, ["claim_strength", "run_directory_at_generation"])
    if "retained_run_directory" in column.columns:
        raise AssertionError("The disposable generation run directory must not be labeled retained.")
    if len(column) != 1:
        raise AssertionError("Current ePC-SAFT column evidence must contain exactly one row.")
    row = column.iloc[0]
    if str(row["thermo_model"]) != "epcsaft_ionic" or str(row["chemical_equilibrium_model"]) != "legacy":
        raise AssertionError("Current ePC-SAFT column row must use the ionic fixed-chemistry lane.")
    if not str(row["validation_pass"]).lower() == "true":
        raise AssertionError("Current ePC-SAFT column row did not pass its retained checks.")
    if str(row["stopped_by"]) != "none" or str(row["outcome"]) != "evaluated":
        raise AssertionError("Current ePC-SAFT column row has an incomplete status.")
    if not 0.0 <= float(row["capture_pct"]) <= 100.0:
        raise AssertionError("Current ePC-SAFT column capture is outside physical bounds.")
    if float(row["boundary_residual_norm"]) > 1.0:
        raise AssertionError("Current ePC-SAFT column boundary residual exceeds 1.0.")
    if max(
        float(row["co2_conservation_relative_residual"]),
        float(row["h2o_conservation_relative_residual"]),
    ) > float(row["conservation_relative_tolerance"]):
        raise AssertionError("Current ePC-SAFT column row fails component conservation.")
    if int(row["invalid_state_count"]) or int(row["guard_penalty_count"]):
        raise AssertionError("Current ePC-SAFT column row contains invalid or guarded states.")
    if row["dataset_content_sha256"] != ionic["dataset_content_sha256"] or row[
        "parameter_document_content_sha256"
    ] != ionic["parameter_document_content_sha256"]:
        raise AssertionError("Column and ionic-state parameter content identities disagree.")


def _check_c_case_benchmark() -> None:
    data = pd.read_csv(TABLES / "nccc_one_bed_accepted_results.csv")
    _require_columns(
        data,
        [
            "case_id",
            "thermo_model",
            "success",
            "capture_error_pct",
            "runtime_s",
            "campaign_year",
        ],
    )
    expected_cases = {"K18", "K19", "1C", "2C", "3C", "4C", "5C", "6C", "7C"}
    got_cases = set(data["case_id"].astype(str))
    if got_cases != expected_cases:
        raise AssertionError(
            f"Expected accepted one-bed cases {sorted(expected_cases)!r}, got {sorted(got_cases)!r}."
        )
    counts = data.groupby("thermo_model")["case_id"].nunique().to_dict()
    if counts.get("ideal_henry") != 9 or counts.get("epcsaft_ionic") != 9:
        raise AssertionError(
            f"Expected both thermo lanes to cover 9 accepted one-bed cases, got {counts!r}."
        )
    if not data["success"].astype(str).str.lower().eq("true").all():
        raise AssertionError("All accepted one-bed rows must be successful.")

    attempted = pd.read_csv(TABLES / "nccc_one_bed_all_attempted_results.csv")
    _require_columns(
        attempted,
        [
            "case_id",
            "thermo_model",
            "success",
            "capture_pct",
            "runtime_s",
            "boundary_residual_norm",
        ],
    )
    accepted_by_gate = attempted.loc[
        attempted["success"].astype(str).str.lower().eq("true")
        & attempted["boundary_residual_norm"].le(1.0)
        & attempted["capture_pct"].between(0.0, 100.0, inclusive="both")
        & attempted["runtime_s"].le(90.0),
        ["case_id", "thermo_model"],
    ]
    expected_pairs = set(accepted_by_gate.itertuples(index=False, name=None))
    accepted_pairs = set(
        data[["case_id", "thermo_model"]].itertuples(index=False, name=None)
    )
    if accepted_pairs != expected_pairs:
        raise AssertionError(
            "Accepted one-bed artifact is not the row-level gate applied to all attempted rows."
        )


def _check_full_species_ionic_sweep() -> None:
    data = pd.read_csv(TABLES / "full_species_ionic_2017_c_case_sweep.csv")
    _require_columns(
        data,
        [
            "case_id",
            "nccc_dataset",
            "data_type",
            "thermo_model",
            "epcsaft_dataset_name",
            "epcsaft_config",
            "success",
            "runtime_s",
            "co2_capture_pct",
            "target_co2_capture_pct",
            "capture_error_pct_pt",
            "invalid_state_count",
            "guard_penalty_count",
            "epcsaft_chemistry_solve_s",
            "epcsaft_chemistry_max_mass_residual",
            "epcsaft_chemistry_max_reaction_residual",
            "epcsaft_chemistry_max_charge_residual",
            "epcsaft_chemistry_failed_count",
            "raw_result_csv",
            "benchmark_command",
        ],
    )
    expected_cases = {"1C", "2C", "3C", "4C", "5C", "6C", "7C"}
    got_cases = set(data["case_id"].astype(str))
    if got_cases != expected_cases:
        raise AssertionError(
            f"Expected full-species 2017 C sweep cases {sorted(expected_cases)!r}, got {sorted(got_cases)!r}."
        )
    if set(data["nccc_dataset"].astype(str)) != {"2017"} or set(
        data["data_type"].astype(str)
    ) != {"mass"}:
        raise AssertionError(
            "Full-species ionic sweep must use the corrected 2017 mass-input C-case data."
        )
    if set(data["thermo_model"].astype(str)) != {
        "epcsaft_reactive_nine_activity_rebased"
    }:
        raise AssertionError(
            "Full-species ionic sweep must use the nine-species activity-rebased ePC-SAFT model."
        )
    if set(data["epcsaft_dataset_name"].astype(str)) != {"MEA_CO2_H2O_ionic_fit"}:
        raise AssertionError(
            "Full-species ionic sweep must use the MEA_CO2_H2O_ionic_fit parameter dataset."
        )
    if set(data["epcsaft_config"].astype(str)) != {
        "2025_Figiel_empirical_fitted_Born_SSM_DS"
    }:
        raise AssertionError(
            "Full-species ionic sweep must use the selected 2025 Figiel ePC-SAFT configuration."
        )
    if not data["success"].astype(str).str.lower().eq("true").all():
        raise AssertionError(
            "Every full-species ionic C-case row is expected to converge."
        )
    if (
        data[
            [
                "invalid_state_count",
                "guard_penalty_count",
                "epcsaft_chemistry_failed_count",
            ]
        ]
        .sum()
        .sum()
        != 0
    ):
        raise AssertionError(
            "Full-species ionic sweep should have zero invalid states, guard penalties, and chemistry failures."
        )
    if data["epcsaft_chemistry_max_mass_residual"].max() > 1e-7:
        raise AssertionError("Full-species ionic sweep mass residual exceeds 1e-7.")
    if data["epcsaft_chemistry_max_reaction_residual"].max() > 1e-7:
        raise AssertionError("Full-species ionic sweep reaction residual exceeds 1e-7.")
    if data["epcsaft_chemistry_max_charge_residual"].max() > 1e-10:
        raise AssertionError("Full-species ionic sweep charge residual exceeds 1e-10.")
    if data["runtime_s"].mean() < 120.0:
        raise AssertionError(
            "Full-species ionic sweep no longer supports the documented slow-path timing boundary."
        )


def _check_accuracy_credibility_tables() -> None:
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
    if not set(registry["evidence_class"]).issubset(
        {"primary", "diagnostic", "recovery"}
    ):
        raise AssertionError("Validation registry has an unexpected evidence class.")

    gate = pd.read_csv(TABLES / "primary_validation_gate.csv")
    _require_columns(
        gate,
        ["case_id", "thermo_model", "primary_validation", "no_case_specific_tuning"],
    )
    if gate["case_id"].astype(str).str.startswith("K").any():
        raise AssertionError(
            "Main-branch primary validation gate must not include K-case rows."
        )
    if not gate["primary_validation"].astype(str).str.lower().eq("true").all():
        raise AssertionError(
            "All gate rows are expected to be primary one-bed C validation rows."
        )
    if not gate["no_case_specific_tuning"].astype(str).str.lower().eq("true").all():
        raise AssertionError(
            "Primary validation rows must pass the no-case-specific-tuning gate."
        )

    summary = pd.read_csv(TABLES / "primary_validation_gate_summary.csv")
    _require_columns(summary, ["evidence_group", "thermo_model", "gate_pass"])
    if not summary["gate_pass"].astype(str).str.lower().eq("true").all():
        raise AssertionError("Primary validation gate summary contains a failed gate.")

    temp = pd.read_csv(TABLES / "nccc_2017_epcsaft_temperature_overlay_metrics.csv")
    _require_columns(temp, ["case_id", "thermo_model", "capture_error_pct", "plot_png"])
    if set(temp["case_id"].astype(str)) != {"1C", "2C", "3C", "4C", "5C", "6C"}:
        raise AssertionError(
            "Temperature-profile gallery should include accepted 2017 one-bed C cases 1C through 6C."
        )
    if set(temp["thermo_model"].astype(str)) != {"epcsaft_ionic"}:
        raise AssertionError("Temperature-profile gallery should be ePC-SAFT only.")


def _check_method_contrast() -> None:
    data = pd.read_csv(TABLES / "method_case_contrast.csv")
    _require_columns(data, ["scenario", "case_id", "method", "success", "runtime_s"])
    if data.empty:
        raise AssertionError("Method contrast table is empty.")
    if not {"Shooting", "Collocation BVP", "Finite difference"}.issubset(
        set(data["method"])
    ):
        raise AssertionError(
            "Method contrast table must include shooting, collocation BVP, and finite difference rows."
        )
    if (
        data["scenario"]
        .astype(str)
        .str.contains("K case|intercool|staged", case=False, regex=True)
        .any()
    ):
        raise AssertionError(
            "Main-branch method contrast must not depend on staged/intercooled K-case evidence."
        )


def _check_profile_index() -> None:
    index_path = TABLES / "nccc_2017_epcsaft_temperature_profile_index.csv"
    if not index_path.exists():
        raise AssertionError(
            "Missing 2017 ePC-SAFT temperature profile index. Run render_c_case_campaign_temperature_gallery.py."
        )
    data = pd.read_csv(index_path)
    _require_columns(
        data, ["case_id", "thermo_model", "profile_png", "clean_profile", "caveat"]
    )
    if data.empty:
        raise AssertionError("Clean profile index is empty.")
    for raw_path in data["profile_png"]:
        path = ROOT / raw_path
        if not path.exists():
            raise AssertionError(
                f"Profile PNG listed in index does not exist: {raw_path}"
            )


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
                raise AssertionError(
                    f"Profile CSV directory listed in {table_path.name} does not exist: {raw_path}"
                )
            if not (path / "profile_manifest.json").exists():
                raise AssertionError(
                    f"Profile CSV directory lacks manifest: {raw_path}"
                )


def _check_latex_paths() -> None:
    for tex_path in _tex_dependency_closure(DOCS_LATEX / "main.tex"):
        text = tex_path.read_text(encoding="utf-8")
        if "docs/benchmark_figures" in text or "benchmark_figures/" in text:
            raise AssertionError(
                f"{tex_path.relative_to(ROOT)} still references docs benchmark figure paths."
            )
        for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            target = match.group(1)
            if target.startswith("Figures/") or target.startswith("figs/"):
                raise AssertionError(
                    f"LaTeX figure path uses stale figure directory casing from {tex_path.relative_to(ROOT)}: {target}"
                )
            if not any(path.exists() for path in _latex_graphic_candidates(target)):
                raise AssertionError(
                    f"LaTeX figure path does not resolve from {tex_path.relative_to(ROOT)}: {target}"
                )


def _check_latex_pdf_is_current() -> None:
    _check_one_latex_pdf_is_current(
        "main.tex",
        "builds/main.pdf",
        "uv run python docs/latex/scripts/latex_workflows.py build",
    )


def _check_one_latex_pdf_is_current(
    tex_name: str, pdf_name: str, build_command: str
) -> None:
    root_tex = DOCS_LATEX / tex_name
    pdf = DOCS_LATEX / pdf_name
    if not pdf.exists():
        raise AssertionError(f"Missing docs/latex/{pdf_name}. Run {build_command}.")
    sources = set()
    for pattern in ("*.bib", "*.bst", "*.cls", "*.sty"):
        sources.update(DOCS_LATEX.glob(pattern))
    for tex_path in _tex_dependency_closure(root_tex):
        sources.add(tex_path)
        text = tex_path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            for candidate in _latex_graphic_candidates(match.group(1)):
                if candidate.exists():
                    sources.add(candidate)
                    break

    newer = [
        path
        for path in sources
        if path.exists() and path.stat().st_mtime > pdf.stat().st_mtime
    ]
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
            raise AssertionError(
                f"Final table still points to removed docs benchmark paths: {path.name}"
            )


def _check_old_docs_benchmark_removed() -> None:
    if (ROOT / "docs" / "benchmark_figures").exists():
        raise AssertionError(
            "Old docs/benchmark_figures directory should not exist after migration."
        )


def _require_existing(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise AssertionError(
            "Missing required analysis artifacts:\n" + "\n".join(missing)
        )


def _require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise AssertionError(f"Missing required columns: {missing!r}")


if __name__ == "__main__":
    raise SystemExit(main())
