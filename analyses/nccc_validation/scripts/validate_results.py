from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
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
ISSUE40_INPUT = ANALYSIS / "inputs/issue40_apparent_true_species.json"
ISSUE40_TABLE = TABLES / "issue40_apparent_true_species.csv"
ISSUE40_SUMMARY = TABLES / "issue40_apparent_true_species_summary.json"
ISSUE40_REPORT = FINAL / "reports/issue40_apparent_true_species.md"
ISSUE41_INPUT = ANALYSIS / "inputs/issue41_reversible_kinetics.json"
ISSUE41_TABLES = {
    "stoichiometry": TABLES / "issue41_stoichiometry.csv",
    "source_rate_evidence": TABLES / "issue41_source_rate_evidence.csv",
    "raw_observations": TABLES / "issue41_raw_rate_observations.csv",
    "provider_equilibrium_relationships": TABLES / "issue41_provider_equilibrium_relationships.csv",
    "estimation_validation_partition": TABLES / "issue41_estimation_validation_partition.csv",
    "packet_bound_comparison": TABLES / "issue41_packet_bound_comparison.csv",
}
ISSUE41_SUMMARY = TABLES / "issue41_reversible_kinetics_summary.json"
ISSUE41_REPORT = FINAL / "reports/issue41_reversible_kinetics.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue17-only", action="store_true")
    parser.add_argument("--issue40-only", action="store_true")
    parser.add_argument("--issue41-only", action="store_true")
    args = parser.parse_args()
    if args.issue17_only:
        _check_issue17_enhancement_comparison()
        print("Issue 17 retained outputs are internally consistent.")
        return 0
    if args.issue40_only:
        _check_issue40_apparent_true_species()
        print("Issue 40 retained-output/source-lineage consistency checks passed.")
        return 0
    if args.issue41_only:
        _check_issue41_reversible_kinetics()
        print("Issue 41 source-rate and packet-bound consistency checks passed.")
        return 0

    checks = [
        _check_issue17_enhancement_comparison,
        _check_issue40_apparent_true_species,
        _check_issue41_reversible_kinetics,
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


def _check_issue40_apparent_true_species() -> None:
    _require_existing([ISSUE40_INPUT, ISSUE40_TABLE, ISSUE40_SUMMARY, ISSUE40_REPORT])
    config = json.loads(ISSUE40_INPUT.read_text(encoding="utf-8"))
    summary = json.loads(ISSUE40_SUMMARY.read_text(encoding="utf-8"))
    data = pd.read_csv(ISSUE40_TABLE, keep_default_na=False)

    def git_blob_sha256(revision: str, path: Path) -> str:
        result = subprocess.run(
            ["git", "show", f"{revision}:{path.relative_to(ROOT)}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return hashlib.sha256(result.stdout).hexdigest()

    source_commit = summary.get("source_repository_commit", "")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if not source_commit or source_commit == current_commit:
        raise AssertionError("Issue 40 retained results must record a distinct source commit from the result-artifact commit.")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT, capture_output=True
    ).returncode != 0:
        raise AssertionError("Issue 40 source repository commit is not a local commit.")
    if subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, current_commit], cwd=ROOT).returncode != 0:
        raise AssertionError("Issue 40 source repository commit is not an ancestor of the retained result commit.")
    if summary.get("source_worktree_clean_at_generation") is not True:
        raise AssertionError("Issue 40 source commit was not generated from a clean worktree.")
    source_inputs = summary.get("source_inputs", {})
    actual_source_inputs = {
        "config_sha256": hashlib.sha256(ISSUE40_INPUT.read_bytes()).hexdigest(),
        "issue33_table_sha256": hashlib.sha256((TABLES / "issue33_concentration_basis.csv").read_bytes()).hexdigest(),
        "issue33_summary_sha256": hashlib.sha256((TABLES / "issue33_concentration_basis_summary.json").read_bytes()).hexdigest(),
        "retained_profile_sha256": hashlib.sha256((ANALYSIS / "inputs/retained_reactive_case3c/film_states.csv").read_bytes()).hexdigest(),
    }
    expected_inputs = config["source_inputs"]
    if (
        actual_source_inputs["issue33_table_sha256"] != expected_inputs["issue33_table"]["sha256"]
        or actual_source_inputs["issue33_summary_sha256"] != expected_inputs["issue33_summary"]["sha256"]
        or actual_source_inputs["retained_profile_sha256"] != expected_inputs["retained_profile"]["sha256"]
    ):
        raise AssertionError("Issue 40 current source input hash disagrees with the input record.")
    if source_inputs != actual_source_inputs:
        raise AssertionError("Issue 40 recorded source input hashes are stale.")
    issue33_summary = json.loads(
        (TABLES / "issue33_concentration_basis_summary.json").read_text(encoding="utf-8")
    )
    if issue33_summary.get("source_table_sha256") != actual_source_inputs["issue33_table_sha256"]:
        raise AssertionError("Issue 40 Issue 33 summary hash does not match the current Issue 33 table.")
    if summary.get("generator_sha256") != git_blob_sha256(
        source_commit, ANALYSIS / "scripts/resolve_issue40_apparent_true_species.py"
    ):
        raise AssertionError("Issue 40 generator hash does not match the recorded source commit.")
    if actual_source_inputs["config_sha256"] != git_blob_sha256(source_commit, ISSUE40_INPUT):
        raise AssertionError("Issue 40 input hash does not match the recorded source commit.")
    if summary.get("source_revision_protocol") != config["reproduction"]["source_revision_protocol"]:
        raise AssertionError("Issue 40 source revision protocol is stale.")

    expected_species = [
        "carbon-dioxide", "monoethanolamine", "water", "protonated-monoethanolamine",
        "carbamate-anion", "bicarbonate-anion", "carbonate-anion", "hydronium-cation", "hydroxide-anion",
    ]
    if len(data) != 5 or set(data["source_row_id"]) != {
        "Putta2016_1M", "Putta2016_5M", "Case3C_position_0", "Case3C_position_0.5", "Case3C_position_1"
    }:
        raise AssertionError("Issue 40 must retain the two source labels and three Case 3C positions.")
    if data["scientific_admission"].ne("basis_unresolved").any():
        raise AssertionError("Issue 40 must not scientifically admit a source-basis-unresolved row.")
    if data["packet_bound_status"].value_counts().to_dict() != {"not_attempted": 4, "evaluated": 1}:
        raise AssertionError("Issue 40 packet row accounting changed.")
    p1 = data.loc[data["source_row_id"] == "Case3C_position_1"].iloc[0]
    if not (
        abs(float(p1["source_loaded_analytical_MEA_mol_L"]) - 4.889309897097635) <= 1.0e-15
        and abs(float(p1["source_free_MEA_mol_L"]) - 2.491683471902737) <= 1.0e-15
        and p1["packet_bound_status"] == "evaluated"
        and p1["inverse_mapping_status"] == "evaluated_packet_equilibrium_roundtrip"
        and p1["forward_transform_status"] == "evaluated"
        and p1["inverse_replay_status"] == "identity_pass"
        and p1["forward_replay_status"] == "identity_pass"
        and p1["replay_branch_match_status"] == "identity_pass"
        and p1["branch_identity"] == p1["replay_branch_identity"]
        and p1["packet_density_status"] == "ePC-SAFT phase density; true-state mapping diagnostic only"
        and p1["diagnostic_density_marker"] == "source density does not define prepared or analytical basis"
        and float(p1["inverse_max_abs_residual"]) <= 1.0e-10
        and float(p1["charge_residual"]) <= 1.0e-10
        and abs(float(p1["issue33_reconstructed_true_species_loading_mol_CO2_per_mol_MEA"]) - 0.24999627615967537) <= 1.0e-15
        and abs(float(p1["apparent_total_inorganic_carbon_to_analytical_MEA_flow_ratio_mol_per_mol"]) - 0.25000000000000006) <= 1.0e-15
        and abs(float(p1["apparent_minus_issue33_loading_mol_per_mol"]) - 3.7238403246819818e-06) <= 1.0e-15
        and p1["loading_apparent_difference_status"] == "distinct inputs; neither relabelled nor fitted"
    ):
        raise AssertionError("Issue 40 Position 1 packet mapping or exact source values failed.")
    if summary["species_order"] != expected_species or summary["gates"] != {
        "fixed_nine_species_order": True,
        "single_liquid_only": True,
        "vle_fugacity_equality_imposed": False,
        "position_1_exact_source_analytical_mol_L": 4.889309897097635,
        "position_1_exact_source_free_MEA_mol_L": 2.491683471902737,
        "position_1_issue33_reconstructed_true_species_loading": 0.24999627615967537,
        "position_1_apparent_flow_ratio": 0.25000000000000006,
        "position_1_apparent_minus_issue33_loading": 3.7238403246819818e-06,
        "loading_and_apparent_ratio_distinct_not_relabelled_or_fitted": True,
        "position_1_source_basis_unresolved": True,
        "packet_evaluated_at_least_one_row": True,
        "analytical_and_elemental_residual_pass": True,
        "mole_fraction_normalization_pass": True,
        "charge_residual_pass": True,
        "deterministic_replay_pass": True,
        "replay_branch_match_pass": True,
        "no_capture_inference": True,
        "no_thermo_or_kinetic_fit": True,
    }:
        raise AssertionError("Issue 40 summary gates or species order changed.")
    for identity in ("outer_sha256", "parameter_document_sha256", "engine_wheel_sha256", "state_packet_sha256", "parameter_fingerprint", "chemistry_sha256"):
        expected = config["bundle"].get(identity)
        if expected is not None and summary["bundle"].get(identity) != expected:
            raise AssertionError(f"Issue 40 bundle identity changed: {identity}")
    if summary["row_counts"] != {
        "source_rows": 5, "literature_label_rows": 2, "retained_case3c_rows": 3,
        "packet_candidate_rows": 1, "packet_evaluated_rows": 1, "packet_non_evaluable_rows": 0,
        "packet_not_attempted_rows": 4, "scientifically_admitted_rows": 0, "basis_unresolved_rows": 5,
    }:
        raise AssertionError("Issue 40 row counts changed.")
    result_hash = hashlib.sha256(ISSUE40_TABLE.read_bytes()).hexdigest()
    if summary.get("result_table_sha256") != result_hash or summary["packet_state_evidence"]["non_evaluable_state_count"] != 31:
        raise AssertionError("Issue 40 retained result hash or historical failure accounting is stale.")
    for _, row in data.loc[data["source_row_id"].astype(str).str.startswith("Putta")].iterrows():
        inputs = json.loads(row["apparent_inputs_json"])
        intervals = json.loads(row["apparent_input_reporting_intervals_json"])
        if inputs != {"CO2": None, "H2O": None, "MEA": None, "normalized_CO2_MEA_H2O": None, "status": "not_reported_source_label_only"}:
            raise AssertionError("Issue 40 Putta source-label row must retain structured null apparent inputs.")
        if any(intervals[key] != {"interval": None, "status": "not_reported"} for key in ("CO2_mol_s", "MEA_mol_s", "H2O_mol_s", "temperature_K", "pressure_Pa")):
            raise AssertionError("Issue 40 Putta source-label row must retain structured null interval metadata.")
        if any(row[key] != "not_reported" for key in ("temperature_K", "pressure_Pa")) or any(row[key] != "not_reported_source_label_only" for key in ("temperature_reporting_interval", "pressure_reporting_interval")):
            raise AssertionError("Issue 40 Putta source-label row has incomplete T/P not-reported metadata.")
    if "retained-output/source-lineage consistency" not in ISSUE40_REPORT.read_text(encoding="utf-8"):
        raise AssertionError("Issue 40 report must describe the focused validator as a lineage check.")
    if any(data[column].eq("").any() for column in ("workers", "machine", "run_id", "reproduction_command")):
        raise AssertionError("Every Issue 40 row must retain run identity and reproduction metadata.")


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


def _check_issue41_reversible_kinetics() -> None:
    _require_existing([ISSUE41_INPUT, ISSUE41_SUMMARY, ISSUE41_REPORT, *ISSUE41_TABLES.values()])
    config = json.loads(ISSUE41_INPUT.read_text(encoding="utf-8"))
    summary = json.loads(ISSUE41_SUMMARY.read_text(encoding="utf-8"))
    if config.get("schema_version") != "issue41_source_rate_evidence_v1":
        raise AssertionError("Issue 41 input schema changed.")
    if config.get("species_order") != [
        "CO2", "MEA", "H2O", "MEAH+", "MEACOO-", "HCO3-", "CO3^2-", "H3O+", "OH-"
    ]:
        raise AssertionError("Issue 41 species order changed.")
    if summary.get("input_sha256") != hashlib.sha256(ISSUE41_INPUT.read_bytes()).hexdigest():
        raise AssertionError("Issue 41 summary input hash is stale.")
    generator = ANALYSIS / "scripts/resolve_issue41_reversible_kinetics.py"
    if summary.get("generator_sha256") != hashlib.sha256(generator.read_bytes()).hexdigest():
        raise AssertionError("Issue 41 generator hash is stale.")
    source_revision = summary.get("source_revision", "")
    if not source_revision or subprocess.run(["git", "cat-file", "-e", f"{source_revision}^{{commit}}"], cwd=ROOT, capture_output=True).returncode != 0:
        raise AssertionError("Issue 41 source revision is not a local commit.")
    current_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", source_revision, current_revision], cwd=ROOT, capture_output=True).returncode != 0:
        raise AssertionError("Issue 41 source revision is not an ancestor of the retained result commit.")
    if summary.get("source_worktree_dirty_during_generation") is not False:
        raise AssertionError("Issue 41 retained outputs were not generated from a clean source worktree.")
    source_blob_hashes = {
        "input": hashlib.sha256(subprocess.run(["git", "show", f"{source_revision}:analyses/nccc_validation/inputs/issue41_reversible_kinetics.json"], cwd=ROOT, check=True, capture_output=True).stdout).hexdigest(),
        "generator": hashlib.sha256(subprocess.run(["git", "show", f"{source_revision}:analyses/nccc_validation/scripts/resolve_issue41_reversible_kinetics.py"], cwd=ROOT, check=True, capture_output=True).stdout).hexdigest(),
    }
    if source_blob_hashes != {"input": summary["input_sha256"], "generator": summary["generator_sha256"]}:
        raise AssertionError("Issue 41 retained source revision/blob lineage is stale or tampered.")
    _check_issue41_external_provenance(config, summary)
    expected_gates = {
        "fixed_nine_species_order": True, "fixed_reaction_projections": True,
        "source_pdf_hashes_verified": True, "bundle_outer_and_member_hashes_match": True,
        "bundle_identity_matches_input": True, "source_f1_f2_coefficients_recovered": True,
        "source_f3_coefficient_recovered": False, "source_printed_third_order_s_minus_2_rejected": True,
        "source_observations_row_level_available": False, "estimation_validation_partition_predeclared_only": True,
        "issue40_basis_unresolved_preserved": True, "packet_bound_scientific_admission": False,
        "packet_activity_closure_attempted": False, "detailed_balance_evaluable": False,
        "reaction_timescale_evaluable": False, "physical_reactive_film_adoption": False, "supported_negative": True,
    }
    if summary.get("gates") != expected_gates:
        raise AssertionError("Issue 41 gates changed.")
    expected_bundle = {
        "outer_sha256": "4139fecd9b5192e7cadd12883d2ff1bff71c20d74950af5256e4f0447995f27b",
        "parameter_document_sha256": "2666914f0f9cfebdf230e96565de843f9aadc9424035c940883147ff66af035c",
        "engine_wheel_sha256": "d7b4fc5ba5cbf0e979b65af83442d565496d11b771bb559233ad9dc3a4f8414a",
        "state_packet_sha256": "41017bcf727a486a8f3feb280e19c111a15c5dda5a3cca4e8c7dc5b051168fef",
        "chemistry_sha256": "1989f3e6c8fa567a019dcdbceb4bbcf26d9ca48aec3f640dad1134bdd1fd4e7c",
        "parameter_fingerprint": "sha256:c1fc2665e94d136eb85f27c793b7defbd16d1d82cb3173cb50a9aaf6513c8940",
    }
    if any(summary.get("bundle", {}).get(key) != value for key, value in expected_bundle.items()):
        raise AssertionError("Issue 41 bundle identity changed.")
    expected_counts = {
        "stoichiometry": 8, "source_rate_evidence": 5, "raw_observation_inventory": 23,
        "provider_equilibrium_relationships": 9, "estimation_validation_partition": 5,
        "packet_bound_comparison": 5, "scientifically_admitted_packet_rows": 0,
    }
    if summary.get("row_counts") != expected_counts:
        raise AssertionError("Issue 41 row counts changed.")
    tables = {key: pd.read_csv(path, keep_default_na=False) for key, path in ISSUE41_TABLES.items()}
    stoich = tables["stoichiometry"]
    if len(stoich) != 8 or set(stoich["reaction_id"]) != {"R1", "R2", "R3", "R4", "R5", "F1", "F2", "F3"} or not stoich["stoichiometry_balance_pass"].astype(str).str.lower().eq("true").all():
        raise AssertionError("Issue 41 stoichiometry table is incomplete or unbalanced.")
    rates = tables["source_rate_evidence"]
    if len(rates) != 5 or not rates["dimensional_reconstruction_pass"].astype(str).str.lower().eq("true").all():
        raise AssertionError("Issue 41 source-rate table is incomplete.")
    if len(rates.loc[rates["coefficient_status"] == "recovered"]) != 4 or len(rates.loc[rates["reaction_id"] == "F3"]) != 1 or rates.loc[rates["reaction_id"] == "F3", "coefficient_status"].iloc[0] != "unavailable":
        raise AssertionError("Issue 41 F1/F2/F3 source-rate statuses changed.")
    raw = tables["raw_observations"]
    if len(raw) != 23 or not raw["raw_rate_value_available"].astype(str).str.lower().eq("false").all():
        raise AssertionError("Issue 41 raw observation inventory must remain non-row-level.")
    partition = tables["estimation_validation_partition"]
    if len(partition) != 5 or partition["dataset_id"].duplicated().any() or not partition["status"].eq("predeclared_only_no_row_ids").all() or not bool(partition[["row_ids_used", "rate_values_used", "uncertainty_weights_used"]].apply(lambda col: col.astype(str).str.lower().eq("false").all()).all()):
        raise AssertionError("Issue 41 estimation/validation partition changed.")
    provider = tables["provider_equilibrium_relationships"]
    if len(provider) != 9 or not provider["detailed_balance_status"].eq("not_evaluable_basis_unresolved").all() or not provider[["lnQ", "detailed_balance_residual", "detailed_balance_pass"]].eq("").all().all():
        raise AssertionError("Issue 41 provider relationship table evaluated unavailable Q data.")
    packet = tables["packet_bound_comparison"]
    if len(packet) != 5 or not packet["source_basis_status"].eq("basis_unresolved").all() or not packet["scientific_admission"].eq("basis_unresolved").all() or not packet["lnQ"].eq("").all() or not packet["detailed_balance_residual"].eq("").all() or not packet["detailed_balance_pass"].eq("").all():
        raise AssertionError("Issue 41 packet-bound comparison crossed the basis-admission boundary.")
    for key, path in ISSUE41_TABLES.items():
        if summary["output_sha256"][key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise AssertionError(f"Issue 41 output hash is stale: {key}")
    if summary["output_sha256"]["report"] != hashlib.sha256(ISSUE41_REPORT.read_bytes()).hexdigest():
        raise AssertionError("Issue 41 report hash is stale.")
    report = ISSUE41_REPORT.read_text(encoding="utf-8")
    if "supported-negative" not in report or "No physical reactive film is adopted" not in report or "No bundle provenance mismatch was found" not in report:
        raise AssertionError("Issue 41 report does not state its evidence boundary.")


def _check_issue41_external_provenance(config: dict, summary: dict) -> None:
    for source in config["source_documents"]:
        path = source.get("local_pdf_path")
        expected = source.get("source_pdf_sha256")
        if path is None:
            if expected is not None:
                raise AssertionError(f"Issue 41 source {source['id']} has an unresolvable PDF hash.")
            continue
        source_path = Path(path)
        if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Issue 41 source PDF is missing or changed: {source['id']}")
    bundle_config = config["bundle"]
    bundle_path = Path(bundle_config["path"])
    if not bundle_path.is_file() or hashlib.sha256(bundle_path.read_bytes()).hexdigest() != bundle_config["outer_sha256"]:
        raise AssertionError("Issue 41 authorized bundle is missing or changed.")
    with zipfile.ZipFile(bundle_path) as archive:
        manifest_names = [name for name in archive.namelist() if name.endswith("/bundle.json")]
        if len(manifest_names) != 1:
            raise AssertionError("Issue 41 bundle manifest is not unique.")
        manifest_name = manifest_names[0]
        prefix = manifest_name[: -len("bundle.json")]
        manifest = json.loads(archive.read(manifest_name))
        for item in manifest["files"]:
            member = prefix + item["path"]
            data = archive.read(member)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise AssertionError(f"Issue 41 bundle member is missing or changed: {item['path']}")
        expected = {
            "outer_sha256": bundle_config["outer_sha256"],
            "parameter_document_sha256": bundle_config["parameter_document_sha256"],
            "engine_wheel_sha256": bundle_config["engine_wheel_sha256"],
            "state_packet_sha256": bundle_config["state_packet_sha256"],
        }
        if any(manifest[key] != value for key, value in expected.items()):
            raise AssertionError("Issue 41 bundle manifest identity changed.")
        chemistry_path = prefix + "chemistry/reaction-system.json"
        if hashlib.sha256(archive.read(chemistry_path)).hexdigest() != bundle_config["chemistry_sha256"]:
            raise AssertionError("Issue 41 bundle chemistry member identity changed.")
    if summary.get("bundle", {}).get("outer_sha256") != bundle_config["outer_sha256"]:
        raise AssertionError("Issue 41 summary bundle identity is stale.")


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
