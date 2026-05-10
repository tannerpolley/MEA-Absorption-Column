import importlib.util
import sys
from pathlib import Path

import pandas as pd

from mea_absorption_column.benchmark import BenchmarkSettings


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analyses" / "nccc_validation"
FINAL = ANALYSIS / "results" / "final"


def test_nccc_validation_analysis_layout_exists():
    expected = [
        ANALYSIS / "README.md",
        ANALYSIS / "analysis.yaml",
        ANALYSIS / "scripts" / "generate_data.py",
        ANALYSIS / "scripts" / "render_figures.py",
        ANALYSIS / "scripts" / "collect_clean_profiles.py",
        ANALYSIS / "scripts" / "validate_results.py",
        FINAL / "tables" / "verified_c_case_thermo_benchmark.csv",
        FINAL / "tables" / "validation_evidence_registry.csv",
        FINAL / "tables" / "primary_validation_gate.csv",
        FINAL / "tables" / "calibration_holdout_metrics.csv",
        FINAL / "tables" / "method_case_contrast.csv",
        FINAL / "tables" / "clean_temperature_profile_index.csv",
        FINAL / "figures" / "c_case_thermo_benchmark.pdf",
        FINAL / "figures" / "error_regime_capture_error.pdf",
        FINAL / "figures" / "calibration_uncertainty_band.pdf",
        FINAL / "figures" / "method_case_solver_contrast.pdf",
    ]

    missing = [path for path in expected if not path.exists()]
    assert missing == []


def test_benchmark_default_output_is_analysis_run_folder():
    default = BenchmarkSettings().output_dir.as_posix()

    assert default == "analyses/nccc_validation/results/runs/benchmark"


def test_clean_profile_generator_defaults_to_sixty_second_case_timeout():
    script = ANALYSIS / "scripts" / "generate_clean_profile_csvs.py"
    spec = importlib.util.spec_from_file_location("generate_clean_profile_csvs", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args([])

    assert args.per_case_timeout_s == 60.0


def test_old_docs_benchmark_gallery_is_removed():
    assert not (ROOT / "docs" / "benchmark_figures").exists()

    checked_files = [ROOT / "README.md", ROOT / "docs" / "latex" / "main.tex"]
    checked_files.extend((ROOT / "docs" / "latex" / "sections").glob("*.tex"))
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        assert "docs/benchmark_figures" not in text
        assert "benchmark_figures/" not in text


def test_final_clean_profile_index_points_to_existing_pngs():
    data = pd.read_csv(FINAL / "tables" / "clean_temperature_profile_index.csv")

    assert not data.empty
    assert {"case_id", "thermo_model", "profile_png", "clean_profile", "caveat"} <= set(data.columns)
    for profile_png in data["profile_png"]:
        assert (ROOT / profile_png).exists()


def test_primary_final_tables_use_analysis_paths_and_metadata():
    c_cases = pd.read_csv(FINAL / "tables" / "verified_c_case_thermo_benchmark.csv")
    gate = pd.read_csv(FINAL / "tables" / "primary_validation_gate.csv")
    method_contrast = pd.read_csv(FINAL / "tables" / "method_case_contrast.csv")

    assert c_cases["case_id"].nunique() == 7
    assert set(c_cases["thermo_model"]) == {"ideal_henry", "epcsaft_ionic"}
    assert set(gate["case_id"].astype(str)) == {"1C", "2C", "3C", "4C", "5C", "6C"}
    assert set(gate["thermo_model"]) == {"epcsaft_ionic"}
    assert not gate["case_id"].astype(str).str.startswith("K").any()
    assert {"Shooting", "Collocation BVP", "Finite difference"} <= set(method_contrast["method"])
    assert c_cases["artifact"].str.startswith("analyses/nccc_validation/results/final/tables/").all()
