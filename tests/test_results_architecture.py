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
        FINAL / "tables" / "verified_staged_kcase_benchmark.csv",
        FINAL / "tables" / "clean_temperature_profile_index.csv",
        FINAL / "figures" / "c_case_thermo_benchmark.pdf",
        FINAL / "figures" / "staged_kcase_capture_error.pdf",
        FINAL / "figures" / "staged_epcsaft_smoke_capture_error.pdf",
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

    checked_files = [
        ROOT / "README.md",
        ROOT / "docs" / "latex" / "main.tex",
        ROOT / "docs" / "latex" / "revised_benchmark_results.tex",
    ]
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
    staged = pd.read_csv(FINAL / "tables" / "verified_staged_kcase_benchmark.csv")

    assert c_cases["case_id"].nunique() == 7
    assert staged["case_id"].nunique() == 19
    for column in ["beds", "intercoolers", "staged_beds", "intercooler_assumption"]:
        assert column in staged.columns
    for data in (c_cases, staged):
        assert data["artifact"].str.startswith("analyses/nccc_validation/results/final/tables/").all()
