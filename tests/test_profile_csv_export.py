import json

import numpy as np
import pandas as pd

from mea_absorption_column.benchmark import _write_profile_rerun_files
from mea_absorption_column.misc.Save_Run_Outputs import (
    build_profile_coordinate_frame,
    make_dfs_dict,
    write_profile_csvs,
)


def test_profile_coordinate_frame_spans_global_staged_height():
    frame = build_profile_coordinate_frame([0.0, 0.5, 1.0], total_packed_height_m=30.0, beds=3)

    assert list(frame.columns) == ["Position", "height_m", "bed_id", "bed_position_m"]
    assert frame["Position"].tolist() == [0.0, 0.5, 1.0]
    assert frame["height_m"].tolist() == [0.0, 15.0, 30.0]
    assert frame["bed_id"].tolist() == [1, 2, 3]


def test_make_dfs_dict_preserves_solver_row_order(tmp_path):
    output_dict = {
        "T": np.array([[320.0, 318.0], [315.0, 316.0]]),
    }
    keys_dict = {"T": ["Tl", "Tv"]}
    coordinate_frame = build_profile_coordinate_frame([0.0, 1.0], total_packed_height_m=10.0, beds=1)

    dfs = make_dfs_dict(output_dict, keys_dict, [0.0, 1.0], coordinate_frame=coordinate_frame)
    frame = dfs["T"]

    assert frame["Position"].tolist() == [0.0, 1.0]
    assert frame["Tl"].tolist() == [320.0, 315.0]
    assert frame["Tv"].tolist() == [318.0, 316.0]


def test_write_profile_csvs_writes_one_file_per_legacy_sheet(tmp_path):
    profiles = {
        "T": pd.DataFrame(
            {
                "Position": [1.0, 0.0],
                "height_m": [10.0, 0.0],
                "bed_id": [1, 1],
                "bed_position_m": [10.0, 0.0],
                "Tl": [320.0, 315.0],
                "Tv": [318.0, 316.0],
            }
        ),
        "CO2": pd.DataFrame({"Position": [1.0, 0.0], "DF_CO2": [1.2, 0.8]}),
    }

    export = write_profile_csvs(
        profiles,
        tmp_path,
        {
            "case_id": "3C",
            "case_source": "C_cases_data",
            "method": "scipy-bvp",
            "thermo_model": "ideal_henry",
            "profile_status": "clean",
        },
    )

    assert export["profile_csv_status"] == "written"
    assert set(export["profile_csv_files"].split(";")) == {"T.csv", "CO2.csv"}
    assert (tmp_path / "T.csv").exists()
    assert (tmp_path / "CO2.csv").exists()
    assert (tmp_path / "profile_manifest.csv").exists()
    manifest = json.loads((tmp_path / "profile_manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_id"] == "3C"
    assert manifest["profile_csv_files"] == ["T.csv", "CO2.csv"]


def test_profile_rerun_file_is_portable_bash(tmp_path):
    profile_dir = tmp_path / "profiles" / "3C"
    profile_dir.mkdir(parents=True)

    _write_profile_rerun_files(
        profile_dir,
        {
            "case_source": "C_cases_data",
            "case_id": "3C",
            "method": "scipy-bvp",
            "thermo_model": "ideal_henry",
        },
        tmp_path / "results",
        {},
    )

    rerun = profile_dir / "rerun_profile.sh"
    assert rerun.exists()
    assert rerun.stat().st_mode & 0o111
    assert not (profile_dir / "rerun_profile.ps1").exists()
    text = rerun.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "git -C" in text
    assert "uv run python" in text
