from __future__ import annotations

import pandas as pd


def test_nccc_master_case_table_maps_source_k_rows_to_appendix_style_cases():
    master = pd.read_csv("data/reference/nccc_master_cases.csv")

    assert len(master) == 23
    assert set(master["legacy_case_id"]) == {f"K{i}" for i in range(1, 24)}

    mapping = dict(zip(master["legacy_case_id"], master["appendix_style_case_id"]))
    assert mapping["K1"] == "1A"
    assert mapping["K3"] == "3A"
    assert mapping["K13"] == "1B"
    assert mapping["K18"] == "1C"
    assert mapping["K20"] == "3C"
    assert mapping["K22"] == "3D"
    assert mapping["K23"] == "4D"

    k3 = master.loc[master["legacy_case_id"] == "K3"].iloc[0]
    assert k3["beds"] == 3
    assert k3["intercoolers"] == 2
    assert k3["capture_gas_side_pct"] == 83.57
