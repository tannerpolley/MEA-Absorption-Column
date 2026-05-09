from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from mea_absorption_column.benchmark import (
    _filter_case_ids,
    _nccc_case_source,
    _run_one_case_in_process,
    load_case_data,
    settings_from_payload,
)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("Usage: python -m mea_absorption_column.benchmark_worker INPUT_JSON")
    input_path = Path(argv[0])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    settings = settings_from_payload(payload["settings"])
    c_cases, nccc_cases, srp_cases = load_case_data(settings.c_case_dataset, settings.nccc_dataset)
    case_source = payload["case_source"]
    nccc_case_source = _nccc_case_source(settings.nccc_dataset)
    data_by_source = {
        "C_cases_data": c_cases,
        "C_cases_campaign_inputs": c_cases,
        "NCCC_Data": nccc_cases,
        nccc_case_source: nccc_cases,
        "SRP_method_cases": srp_cases,
    }
    df = data_by_source[case_source]
    if case_source in {"C_cases_data", "C_cases_campaign_inputs"}:
        df = _filter_case_ids(df, settings.c_case_ids, case_source)
    elif case_source in {"NCCC_Data", nccc_case_source}:
        df = _filter_case_ids(df, settings.nccc_case_ids, case_source)
    else:
        df = _filter_case_ids(df, settings.srp_case_ids, case_source)
    row = _run_one_case_in_process(
        df=df,
        run=int(payload["run"]),
        case_source=case_source,
        method=payload["method"],
        thermo_model=payload["thermo_model"],
        settings=settings,
    )
    output_path = Path(payload["output_path"])
    output_path.write_text(json.dumps(_json_clean(row)), encoding="utf-8")
    return 0


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if hasattr(value, "item"):
        return _json_clean(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
