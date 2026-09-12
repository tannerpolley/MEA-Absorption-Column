from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from mea_absorption_column.benchmark import (
    _filter_case_ids,
    _nccc_case_source,
    _run_one_case_in_process,
    load_case_data,
    settings_from_payload,
)
from mea_absorption_column.config.column import CapabilityRefusal, ConfigurationError, verify_worker_identity


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("Usage: python -m mea_absorption_column.benchmark_worker INPUT_JSON")
    input_path = Path(argv[0])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output_path = Path(payload["output_path"])
    worker_identity = None
    try:
        if payload.get("runtime_identity") is not None:
            worker_identity = verify_worker_identity(payload["runtime_identity"])
    except CapabilityRefusal as exc:
        output_path.write_text(json.dumps(_json_clean({
            "case_id": str((payload.get("resolved_case") or {}).get("case_id", payload.get("run", "unknown"))),
            "case_source": payload.get("case_source", ""),
            "method": payload.get("method", ""),
            "thermo_model": payload.get("thermo_model", ""),
            "success": False,
            "message": str(exc),
            "failure_kind": "capability_refusal",
            "jacobian_status": "capability_refusal",
        })), encoding="utf-8")
        return 0
    if payload.get("task") == "conserved_preparation":
        from mea_absorption_column.column import _prepare_conserved_column_in_process
        from mea_absorption_column.config.column import resolve_column_config
        try:
            result = _prepare_conserved_column_in_process(
                resolve_column_config(payload["config"])
            )
            result.pop("assembly", None)
            output_path.write_text(json.dumps(_json_clean(result)), encoding="utf-8")
        except CapabilityRefusal as exc:
            output_path.write_text(json.dumps(_json_clean({
                "failure_kind": "capability_refusal", "message": str(exc)
            })), encoding="utf-8")
        except (ConfigurationError, OSError, RuntimeError, ValueError) as exc:
            output_path.write_text(json.dumps(_json_clean({
                "failure_kind": "preparation_failed", "message": str(exc)
            })), encoding="utf-8")
        return 0
    settings = settings_from_payload(payload["settings"])
    if settings.cache_policy == "disabled":
        os.environ["MEA_EPCSAFT_DISABLE_CACHE"] = "1"
    case_source = payload["case_source"]
    resolved_case = payload.get("resolved_case")
    if resolved_case is not None:
        df = pd.DataFrame([resolved_case["values"]], index=[resolved_case["case_id"]])
        worker_run = 0
    else:
        c_cases, nccc_cases, srp_cases = load_case_data(settings.c_case_dataset, settings.nccc_dataset)
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
        worker_run = int(payload["run"])
    row = _run_one_case_in_process(
        df=df,
        run=worker_run,
        case_source=case_source,
        method=payload["method"],
        thermo_model=payload["thermo_model"],
        settings=settings,
    )
    if worker_identity is not None:
        row["worker_identity"] = worker_identity
    output_path.write_text(json.dumps(_json_clean(row)), encoding="utf-8")
    return 0


def _json_clean(value):
    if isinstance(value, pd.DataFrame):
        return {
            "__json_type__": "dataframe",
            "columns": [str(column) for column in value.columns],
            "index": [_json_clean(item) for item in value.index.tolist()],
            "data": _json_clean(value.to_numpy()),
        }
    if isinstance(value, pd.Series):
        return {
            "__json_type__": "series",
            "name": value.name,
            "index": [_json_clean(item) for item in value.index.tolist()],
            "data": _json_clean(value.to_numpy()),
        }
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if hasattr(value, "item"):
        return _json_clean(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return {"value_class": "nan" if math.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
