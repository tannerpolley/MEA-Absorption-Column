from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = ROOT / "docs" / "papers" / "md"
K_SOURCE = PAPER_DIR / "Morgan et al. - 2018 - Development of a Rigorous Modeling Framework for Solvent-Based CO2 Capture.md"
AD_SOURCE = PAPER_DIR / "Morgan et al.md"
DATA_DIR = ROOT / "src" / "mea_absorption_column" / "data"
SOURCE_2014_OUT = DATA_DIR / "NCCC_2014_cases.csv"
SOURCE_2017_OUT = DATA_DIR / "NCCC_2017_cases.csv"
SOURCE_2014_NO_INTERCOOLER_OUT = DATA_DIR / "NCCC_2014_no_intercooler_cases.csv"
SOURCE_2017_NO_INTERCOOLER_OUT = DATA_DIR / "NCCC_2017_no_intercooler_cases.csv"
MODEL_2014_OUT = DATA_DIR / "NCCC_2014_model_inputs_mass.csv"
MODEL_2017_OUT = DATA_DIR / "NCCC_2017_model_inputs_mass.csv"
COMBINED_OUT = DATA_DIR / "NCCC_combined_case_catalog.csv"
NO_INTERCOOLER_OUT = DATA_DIR / "NCCC_no_intercooler_case_options.csv"


def main() -> int:
    source_2014 = pd.DataFrame(_extract_k_source_cases(K_SOURCE.read_text(encoding="utf-8")))
    source_2017 = pd.DataFrame(_extract_ad_source_cases(AD_SOURCE.read_text(encoding="utf-8")))
    source_2014 = _sort_cases(source_2014)
    source_2017 = _sort_cases(source_2017)
    source_2014.to_csv(SOURCE_2014_OUT, index=False)
    source_2017.to_csv(SOURCE_2017_OUT, index=False)

    model_2014 = _model_inputs_from_source(source_2014)
    model_2017 = _model_inputs_from_source(source_2017)
    model_2014.to_csv(MODEL_2014_OUT, index=False)
    model_2017.to_csv(MODEL_2017_OUT, index=False)

    source_2014[source_2014["intercoolers"].eq(0)].to_csv(SOURCE_2014_NO_INTERCOOLER_OUT, index=False)
    source_2017[source_2017["intercoolers"].eq(0)].to_csv(SOURCE_2017_NO_INTERCOOLER_OUT, index=False)

    rows = _catalog_rows_from_source(source_2014) + _catalog_rows_from_source(source_2017)

    data = pd.DataFrame(rows)
    data["case_sort"] = data["case_no"].map(_case_sort_key)
    data = data.sort_values(["campaign_year", "case_sort"]).drop(columns="case_sort")
    data.to_csv(COMBINED_OUT, index=False)

    no_intercooler = data[data["intercoolers"].eq(0)].copy()
    no_intercooler.to_csv(NO_INTERCOOLER_OUT, index=False)

    for path, frame in (
        (SOURCE_2014_OUT, source_2014),
        (SOURCE_2017_OUT, source_2017),
        (MODEL_2014_OUT, model_2014),
        (MODEL_2017_OUT, model_2017),
        (COMBINED_OUT, data),
        (NO_INTERCOOLER_OUT, no_intercooler),
    ):
        print(f"Wrote {path} ({len(frame)} rows)")
    return 0


def _extract_k_source_cases(text: str) -> list[dict[str, object]]:
    occurrences: dict[str, list[list[str]]] = {}
    for case_no, values in _iter_case_rows(text, r"K\d+"):
        occurrences.setdefault(case_no, []).append(values)

    rows: list[dict[str, object]] = []
    for case_no in sorted(occurrences, key=_case_sort_key):
        parts = occurrences[case_no]
        absorber = _first(parts, lambda values: len(values) == 9 and "(" in values[-1])
        stripper = _first(parts, lambda values: len(values) == 7)
        capture = _first(parts, lambda values: len(values) == 4 and _is_percent_row(values))
        lean_loading = _last(parts, lambda values: len(values) == 4 and values is not capture)
        if absorber is None or stripper is None or capture is None or lean_loading is None:
            raise ValueError(f"Could not find complete 2014 case data for {case_no}")
        beds, intercoolers = _parse_beds(absorber[8])
        rows.append(
            {
                "case_no": case_no,
                "campaign_year": 2014,
                "source_file": K_SOURCE.name,
                "source_subset": "2014 K campaign",
                "configuration": _configuration_label(beds, intercoolers),
                "absorber_beds": beds,
                "intercoolers": intercoolers,
                "absorber_lean_solvent_flow_kg_h": _num(absorber[0]),
                "absorber_lean_solvent_temp_C": _num(absorber[1]),
                "absorber_lean_loading_mol_co2_per_mol_mea": _num(absorber[2]),
                "absorber_nominal_lean_solvent_mea_weight_fraction": _num(absorber[3]),
                "absorber_flue_gas_flow_kg_h": _num(absorber[4]),
                "absorber_inlet_gas_temp_C": _num(absorber[5]),
                "absorber_inlet_gas_co2_mol_pct": _num(absorber[6]),
                "absorber_pressure_kPa": _num(absorber[7]),
                "stripper_rich_solvent_flow_kg_h": _num(stripper[0]),
                "stripper_inlet_solvent_temp_C": _num(stripper[1]),
                "stripper_outlet_solvent_temp_C": _num(stripper[2]),
                "stripper_rich_loading_mol_co2_per_mol_mea": _num(stripper[3]),
                "stripper_rich_mea_weight_fraction": _num(stripper[4]),
                "stripper_operating_pressure_kPa": _num(stripper[5]),
                "stripper_reboiler_duty_kW": _num(stripper[6]),
                "capture_liquid_side_pct": _num(capture[0]),
                "capture_gas_side_pct": _num(capture[1]),
                "original_model_capture_pct": _num(capture[2]),
                "composition_uncertainty_model_capture_pct": _num(capture[3]),
                "lean_loading_original_data": _num(lean_loading[0]),
                "lean_loading_original_model": _num(lean_loading[1]),
                "lean_loading_composition_uncertainty_data": _num(lean_loading[2]),
                "lean_loading_composition_uncertainty_model": _num(lean_loading[3]),
            }
        )
    return rows


def _extract_ad_source_cases(text: str) -> list[dict[str, object]]:
    occurrences: dict[str, list[list[str]]] = {}
    for case_no, values in _iter_case_rows(text, r"[0-9]+[ABCD]"):
        occurrences.setdefault(case_no, []).append(values)

    result_rows = _extract_ad_result_rows(text)
    rows: list[dict[str, object]] = []
    for case_no in sorted(occurrences, key=_case_sort_key):
        parts = occurrences[case_no]
        part1 = _first(parts, _is_ad_part1)
        part2 = _first(parts, _is_ad_part2)
        part3 = _first(parts, _is_ad_part3)
        heat_exchanger = _first(parts, lambda values: len(values) == 4)
        stripper = _first(parts, _is_ad_stripper)
        temperature_profile = _first(parts, lambda values: len(values) == 12)
        if part1 is None or part2 is None or part3 is None:
            raise ValueError(f"Could not find complete 2017 absorber data for {case_no}")
        beds, intercoolers = _parse_beds(part3[5])
        summary_capture, model = result_rows.get(case_no, (None, None))
        row = {
            "case_no": case_no,
            "campaign_year": 2017,
            "source_file": AD_SOURCE.name,
            "source_subset": _ad_subset(case_no),
            "configuration": _configuration_label(beds, intercoolers),
            "absorber_beds": beds,
            "intercoolers": intercoolers,
            "absorber_lean_solvent_flow_kg_h": _num(part1[0]),
            "absorber_flue_gas_flow_kg_h": _num(part1[1]),
            "absorber_lean_solvent_temp_C": _num(part1[2]),
            "absorber_inlet_gas_temp_C": _num(part1[3]),
            "absorber_top_pressure_kPa": _num(part1[4]),
            "absorber_inlet_gas_pressure_kPa": _num(part1[5]),
            "absorber_lean_loading_mol_co2_per_mol_mea": _num(part1[6]),
            "absorber_lean_solvent_mea_weight_fraction": _num(part2[0]),
            "absorber_inlet_gas_co2_mole_fraction": _num(part2[1]),
            "absorber_inlet_gas_o2_mole_fraction": _num(part2[2]),
            "absorber_rich_solvent_flow_kg_h": _num(part2[3]),
            "absorber_rich_solvent_temp_C": _num(part2[4]),
            "absorber_outlet_gas_temp_C": _num(part2[5]),
            "absorber_rich_solvent_pressure_outlet_kPa": _num(part2[6]),
            "absorber_rich_solvent_pressure_after_pump_kPa": _num(part3[0]),
            "absorber_rich_loading_mol_co2_per_mol_mea": _num(part3[1]),
            "absorber_rich_mea_weight_fraction": _num(part3[2]),
            "absorber_outlet_gas_co2_mole_fraction": _num(part3[3]),
            "absorber_outlet_gas_o2_mole_fraction": _num(part3[4]),
            "absorber_capture_pct_avg": _num(part3[6]),
            "absorber_capture_pct_std": _capture_std(part3[6]),
            "summary_table_capture_pct": summary_capture,
            "reported_model_capture_pct": model,
        }
        if heat_exchanger is not None:
            row.update(
                {
                    "heat_exchanger_rich_solvent_inlet_temp_C": _num(heat_exchanger[0]),
                    "heat_exchanger_rich_solvent_outlet_temp_C": _num(heat_exchanger[1]),
                    "heat_exchanger_lean_solvent_inlet_temp_C": _num(heat_exchanger[2]),
                    "heat_exchanger_lean_solvent_outlet_temp_C": _num(heat_exchanger[3]),
                }
            )
        if stripper is not None:
            row.update(
                {
                    "stripper_rich_solvent_inlet_temp_C": _num(stripper[0]),
                    "stripper_rich_solvent_inlet_pressure_kPa": _num(stripper[1]),
                    "stripper_top_pressure_kPa": _num(stripper[2]),
                    "stripper_reboiler_temp_C": _num(stripper[3]),
                    "stripper_lean_solvent_pressure_after_pump_kPa": _num(stripper[4]),
                    "stripper_reboiler_duty_kW": _num(stripper[5]),
                }
            )
        if temperature_profile is not None:
            row.update({f"absorber_temperature_profile_C_{idx:02d}": _num(value) for idx, value in enumerate(temperature_profile, 1)})
        rows.append(row)
    return rows


def _catalog_rows_from_source(source: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for _, row in source.iterrows():
        is_2014 = int(row["campaign_year"]) == 2014
        rows.append(
            {
                "case_no": row["case_no"],
                "campaign_year": int(row["campaign_year"]),
                "source_file": row["source_file"],
                "source_subset": row["source_subset"],
                "configuration": row["configuration"],
                "absorber_beds": int(row["absorber_beds"]),
                "intercoolers": int(row["intercoolers"]),
                "lean_solvent_flow_kg_h": row["absorber_lean_solvent_flow_kg_h"],
                "flue_gas_flow_kg_h": row["absorber_flue_gas_flow_kg_h"],
                "lean_solvent_temp_C": row["absorber_lean_solvent_temp_C"],
                "inlet_gas_temp_C": row["absorber_inlet_gas_temp_C"],
                "absorber_pressure_kPa": row["absorber_pressure_kPa"] if is_2014 else row["absorber_top_pressure_kPa"],
                "lean_loading_mol_co2_per_mol_mea": row["absorber_lean_loading_mol_co2_per_mol_mea"],
                "w_MEA": (
                    row["absorber_nominal_lean_solvent_mea_weight_fraction"]
                    if is_2014
                    else row["absorber_lean_solvent_mea_weight_fraction"]
                ),
                "y_CO2": (
                    row["absorber_inlet_gas_co2_mol_pct"] / 100.0
                    if is_2014
                    else row["absorber_inlet_gas_co2_mole_fraction"]
                ),
                "y_O2": None if is_2014 else row["absorber_inlet_gas_o2_mole_fraction"],
                "co2_capture_pct": row["capture_gas_side_pct"] if is_2014 else row["absorber_capture_pct_avg"],
                "co2_capture_std_pct": None if is_2014 else row["absorber_capture_pct_std"],
                "reported_model_capture_pct": row["original_model_capture_pct"] if is_2014 else row["reported_model_capture_pct"],
            }
        )
    return rows


def _model_inputs_from_source(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in _catalog_rows_from_source(source):
        lean_temp_c = row["lean_solvent_temp_C"]
        lean_temp_imputed = False
        if pd.isna(lean_temp_c) and int(row["campaign_year"]) == 2017 and row["case_no"] in {"1C", "2C", "3C", "3D"}:
            lean_temp_c = 45.0
            lean_temp_imputed = True
        output = {
            "case_no": row["case_no"],
            "L": row["lean_solvent_flow_kg_h"] / 3600.0,
            "G": row["flue_gas_flow_kg_h"] / 3600.0,
            "alpha": row["lean_loading_mol_co2_per_mol_mea"],
            "w_MEA": row["w_MEA"],
            "y_CO2": row["y_CO2"],
            "Tl": None if pd.isna(lean_temp_c) else lean_temp_c + 273.15,
            "Tv": None if pd.isna(row["inlet_gas_temp_C"]) else row["inlet_gas_temp_C"] + 273.15,
            "P": row["absorber_pressure_kPa"] * 1000.0,
            "Beds": row["absorber_beds"],
            "Intercoolers": row["intercoolers"],
            "CO2  %": row["co2_capture_pct"],
            "lean_solvent_temp_imputed": lean_temp_imputed,
            "lean_solvent_temp_imputed_C": lean_temp_c if lean_temp_imputed else None,
        }
        if row["y_O2"] is not None and not pd.isna(row["y_O2"]):
            output["y_O2"] = row["y_O2"]
        rows.append(output)
    return pd.DataFrame(rows)


def _extract_ad_result_rows(text: str) -> dict[str, tuple[float | None, float | None]]:
    results: dict[str, tuple[float | None, float | None]] = {}
    for case_id, values in _iter_case_rows(text, r"[0-9]+[ABCD]"):
        if (
            len(values) == 7
            and case_id not in results
            and _is_float(values[5])
            and _is_float(values[6])
            and 0.0 <= float(values[5]) <= 100.0
            and 0.0 <= float(values[6]) <= 100.0
        ):
            results[case_id] = (_num(values[5]), _num(values[6]))
        if (
            len(values) == 6
            and case_id not in results
            and _is_float(values[4])
            and _is_float(values[5])
            and 0.0 <= float(values[4]) <= 100.0
            and 0.0 <= float(values[5]) <= 100.0
        ):
            results[case_id] = (_num(values[4]), _num(values[5]))
    return results


def _split_row(row: str) -> list[str]:
    return [value.strip().replace(",", "") for value in row.split("&")]


def _iter_case_rows(text: str, case_pattern: str):
    pattern = re.compile(rf"^\\hline\s+({case_pattern})\s*&\s*(.+?)\s*\\\\\s*$")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            yield match.group(1), _split_row(match.group(2))


def _first(rows: list[list[str]], predicate) -> list[str] | None:
    return next((row for row in rows if predicate(row)), None)


def _last(rows: list[list[str]], predicate) -> list[str] | None:
    return next((row for row in reversed(rows) if predicate(row)), None)


def _sort_cases(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["case_sort"] = data["case_no"].map(_case_sort_key)
    return data.sort_values("case_sort").drop(columns="case_sort")


def _is_percent_row(values: list[str]) -> bool:
    numeric = [_num(value) for value in values]
    return all(value is not None and 0.0 <= value <= 100.0 for value in numeric)


def _is_ad_part1(values: list[str]) -> bool:
    return (
        len(values) == 7
        and _is_temp_or_na(values[2])
        and _is_float(values[4])
        and float(values[4]) > 90.0
        and _is_float(values[6])
    )


def _is_ad_part2(values: list[str]) -> bool:
    return (
        len(values) == 7
        and _is_float(values[0])
        and _is_float(values[1])
        and _is_float(values[2])
        and 0.0 <= float(values[0]) <= 1.0
        and 0.0 <= float(values[1]) <= 1.0
        and 0.0 <= float(values[2]) <= 1.0
    )


def _is_ad_part3(values: list[str]) -> bool:
    return len(values) == 7 and "(" in values[5]


def _is_ad_stripper(values: list[str]) -> bool:
    return (
        len(values) == 6
        and _is_float(values[0])
        and _is_float(values[1])
        and 90.0 <= float(values[0]) <= 130.0
        and 100.0 <= float(values[1]) <= 600.0
    )


def _is_temp_or_na(value: str) -> bool:
    return value == "NA" or _is_float(value)


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _num(value: str) -> float | None:
    cleaned = value.strip().replace("$", "").replace("{", "").replace("}", "")
    cleaned = cleaned.split("\\pm")[0].strip()
    if cleaned == "NA" or cleaned == "":
        return None
    return float(cleaned)


def _capture_std(value: str) -> float | None:
    if "\\pm" not in value:
        return None
    return _num(value.split("\\pm", 1)[1])


def _parse_beds(value: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\s*\((\d+)\)", value)
    if not match:
        raise ValueError(f"Could not parse bed/intercooler value: {value}")
    return int(match.group(1)), int(match.group(2))


def _configuration_label(beds: int, intercoolers: int) -> str:
    bed_word = "bed" if beds == 1 else "beds"
    if intercoolers == 0:
        return f"{beds} {bed_word}, no intercooling"
    intercooler_word = "intercooler" if intercoolers == 1 else "intercoolers"
    return f"{beds} {bed_word}, {intercoolers} {intercooler_word}"


def _ad_subset(case_id: str) -> str:
    suffix = case_id[-1]
    return {
        "A": "2017 SDoE iteration 1",
        "B": "2017 SDoE iteration 2",
        "C": "2017 one-bed validation",
        "D": "2017 two-bed validation",
    }[suffix]


def _case_sort_key(case_id: str) -> tuple[int, str, int]:
    if case_id.startswith("K"):
        return (0, "K", int(case_id[1:]))
    return (1, case_id[-1], int(case_id[:-1]))


if __name__ == "__main__":
    raise SystemExit(main())
