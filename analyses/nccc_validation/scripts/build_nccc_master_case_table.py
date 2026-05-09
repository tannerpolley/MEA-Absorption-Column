from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ZIP_SOURCE = Path(r"C:\Users\Tanner\Documents\git\Lithium_Extraction\data\PDF_Export_08052026.zip")
DOC_MD_DIR = ROOT / "docs" / "paper" / "md"
REFERENCE_DIR = ROOT / "data" / "reference"
ANALYSIS_INPUT_DIR = ROOT / "analyses" / "nccc_validation" / "data" / "input"
PACKAGE_NCCC = ROOT / "src" / "mea_absorption_column" / "data" / "NCCC_Data.csv"

MORGAN_2018_NAME = (
    "Morgan et al. - 2018 - Development of a Rigorous Modeling Framework for "
    "Solvent-Based CO2 Capture.md"
)
MOORE_2021_NAME = "Moore et al. - 2021 - Advanced absorber heat integration via heat exchange packings.md"

A_K_ROWS = [f"K{i}" for i in range(1, 13)] + ["K14", "K15", "K16"]
K_TO_APPENDIX_STYLE = {
    **{legacy: f"{idx}A" for idx, legacy in enumerate(A_K_ROWS, start=1)},
    "K13": "1B",
    "K17": "1D",
    "K18": "1C",
    "K19": "2C",
    "K20": "3C",
    "K21": "2D",
    "K22": "3D",
    "K23": "4D",
}


def main() -> None:
    DOC_MD_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = _extract_source_markdown()
    absorber = _parse_absorber_table(docs["morgan_2018"])
    capture = _parse_capture_table(docs["morgan_2018"])
    package = pd.read_csv(PACKAGE_NCCC).rename(columns={"Runs": "legacy_case_id", "CO2  %": "package_capture_gas_side_pct"})

    master = absorber.merge(capture, on="legacy_case_id", how="left", validate="one_to_one")
    master = master.merge(
        package[
            [
                "legacy_case_id",
                "L",
                "G",
                "alpha",
                "w_MEA",
                "y_CO2",
                "Tl",
                "Tv",
                "P",
                "package_capture_gas_side_pct",
            ]
        ],
        on="legacy_case_id",
        how="left",
        validate="one_to_one",
    )
    master = master.sort_values("legacy_case_number").drop(columns=["legacy_case_number"])

    output = REFERENCE_DIR / "nccc_master_cases.csv"
    analysis_copy = ANALYSIS_INPUT_DIR / "nccc_master_cases.csv"
    master.to_csv(output, index=False)
    master.to_csv(analysis_copy, index=False)
    _write_crosswalk_note(master)

    print(f"Wrote {output}")
    print(f"Wrote {analysis_copy}")
    print(f"Wrote {DOC_MD_DIR / 'nccc_case_crosswalk.md'}")


def _extract_source_markdown() -> dict[str, str]:
    if not ZIP_SOURCE.exists():
        raise FileNotFoundError(f"Missing source zip: {ZIP_SOURCE}")

    outputs = {
        MORGAN_2018_NAME: DOC_MD_DIR / "morgan_2018_supporting_information.md",
        MOORE_2021_NAME: DOC_MD_DIR / "moore_2021_advanced_absorber_heat_integration.md",
    }
    source_texts: dict[str, str] = {}
    with zipfile.ZipFile(ZIP_SOURCE) as archive:
        names = set(archive.namelist())
        missing = [name for name in outputs if name not in names]
        if missing:
            raise FileNotFoundError(f"Missing source markdown entries in {ZIP_SOURCE}: {missing}")
        for name, target in outputs.items():
            text = archive.read(name).decode("utf-8", errors="replace")
            target.write_text(text, encoding="utf-8")
            if name == MORGAN_2018_NAME:
                source_texts["morgan_2018"] = text
            elif name == MOORE_2021_NAME:
                source_texts["moore_2021"] = text
    return source_texts


def _parse_absorber_table(text: str) -> pd.DataFrame:
    rows = []
    for match in re.finditer(r"\\hline\s+(K\d+)\s*&\s*([^\\]+?)\\\\", _section(text, "Table S1", "Table S2")):
        values = [part.strip() for part in match.group(2).split("&")]
        if len(values) != 9:
            continue
        beds_match = re.match(r"(\d+)\s*\((\d+)\)", values[8])
        if beds_match is None:
            raise ValueError(f"Could not parse bed/intercooler field for {match.group(1)}: {values[8]}")
        legacy_case_id = match.group(1)
        appendix_id = K_TO_APPENDIX_STYLE.get(legacy_case_id, "")
        rows.append(
            {
                "legacy_case_id": legacy_case_id,
                "legacy_case_number": int(legacy_case_id[1:]),
                "appendix_style_case_id": appendix_id,
                "appendix_case_source": "mapped from 2018 K-row by bed/intercooler group and Appendix-C ordering"
                if appendix_id
                else "",
                "case_group": _case_group(appendix_id, int(beds_match.group(1)), int(beds_match.group(2))),
                "lean_solvent_flow_kg_hr": float(values[0]),
                "lean_solvent_flow_kg_s": float(values[0]) / 3600.0,
                "lean_solvent_temperature_C": float(values[1]),
                "lean_solvent_temperature_K": float(values[1]) + 273.15,
                "lean_loading_mol_co2_per_mol_mea": float(values[2]),
                "lean_mea_weight_fraction": float(values[3]),
                "gas_flow_kg_hr": float(values[4]),
                "gas_flow_kg_s": float(values[4]) / 3600.0,
                "gas_temperature_C": float(values[5]),
                "gas_temperature_K": float(values[5]) + 273.15,
                "inlet_co2_mol_percent": float(values[6]),
                "inlet_co2_mole_fraction": float(values[6]) / 100.0,
                "pressure_kPa": float(values[7]),
                "pressure_Pa": float(values[7]) * 1000.0,
                "beds": int(beds_match.group(1)),
                "intercoolers": int(beds_match.group(2)),
                "source_table": "Morgan et al. 2018 Supporting Information Table S1",
            }
        )
    if len(rows) != 23:
        raise ValueError(f"Expected 23 absorber rows, found {len(rows)}")
    return pd.DataFrame(rows)


def _parse_capture_table(text: str) -> pd.DataFrame:
    rows = []
    for match in re.finditer(r"\\hline\s+(K\d+)\s*&\s*([^\\]+?)\\\\", _section(text, "Table S3", "Table S4")):
        values = [part.strip() for part in match.group(2).split("&")]
        if len(values) != 4:
            continue
        rows.append(
            {
                "legacy_case_id": match.group(1),
                "capture_liquid_side_pct": float(values[0]),
                "capture_gas_side_pct": float(values[1]),
                "morgan_original_model_capture_pct": float(values[2]),
                "morgan_uncertainty_model_capture_pct": float(values[3]),
                "capture_source_table": "Morgan et al. 2018 Supporting Information Table S3",
            }
        )
    if len(rows) != 23:
        raise ValueError(f"Expected 23 capture rows, found {len(rows)}")
    return pd.DataFrame(rows)


def _section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        raise ValueError(f"Could not find section {start_marker!r} to {end_marker!r}")
    return text[start:end]


def _case_group(appendix_id: str, beds: int, intercoolers: int) -> str:
    if appendix_id.endswith("A"):
        return "A: three beds with two intercoolers"
    if appendix_id.endswith("B"):
        return "B: three beds without intercoolers"
    if appendix_id.endswith("C"):
        return "C: one bed without intercoolers"
    if appendix_id.endswith("D"):
        return f"D: two beds with {intercoolers} intercooler(s)"
    return f"K-only: {beds} beds with {intercoolers} intercooler(s)"


def _write_crosswalk_note(master: pd.DataFrame) -> None:
    mapped = master[["legacy_case_id", "appendix_style_case_id", "case_group", "beds", "intercoolers"]].copy()
    rows = "\n".join(
        f"| {row.legacy_case_id} | {row.appendix_style_case_id or '-'} | {row.case_group} | "
        f"{row.beds} | {row.intercoolers} |"
        for row in mapped.itertuples(index=False)
    )
    note = f"""# NCCC K-Case And Appendix-Style Crosswalk

This folder contains source markdown extracted from the local literature export used for the NCCC
intercooling and validation review. The canonical machine-readable case table is:

- `data/reference/nccc_master_cases.csv`
- `analyses/nccc_validation/data/input/nccc_master_cases.csv`

The Morgan et al. 2018 supporting-information table names the NCCC runs as `K1` through `K23`.
The Appendix-C-style A-D labels used by the temperature-profile plotting workflow are a plotting
nomenclature, not a replacement for the source K identifiers. The crosswalk below maps K rows by
bed/intercooler group and the existing Appendix-C plotting order.

| 2018 K case | Appendix-style case | Group | Beds | Intercoolers |
| --- | --- | --- | ---: | ---: |
{rows}

Notes:

- `A` rows are the 15 three-bed/two-intercooler K cases: K1-K12 and K14-K16.
- The 2018 K set contains one three-bed/no-intercooler absorber row, K13, mapped here to `1B`.
- The 2018 K set contains three one-bed rows, K18-K20, mapped here to `1C`-`3C`; the separate
  seven one-bed C-case dataset remains a different validation source.
- `D` rows cover the two-bed cases, with K22-K23 carrying one intercooler.
"""
    (DOC_MD_DIR / "nccc_case_crosswalk.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
