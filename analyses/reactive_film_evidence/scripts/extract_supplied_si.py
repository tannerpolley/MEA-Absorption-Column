from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree


FIELDS = [
    "table_id",
    "mixture",
    "T_K",
    "co2_loading_mol_per_mol_mea",
    "co2_loading_uncertainty",
    "mea_wt_pct",
    "sugar_wt_pct",
    "water_wt_pct",
    "property_or_metric",
    "units",
    "value",
    "comparison_or_reference",
    "source_locator",
    "evidence_label",
    "claim_scope",
    "K_G_x10e4_mol_m2_kPa_s",
    "D_CO2_x10e9_m2_s",
    "H_CO2_kPa_m3_kmol",
    "k_l_x10e5_m_s",
    "k_g_x10e2_m_s",
    "k_ov_s_1",
    "Ha",
    "E_inf",
    "zotero_collection_key",
    "zotero_parent_key",
    "zotero_attachment_key",
    "local_companion_locator",
]
RAMEZANI_PROVENANCE = ("9T8XETPA", "VMZKH34U", "NCLG54A3", "Downloads/Zotero-Supplements/Ramezani_2021_Supporting_Information.pdf")
GANESAN_PROVENANCE = ("9T8XETPA", "KVLELL3S", "77DLGVMR", "Downloads/Zotero-Supplements/Ganesan_2026_Supplementary_Information.pdf")
assert RAMEZANI_PROVENANCE[2] == "NCLG54A3"


def _text(path: Path) -> str:
    return subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _row(**values: str) -> dict[str, str]:
    source = values.get("table_id", "")
    if source in {"S1", "S2", "S3", "S4", "S5"}:
        provenance = RAMEZANI_PROVENANCE
    elif source in {"S6.1", "S6.2"}:
        provenance = GANESAN_PROVENANCE
    else:
        provenance = ("", "", "", "")
    defaults = dict(zip(FIELDS[-4:], provenance))
    defaults.update(values)
    return {field: defaults.get(field, "") for field in FIELDS}


def _numbers(line: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])(?:\d+\.\d+|\d+)(?![A-Za-z])|-", line)


S5_FIELDS = FIELDS[15:23]
assert S5_FIELDS == [
    "K_G_x10e4_mol_m2_kPa_s",
    "D_CO2_x10e9_m2_s",
    "H_CO2_kPa_m3_kmol",
    "k_l_x10e5_m_s",
    "k_g_x10e2_m_s",
    "k_ov_s_1",
    "Ha",
    "E_inf",
]


def _docx_table_rows(path: Path, table_index: int) -> list[list[str]]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    table = root.findall(".//w:tbl", namespace)[table_index]
    return [
        ["".join(cell.itertext()).strip() for cell in row.findall("./w:tc", namespace)]
        for row in table.findall("./w:tr", namespace)
    ]


def extract_ramezani(path: Path, docx_path: Path, output: Path) -> None:
    text = _text(path)
    rows: list[dict[str, str]] = []

    s1 = text.split("Table S2", 1)[0]
    for line in s1.splitlines():
        values = _numbers(line)
        if len(values) == 7 and values[0].startswith(("298.", "313.", "323.", "333.", "343.")):
            names = ["this_work", "Bernhardsen_et_al", "Hartono_et_al", "Han_et_al", "water_this_work", "Spieweck_et_al"]
            for name, value in zip(names, values[1:]):
                rows.append(_row(
                    table_id="S1",
                    mixture="30 wt% MEA / pure water comparison",
                    T_K=values[0],
                    property_or_metric="density",
                    units="g/cm3",
                    value=value,
                    comparison_or_reference=name,
                    source_locator="Ramezani2021 supplied SI PDF p. 2, Table S1",
                    evidence_label="verified",
                    claim_scope="retained_property_observation_or_comparison",
                ))

    def property_table(table_id: str, property_name: str, units: str, start: str, end: str) -> None:
        section = text.split(start, 1)[1].split(end, 1)[0]
        for line in section.splitlines():
            values = _numbers(line)
            if len(values) not in (10, 11) or not values[0].startswith(("298.", "313.", "323.", "333.", "343.")):
                continue
            T, mea, water, water_u, sugar = values[:5]
            sugar_u = values[5] if len(values) == 11 else ""
            data = values[6:] if len(values) == 11 else values[5:]
            for loading, value in zip(("0", "0.100", "0.200", "0.300", "0.400"), data):
                rows.append(_row(
                    table_id=table_id,
                    mixture="30 wt% MEA + sugar" if sugar != "0" else "30 wt% MEA",
                    T_K=T,
                    co2_loading_mol_per_mol_mea=loading,
                    co2_loading_uncertainty={"0": "", "0.100": "0.001", "0.200": "0.003", "0.300": "0.004", "0.400": "0.005"}[loading],
                    mea_wt_pct=mea,
                    sugar_wt_pct=sugar,
                    water_wt_pct=water,
                    property_or_metric=property_name,
                    units=units,
                    value=value,
                    source_locator=f"Ramezani2021 supplied SI PDF p. {'3' if table_id == 'S2' else '4'}, Table {table_id}",
                    evidence_label="verified",
                    claim_scope="retained_property_observation",
                ))

    property_table("S2", "density", "g/cm3", "Table S2", "Table S3")

    s3 = text.split("Table S3", 1)[1].split("Table S4", 1)[0]
    for line in s3.splitlines():
        values = _numbers(line)
        if len(values) == 5 and values[0].startswith(("298.", "313.", "323.", "333.", "343.")):
            for name, value in zip(("this_work", "Bernhardsen_et_al", "Hartono_et_al", "Amundsen_et_al"), values[1:]):
                rows.append(_row(
                    table_id="S3",
                    mixture="30 wt% MEA",
                    T_K=values[0],
                    property_or_metric="viscosity",
                    units="mPa s",
                    value=value,
                    comparison_or_reference=name,
                    source_locator="Ramezani2021 supplied SI PDF p. 3, Table S3",
                    evidence_label="verified",
                    claim_scope="retained_property_observation_or_comparison",
                ))

    property_table("S4", "viscosity", "mPa s", "Table S4", "Table S5")

    mixture = ""
    for values in _docx_table_rows(docx_path, 4):
        if len(values) == 1 and values[0].startswith("30 wt% MEA"):
            mixture = values[0]
            continue
        if len(values) != 10 or not values[0][:1].isdigit() or "." not in values[0]:
            continue
        loading_parts = values[1].replace(" ", "").split("±")
        loading, uncertainty = loading_parts[0], loading_parts[1] if len(loading_parts) == 2 else ""
        rows.append(_row(
            table_id="S5",
            mixture=mixture,
            T_K=values[0],
            co2_loading_mol_per_mol_mea=loading,
            co2_loading_uncertainty=uncertainty,
            mea_wt_pct="30",
            sugar_wt_pct="0" if mixture == "30 wt% MEA" else re.search(r"\+ (\d+) wt% sugar", mixture).group(1),
            property_or_metric="kinetics_state",
            units="reported Table S5 columns retained individually",
            source_locator="Ramezani2021 supplied SI DOCX XML Table 5; PDF pp. 5-8 Table S5",
            evidence_label="verified",
            claim_scope="retained_kinetics_observation",
            **dict(zip(S5_FIELDS, values[2:])),
        ))

    _write(output, rows)
    assert sum(row["table_id"] == "S1" for row in rows) == 30
    assert sum(row["table_id"] == "S2" for row in rows) == 125
    assert sum(row["table_id"] == "S3" for row in rows) == 20
    assert sum(row["table_id"] == "S4" for row in rows) == 125
    assert sum(row["table_id"] == "S5" for row in rows) == 125
    assert all(all(row[field] for field in S5_FIELDS) for row in rows if row["table_id"] == "S5")


def extract_ganesan(path: Path, output: Path) -> None:
    text = _text(path)
    rows: list[dict[str, str]] = []
    s61_values = {
        "0.048": {"333.15": "24.21"},
        "0.063": {"313.15": "10.31", "323.15": "19.25", "333.15": "45.48"},
        "0.092": {"303.15": "5.65", "313.15": "17.13", "323.15": "49.01", "333.15": "120.2"},
        "0.091": {"313.15": "19.87"},
        "0.131": {"303.15": "12.21", "313.15": "40.57", "323.15": "125.13", "333.15": "217.14"},
        "0.188": {"303.15": "30.63", "313.15": "70.80", "323.15": "267.12", "333.15": "625.72"},
        "0.192": {"333.15": "562.08"},
        "0.157": {"303.15": "25.15", "313.15": "61.14", "323.15": "162.64", "333.15": "477.87"},
        "0.223": {"303.15": "44.18", "313.15": "117.75", "323.15": "311.62", "333.15": "842.01"},
        "0.251": {"303.15": "66.87", "313.15": "182.39", "323.15": "499.53", "333.15": "1306.08"},
        "0.269": {"313.15": "261.54"},
        "0.275": {"303.15": "102.34", "313.15": "261.43", "323.15": "705.22", "333.15": "1796"},
        "0.304": {"313.15": "408.31"},
        "0.308": {"313.15": "413.37"},
        "0.314": {"313.15": "418.46"},
    }
    for loading, by_temperature in s61_values.items():
        for T, value in by_temperature.items():
            rows.append(_row(
                table_id="S6.1",
                mixture="5 M MEA (30 wt%)",
                T_K=T,
                co2_loading_mol_per_mol_mea=loading,
                property_or_metric="equilibrium_pCO2",
                units="ppm",
                value=value,
                source_locator="Ganesan2026 supplied SI PDF p. 10, Table S6.1",
                evidence_label="verified",
                claim_scope="retained_VLE_observation",
            ))
    section = text.split("Table S6.2", 1)[1]
    for line in section.splitlines():
        values = _numbers(line)
        if len(values) == 7 and values[0].startswith(("293.", "295.", "297.", "298.", "300.", "301.", "302.", "303.", "304.", "305.", "307.", "308.", "310.", "311.", "312.", "314.", "315.")):
            for name, value, units in zip(("m", "kg", "kL", "Ha", "m_kL_Ei"), values[2:], ("dimensionless", "10^-3 m/s", "10^-5 m/s", "dimensionless", "10^-3 m/s")):
                rows.append(_row(
                    table_id="S6.2",
                    mixture="5 M MEA (30 wt%)",
                    T_K=values[0],
                    co2_loading_mol_per_mol_mea=values[1],
                    property_or_metric=name,
                    units=units,
                    value=value,
                    source_locator="Ganesan2026 supplied SI PDF p. 10-11, Table S6.2",
                    evidence_label="verified",
                    claim_scope="retained_specific_absorption_flux_input",
                ))
    _write(output, rows)
    assert sum(row["table_id"] == "S6.1" for row in rows) == 38
    assert sum(row["table_id"] == "S6.2" for row in rows) == 155


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramezani", type=Path, required=True)
    parser.add_argument("--ramezani-docx", type=Path, required=True)
    parser.add_argument("--ganesan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract_ramezani(args.ramezani, args.ramezani_docx, args.output_dir / "ramezani2021_si_rows.csv")
    extract_ganesan(args.ganesan, args.output_dir / "ganesan2026_si_rows.csv")


if __name__ == "__main__":
    main()
