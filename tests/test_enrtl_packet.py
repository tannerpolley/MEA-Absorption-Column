from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest

import mea_absorption_column_enrtl_packet as packet


ROOT = Path(__file__).resolve().parents[1]


def _sandbox(tmp_path, monkeypatch):
    data = tmp_path / "data"
    shutil.copytree(ROOT / "src/mea_absorption_column_enrtl_packet/data/enrtl_packet", data / "enrtl_packet")
    shutil.copy(
        ROOT / "src/mea_absorption_column_enrtl_packet/data/enrtl_packet_adoption_receipt.json",
        data / "enrtl_packet_adoption_receipt.json",
    )
    monkeypatch.setattr(packet, "_resource", lambda *parts: data.joinpath(*parts))
    return data


def _refresh_hashes(data):
    manifest_path = data / "enrtl_packet/manifest.json"
    payload_path = data / "enrtl_packet/payload.json"
    manifest = json.loads(manifest_path.read_text())
    content = payload_path.read_bytes()
    for item in manifest["files"]:
        if item["path"] == "payload.json":
            item["bytes"] = len(content)
            item["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    receipt_path = data / "enrtl_packet_adoption_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for name, path in (("manifest.json", manifest_path), ("payload.json", payload_path)):
        content = path.read_bytes()
        receipt["files"][name]["bytes"] = len(content)
        receipt["files"][name]["sha256"] = hashlib.sha256(content).hexdigest()
    receipt["manifest_sha256_external"] = receipt["files"]["manifest.json"]["sha256"]
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


def test_source_resource_load_and_contract():
    manifest, payload = packet.validate_packet()
    assert manifest["files"][0]["path"] == "payload.json"
    assert payload["fit"]["observation_counts"]["vle"] == 240
    assert packet.load_packet(
        expected_model="mea-enrtl-six-parameter-corrected-dh-v1",
        expected_species_order=manifest["model"]["species_order"],
        expected_reaction_order=manifest["model"]["reaction_order"],
        expected_property_basis="true",
        expected_domain="calibration_VLE",
        result_identity="baseline-six-parameter-physical-reaction-coordinates",
    )["identity"] == payload["identity"]


def test_changed_payload_is_rejected(tmp_path, monkeypatch):
    data = _sandbox(tmp_path, monkeypatch)
    payload = data / "enrtl_packet/payload.json"
    payload.write_bytes(payload.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        packet.validate_packet()


@pytest.mark.parametrize("case", ["model", "species", "reaction", "basis", "domain", "baseline", "unavailable", "counts"])
def test_semantic_tampering_is_rejected_after_hash_refresh(tmp_path, monkeypatch, case):
    data = _sandbox(tmp_path, monkeypatch)
    manifest_path = data / "enrtl_packet/manifest.json"
    payload_path = data / "enrtl_packet/payload.json"
    manifest = json.loads(manifest_path.read_text())
    payload = json.loads(payload_path.read_text())
    if case == "model":
        manifest["model"]["identity"] = "wrong"
    elif case == "species":
        manifest["model"]["species_order"] = ["wrong"]
    elif case == "reaction":
        manifest["model"]["reaction_order"] = ["wrong"]
    elif case == "basis":
        manifest["model"]["bases"]["property_basis"] = "apparent"
    elif case == "domain":
        manifest["property_domains"]["calibration_VLE"]["loading"] = "wrong"
    elif case == "baseline":
        payload["baseline_identity"] = "wrong"
    elif case == "unavailable":
        manifest["unavailable_properties"].pop()
    else:
        payload["fit"]["observation_counts"]["vle"] = 241
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    _refresh_hashes(data)
    with pytest.raises(ValueError):
        packet.validate_packet()


def test_non_finite_payload_is_rejected_after_hash_refresh(tmp_path, monkeypatch):
    data = _sandbox(tmp_path, monkeypatch)
    payload_path = data / "enrtl_packet/payload.json"
    payload = json.loads(payload_path.read_text())
    payload["parameters"][0]["value"] = float("nan")
    payload_path.write_text(json.dumps(payload))
    _refresh_hashes(data)
    with pytest.raises(ValueError):
        packet.validate_packet()


def test_substituted_manifest_is_rejected(tmp_path, monkeypatch):
    data = _sandbox(tmp_path, monkeypatch)
    manifest = data / "enrtl_packet/manifest.json"
    value = json.loads(manifest.read_text())
    value["model"]["identity"] = "wrong"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        packet.validate_packet()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_model": "wrong"},
        {"expected_species_order": ["wrong"]},
        {"expected_reaction_order": ["wrong"]},
        {"expected_property_basis": "apparent"},
        {"expected_domain": "not-declared"},
        {"result_identity": "wrong"},
        {"requested_property": "total_reacting_solution_heat_capacity"},
    ],
)
def test_declared_boundaries_are_rejected(kwargs):
    with pytest.raises(ValueError):
        packet.load_packet(**kwargs)
