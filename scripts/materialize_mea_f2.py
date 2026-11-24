"""Materialize the adopted F2 inputs from the Amine-Thermodynamics Git record."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout")
    parser.add_argument("commit")
    args = parser.parse_args()
    git = ["git", "-C", args.checkout]
    commit = subprocess.check_output([*git, "rev-parse", "--verify", "--end-of-options",
                                      f"{args.commit}^{{commit}}"], text=True).strip()

    def read(path):
        return subprocess.check_output([*git, "show", f"{commit}:{path}"])

    spec_path = Path("analyses/solvent_specification_mea/MEA.toml")
    history_path = "analyses/mea_parameter_bundle/results/parameter-record-history.csv"
    spec = tomllib.loads(read(spec_path).decode())
    history = list(csv.DictReader(read(history_path).decode().splitlines()))
    sources, contents = {}, {}
    for key in ("endpoint", "declarations", "caloric"):
        path = str(spec_path.parent / spec[key]["path"])
        contents[key] = read(path)
        digest = sha256(contents[key]).hexdigest()
        if digest != spec[key]["sha256"]:
            raise ValueError(f"SHA-256 mismatch against {spec_path}: {path}")
        sources[key] = {"path": path, "sha256": digest}

    parameter_hash = sources["endpoint"]["sha256"]
    if parameter_hash != history[-1]["candidate_sha256"]:
        raise ValueError(f"Parameter SHA-256 disagrees with the last row of {history_path}")
    chemistry = tomllib.loads(contents["declarations"].decode())["chemistry"]
    for reactions in chemistry["reactions"].values():
        for row in reactions:
            correlation = row["correlation"]
            if any(correlation.pop(key, 0) != 0 for key in ("e", "f")):
                raise ValueError(f"Nonzero e or f coefficient in {row['name']}")
    reference = chemistry["neutral_references"]["aqueous"]
    if reference.pop("species_reference", {}):
        raise ValueError("Non-empty neutral-reference species_reference")
    if commit != "761e38ae866419bebb5895574e9c07645fbf6e34":
        raise ValueError(f"Commit is not the bound F2 source: {commit}")

    provenance = {"repository": "https://github.com/tannerpolley/Amine-Thermodynamics",
                  "commit": commit, "specification": str(spec_path), "parameter_history": history_path, "sources": sources}
    record = {"schema": "absorber-engine-reactions-v1", "parameter_sha256": parameter_hash, "provenance": provenance,
              "neutral_reference": reference, "reactions": chemistry["reactions"]["aqueous"]}
    target = Path(__file__).resolve().parents[1] / "src/mea_absorption_column/data/epcsaft_datasets/MEA_F2_ae17803e"
    target.mkdir(exist_ok=True)
    (target / "parameters.json").write_bytes(contents["endpoint"])
    (target / "ideal-gas-thermochemistry.json").write_bytes(contents["caloric"])
    (target / "engine-reactions.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
