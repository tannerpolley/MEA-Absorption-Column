"""Materialize the MEA fit reported in the Amine-Thermodynamics manuscript (fit A) from its Git record."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tomllib

FIT = "analyses/amines/results/mea/reported/fit"
# The MEA route binds this specification and replaces only its parameters and reactions.
CALORIC = "analyses/solvent_specification_mea/inputs/caloric.json"
DECLARATIONS = "analyses/solvent_specification_mea/declarations.toml"


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

    selection = json.loads(read(f"{FIT}/selection.json"))
    fit_bytes = read(f"{FIT}/{selection['selected']}.json.gz")
    fit = json.loads(gzip.decompress(fit_bytes))
    if not (fit["replay_passed"] and fit["endpoint_fingerprint"] == selection["endpoint_fingerprint"]):
        raise ValueError("Selected fit record does not match its selection")
    caloric = read(CALORIC)
    parameters = (json.dumps(fit["endpoint"]["parameters"], indent=2, sort_keys=True) + "\n").encode()
    reactions = fit["endpoint"]["reactions"]["aqueous"]
    for row in reactions:
        if any(row["correlation"].pop(key, 0) != 0 for key in ("e", "f")):
            raise ValueError(f"Nonzero e or f coefficient in {row['name']}")
    neutral_reference = tomllib.loads(read(DECLARATIONS).decode())["chemistry"]["neutral_references"]["aqueous"]
    if neutral_reference.pop("species_reference", {}):
        raise ValueError("Non-empty neutral-reference species_reference")
    record = {"schema": "absorber-engine-reactions-v1", "parameter_sha256": sha256(parameters).hexdigest(),
              "provenance": {"repository": "https://github.com/tannerpolley/Amine-Thermodynamics", "commit": commit,
                             "selection": f"{FIT}/selection.json", "selected": selection["selected"],
                             "fit_record": {"path": f"{FIT}/{selection['selected']}.json.gz", "sha256": sha256(fit_bytes).hexdigest()},
                             "endpoint_fingerprint": fit["endpoint_fingerprint"], "fit_wheel_sha256": fit["wheel"],
                             "caloric": {"path": CALORIC, "sha256": sha256(caloric).hexdigest()}},
              "neutral_reference": neutral_reference, "reactions": reactions}
    target = Path(__file__).resolve().parents[1] / "src/mea_absorption_column/data/epcsaft_datasets" / f"MEA_{selection['selected']}_{fit['endpoint_fingerprint'][:8]}"
    target.mkdir(exist_ok=True)
    (target / "parameters.json").write_bytes(parameters)
    (target / "ideal-gas-thermochemistry.json").write_bytes(caloric)
    (target / "engine-reactions.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
