"""Compare cold and neighboring-temperature starts at the exact failed state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--failure', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model_root.resolve()/'src'))
    from mea_absorption_column.Thermodynamics.reactive_bundle import DATASET, ReactiveLiquid
    failures = json.loads(args.failure.read_text())['failures']
    if len(failures) != 2:
        raise ValueError('This probe expects exactly one nested cold-anchor failure and its outer column state')
    anchor, target = failures[0], failures[-1]
    if (anchor['temperature_K'] != target['temperature_K'] or anchor['pressure_Pa'] != target['pressure_Pa']
            or anchor['options'].get('loading_anchor') is not None or target['options'].get('loading_anchor') is None):
        raise ValueError('Failure records do not identify a matched cold anchor and column target')
    temperature, pressure = target['temperature_K'], target['pressure_Pa']
    liquid = ReactiveLiquid(DATASET, reuse_states=True)
    records = []

    def evaluate(label, t, amounts):
        row = dict(label=label, temperature_K=t, pressure_Pa=pressure, apparent_amounts_mol=amounts)
        try:
            result = liquid.solve(t, pressure, amounts, state_input_derivatives=True)
            row.update(success=True, density_mol_m3=result['density_mol_m3'],
                amounts_mol=result['amounts_mol'].tolist(), feed_amounts_mol=result['feed_amounts_mol'].tolist(),
                fugacity_CO2_Pa=float(result['fugacities_pa'][0]), evidence=result['evidence'])
        except Exception as error:
            row.update(success=False, error=str(error))
        records.append(row)
        print(json.dumps(row), flush=True)
        return row['success']

    evaluate('exact_anchor_cold', temperature, anchor['apparent_amounts_mol'])
    # A 0.01 K neighboring state tests initialization, not an altered target.
    if evaluate('neighbor_anchor_cold', temperature+.01, anchor['apparent_amounts_mol']):
        if evaluate('exact_anchor_warm', temperature, anchor['apparent_amounts_mol']):
            evaluate('exact_column_state_warm', temperature, target['apparent_amounts_mol'])
    (args.failure.parent/'state_probe.json').write_text(json.dumps(records, indent=2)+'\n')
    (args.failure.parent/'probe_identity.json').write_text(json.dumps(dict(
        failure_sha256=hashlib.sha256(args.failure.read_bytes()).hexdigest(),
        model_root=str(args.model_root.resolve()),
        model_identity=json.loads((args.failure.parent/'identity.json').read_text()),
        scope='State-only feasibility with a direct warm start; not a replay of the full column loading path'), indent=2)+'\n')
    if len(records) != 4 or not all(row['success'] for row in records[1:]):
        raise SystemExit('State-only diagnostic incomplete; retained records do not establish warm feasibility')


if __name__ == '__main__':
    main()
