"""Retain the exact differences from the correlation fit-A article."""
import json
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parent
CORRELATION = RUNS.with_name('fitA_article_20261128')
STUDIES = {
    'campaign': ('figures/campaign/summary.csv', 'case_id'),
    'refinement': ('figures/refinement/summary.csv', 'mesh_points'),
    'thermodynamics': ('figures/thermodynamics/summary.csv', 'run'),
    'transport': ('figures/transport/summary.csv', 'run'),
    'transport_initialization': ('figures/transport/initialization_control.csv', 'run'),
    'operating': ('figures/operating/summary.csv', 'scenario'),
    'timing': ('timing.csv', 'run'),
}


def main():
    rows = []

    def add(study, row, quantity, before, after):
        rows.append((study, row, quantity, before, after, after-before))

    for study, (file, index) in STUDIES.items():
        before, after = [pd.read_csv(root/file, float_precision='round_trip').set_index(index)
                         for root in (CORRELATION, RUNS)]
        numerical = before.select_dtypes('number').columns.intersection(after.select_dtypes('number').columns)
        for row in before.index:
            for quantity in numerical:
                a, b = before.at[row, quantity], after.at[row, quantity]
                if pd.notna(a) and pd.notna(b):
                    add(study, row, quantity, a, b)
        if study == 'campaign':
            add(study, 'all seven', 'capture_MAE_pp', before.capture_error_pct.abs().mean(),
                after.capture_error_pct.abs().mean())
        if study == 'refinement':
            for quantity in ('capture_pct', 'peak_liquid_temperature_K', 'peak_vapor_temperature_K'):
                add(study, '41 minus 21', quantity, before.at[41, quantity]-before.at[21, quantity],
                    after.at[41, quantity]-after.at[21, quantity])
        if study == 'timing':
            for quantity in ('runtime_s', 'total_wall_including_seed_and_outputs_s', 'peak_rss_mib'):
                add(study, 'median', quantity, before[quantity].median(), after[quantity].median())
    for run in ('energy_paired_replay', 'energy_dense_41', 'energy_dense_81'):
        records = [json.loads((root/run/'energy_check.json').read_text()) for root in (CORRELATION, RUNS)]
        add('energy', run, 'dense_range_W', records[0]['dense']['range_W'], records[1]['range_W'])
    result = pd.DataFrame(rows, columns=['study', 'row', 'quantity', 'correlation_fit_A', 'epcsaft_fit_A', 'change'])
    result.to_csv(RUNS/'article_comparison.csv', index=False)


if __name__ == '__main__':
    main()
