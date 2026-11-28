"""Compare fit-A enthalpy increments to Kim calorimeter dose intervals."""
import argparse
from functools import cache
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from mea_absorption_column.Thermodynamics.reactive_bundle import (
    FIT_A, ReactiveLiquid, bulk_vapor_state, pure_vapor_enthalpy,
)


OBSERVATIONS = 'data/reference/MEA/observations/calorimetry/MEA_heat_of_absorption_observations.csv'
RETAINED = 'analyses/amines/results/mea/reported/fit/heat-comparison.json.gz'
STEAM_TABLE = 'https://www.nist.gov/system/files/documents/srd/NISTIR5078-Tab1.pdf'


def compare(amine, output):
    source, retained_path = amine/OBSERVATIONS, amine/RETAINED
    observations = pd.read_csv(source)
    kim = observations[observations.source.eq('Kim and Svendsen 2007') & (observations.mea_mass_fraction == .30)]
    retained = json.loads(gzip.decompress(retained_path.read_bytes()))
    assert retained['endpoint_fingerprint'].startswith('196faf11')
    runtime = ReactiveLiquid(FIT_A, loading_anchor=.25, reuse_states=True)
    water = .70*runtime.molar_masses[1]/(.30*runtime.molar_masses[2])
    @cache
    def liquid_h(temperature, pressure, loading):
        return runtime.system_enthalpy(temperature, pressure, (loading, 1., water))

    @cache
    def gas_h(temperature, pressure):
        return pure_vapor_enthalpy(temperature, pressure, 'carbon-dioxide', FIT_A)

    rows = []
    for (temperature, run), group in kim.groupby(['temperature_K', 'run']):
        previous = 0.
        for row in group.itertuples():
            end = float(row.co2_loading_mol_per_mol_mea)
            record = dict(source='Kim and Svendsen 2007', id=row.record_id, run=int(run), temperature_K=temperature,
                          start_loading=previous, end_loading=end, observed_release_kJ_mol=row.dh_kj_per_mol_co2,
                          uncertainty_pct=row.measurement_uncertainty_relative_percent)
            if temperature > 353.15:
                record.update(status='outside_fit_A_temperature_range')
            else:
                record.update(status='available', pressure_Pa=300000.)
                for label, pressure in [('release_kJ_mol', 300000.), ('release_at_10bar_kJ_mol', 1000000.)]:
                    h0, solver = liquid_h(temperature, pressure, previous)
                    h1, _ = liquid_h(temperature, pressure, end)
                    record[label] = (gas_h(temperature, pressure)-(h1-h0)/(end-previous))/1000.
                record['linear_solver'] = solver
            rows.append(record)
            previous = end

    kim2014 = {item['record_id']: item for item in retained['rows']['rows'] if item['source'] == 'Kim2014'}
    for item in kim2014.values():
        if item['T_K'] <= 353.15:
            rows.append(dict(source='Kim et al. 2014', id=item['record_id'], temperature_K=item['T_K'],
                             start_loading=item['previous_loading'], end_loading=item['loading'],
                             observed_release_kJ_mol=item['observed_kJ_mol'], release_kJ_mol=item['predicted_kJ_mol'],
                             pressure_Pa=retained['rows']['system_pressure_Pa'], status=item['status'],
                             diagnostic=item.get('diagnostic')))
    excluded = observations[observations.source.eq('Kim et al. 2014') & (observations.temperature_K > 353.15)]
    for row in excluded.itertuples():
        archived = kim2014.get(row.record_id, {})
        rows.append(dict(source='Kim et al. 2014', id=row.record_id, temperature_K=row.temperature_K,
                         start_loading=archived.get('previous_loading'),
                         end_loading=row.co2_loading_mol_per_mol_mea, observed_release_kJ_mol=row.dh_kj_per_mol_co2,
                         retained_release_outside_fit_A_kJ_mol=archived.get('predicted_kJ_mol'),
                         status='outside_fit_A_temperature_range'))

    doses = pd.DataFrame(rows)
    doses['error_kJ_mol'] = doses.release_kJ_mol-doses.observed_release_kJ_mol
    doses['relative_error_pct'] = 100.*doses.error_kJ_mol/doses.observed_release_kJ_mol
    doses['within_stated_uncertainty'] = (doses.relative_error_pct.abs() <= doses.uncertainty_pct).where(
        doses.uncertainty_pct.notna() & doses.relative_error_pct.notna())
    doses.to_csv(output/'C5_absorption_enthalpy.csv', index=False)
    results = []
    for (source_name, t), group in doses.groupby(['source', 'temperature_K']):
        valid = group[group.status == 'available']
        entry = dict(source=source_name, temperature_K=t, available=len(valid),
                     status_counts=group.status.value_counts().to_dict())
        if len(valid):
            entry.update(mae_kJ_mol=valid.error_kJ_mol.abs().mean(), aard_pct=valid.relative_error_pct.abs().mean(),
                         max_abs_relative_error_pct=valid.relative_error_pct.abs().max())
            if valid.uncertainty_pct.notna().any():
                entry.update(within_2_2_pct=int(valid.within_stated_uncertainty.sum()),
                             max_3_to_10bar_change_kJ_mol=(valid.release_at_10bar_kJ_mol-valid.release_kJ_mol).abs().max())
        results.append(entry)

    latent = []
    for t, steam in [(313.15, 2406.), (353.15, 2308.)]:
        p = 109500.
        v = bulk_vapor_state(t, p, (2., 1.67, 20., 2.), str(FIT_A), True)
        for loading in (.10, .25, .40, .50):
            l = runtime.solve(t, p, (loading, 1., water), caloric=True)
            latent.append(dict(temperature_K=t, pressure_Pa=p, loading=loading,
                               H2O_transfer_kJ_mol=(v['enthalpy_partials'][2]-l['enthalpy_partials'][2])/1000.,
                               pure_water_latent_kJ_mol=steam*runtime.molar_masses[2],
                               liquid_linear_solver=l['evidence']['linear_solver'],
                               vapor_linear_solver=v['linear_solver']))
    pd.DataFrame(latent).to_csv(output/'water_latent_enthalpy.csv', index=False)
    summary = dict(
        statistics=results, steam_table=STEAM_TABLE,
        kim2007_basis='Reported positive release. 30 wt % unloaded MEA. Dose interval enthalpy difference minus pure CO2 '
                       'vapor reference at test T. Nominal 3 bar feed and 10 bar sensitivity. Dose-specific reactor '
                       'pressures were not tabulated, and reactor pressure rose during each series.',
        kim2014_basis='Retained Amine-Thermodynamics comparison on the 1.734 MPa common-pressure basis chosen there. '
                      'Kim 2014 did not report dose-level heat uncertainty. 1.734 MPa was not a Kim test pressure.',
        kim2014_retained_wheel_sha256=retained['wheel'], kim2014_retained_engine_commit=retained['rows']['engine_commit'],
        source_sha256={str(path.relative_to(amine)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (source, retained_path)})
    (output/'C5.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('amine', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    compare(args.amine, args.output)
