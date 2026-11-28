import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from mea_absorption_column.Run_Model import run_model
from mea_absorption_column.Thermodynamics.reactive_bundle import CORRELATION_SOURCES
from mea_absorption_column.benchmark import BenchmarkSettings, run_benchmark

out, test = Path(sys.argv[1]), sys.argv[2]
settings = dict(mesh_points=21, tol=0.5, bc_tol=.001, max_nodes=1000,
                chemical_equilibrium_model='epcsaft_reactive_nine', thermal_state_mode='temperature',
                transform_mode='raw', reactive_reuse_states=True)
source_keys = [*CORRELATION_SOURCES, 'co2_vapor_fugacity_basis', 'water_vaporization_enthalpy']
if test == 'run_model':
    cases = pd.read_csv('src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv', index_col=0)
    retained = Path('analyses/nccc_validation/results/runs/fitA_epcsaft_properties_20261128/inlet_humidity_3C')
    result = run_model(cases, run=cases.index.get_loc('3C'), method='scipy-bvp',
                       thermo_model='epcsaft_reactive_nine', return_details=True,
                       solver_settings={**settings, 'return_profiles': True})
    profiles = result.pop('_profiles')
    reference = pd.read_csv(retained/'Fv.csv').iloc[0]
    actual = profiles['Fv'].iloc[0]
    metrics = {key: result[key] for key in ('case_id', 'thermo_model', 'chemical_equilibrium_model',
               'success', 'message', 'capture_pct', 'runtime_s', 'max_rms_residual', 'gas_flow_basis',
               'continuation_path', 'epcsaft_chemistry_last_evidence', *source_keys)}
    metrics['inlet_y_H2O'] = float(actual.Fv_H2O / actual.Fv_T)
    metrics['retained_y_H2O'] = float(reference.Fv_H2O / reference.Fv_T)
    metrics['inlet_mol_s'] = actual.drop('Position', errors='ignore').to_dict()
    metrics['retained_inlet_mol_s'] = reference.drop('Position').to_dict()
    metrics['verification'] = {'command': sys.argv, 'wall_limit_s': 150}
    print(json.dumps(metrics, default=lambda x: x.item(), indent=2), flush=True)
    (out/'run_model.json').write_text(json.dumps(metrics, default=lambda x: x.item(), indent=2)+'\n')
    assert result['success'] and result['continuation_path'] == 'ideal_henry->epcsaft_reactive_nine'
    np.testing.assert_allclose(actual[reference.index[1:]].astype(float), reference.iloc[1:].astype(float),
                               rtol=0, atol=1e-9)
else:
    control = test.startswith('benchmark_43990')
    choices = dict(CORRELATION_SOURCES) if control else dict.fromkeys(CORRELATION_SOURCES, 'epcsaft')
    if control:
        settings.update(choices, water_vaporization_enthalpy=43990.)
        if test.endswith('_henry'):
            seed = Path('analyses/nccc_validation/results/runs/fitA_article_20261128/coarse/initial_scaled.csv')
            settings['initial_guess_scaled'] = pd.read_csv(seed, float_precision='round_trip').to_numpy().T.tolist()
    run_benchmark(BenchmarkSettings(
        methods=('scipy-bvp',), thermo_models=('epcsaft_reactive_nine',),
        c_case_limit=0, nccc_dataset='2017', nccc_case_ids=('3C',), data_type='mass',
        output_dir=out/test, solver_settings=settings, subprocess_timeout_s=120,
    ))
    record = pd.read_csv(out/test/'benchmark_results.csv').iloc[0]
    print(record[['case_id', 'success', 'message', 'capture_pct', 'runtime_s', *source_keys]].to_string(), flush=True)
    assert record['success'] and record['case_id'] == '3C'
    assert record[list(choices)].to_dict() == choices
    assert record.co2_vapor_fugacity_basis == 'CO2/MEA/H2O reduced composition, rescaled to wet CO2 mole fraction'
    assert record.water_vaporization_enthalpy == 43990. if control else pd.isna(record.water_vaporization_enthalpy)
