"""Replay source records and release gates without overwriting bundle-1 evidence."""
from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

import resolve_issue36_film_input_release as release
import resolve_issue42_film_transport as transport

ROOT = release.ROOT
RUN = release.ANALYSIS / 'results/runs/issue78_bundle3'


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def replay(stage: str) -> None:
    revision, dirty = release.git_revision()
    metadata = {
        'source_revision': revision,
        'generator_sha256': release.sha256_path(Path(__file__)),
        'exact_command': f'uv run --frozen python {Path(__file__).relative_to(ROOT)} --stage {stage}',
        'machine': platform.platform(), 'workers': 1,
        'run_id': f'issue78_bundle3_{stage}_{revision[:12]}',
    }
    output = RUN / stage
    output.mkdir(parents=True, exist_ok=True)
    if stage == 'transport':
        config = transport.load_json(transport.INPUT)
        metadata['input_sha256'] = transport.sha256(transport.INPUT)
        documents = transport.validate_source_documents(config)
        i35, i40, i41, dependencies = transport.validate_dependencies(config)
        replay35 = transport.replay_issue35(config)
        rows = transport.source_rows(config, metadata)
        comparison = transport.comparison_rows(config, dependencies['issue40_rows'], metadata)
        transport.write_csv(output / 'transport_inputs.csv', rows, transport.SOURCE_FIELDS)
        transport.write_csv(output / 'transport_comparison.csv', comparison, transport.COMPARISON_FIELDS)
        report = output / 'transport.md'
        report.write_text('Bundle 3 source correction: loaded density and viscosity are not admitted; no loaded row-level uncertainty. All five transport states remain blocked; no flux or uncertainty interval is calculated. Amundsen2009 pp. 3096-3100.\n')
        summary = {
            'metadata': metadata, 'input': config,
            'row_counts': {
                'transport_records': len(rows), 'declared_comparison_states': len(comparison),
                'evaluated_comparison_states': sum(r['attempt_status'] != 'not_attempted' for r in comparison),
                'scientifically_admitted_states': sum(r['decision'] == 'admitted' for r in comparison),
            },
            'source_documents_verified': documents, 'issue35_replay': replay35,
            'gates': {'physical_transport_adoption': False, 'no_package_or_parameter_identity_used': True, 'no_capture_inference': True},
            'output_sha256': {'transport_comparison': transport.sha256(output / 'transport_comparison.csv'), 'report': transport.sha256(report)},
        }
    else:
        config = release.load_json(release.INPUT)
        metadata['input_sha256'] = release.sha256_path(release.INPUT)
        owner = release.validate_work_package_a(config)
        guard = release.validate_release_guard(config)
        bundle_path = Path(config['bundle']['path'])
        bundle = (release.validate_bundle(config) if bundle_path.is_file() else {
            'verification_status': 'not_run_missing_archive',
            'path': str(bundle_path), 'expected_sha256': config['bundle']['outer_sha256'],
        })
        input_bytes = subprocess.check_output(['git', '-C', owner['local_checkout'], 'show',
            f"{owner['owner_revision']}:{owner['files'][0]['path']}"])
        properties = json.loads(input_bytes)['property_inputs']
        observation = properties[0]
        observation_bytes = subprocess.check_output(['git', '-C', owner['local_checkout'], 'show',
            f"{owner['owner_revision']}:{observation['data_path']}"])
        release.require(release.sha256_bytes(observation_bytes) == observation['data_sha256'],
                        'Bundle-3 property observation hash changed')
        observations = list(csv.DictReader(io.StringIO(observation_bytes.decode())))
        correction = []
        for prop in properties:
            name = 'density' if 'density' in prop['input_id'] else 'viscosity'
            loaded = prop['loading_state'] == 'loaded'
            selected = [row for row in observations if row['property'] == name
                        and (float(row['co2_loading_mol_per_mol_mea'] or 0) > 0) == loaded]
            release.require(not loaded or prop['applicable_uncertainty'] is None,
                            'Loaded row-level uncertainty was admitted')
            correction.append({'property': name, 'loading_state': prop['loading_state'],
                'source_rows': len(selected), 'bundle1_admitted_rows': len(selected),
                'bundle3_admitted_rows': 0 if loaded else len(selected),
                'bundle3_row_uncertainty': prop['applicable_uncertainty'],
                'bundle3_admission_status': prop['admission_status']})
        save(output / 'property_admission.json', {'observation_path': observation['data_path'],
            'observation_sha256': observation['data_sha256'], 'comparison': correction})
        dependencies = release.validate_dependency_files(config)
        rows = release.build_rows(config, dependencies, metadata)
        transport.write_csv(output / 'release_states.csv', rows, release.RESULT_FIELDS)
        summary = {
            'metadata': metadata, 'input': config, 'work_package_a': owner,
            'release_guard': guard, 'bundle': bundle,
            'row_counts': {'declared_states': len(rows), 'blocked_states': sum(r['admission_decision'] != 'admitted' for r in rows), 'admitted_states': sum(r['admission_decision'] == 'admitted' for r in rows), 'release_files_created': 0},
            'uncertainty_propagation_attempted': False,
            'downstream_issue30': 'blocked',
        }
        # Exercise rejection as well as the normal absent-file guard. The
        # temporary sentinel is confined to this run, never a release path.
        sentinel = output / 'release_guard_sentinel.json'
        sentinel.write_text('{}\n')
        try:
            guarded = {**config, 'release_guard': {**config['release_guard'], 'external_paths': [str(sentinel)]}}
            try:
                release.validate_release_guard(guarded)
            except ValueError as error:
                summary['release_guard_existing_file_rejected'] = str(error)
            else:
                raise AssertionError('release guard accepted an existing release file')
        finally:
            sentinel.unlink()
    summary['source_worktree_dirty'] = dirty
    save(output / 'summary.json', summary)
    print(json.dumps(summary['row_counts'], indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['transport', 'release'], required=True)
    parser.add_argument('--execute', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        replay(args.stage)
    else:
        execution = {'wall_limit_s': 120}
        directory = RUN / args.stage
        directory.mkdir(parents=True, exist_ok=True)
        save(directory / 'execution.json', {'execution': execution})
        with (directory / 'run.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__)), '--stage', args.stage, '--execute'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=execution['wall_limit_s'], check=True)
        print((directory / 'run.log').read_text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
