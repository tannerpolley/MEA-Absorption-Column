# NCCC Validation Analysis

This analysis owns the reviewer-response validation evidence for the MEA absorber model. It separates disposable benchmark runs from curated final evidence so completed results can be inspected without digging through temporary solver folders.

## Layout

- `results/runs/`: disposable benchmark runs and long diagnostics. This folder is ignored by Git.
- `inputs/retained_reactive_case3c/`: tracked 21-state film input and certified 44-state reactive-speciation table used by the retained calculations.
- `results/final/tables/`: accepted result CSVs, diagnostic CSVs, plot-ready tables, and profile indexes.
- `results/final/figures/`: retained PDF benchmark figures.
- `results/final/profiles/<case_id>/<thermo_model>/`: clean temperature-profile PNGs for quick visual review.
- `results/final/reports/`: Markdown or CSV summaries for accepted rows, fallback rows, and unresolved diagnostics.
- `results/runs/<run_id>/profiles/<case_source>/<case_id>/<method>/<thermo_model>/`: requested dense profile CSV exports. These are the CSV replacement for the older `Profiles.xlsx` workbook: each old workbook sheet is written as its own CSV, with `Position`, `height_m`, `bed_id`, and `bed_position_m` coordinate columns.

## Commands

When running from a Git worktree with the shared Codex venv, set the source path first so subprocess benchmark workers import this worktree:

```bash
export PYTHONPATH="src"
```

## Script Inventory

| Script | Role | Runs model? | ePC-SAFT dependency |
| --- | --- | --- | --- |
| `generate_data.py` | Converts curated run/final CSVs into raw, verified, and plot-ready tables. | No. | No direct dependency; may process existing ePC-SAFT rows. |
| `render_figures.py` | Renders final manuscript figures from final tables. | No. | No direct dependency; may plot existing ePC-SAFT rows. |
| `collect_clean_profiles.py` | Refreshes the clean temperature-profile PNG gallery and index. | Sometimes. | Required only when rerunning or collecting `epcsaft_*` lanes. |
| `run_case_profile.py` | Runs one case and exports dense `Profiles.xlsx`-style CSVs. | Yes. | Required for `epcsaft_ionic`. |
| `generate_clean_profile_csvs.py` | Runs accepted clean rows with per-case timeouts and writes dense profile CSV folders. | Yes. | Required only for ePC-SAFT rows in the selected suite. |
| `render_c_case_campaign_temperature_gallery.py` | Renders the corrected one-bed C-case temperature overlay gallery from a completed campaign-input benchmark run. | No. | No direct dependency; the source run may include ePC-SAFT rows. |
| `probe_reactive_epcsaft_speciation.py` | Archived probe for the superseded reactive package interface. | No supported current run. | Retained for provenance pending typed-equilibrium migration. |
| `generate_epcsaft_v02_validation.py` | Writes the current neutral-versus-ionic contribution table and exactly one fixed-chemistry Case 3C column row. | Yes, one column row. | Requires the locked ePC-SAFT 0.2 wheel and `MEA_CO2_H2O_ionic_fit` parameter document. |
| `validate_results.py` | Checks final tables, figures, profile indexes, and stale path regressions. | No. | No direct dependency. |
| `analyze_retained_reactive_case3c.py` | Builds the controlled retained-versus-prior Case 3C fugacity, speciation, parameter, mesh, and transfer-sensitivity tables. | Reads completed column runs and evaluates bounded liquid states. | Requires the retained ePC-SAFT wheel and parameter documents. |
| `render_retained_reactive_case3c_diagnosis.py` | Renders the reviewed Case 3C diagnostic figure from retained tables. | No. | No direct dependency. |
| `analyze_enhancement_consistency.py` | Compares explicit and complete Gaspar-style enhancement calculations at retained Case 3C states under fixed thermodynamic parameters. | Evaluates 21 retained liquid states. | Requires the retained ePC-SAFT wheel, parameter document, and reactive table. |
| `render_enhancement_consistency.py` | Renders enhancement and fixed-state flux comparisons from the retained enhancement table. | No. | No direct dependency. |
| `analyze_issue17_enhancement_comparison.py` | Evaluates the four issue 17 enhancement equations on 21 retained current-reconstruction fugacity-only states and applies the staged numerical and physical gates. | Evaluates 84 fixed-state rows. | No direct dependency; reads the retained fixed-state table. |
| `render_issue17_enhancement_comparison.py` | Renders the three issue 17 equation-comparison figures from the retained 84-row table. | No. | No direct dependency. |
| `resolve_concentration_bases.py` | Reconstructs prepared, loaded analytical, and free-MEA concentration bases for Putta/Luo labels and retained NCCC Case 3C states. | Reconstructs retained nine-species states. | Requires the certified reactive ePC-SAFT table. |
| `resolve_issue35_transport.py` | Reconstructs source-labeled diffusivity, density, and viscosity inputs and retains the unequal-ion closure decision. | No. | No. |

Henry-only validation does not evaluate ePC-SAFT. The project dependency pins an immutable ePC-SAFT 0.2 wheel, while the MEA parameter inputs remain vendored under `src/mea_absorption_column/data/epcsaft_datasets/` and are converted to the strict parameter-document API by the absorber adapter.

Typical ePC-SAFT diagnostic environment:

```bash
export MEA_EPCSAFT_DATASET_NAME="MEA_CO2_H2O_ionic_fit"
```

Runtime derivative and Born-model option matrices are not supported by API 0.2. CppAD is the sole production derivative authority; alternative model families require separately identified parameter documents.

Regenerate the current ePC-SAFT 0.2 fixed-state table and one bounded ionic/fixed-chemistry column row:

```bash
uv run python analyses/nccc_validation/scripts/generate_epcsaft_v02_validation.py
```

This command writes `epcsaft_v02_contribution_table.csv` and
`epcsaft_v02_column_row.csv`. The provider parameter fingerprint is labeled
checkout-path-local; the dataset and generated parameter-document SHA-256
values are the portable identities. `run_directory_at_generation` records the
disposable run path used by the command; it is not a retained artifact. The
final CSVs and their immutable identities are retained. The two current tables
validate fixed chemistry and configuration plumbing only. They do not
establish predictive reactive chemistry, predictive absorber performance, or
parameter accuracy.
The incompatible pre-0.2 generators were removed; their tables remain listed
under `historical_outputs` in `analysis.yaml` for provenance only.

Regenerate plot-ready tables from curated inputs or available run folders:

```bash
uv run python analyses/nccc_validation/scripts/generate_data.py
```

Render final figures from the final tables:

```bash
uv run python analyses/nccc_validation/scripts/render_figures.py
```

Refresh the clean profile index without rerunning simulations:

```bash
uv run python analyses/nccc_validation/scripts/collect_clean_profiles.py --collect-existing
```

Run one specific case and export dense per-variable profile CSVs:

```bash
uv run python analyses/nccc_validation/scripts/run_case_profile.py --case-source C_cases_data --case-id 3C --method scipy-bvp --thermo-model ideal_henry --output-dir analyses/nccc_validation/results/runs/manual_case_profiles
```

Full benchmark runs can also request these dense profile CSVs:

```bash
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry --c-case-ids 3C --nccc-case-limit 0 --profile-csvs --subprocess-timeout-s 60 --output-dir analyses/nccc_validation/results/runs/profile_csv_probe
```

Run the corrected one-bed NCCC C-case campaign table and regenerate the 1C--7C temperature overlay gallery:

```bash
export MEA_EPCSAFT_DATASET_NAME="MEA_CO2_H2O_ionic_fit"
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_ionic --c-case-dataset campaign --c-case-ids 1C 2C 3C 4C 5C 6C 7C --nccc-case-limit 0 --srp-case-limit 0 --staged-beds false --mesh-points 51 --tol 0.5 --bc-tol 0.001 --max-nodes 1000 --subprocess-timeout-s 120 --profile-csvs --profile-pngs --output-dir analyses/nccc_validation/results/runs/c_case_campaign_temperature_gallery
uv run python analyses/nccc_validation/scripts/render_c_case_campaign_temperature_gallery.py --run-dir analyses/nccc_validation/results/runs/c_case_campaign_temperature_gallery
```

The campaign dataset is stored separately as `src/mea_absorption_column/data/C_cases_campaign_inputs.csv`; the older `C_cases_data.csv` remains available through the default `--c-case-dataset legacy` path for reproducibility.

Run the favorable SRP-style method-comparison case across shooting, collocation BVP, and finite difference:

```bash
uv run python -m mea_absorption_column.benchmark --methods single scipy-bvp finite --thermo-models ideal_henry --c-case-limit 0 --nccc-case-limit 0 --srp-case-limit 1 --mesh-points 21 --tol 0.5 --bc-tol 0.001 --max-runtime-s 30 --seed-from-shooting --subprocess-timeout-s 60 --output-dir analyses/nccc_validation/results/runs/srp_method_slice
```

The current SRP/NCCC method contrast is summarized in `results/final/reports/solver_method_contrast_srp_3c.md`.

Each generated profile folder includes `profile_manifest.json`, `profile_manifest.csv`, `run_spec.json`, and `rerun_profile.sh` so a single case can be rerun from that folder without launching the entire benchmark sweep.

Generate dense profile CSVs for the accepted validation rows with 60-second per-case timeouts and an incremental log:

```bash
uv run python analyses/nccc_validation/scripts/generate_clean_profile_csvs.py --suite all --output-dir analyses/nccc_validation/results/runs/clean_profile_csvs
```

This script writes each case row as soon as it finishes. If a case exceeds the default 60-second subprocess timeout or errors, the row is logged as a failed diagnostic result and the script continues to the next case. Use `--per-case-timeout-s <seconds>` only when intentionally running a longer diagnostic probe.
It also writes `profile_runtime_index.csv` and refreshes each profile folder's `profile_manifest.json` / `profile_manifest.csv` with `runtime_s` and a human-readable `runtime_label`.

Validate the analysis results used by the manuscript:

```bash
uv run python analyses/nccc_validation/scripts/validate_results.py
```

Generate and render the retained reactive Case 3C diagnosis after its named
run folders are present:

```bash
MEA_EPCSAFT_REACTIVE_TABLE=analyses/nccc_validation/inputs/retained_reactive_case3c/speciation_table.csv \
uv run python analyses/nccc_validation/scripts/analyze_retained_reactive_case3c.py
uv run python analyses/nccc_validation/scripts/render_retained_reactive_case3c_diagnosis.py
```

Run the enhancement/interface consistency experiment from issue 16:

```bash
MEA_EPCSAFT_DATASET_NAME=MEA_CO2_H2O_retained_predictive \
MEA_EPCSAFT_REACTIVE_TABLE=analyses/nccc_validation/inputs/retained_reactive_case3c/speciation_table.csv \
uv run python analyses/nccc_validation/scripts/analyze_enhancement_consistency.py
uv run python analyses/nccc_validation/scripts/render_enhancement_consistency.py
uv run pytest -q analyses/nccc_validation/tests/test_enhancement_consistency.py
```

Regenerate the issue 17 fixed-state comparison and its three retained-table
figures:

```bash
uv run python analyses/nccc_validation/scripts/analyze_issue17_enhancement_comparison.py
uv run python analyses/nccc_validation/scripts/render_issue17_enhancement_comparison.py
uv run pytest -q analyses/nccc_validation/tests/test_issue17_enhancement_comparison.py
```

The historical 89.832629% row lacks a retained ePC-SAFT wheel identity and
dense profile. Issue 17 therefore uses the separately named
`current_fugacity_only_reconstruction_b2d6636` input snapshot; its exact
identity and the historical gap are recorded in
`inputs/issue17_enhancement_comparison/identity.json`. The retained 101-state
profile supplies the 21 exact positions at 0.05 intervals, including both
boundaries.

Run the three-position direct-boundary numerical gate:

```bash
MEA_EPCSAFT_DATASET_NAME=MEA_CO2_H2O_retained_predictive \
MEA_EPCSAFT_REACTIVE_TABLE=analyses/nccc_validation/inputs/retained_reactive_case3c/speciation_table.csv \
uv run python analyses/nccc_validation/scripts/analyze_reactive_film.py --numerical-gate --case-timeout-s 10
```

This issue 16 command consumes the public exact fixed-\(T,P\) derivative and
writes `issue16_exact_reactive_film_*` outputs. It retains every failed row and
returns nonzero when a physical or numerical gate fails. The prior Stage A
tables remain unchanged. Gate and immutable-input identities are in
`issue16_reactive_film_gate.csv` and
`inputs/issue16_reactive_film_identity.json`;
all placeholder-dependent evidence is `provisional_concept_only`.

Resolve the Issue 33 concentration bases and retain the source table:

```bash
MEA_EPCSAFT_REACTIVE_TABLE=analyses/nccc_validation/inputs/retained_reactive_case3c/speciation_table.csv \
uv run python analyses/nccc_validation/scripts/resolve_concentration_bases.py
```

The input record `inputs/issue33_concentration_basis.json` keeps the equations,
source locators, Zotero attachment hashes, density observations, uncertainty,
and admission rules. The output table reports Putta's nominal 1 M and 5 M
labels plus Case 3C positions 0, 0.5, and 1. Position 1 retains the computed
`4.889309897097635 mol/L` analytical MEA concentration and its separate free
MEA concentration; it is not rounded or admitted as exact 5 M. Every row
remains `basis_unresolved` until the missing preparation-temperature and
prepared-to-loaded-volume evidence is supplied.

Regenerate the Issue 34 source-faithful kinetics and reaction-partition record:

```bash
uv run python analyses/nccc_validation/scripts/resolve_issue34_kinetics.py
```

This retains the Putta F1/F2 relationships, rejects the printed third-order
`s^-2` coefficient unit by dimensional reconstruction, records the unavailable
Gondal F3 coefficient as a supported negative, and keeps unresolved film
timescales and aggregate rate comparisons out of physical film adoption. The
source-correlation arithmetic reconstruction table is explicitly not a
physical kinetic or transport sensitivity study. It also refreshes the
source-faithful report at
`results/final/reports/issue34_reaction_kinetics.md`.

Regenerate the Issue 35 source-faithful transport-input and unequal-ion closure record:

```bash
uv run python analyses/nccc_validation/scripts/resolve_issue35_transport.py
```

This retains the Luo/Snijder/Amundsen source reconstructions, density and viscosity observations, the blocked Putta N2O analogy, and the unattributed legacy ion-scalar rejection. Candidate B remains non-executable until a complete mobility law, unequal-ion inputs, and an accepted true-species basis are available. No physical flux comparison or transport adoption is performed.

Morgan's 7.3% campaign uncertainty is used only for the diagnostic unloaded
concentration at each local state temperature; it is not a prepared-concentration
result or a loaded-versus-5 M admission basis. The NCCC preparation temperature is
unreported, so true prepared concentration remains unresolved. Amundsen's
instrument density uncertainty is kept separate from its combined relative
estimates, which are converted row-wise for the diagnostic local-state density.

## Result Semantics

The clean profile gallery contains accepted validation rows and explicitly accepted fallback rows used in the manuscript. Diagnostic or unresolved rows stay in final tables and reports, but they are not mixed into the clean profile gallery unless the caveat is explicit in the profile index.

The dense profile CSV folders are run outputs, not summary tables. Use them to inspect how internal variables change with column position; use `verified_*.csv`, `raw_*.csv`, and `plot_*.csv` for manuscript validation metrics and figure generation.
