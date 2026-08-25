# Reproduce the Manuscript Evidence

This document gives the command chain for rebuilding the curated manuscript artifacts without rerunning slow full-species simulations by default. Run commands from the repository root.

## Environment

```bash
uv sync --group test
export PYTHONPATH="src"
export MEA_EPCSAFT_DATASET_NAME="MEA_CO2_H2O_ionic_fit"
uv run python scripts/check_epcsaft_integration.py --mode final
```

`pyproject.toml` pins the ePC-SAFT 0.2 wheel built from Engine commit
`b64f6df906489cb132792641c5be2ee8f404b114`; the final integration gate
requires wheel SHA-256
`a505d20ac9019e9b4edbad16df80166b6e082170ef2a2f002d897269e756d0d9`.
The downstream adapter uses the public `Parameters`, `Mixture`, and `State`
API, and the package uses CppAD as its sole production derivative authority.
Runtime derivative and Born-model switches from the superseded API are
intentionally unsupported. Henry-law checks do not evaluate ePC-SAFT. The
wheel hash is reproducible; final upstream release provenance still requires
an Engine commit and clean-tree receipt and is not inferred from the filename.

## Refresh Curated Tables, Figures, and Profiles

These commands rebuild plot-ready tables and figures from existing committed results or already completed run folders:

```bash
uv run python analyses/nccc_validation/scripts/generate_nccc_one_bed_artifacts.py
uv run python analyses/nccc_validation/scripts/render_source_backed_temperature_capture_gallery.py
uv run python analyses/nccc_validation/scripts/generate_data.py
uv run python analyses/nccc_validation/scripts/render_figures.py
uv run python analyses/nccc_validation/scripts/collect_clean_profiles.py --collect-existing
uv run python analyses/nccc_validation/scripts/validate_results.py
uv run python docs/latex/scripts/latex_workflows.py sync-figures
```

## Routine ePC-SAFT C-Case Campaign

This reruns the 2017 C-case campaign used for temperature overlays. It is slower than table rendering but much faster than the full activity-coupled path.

```bash
uv run python -m mea_absorption_column.benchmark --methods scipy-bvp --thermo-models ideal_henry epcsaft_ionic --c-case-dataset campaign --c-case-ids 1C 2C 3C 4C 5C 6C 7C --nccc-case-limit 0 --srp-case-limit 0 --staged-beds false --mesh-points 51 --tol 0.5 --bc-tol 0.001 --max-nodes 1000 --subprocess-timeout-s 120 --profile-csvs --profile-pngs --output-dir analyses/nccc_validation/results/runs/c_case_campaign_temperature_gallery
uv run python analyses/nccc_validation/scripts/render_c_case_campaign_temperature_gallery.py --run-dir analyses/nccc_validation/results/runs/c_case_campaign_temperature_gallery
```

## Full Activity-Coupled Evidence

The committed evidence for the slow full path is:

```text
analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv
```

Do not rerun this path during a normal manuscript refresh. These rows were generated with the superseded reactive package interface and are retained only as archived numerical-feasibility evidence. The `epcsaft_reactive_*` modes now fail closed. Regeneration requires independently sourced dimensionless reaction constants, an explicit standard-state conversion, and migration to the ePC-SAFT 0.2 typed chemical-equilibrium API.

Expected evidence fields include case id, model labels, capture, capture error, runtime, chemistry-solve time, residuals, guard counts, Python version, platform, package versions, exact command, and relevant environment variables.

## LaTeX Build

```bash
docs/latex/scripts/build_main.sh
uv run python docs/latex/scripts/check_main_pdf_fresh.py
```

The build first projects the Zotero Companion path-sanitized Better BibTeX
export at `/home/tnnrpolley21/Zotero/exports/references.bib` into the
Git-tracked `docs/latex/references.bib` snapshot. Refresh the central export
with `zotero-companion bibliography-export --apply --json`; edit article
metadata in Zotero, never in either `.bib` file. The separate Zotero auto-export
under `Documents/Papers` currently includes attachment paths and is not safe for
Git or Overleaf. The repository snapshot keeps Overleaf self-contained.

The source of truth is `docs/latex`, except for the Zotero-owned central
bibliography. Use the strict Overleaf mirror sync only after the local build and
freshness check pass.
