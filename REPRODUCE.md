# Reproduce the Manuscript Evidence

This document gives the command chain for rebuilding the curated manuscript figures, tables and PDF without rerunning slow full-species simulations by default. Run commands from the repository root.

## Fallback edition — 2026-09-03

`codex/fallback-manuscript` is based on **`0d4e552e15fdf9ee17edc64908c3a04a6525f98b` (2026-08-25)**, the merged MEA-only freeze with the nine-case evidence. The earlier `nce-mea-only-pre-mdea-2026-08-25` tag points to a different, eight-case version and is not this baseline. The revised `docs/latex` tree, reviewer notebook, review records, and the user-selected runtime update below are an **uncommitted overlay**; checking out the baseline alone does not recreate that overlay. No new commit or push was made. The historical aggregate figures and tables have not been replaced with new-runtime calculations; the rebuilt PDF reports the coupled nine-species verification and sensitivity studies separately.

Start the manuscript-specific notebook without installing the model environment:

```bash
python3 docs/latex/scripts/test_reviewer_checklist.py
python3 docs/latex/scripts/reviewer_checklist.py --serve --port 40629
```

Open <http://127.0.0.1:40629/reviewers>. Use `--port 0` if that port is occupied. The notebook reads only this worktree, reloads assessments every three seconds, and flags changed manuscript evidence for reassessment. Edit `docs/latex/scripts/reviewer_checklist.json` to record a reviewed change; scores are internal coverage assessments, not acceptance of numerical results.

For this frozen draft, build from retained figures and bibliography without refreshing Zotero, syncing figures, or publishing to the shared Overleaf mirror:

```bash
(cd docs/latex && TEXMFHOME=/home/tnnrpolley21/texmf latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=builds main.tex)
python3 docs/latex/scripts/check_main_pdf_fresh.py
```

This requires XeLaTeX, latexmk, BibTeX, and the CAS class files in the local TeX installation. It rebuilds the document, not the calculations; identical PDF bytes across TeX installations are not promised.

### Integrated visual pilot

The fallback includes the presentation work from the task “Quarto MEA manuscript visual pilot”: commit `63678e33557c9f775cdb4fe34f7aa00e72275904` and its later archive snapshot `2fdd4fd53f9cc54e3aaf208de4f41229387f4fea`.
The named pilot branch still points to the August base; use these immutable commit identities when inspecting the visual source.
The pilot ultimately retained raw CAS XeLaTeX for the manuscript, so no Quarto manuscript renderer was added.
The port adopts the native TikZ workflow, table layouts, semantic equation and unit notation, cross-references, and reduced appendix whitespace.
It preserves this fallback's reviewer additions, phase-specific boundary conditions, numerical values, quantitative figure files, bibliography, and scientific limitations.
The original workflow PNG remains retained but is no longer included in the article; its editable replacement is `docs/latex/figures/tikz/model-framework-flowchart.tex`.
Local adaptations preserve CAS float placement, reserve table-footnote space, keep parameter units with their table, and fix chemical spacing and the T1 Celsius glyph.
This visual integration is part of the uncommitted overlay, not a new model run or an approval of the historical results.

**Historical numerical replication remains unverified.** The exact old wheel (`a505d20ac9019e9b4edbad16df80166b6e082170ef2a2f002d897269e756d0d9`, recorded Engine `b64f6df906489cb132792641c5be2ee8f404b114`) was not recovered. The user instead selected the latest MEA-Thermodynamics parameter notebook and a compatible modern wheel. The replacement environment works; it does not establish the provenance or correctness of the historical outputs. Energy-sign, temperature-chain-rule and temperature-initialization defects are now repaired and checked. Four source-backed Henry cases converged. The old-wheel/unseeded reactive attempt failed certification, but both its exact stopping state and the independent working reference now pass with the supplied runtime and loading path. Historical manuscript values remain unreplaced; see [the check record and exact commands](analyses/nccc_validation/results/reviewer_energy/README.md).


## Seven-case comparison and measured runtime — 2026-09-04

The investigator requested manuscript use of the supplied seven-case package, README SHA-256 `95c7b82e676bc2374b2ae9704237f6d8958e7f24535800141dc37add36e1e568`, for capture/temperature comparison and measured runtime, subject to its coarse-mesh scope, 6C/7C discrepancies and separately retained Engine test failure. This is a content-identified working snapshot, not a new commit.

The byte-identical package is retained in `analyses/nccc_validation/figures/reactive_parallel`. Its summary gives seven converged cases and 5.85483 percentage-point capture MAE. The selected run directories named in that summary are copied under the same relative `analyses/nccc_validation/results/runs/` paths here. Original provenance keeps the ac57 absolute paths: map that checkout prefix to this bad5 checkout to locate the copied files. No run identity was regenerated. Morgan Table C2 observations and their coordinate conversion are retained with the figure package. Case 1C uses the declared initial reaction extent fraction 0.0001; the supporting initialization probe is retained.

The campaign uses its per-run wheels (3a7391… for 1C–6C, b011d0… for 7C), not the optimized timing wheel. Refined Case 3C capture 91.54839% remains separate from campaign capture 91.54329%.

Runtime evidence is retained in `analyses/nccc_validation/results/runs/runtime_diagnostics_20260904/`: summary, profile comparisons, notebook, candidate reference and all three `optimized_plain_*` runs with identities, initialization, diagnostics and numerical tables. Median BVP wall time is 34.6569113731 s. The fastest BVP run (`optimized_plain_3`) records 34.4052288532 s wall, 34.231582059 s CPU, 51.4534527540 s including fresh initialization/profiles and 203.51171875 MiB peak RSS. These timings do not replace the original refined-mesh timings, which remain in their original evidence. The runtime notebook's unimported full native trace remains owned by the source analysis; it is not needed for the reported three-repeat result.

Exact optimized wheel: `/home/tnnrpolley21/.cache/epcsaft/wheels/bafc4375476a39f08f5bc43cc6ea4b034d1ca956730910a0e753613007e8d12f/epcsaft-0.2.0.dev0-cp313-cp313-linux_x86_64.whl`, SHA-256 `9b538f4defd5af661cd736af03760a55adb59b231d674f96c1e0f7d67350689d`. Engine branch `codex/reactive-evaluation-reuse` was based on `4563c6a89f8837ebb1bc24408b7177fa8d209e9d` with uncommitted optimizations; that base commit alone does not identify the measured source. The separately copied restored-build confirmation uses wheel `91632d2812429cbd293aae70fe8d4efb00000efe2377a91546dd7374dca67ee4` and takes 35.7818 s BVP / 52.8563 s total.

The optimized implementation retains one failing temperature-boundary test (`test_solved_temperature_boundaries_certify_bubble_dew_and_total_derivatives`): the vapor-reference center reaches its iteration limit at 296.7724020931804 K, pressure residual −6.03e−7 Pa against the 1.2e−7 Pa outer target. The source notebook documents 114 affected checks passing and three of four additional slow checks passing. This does not invalidate the retained column result, and it is not a claim that every Engine check passes.

The repository lock and final-integration pin still select b011d0…, not the optimized wheel. The manuscript update does not change that environment. For a new isolated timing check, use a new label and verify the hash first; the copied driver retains the source settings:

```bash
runtime_wheel=/home/tnnrpolley21/.cache/epcsaft/wheels/bafc4375476a39f08f5bc43cc6ea4b034d1ca956730910a0e753613007e8d12f/epcsaft-0.2.0.dev0-cp313-cp313-linux_x86_64.whl
sha256sum "$runtime_wheel"
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD/src" \
  uv run --isolated --no-project --python 3.13 \
  --with "$runtime_wheel" --with numpy==2.4.4 --with scipy==1.17.1 \
  --with pandas==3.0.2 --with matplotlib==3.10.9 --with openpyxl==3.1.5 --with casadi==3.7.2 \
  analyses/nccc_validation/scripts/diagnose_case3c_runtime.py \
  manuscript_fast_path_confirmation --plain --wheel "$runtime_wheel"
```

The command selects native Jacobians, exact state reuse, temperature/raw coordinates, dry-saturated feed and reported dry gas mass, with 21 initial nodes, tolerance 0.5, boundary tolerance 0.001 and 1000 maximum nodes. It was not rerun during this editorial integration. The run identities retain the evaluated source-file hashes; no numerical reproduction is inferred from rebuilding the PDF.

## Environment

```bash
(set -e
test "$(sha256sum /home/tnnrpolley21/Workspaces/Engineering/ePC-SAFT-project/build/candidate-wheels/balanced-state-actions/9e1bef97fbea5c6f465612ae27b054192f91f19c/epcsaft-0.2.0.dev0-cp313-cp313-linux_x86_64.whl | cut -d ' ' -f 1)" = b011d0f9d492e9db197f67cc0ae6781ac636fa3278805ddf1d6a05ecd167074b
uv sync --frozen --group test
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
uv run --frozen python scripts/check_epcsaft_integration.py --mode final
uv run --frozen pytest tests/test_column_energy.py tests/test_thermodynamics_adapter.py tests/test_epcsaft_reactive_chemistry.py tests/test_robust_convergence.py -q
)
```

`pyproject.toml` pins the non-editable CPython 3.13 Linux x86-64 ePC-SAFT wheel from Engine commit
`9e1bef97fbea5c6f465612ae27b054192f91f19c`; the final integration gate
requires wheel SHA-256
`b011d0f9d492e9db197f67cc0ae6781ac636fa3278805ddf1d6a05ecd167074b`.
Another machine needs these exact wheel bytes at the pinned path, or an explicit path-only dependency/lock update retaining this hash.
The installed native extension SHA-256 is `b5f97d49eb9439da84312dbeacb8ac0bae26ce6939562339a3c73d842fccce34`. This is the implementation/build identity, not a later merge commit. The source notebook's older export wheel is not the current runtime.
The downstream adapter uses the public `Parameters`, `Mixture`, and `State`
API and `equilibrium.solve`; no mutable sibling source import is used.
The unused `polar: none` model-family declaration was removed from the historical adapter; no polar parameters were added to the new set.
Henry-law checks do not evaluate ePC-SAFT. Wheel identity is not inferred from its filename.

## Selected nine-species runtime

The readable [selected parameter supporting record](docs/selected-reactive-parameters.md) tabulates the exact retained component, pair, association and reaction inputs, source distinctions, standard state and domains. It documents the replacement inputs without relabeling the historical results.

Source: `/home/tnnrpolley21/Workspaces/Engineering/MEA-Thermodynamics/analyses/mea_parameter_bundle/notebook.html#parameter-change-history`.
The user selected its current `results/selected-current-best-parameters.json`, SHA-256 `a9186c93759f2e2c02a6c913350ad06a244fff3f82503820c9962b3df8dd40d9`.
An exact copy is retained as `src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle/parameters.json`, alongside the matching source `reaction-system.json` (SHA-256 `810dfec15760cf74451df91743d6e63684cee93ddaf3e1ff4e42bf4a686afe29`), `anchored-reference-thermochemistry.json` (SHA-256 `a24a6b3c8b506fc659fc1bbd8a470b55919ba93da23eea27ffdf882645706185`), `adoption-receipt.json`, and `bundle.json` inventory from the verified Orchestrator handoff. All inventory hashes are checked before use. Its export-time wheel identity is provenance, not the separately pinned installed build.
No upstream fitting or publication files were edited.

Select `thermo_model="epcsaft_reactive_nine"` through `run_model`, or `--thermo-models epcsaft_reactive_nine` through the existing benchmark CLI.
Direct `abs_column` callers must select the same `chemical_equilibrium_model`.
This mode jointly solves the nine-species liquid EOS, five reactions, pressure, balances, and charge using the selected source-standard-state conversion.
Typed R2/R4/R5 correlations from the selected parameter file supersede source correlations as a whole; their molality conversion is not applied twice. R1/R3 use the source export. The five effective ln K values match the previous implementation at 298.15, 318.15, 325 and 353.15 K; this adoption does not refit or alter them. The common selected calculation domain is 293.15–393.15 K.
This is an explicitly selected working parameter set, not an independently validated column model; the source notebook's full publication refresh is incomplete.

The local thermodynamics wrapper reuses the verified loading-path implementation from the Orchestrator's 6c99 source (SHA-256 `f6a3e75646cf30702271af62fe1c4c57fa0932a37243ca1a78ff97e30284379e`) without a sibling import or reactive-film implementation. `reactive_liquid()` caches immutable model inputs only. Each call solves a fresh anchor at CO2/MEA = 0.25 at the requested T/P and MEA/water ratio, then uses positive conservative starts with maximum log-loading step 0.1 and budget 32. Native A1 mode honors the explicit starts; each step must satisfy physical acceptance and certified liquid pressure-root identity. Budget exhaustion or native rejection remains an error, with no automatic anchor retry.

Both the independent 318.15 K reference and the exact previously failed 325 K point now pass in this worktree; values and repetition commands are in [the nine-species replay record](analyses/nccc_validation/results/reviewer_energy/README.md#nine-species-replay-after-the-orchestrator-handoff). The earlier failure was an old-wheel/unseeded invocation, not evidence that the parameter set cannot work. No full current nine-species axial solution is claimed by this local replay.

True concentrations use the solved EOS molar density, and liquid CO2 fugacity uses that same state.
The neutral-vapor approximation, water saturation-pressure closure, empirical hydraulic/transport correlations and conventional enhancement-factor approach are retained. In the coupled mode, the film resistance now uses the EOS bulk fugacity/free-CO2 concentration ratio under a frozen-bulk-activity approximation, not an unrelated empirical Henry coefficient or a differential reactive-loading coefficient. The empirical enhancement-only CO2 divisor remains specified separately. Shared expressions expose native/AD derivatives without copying the transport or energy equations.
Enhancement still uses the first six true-species concentrations; this is not a nine-species reactive-film model.
Saved `Fl`, `Cl`, and `x` profiles include carbonate, hydronium, and hydroxide, with a separate true-species EOS density field.
True molar flow is normalized by the conserved MEA nitrogen flow, not by the empirical hydraulic molar density.
Rejected reactive states propagate as errors, without Henry-law or finite-RHS-penalty substitution; failed solver iterates are not promoted merely for a small boundary residual.
Run metadata identifies the selected bundle and retains the last native equilibrium evidence, including its parameter fingerprint, in `epcsaft_chemistry_last_evidence`.
Legacy mass/reaction/charge residual columns are unavailable for this mode rather than assigned zero or populated with native quantities on a different basis.

The focused tests check element/charge conservation at the retained Case 3C inlet, actual column RHS and profile evaluation, legacy flux coupling, visible domain/fugacity failure, and local energy conservation and enthalpy/temperature-state equivalence.
They do not validate selected-reactive full-column convergence, capture accuracy or the archived manuscript numbers.
The bounded runner `analyses/nccc_validation/scripts/run_reviewer_validation.py` records source/input hashes, solver settings, native diagnostics and CPU/wall time in new output directories. The retained [working results](analyses/nccc_validation/results/reviewer_energy/README.md) include four Henry checks, a Case 3C refinement, failed competing-method attempts and the superseded selected-reactive stopping state, alongside the successful nine-species replay. Run new campaigns into distinct directories; do not overwrite historical manuscript figures or tables until matching results are reviewed.

### Current nonisothermal column

Case 3C converges with the selected parameters: 91.543287% capture on the 21-point initial mesh and 91.548387% after 41-point/tighter-tolerance refinement, against 89.50% observed. Native residuals, material/energy closure, exact values and a compact profile figure are retained in [the current result record](analyses/nccc_validation/results/reviewer_energy/README.md#current-nine-species-nonisothermal-result). This is one current case, not a replacement of the historical multi-case manuscript evidence.

The dedicated `run_reactive_column.py` command uses current nine-species ePC-SAFT at every final residual, the empirical energy balance, raw scaled coordinates and a native-thermodynamics/CasADi Jacobian. A converged same-case Henry profile initializes the first run only. `reactive_reuse_states=True` enables exact-state reuse and conservative starts from certified native amounts within one model instance; rejected native states never replace that seed. Its first composition anchor fixes both loading 0.25 and the inlet water/MEA ratio. This extends, rather than changes the default fresh-path behavior described above. The two reference states alone remain distinct from the retained converged axial result.

Use new output directory names for independent reproductions:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 1200s uv run --frozen python analyses/nccc_validation/scripts/run_reactive_column.py --output analyses/nccc_validation/results/runs/reactive_native_21
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 1200s uv run --frozen python analyses/nccc_validation/scripts/run_reactive_column.py --initial-profile analyses/nccc_validation/results/runs/reactive_native_21/solution_scaled.csv --mesh 41 --tol 0.05 --output analyses/nccc_validation/results/runs/reactive_native_41
MPLBACKEND=Agg uv run --frozen python analyses/nccc_validation/figures/reactive_column/scripts/render.py analyses/nccc_validation/results/runs/reactive_native_21 analyses/nccc_validation/results/runs/reactive_native_41
```

Retain stdout/stderr alongside each run for mesh-iteration, residual and thermodynamic-call evidence. The output identity records pin source files, parameter files and Engine identity; the result separates BVP CPU/wall time from total wall time including initialization and profile output. Native call counts in the final result include profile output. The figure renderer checks positivity and reports material, charge and energy conservation from retained profiles without running the model. These results do not retrospectively reproduce the manuscript's historical six-species comparison.

The figure-owned `output/` directory is intentionally available for Git tracking: it contains exact plotted values, scalar results, complete run identities and the local Jacobian timing comparison. Raw run directories remain ignored and should accompany a scientific archive. The present overlay is still uncommitted; no new immutable manuscript revision is claimed.

### R1.8 local thermodynamic sensitivity

Use the accepted 41-node Case 3C profile above as initialization. Run each combination of
`--kij pair/monoethanolamine/water/k_ij` or
`--kij pair/carbon-dioxide/water/k_ij` with `--factor .95` or `--factor 1.05`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 1800s uv run python analyses/nccc_validation/scripts/run_reactive_column.py --initial-profile analyses/nccc_validation/results/runs/reactive_native_41/solution_scaled.csv --mesh 41 --tol .05 --kij pair/carbon-dioxide/water/k_ij --factor 1.05 --output analyses/nccc_validation/results/runs/r18_co2_water_105_guarded
```

Choose a new output directory for every run; the command refuses to replace an existing run.
The public native parameter mapping changes exactly one interaction coefficient, in both
liquid and vapor calculations. It retains the adopted file unchanged and writes the exact
evaluated parameter mapping and fingerprint into the run directory. These ±5% multipliers
are a local sensitivity screen, not fitted values or statistical uncertainty bounds.
Reaction constants, transport correlations, inlet conditions and the enhancement approach stay fixed.

For the reaction-equilibrium extension, replace `--kij ...` with `--reaction R4` or
`--reaction R5`, using the same factors and distinct output directories
`r18_r4_095`, `r18_r4_105`, `r18_r5_095` and `r18_r5_105`.
This multiplies K(T), not its logarithm or each fitted coefficient. The implementation adds
ln(factor) to the R4 natural-log correlation constant or subtracts log10(factor) from
the R5 negative-log10 correlation constant. Temperature slopes remain unchanged.
All other reactions, EOS parameters, source files and physical correlations remain fixed.
The evaluated reaction mapping is retained as `evaluated_reactions.json` beside each run;
the run identity hashes it separately from the unchanged EOS parameter document.

For loading jumps beyond the declared maximum logarithmic step of 0.1, the solver uses its
fresh loading path instead of the last accepted composition. Nearby queries retain conservative
warm starts and exact-state caching. The initial +5% CO2–water attempt and its rejected vapor-density
root are preserved; a fresh loading path solves that same state on the certified liquid root.
No root check is relaxed and no failed equilibrium result is used as a column prediction.

Plot retained outputs without solving the column:

```bash
uv run python analyses/nccc_validation/figures/reactive_column/scripts/render_sensitivity.py analyses/nccc_validation/results/runs/reactive_native_41 analyses/nccc_validation/results/runs/r18_mea_water_095 analyses/nccc_validation/results/runs/r18_mea_water_105 analyses/nccc_validation/results/runs/r18_co2_water_095 analyses/nccc_validation/results/runs/r18_co2_water_105_guarded analyses/nccc_validation/results/runs/r18_r4_095 analyses/nccc_validation/results/runs/r18_r4_105 analyses/nccc_validation/results/runs/r18_r5_095 analyses/nccc_validation/results/runs/r18_r5_105 --initialization-check analyses/nccc_validation/results/runs/r18_co2_water_105_probe
```

Copy the combined retained plot into the manuscript before rebuilding its PDF:

```bash
cp analyses/nccc_validation/figures/reactive_column/output/sensitivity/comparison.pdf docs/latex/figures/reactive-parameter-sensitivity.pdf
```

The figure-owned `output/sensitivity/` contains plotted values, profiles, evaluated parameters and reactions,
the rejected-state comparison and input hashes. Its study summary compares capture in percentage
points and the sampled peak liquid temperature in kelvin against the selected-parameter reference.
Original run identities distinguish the initialization policies; concurrent runtime values are
not used for a method-performance comparison.

### R1.9 retained transport sensitivity

The investigator-authorized transport study is retained in `analyses/transport_sensitivity/`, imported without changing the supplied notebook, inputs, failed attempts or numerical outputs from the `ac57` task. Its `notebook.qmd` owns the scientific interpretation; `figures/response/output/summary.csv` and `profiles.csv` own the exact values. The notebook SHA-256 at promotion is `2985e983a931c327f2968a8455b9196cbda0c9d3e15fdea3f2dff5fbda3f9446`; the summary SHA-256 is `f114cfec689403d27ab1a00c136ba17b375abdfbb8012053f7fa585dfe8d55d7`. This is an uncommitted study snapshot, not a new model default or fitted uncertainty estimate.

The package includes the seven completed points, an initialization control, both failed diffusivity attempts, exact-state checks, six directional-derivative checks and the immutable-wheel check. Original absolute execution paths remain in the retained provenance and run identities. For verifying the imported copies, paths below the original `ac57` `analyses/transport_sensitivity/` prefix map to the same relative paths here; original identities are not rewritten. The source/input archives preserve the model and reference used for those calculations. Study-specific reproduction commands remain in its README; recreating columns in another location requires selecting the retained reference/profile/anchor paths explicitly, not using those historical absolute paths as defaults.

Manuscript builds need only the retained publication figure, without rerunning a column:

```bash
cp analyses/transport_sensitivity/figures/response/output/transport_sensitivity.pdf docs/latex/figures/transport-sensitivity.pdf
```

Both thermodynamic and transport figures are self-contained under `docs/latex/figures`. The article distinguishes their ±5% and ±10% input ranges and compares changes with the retained refinement difference, not a certified error interval. The working reviewer record links the supplied HTML notebook and numerical evidence. No bibliography refresh or Overleaf synchronization accompanies this integration.

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

Do not rerun this path during a normal manuscript refresh. These rows were generated with the superseded reactive package interface and are retained only as archived numerical-feasibility evidence. Legacy reactive aliases still fail closed. The canonical `epcsaft_reactive_nine` mode now uses the selected modern bundle described above; it is not a reproduction of these rows.

Expected evidence fields include case id, model labels, capture, capture error, runtime, chemistry-solve time, residuals, guard counts, Python version, platform, package versions, exact command, and relevant environment variables.

## LaTeX Build

```bash
docs/latex/scripts/build_main.sh
uv run python docs/latex/scripts/check_main_pdf_fresh.py
```

The build first projects the Zotero-owned Better BibTeX auto-export at
`/home/tnnrpolley21/Documents/Papers/references.bib` into the Git-tracked
`docs/latex/references.bib` snapshot. Edit article metadata in Zotero, never in
either `.bib` file. The repository snapshot keeps Overleaf self-contained.

The source of truth is `docs/latex`, except for the Zotero-owned central
bibliography. Use the strict Overleaf mirror sync only after the local build and
freshness check pass.
