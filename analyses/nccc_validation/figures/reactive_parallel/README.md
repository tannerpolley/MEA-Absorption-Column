# Seven-case reactive absorber figures

## Manuscript use

`output/temperature_profiles.pdf` is the seven-case replacement for the old
`docs/latex/figures/case-c-temperature-overlay.png`. It contains actual coupled
nine-species ePC-SAFT column results, conventional enhancement-factor transport,
empirical energy closure, and the reported NCCC packing temperatures. It is not
a resolved reactive-film calculation or an eNRTL comparison.

`output/case_3c_temperature.pdf` provides the matching representative 3C view.
`output/capture_comparison.pdf` compares all seven predicted and reported captures.
PNG companions are available for preview. The CSV files in `output/` retain every
plotted profile, observation, capture, result directory, parameter identity,
wheel identity, and numerical check. `output/provenance.json` hashes the inputs.

Run `scripts/render.py` with the installed analysis environment to refresh these
figures without rerunning the column. The renderer prefers the completed
unchanged-source confirmation runs, uses the controlled Case 3C calculation,
and retains the successful native-initialized 7C calculation. Historical runs
and their source-change warnings remain untouched. Do not archive a figure
with `source_review_required=true` as a clean-source campaign.

The 2026-09-04 completed package has seven successful cases and no outstanding
source-change flags. Predicted captures for 1C-7C are respectively 99.1020,
97.3889, 91.5433, 91.4949, 92.2754, 71.6467, and 64.4674 percent. Their mean
absolute capture error is 5.85483 percentage points. This is numerical
completion, not validation of uniform predictive accuracy.

Case 1C uses a declared initial reaction-extent fraction of 0.0001 instead of
0.001. A retained isolated replay showed that the original and optimized Engine
both failed from the latter start at the same admissible anchor state; three
alternative conservative starts converged to the same stable equilibrium.
The accepted 1C run uses the declared start throughout, with unchanged model,
parameters, tolerances, and source files; it does not silently retry failed
states. See `../../results/runs/reactive_confirmed_20260904/initialization_probe.csv`
and `../../results/runs/reactive_confirmed_20260904/declared_start/1C/`.
This is not a claim that every possible equilibrium initial guess is reliable.

## Source correction

verified: Morgan et al. (2020), DOI 10.1016/j.apenergy.2020.114533,
Appendix C p.22 defines the source coordinate from top to bottom. The absorber
coordinate is bottom to top, so **z = 1 - x**; no fitted shift is used.
Table C2 p.27 contains all 35 temperatures for 1C-7C. They were transcribed into
`inputs/morgan2020_table_c2.csv` and visually checked against the retained PDF.
The existing year-specific temperature CSV agrees for 1C-6C but omits 7C.
The legacy `C_cases_campaign_inputs.csv` is not used for these observations.
Its 35 temperature entries nevertheless agree with Table C2 after the same
coordinate and unit conversions: the two layouts are not conflicting data.

Zotero attachment: HX2358GV. PDF SHA-256:
`f0ac09b3978962b0a3eab52ef062ac942811192c982b201caaae499e1aaf2232`.
The live Zotero service was unavailable; the existing local attachment was read
without acquisition, library mutation, or bibliography changes.

The source labels these measurements absorber temperatures, without a
phase-specific sensor designation in the checked table/figure. The legend
therefore says NCCC packing temperature and distinguishes both predicted
phase temperatures. Source temperatures are converted from Celsius to kelvin.
No temperature at z=1 is fabricated: that end of the curves is the imposed
lean-liquid boundary, not a measurement. No uncertainty bars are invented.

## Suggested caption and interpretation

Temperature and capture comparison for the seven one-bed NCCC 2017 C cases.
Solid and dashed curves denote liquid and vapor temperatures calculated with
the coupled nine-species reactive ePC-SAFT equilibrium model and conventional
enhancement-factor transport. Crosses are the reported absorber temperatures
from Morgan et al. (2020), Table C2, transformed to the bottom-to-top coordinate.
Each panel gives predicted and reported capture. Cases 1C-3C use the documented
318.15 K lean-inlet assumption because the source inlet entries are missing.

These are coarse column calculations (21 initial nodes, tolerance 0.5), not
seven mesh-convergence studies. All included solutions must pass the retained
convergence, positive-species, component-balance, and charge checks. The large
6C/7C capture discrepancies and temperature-shape discrepancies must remain
visible. Do not describe this result as uniformly improved predictive accuracy.
The empirical caloric closure is not a claim of exact axial energy conservation;
its retained net-enthalpy-flow ranges are included in `summary.csv`.

Replacing the old figures also requires replacing their old 88.32% or 89.40%
3C statements with the correct configuration-specific value in `summary.csv`.
The refined 91.54839% sensitivity reference is a different discretization and
must remain identified as such. The historical six-species Henry/ePC-SAFT
comparison cannot simply be relabeled as the new nine-species campaign.

The separate operating-response figure is at
`../reactive_operating/output/operating_response.pdf`. It retains accepted local
L/G, loading, and inlet-temperature perturbations and explicitly identifies the
two missing conditions; it is not a complete sweep or energy/cost optimization.

This package is prepared for manuscript integration. No manuscript agent was
contacted and no manuscript source or environment was modified here.

## Runtime response

The root-level Engine optimization is now measured, not projected: three
sequential fresh-start Case 3C runs took a median **34.66 s** for the BVP
(34.41–34.89 s), or **51.45 s** including initialization and profiles.
The restored experimental wheel independently confirmed 35.78 s / 52.86 s,
with the same 91.54328696744% capture and numerical configuration.
Use the scoped hardware/settings statement and exact wheel identities in
`../../results/runs/runtime_diagnostics_20260904/notebook.qmd` for R1.11.
The seven-case campaign is not being rerun just to replace its runtime example.
The fast Engine retains one explicitly accepted, separately documented
temperature-boundary test regression; do not describe the whole Engine suite
as passing or this implementation as universally optimal.

## Explicit manuscript-agent instructions: replace outdated runtime reporting

Verified against the current sources on 2026-09-04 in
`/home/tnnrpolley21/.codex/worktrees/bad5/MEA-Absorption-Column`, branch
`codex/fallback-manuscript`. The user requests the fastest verified Engine
calculation as the current runtime example. Implement these editorial changes
alongside the seven-case figures above; no new campaign is required. Line
numbers below are inspection locators and may move as the manuscript changes.

### Exact changes to make

| File and locator, relative to the manuscript checkout | Required change |
|---|---|
| `docs/latex/tables/reactive_numerical_verification.tex:21–23` | The rows still report CPU 309.481/515.481 s, BVP wall 309.853/517.139 s, and total wall 418.663/622.446 s. Remove these three timing rows from the reader-facing mesh-refinement table, retain them in their original archived evidence, and report the optimized timing separately as specified below. Keep the actual coarse/refined accuracy and conservation results. Rename the caption to numerical verification rather than implying this table gives current Engine cost. |
| `docs/latex/sections/results.tex:97`, Numerical Solver Behavior | Replace the paragraph leading with 515.481/517.139 s with the optimized runtime paragraph below. Preserve any useful explanation of the original 335/499 equilibrium counts as belonging to the original refinement runs, not the optimized timing runs. Do not attach those counts to the new wheel. |
| `docs/latex/sections/methods.tex:142` | Distinguish historical single-run numerical-verification timings from the three optimized repeats. Add the measured hardware, single-thread settings, fresh initialization, timing boundaries, and process peak RSS. Replace the blanket statements that measurements are single-run and memory was not measured with scope-specific statements. The historical Henry-method timings still have their own startup/output boundaries. |
| `docs/fallback_reviewer_response.md:135–137`, R1.11 | Replace the old 309/517-second response with the fastest observed time, repeat range/median, CPU time, RSS, and mesh/settings below. Explain the exact derivative-work reductions, without claiming a speed ranking against unmatched Henry/shooting/finite-difference runs. Preserve the reviewer's original wording. |
| `docs/latex/scripts/reviewer_checklist.json:269–270`, R1.11 `basis` and `remaining` | Make the same correction, add the retained timing evidence to the relevant evidence entries, and recompute hashes only after checking the changed manuscript. Reassess other entries that reference edited files. The existing live HTML reads this JSON; do not hand-edit numbers into the HTML template. |
| `REPRODUCE.md:40–58` and coupled-column reproduction instructions around line 105 | Add the exact optimized wheel, command, settings, and evidence location below. The existing lock and final integration pin still select the older b011 wheel; ordinary `uv sync`/`uv run --frozen` does not select this optimized wheel automatically. Keep historical reproduction separately labeled. |
| `docs/latex/QA_REPORT.md:65–81` and old 300-second failure entries | Add a dated current correction explaining the verified optimized runtime. Keep dated failure/history entries as history, not current limitations; do not erase the retained failed attempt. |

Do not blindly replace every occurrence of 300, every historical runtime, or
every 91.54839% capture. The refined 41-to-42-node, tolerance-0.05 sensitivity
reference has not been timed with the optimized wheel. Its old cost is not the
current optimized cost, and 34.405 s must not be inserted in that refined column.
The existing historical 5.25/7.70-second medians and method-comparison figure
describe other models and should remain explicitly historical if retained.

### Values and replacement paragraph

Use `optimized_plain_3` as the **fastest observed BVP run**, not an estimate:

- BVP wall **34.4052288532 s**; BVP process CPU **34.231582059 s**.
- Whole calculation for that same run **51.4534527540 s**; peak process RSS
  **203.51171875 MiB**. Do not pair its BVP time with another run's shorter total.
- Three-repeat BVP median **34.6569113731 s**, range **34.4052288532–34.8856194019 s**.
- Capture **91.54328696744%**, two mesh cycles, 21 initial/22 final nodes,
  tolerance **0.5**, boundary tolerance **0.001**, maximum nodes **1000**.
- Final maximum RMS residual **0.125340382574**, maximum scaled boundary
  residual **0**, and no invalid-state penalties. These are coarse settings,
  not a new mesh-convergence claim.

Suggested Results/R1.11 prose (adapt to LaTeX conventions):

> With the optimized Engine, the fastest of three sequential fresh-start Case
> 3C calculations required 34.41 s BVP wall time and 34.23 s process CPU time
> on an AMD Ryzen 5 5500 with single-thread numerical libraries. BVP wall times
> ranged from 34.41 to 34.89 s, with a median of 34.66 s. For the fastest BVP
> run, fresh Henry initialization, calculation and profile export together
> required 51.45 s, and peak process resident memory was 203.51 MiB. The mesh
> increased from 21 to 22 nodes in two mesh iterations at collocation tolerance
> 0.5 and boundary tolerance 0.001. Capture remained 91.54329%, with numerical
> profiles agreeing with the prior implementation within the retained comparison
> tolerance. These measurements describe this discretization on a shared
> machine, not a matched-accuracy ranking of numerical methods.

Explain the acceleration in one Methods sentence: pressure-root iterations
evaluate only pressure and its exact density derivative, accepted states receive
full thermodynamic properties, and exact Hessian actions, shared association
Jacobians, primal-point reuse, and demand-driven source-temperature derivatives
remove unnecessary repeated work. No physical equation or acceptance tolerance
was relaxed. This remains the conventional enhancement-factor column, not a
CasADi or resolved-reactive-film calculation.

### Exact fastest wheel and how to use it

The three-repeat measurements use this non-editable wheel:

```text
/home/tnnrpolley21/.cache/epcsaft/wheels/bafc4375476a39f08f5bc43cc6ea4b034d1ca956730910a0e753613007e8d12f/epcsaft-0.2.0.dev0-cp313-cp313-linux_x86_64.whl
SHA-256: 9b538f4defd5af661cd736af03760a55adb59b231d674f96c1e0f7d67350689d
```

The Engine source is the local `codex/reactive-evaluation-reuse` revision based
on `4563c6a89f8837ebb1bc24408b7177fa8d209e9d`, with uncommitted optimizations;
do not pretend the base commit alone identifies the optimized source. Preserve
the exact wheel and its hash in the reproducibility package. The later restored
wheel, SHA-256 `91632d2812429cbd293aae70fe8d4efb00000efe2377a91546dd7374dca67ee4`,
uses the same optimized approach and independently took 35.7818 s BVP / 52.8563 s
whole calculation. Do not assign the 34.405-second observation to that rebuild.

No rerun is needed for the editorial update. If verifying or generating further
results, this command selects the measured fastest wheel without changing the
manuscript checkout's environment. Use a new label instead of overwriting an
existing result directory:

```bash
cd /home/tnnrpolley21/.codex/worktrees/bad5/MEA-Absorption-Column
runtime_wheel=/home/tnnrpolley21/.cache/epcsaft/wheels/bafc4375476a39f08f5bc43cc6ea4b034d1ca956730910a0e753613007e8d12f/epcsaft-0.2.0.dev0-cp313-cp313-linux_x86_64.whl
sha256sum "$runtime_wheel"
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/home/tnnrpolley21/.codex/worktrees/bad5/MEA-Absorption-Column/src \
  uv run --isolated --no-project --python 3.13 \
  --with "$runtime_wheel" --with numpy==2.4.4 --with scipy==1.17.1 \
  --with pandas==3.0.2 --with matplotlib==3.10.9 --with openpyxl==3.1.5 --with casadi==3.7.2 \
  /home/tnnrpolley21/.codex/worktrees/ac57/MEA-Absorption-Column/analyses/nccc_validation/scripts/diagnose_case3c_runtime.py \
  manuscript_fast_path_confirmation --plain --wheel "$runtime_wheel"
```

Check the printed SHA-256 against the value above before execution. The driver
also checks installed package bytes against the wheel ZIP. It selects
`thermo_model='epcsaft_reactive_nine'`, `method='scipy-bvp'`, fresh Henry
initialization, `jacobian_mode='native'`, and `reactive_reuse_states=True`.
It uses temperature/raw states, dry-saturated vapor composition and reported
dry gas mass flow. Keep these settings and the selected parameter bundle;
using the old wheel, default finite-difference Jacobian, different mesh, or
instrumented trace mode will not reproduce this timing configuration.
CasADi is a pinned import dependency here, not the selected BVP solver.

If making this wheel the manuscript checkout's persistent default, update the
existing `pyproject.toml` wheel source, `uv.lock`, integration wheel/source
identity and `REPRODUCE.md` together through the existing integration procedure.
Preserve all five adopted input-file hashes; do not change model parameters or
disable identity checks. Run the final integration check and focused tests
before claiming adoption. The current local optimized source is uncommitted:
do not invent a committed source identity to satisfy an archive requirement.
This handoff does not claim the manuscript environment has already been updated.

### Evidence transfer and completion checks

The evidence source is
`/home/tnnrpolley21/.codex/worktrees/ac57/MEA-Absorption-Column/analyses/nccc_validation/results/runs/runtime_diagnostics_20260904/`.
Read `summary.csv`, `profile_comparisons.csv`, `notebook.qmd` and
`optimized_plain_1/`, `optimized_plain_2/`, `optimized_plain_3/` (each contains
`identity.json`, `result.json`, `diagnostics.json`, initialization and numerical
tables). Copy the selected evidence into the manuscript repository's established
analysis owner before adding checklist evidence paths: the checklist deliberately
rejects paths outside its checkout. Preserve original hashes and per-run wheel
identities; do not regenerate provenance as though old runs used the new wheel.

The user accepts the fast experimental implementation despite the separately
documented temperature-boundary failure. Keep its disclosure in reproducibility
and Engine notes; do not claim all Engine tests pass or introduce a new approval
gate before applying these manuscript edits. No new runtime figure is requested.

After editing, inspect the rendered numerical table and Results paragraph; run
the existing reviewer-checklist test, manuscript build and PDF-freshness check.
Recheck R1.11 and every checklist entry whose evidence changed, rather than
blindly refreshing hashes. Search the active manuscript and response again for
309.481, 309.853, 515.481, 517.139, 418.663 and 622.446: none should remain
presented as the current optimized runtime. Retained historical records may
contain them. Do not edit generated PDF text dumps directly. Preserve numerical
refinement and sensitivity results and keep the seven-case physical figures
as the manuscript deliverable.
