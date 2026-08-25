# Revision Report

## 2026-08-25 MEA-only pre-MDEA freeze

- Bound the downstream ePC-SAFT 0.2 adapter to Engine commit `b64f6df906489cb132792641c5be2ee8f404b114` and wheel SHA-256 `a505d20ac9019e9b4edbad16df80166b6e082170ef2a2f002d897269e756d0d9`.
- Refreshed the tracked bibliography from the Zotero Companion path-sanitized Better BibTeX export; the separate `Documents/Papers` auto-export was not used because it contains local attachment paths.
- Reserved tag `nce-mea-only-pre-mdea-2026-08-25` for the validated MEA-only manuscript, evidence, and code state before MDEA companion work.
- Kept the archived nine-species activity-rebased calculations as historical numerical-feasibility evidence only; the ePC-SAFT 0.2 reactive route remains fail-closed until independently sourced standard-state reaction constants are admitted.

## Summary

Implemented the submission-readiness plan around the current committed evidence artifacts. The manuscript now separates the routine `epcsaft_ionic` fugacity campaign from the slower full nine-species activity-coupled feasibility path, reports accepted and attempted NCCC rows separately, and adds reviewer-facing reproduction and traceability documents.

## Manuscript Changes

- Added explicit accepted-row gate language: solver success, boundary residual norm at or below 1.0, physical capture in the 0--100\% range, and completion within the configured subprocess timeout. Temperature RMSE is reported but is not an acceptance gate, and guard or invalid-state counts are conditioning diagnostics unless a separate gate is configured.
- Clarified why K19 remains in the accepted validation set despite 166 ePC-SAFT and 183 Henry-law invalid-state and guard-penalty events.
- Removed the fixed driving-force scale framing from the manuscript after the source-backed transport, thermophysical-property, and gas-basis corrections made the previous scale factor unnecessary.
- Replaced informal Section 2.2 phrasing with direct engineering prose for superficial velocity, effective interfacial area, hold-up correlations, and heat-transfer coefficient definitions without changing equations.
- Corrected rendered ion notation for \ce{CO3^{2-}}, \ce{H3O+}, and \ce{OH-}, and noted that those entries are auxiliary diagnostic species unless independently sourced parameters are provided.
- Rewrote the title and abstract around the core claim: ePC-SAFT CO2 fugacity driving forces in a reproducible MEA absorber benchmark.
- Tightened the introduction and novelty ceiling to "to the authors' knowledge" and defined ePC-SAFT, NCCC, BVP, and eNRTL in reviewer-facing terms.
- Revised Sections 3.1--3.3 so shooting, finite difference, and collocation are concise, equation-backed, and tied to the absorber benchmark instead of tutorial prose.
- Kept the ePC-SAFT model description to the five evaluated residual-Helmholtz contribution terms and referenced the detailed ePC-SAFT literature for the full expressions.
- Rebuilt Section 4 around distinct evidence roles: one-bed validation, thermodynamic driving-force comparison, solver behavior, full activity-coupled timing, and limitations.
- Added a generated attempted-row table for K18--K20 and 1C--7C so the accepted-row accuracy aggregate is conditional and auditable.
- Updated Table 3 to include accuracy and timing for the routine campaign and the full activity-coupled path.
- Moved limitation language into the limitations section and removed defensive language from the abstract and main results narrative.

## Evidence Basis

- Routine accepted cases: K18, K19, and 1C--7C from `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv` (18 thermodynamic rows).
- Routine summary values: ePC-SAFT mean absolute capture error 3.32 percentage points and median runtime 7.70 s; Henry-law mean absolute capture error 3.14 percentage points and median runtime 5.25 s.
- Attempted rows: K20 is rejected by solver failure and mesh/domain-guard behavior; both 7C thermodynamic rows satisfy the stated gate.
- Full activity-coupled path: `analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv`, all seven 2017 C rows converged under relaxed feasibility settings, mean runtime 171.102 s, mean chemistry-solve time 127.764 s, mean absolute capture error 6.595 percentage points.
- Full-path Case 3C evidence is preserved as capture error -0.764 percentage points and runtime 139.420 s. The manuscript reports the evidence in prose and tables without exposing internal CSV or script filenames in the body text.
- Solver-method contrast: `analyses/nccc_validation/results/final/tables/method_case_contrast.csv`.

## Reproducibility Updates

- Added `REPRODUCE.md` with environment setup, artifact refresh, routine C-case campaign rerun, optional slow full-path rerun, validation, and LaTeX build commands.
- Added the `generate_nccc_one_bed_artifacts.py` and `sync_latex_figures.ps1` commands to the curated artifact refresh chain.
- Added `docs/code_to_paper_traceability.md` mapping manuscript claims and figures/tables to source artifacts and scripts.
- Updated `README.md`, `analyses/README.md`, `analyses/nccc_validation/README.md`, `analyses/nccc_validation/analysis.yaml`, and `docs/workflow_map.md` to remove local machine paths and distinguish routine ePC-SAFT from full activity-coupled ePC-SAFT.
- Left `references.bib` untouched; the non-Zotero software citation now lives in `docs/latex/software_references.bib` with repository URL, package version, commit hash, and no-DOI status.

## Context / Voice / Metadata Sweep

- Confirmed submission-facing LaTeX sources have no remaining hits for the targeted stale-language and metadata terms from the plan.
- Replaced informal or internal phrases across the manuscript and active workflow docs, including branch-local wording and vague claims about model positioning.
- Removed SRP-style language from the manuscript narrative and described the evidence as a high-liquid-to-gas-ratio probe.
- Kept process-level framing around mass transfer, solver conditioning, local driving force, validation scope, runtime, and reproducibility.
- Confirmed the abstract has no bracketed citations.
- Confirmed local machine paths were removed from README and workflow docs.
- Confirmed figure/table source paths under `docs/latex` exist and the compiled PDF contains the attempted-status table, solver table, timing table, and updated software/data availability statements.
- Broad repository search still finds historical local Windows paths in legacy SRP CSV traceback artifacts and old planning/answer documents. Those are not submission-facing manuscript sources and were left unchanged.

## Remaining Author Verifications

- No archival DOI has been minted for either repository at this pass; mint an archive DOI before journal submission if a permanent identifier is desired.
- The LaTeX availability statements currently cite the repository URLs and the current evidence baseline commits available during the cleanup pass.
- Decide whether the remaining bibliography underfull warning from `output.bbl` is acceptable; it is a minor bibliography line-break warning and does not block compilation.

## Verification

- `.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\generate_nccc_one_bed_artifacts.py` regenerated the one-bed tables and Figure 4 artifacts.
- `docs\latex\scripts\sync_latex_figures.ps1` refreshed the LaTeX figure copy.
- `docs\latex\scripts\build_main.ps1` passed and produced a fresh 30-page PDF.
- `.\.venv\Scripts\python.exe docs\latex\scripts\check_main_pdf_fresh.py` passed.
- `.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py` passed.
- `git diff --check` passed with only line-ending normalization warnings.
- `docs\latex\scripts\sync_to_overleaf_mirror.ps1 -CleanBuildFiles -WhatIf` completed.
- `docs\latex\scripts\sync_to_overleaf_mirror.ps1 -CleanBuildFiles` completed.
- `docs\latex\scripts\test_overleaf_sync.ps1` passed.

## Final Exact-Edit Pass

### Files Changed

- `docs/latex/main.tex`
- `docs/latex/sections/model_framework.tex`
- `docs/latex/sections/methods.tex`
- `docs/latex/sections/results.tex`
- `docs/latex/sections/conclusion.tex`
- `docs/latex/sections/data_availability.tex`
- `docs/latex/sections/code_availability.tex`
- `docs/latex/tables/appendix_parameter_scope.tex`
- `docs/latex/tables/epcsaft_parameter_summary.tex`
- `README.md`
- `analyses/nccc_validation/scripts/generate_accuracy_credibility_artifacts.py`
- `analyses/nccc_validation/scripts/run_full_species_ionic_2017_c_case_sweep.py`
- `analyses/nccc_validation/results/final/figures/*`
- `analyses/nccc_validation/results/final/tables/*`
- `docs/latex/figures/*`

### Exact Edits Applied

- Replaced the abstract with the final concise submission abstract and confirmed it remains under 250 words.
- Defined post-combustion carbon capture as PCC at first use.
- Tightened the thermodynamic overview so ePC-SAFT is described as the liquid-side electrolyte fugacity closure, with the vapor-side calculation using the same fugacity-coefficient framework for neutral species.
- Removed the former hidden driving-force scale statement and replaced the surrounding discussion with the source-backed validation basis used by the current routine campaign.
- Revised transport prose for interfacial area, holdup, mass transfer, heat transfer, molar flux, and enthalpy flux without changing the equations.
- Replaced "full ePC-SAFT fugacity correction" with "ePC-SAFT CO2 fugacity correction" in the validation-method description.
- Updated Figure 2 and Figure 3 captions so each figure has one clear evidence role.
- Updated Figure 5 to use a linear y-axis with enough headroom for failed-run labels.
- Replaced source-path metadata in submission-facing CSV artifacts with portable repository-relative or package-relative notes.
- Rendered carbonate, hydronium, and hydroxide notation consistently in Section 4.4 and the appendix parameter table.
- Replaced the conclusion, Data Availability, and Code Availability text with the final submission reference language.
- Switched the visible ePC-SAFT package URL to `\url{...}` so the Code Availability paragraph builds without an overfull box.

### Numerical Values Preserved

- Routine accepted-row ePC-SAFT mean absolute capture error: 3.32 percentage points.
- Routine accepted-row Henry-law mean absolute capture error: 3.14 percentage points.
- Routine accepted-row ePC-SAFT median runtime: 7.70 s.
- Routine accepted-row Henry-law median runtime: 5.25 s.
- Full activity-coupled mean absolute capture error: 6.595 percentage points.
- Full activity-coupled mean runtime: 171.102 s.
- Full activity-coupled mean chemistry-solve time: 127.764 s.
- Source-backed routine campaign uses the unscaled driving-force path.

### Commands Run

- `.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\generate_nccc_one_bed_artifacts.py`
- `.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\render_figures.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File docs\latex\scripts\sync_latex_figures.ps1`
- `powershell -NoProfile -ExecutionPolicy Bypass -File docs\latex\scripts\build_main.ps1`
- `.\.venv\Scripts\python.exe docs\latex\scripts\check_main_pdf_fresh.py`
- `.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py`
- `git diff --check`

### Build / Validation Results

- LaTeX build passed and produced a fresh `docs/latex/builds/main.pdf`.
- PDF freshness check passed.
- NCCC validation artifact consistency check passed.
- `git diff --check` passed with only line-ending normalization warnings.
- The only remaining LaTeX warning is the existing minor bibliography underfull box in `main.bbl`.

### Grep Results

- No submission-facing hits remain for the requested stale-language phrases, the author-verification marker, the old capture-error unit label, or local Windows user paths.

### PDF Visual Checks

- Page 1: title, abstract, keywords, and introduction opening render cleanly.
- Page 16: Figure 2 renders with the liquid-inlet boundary explanation attached to the caption.
- Page 17: Figure 3 renders as the validation-only temperature-profile overlay.
- Page 18: Figure 4 renders with the revised capture-error axis label.
- Page 19: Figure 5 renders with a linear y-axis and unclipped failed-run labels.
- Page 27: Appendix parameter tables render with readable ePC-SAFT parameter notation and ion formulas.

### Tag / Submission Reference

- The manuscript, Data Availability, Code Availability, and report use `nce-submission-preprint-v2` as the submission reference tag.

## Final Submission Metadata And Figure 4 Update

### Files Changed

- `docs/latex/main.tex`
- `docs/latex/sections/declaration_competing_interest.tex`
- `docs/latex/sections/data_availability.tex`
- `docs/latex/sections/code_availability.tex`
- `docs/latex/sections/results.tex`
- `docs/latex/software_references.bib`
- `docs/latex/thumbnails/cas-email.jpeg`
- `analyses/nccc_validation/scripts/generate_nccc_one_bed_artifacts.py`
- `analyses/nccc_validation/results/final/figures/nccc_one_bed_thermo_benchmark.pdf`
- `analyses/nccc_validation/results/final/figures/nccc_one_bed_thermo_benchmark.png`
- `docs/latex/figures/nccc-one-bed-thermo-benchmark.pdf`
- `docs/code_to_paper_traceability.md`

### Author Metadata Added

- Added Tanner W. Polley's ORCID identifier: `0009-0008-5957-4152`.
- Added BYU author emails for Tanner W. Polley and John D. Hedengren.
- Marked John D. Hedengren as corresponding author.
- Added the CAS template email thumbnail required by the `cas-sc` front matter under `docs/latex/thumbnails/cas-email.jpeg`.

### Declarations Added Or Updated

- Added a CRediT Author Statement.
- Added a Funding statement declaring no specific funding.
- Updated the Declaration of Competing Interest.
- Updated Data Availability to point to the project repository and `nce-submission-preprint-v2`.
- Updated Code Availability to point to the project repository, `nce-submission-preprint-v2`, and the `epcsaft` package release.

### Software Citation Update

- Updated `docs/latex/software_references.bib` for the `epcsaft` software citation.
- The citation now points to the public `ePC-SAFT` GitHub release `v1.5.0`, includes package version 1.5.0, and records commit `e4ee304721ac6670120e69ef260a596ea3c3fe9c`.
- `docs/latex/references.bib` remained untouched because it is Zotero-owned.

### Figure 4 Correction

- Fixed `analyses/nccc_validation/scripts/generate_nccc_one_bed_artifacts.py` so Figure 4 and its summary are derived from the all-attempted artifact by the manuscript's stated gate:
  - `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`
  - `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_summary.csv`
- Removed the hard-coded accepted-case list that excluded a passing 7C result.
- Added validation inside the script so Figure 4 fails unless all 20 expected attempted rows are present exactly once and the accepted artifact equals the row-level gate.
- Regenerated Figure 4 from the gate-derived all-attempted data, removed the horizontal gridlines, and moved the legend to the top middle of the left capture-validation panel.
- Updated `docs/code_to_paper_traceability.md` so the Figure 4 script path points to `generate_nccc_one_bed_artifacts.py`.

### Manuscript Prose Cleanup

- Reconciled the abstract, methods, results, conclusion, timing table, and figure caption with the gate-derived acceptance of 7C and rejection of K20.
- Removed the Section 4.4 sentence that exposed internal CSV and script filenames in manuscript prose.

### Commands Run

- `.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\generate_nccc_one_bed_artifacts.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File docs\latex\scripts\sync_latex_figures.ps1`
- `powershell -NoProfile -ExecutionPolicy Bypass -File docs\latex\scripts\build_main.ps1`
- `.\.venv\Scripts\python.exe docs\latex\scripts\check_main_pdf_fresh.py`
- `.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py`
- `git diff --check`
- `powershell -NoProfile -ExecutionPolicy Bypass -File docs\latex\scripts\sync_to_overleaf_mirror.ps1 -WhatIf -CleanBuildFiles`
- `powershell -NoProfile -ExecutionPolicy Bypass -File docs\latex\scripts\sync_to_overleaf_mirror.ps1 -CleanBuildFiles`
- `powershell -NoProfile -ExecutionPolicy Bypass -File docs\latex\scripts\test_overleaf_sync.ps1`
- `powershell -NoProfile -ExecutionPolicy Bypass -File docs\latex\scripts\test_overleaf_sync.ps1 -RequireCleanMirrorGit -VerifyRemote`

### Build / Validation Results

- LaTeX build passed and produced a fresh 31-page `docs/latex/builds/main.pdf`.
- PDF freshness check passed.
- NCCC validation artifact consistency check passed.
- `git diff --check` passed with only line-ending normalization warnings.
- Local and remote Overleaf sync audits passed.
- Source repository `main` was pushed with tag `nce-submission-preprint-v2`.
- Overleaf mirror `master` was pushed after the source repository.

### PDF Visual Checks

- Page 1: title, author metadata, corresponding-author marker, ORCID line, abstract, and keywords rendered correctly.
- Page 2: the first citation renders as `[1, 2, 3]` and the reference list begins with the same numbered sources.
- Page 17: the attempted-status table renders legibly and reports 7C as accepted.
- Page 19: Figure 4 renders from the gate-derived data and includes 7C.
- Page 21: the full-species timing table renders legibly.
- Page 31: CRediT, Funding, Declaration of Competing Interest, Data Availability, Code Availability, and Generative AI disclosure render correctly.

### Metadata Pass Remaining Issues

- The build still reports nonblocking LaTeX warnings: three empty-anchor warnings from CAS front matter and underfull bibliography lines in `main.bbl`.
- These warnings do not block PDF generation, freshness validation, or Overleaf sync.

### Remaining Issues

- The bibliography still reports one minor underfull box in `main.bbl`. It is isolated to a reference-list line break and does not block the build.
- A reviewer-facing repository cleanup removed local IDE/Codex metadata, captured install logs, old internal planning documents, and unreferenced packaged data artifacts. Script-wired legacy analysis outputs such as the validation-evidence registry, calibration/holdout tables, and uncertainty/error-regime figures remain tracked because pruning them cleanly requires a separate analysis-script cleanup pass.
- The NCCC artifact validator passes. `tests/test_results_architecture.py` now passes all seven tests and asserts that the accepted artifact is exactly the stated gate applied to the attempted artifact; the accepted cases are K18, K19, and 1C--7C for both closures.
