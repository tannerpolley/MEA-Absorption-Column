# Revision Report

## Summary

Implemented the submission-readiness plan around the current committed evidence artifacts. The manuscript now separates the routine `epcsaft_ionic` fugacity campaign from the slower full nine-species activity-coupled feasibility path, reports accepted and attempted NCCC rows separately, and adds reviewer-facing reproduction and traceability documents.

## Manuscript Changes

- Rewrote the title and abstract around the core claim: ePC-SAFT CO2 fugacity driving forces in a reproducible MEA absorber benchmark.
- Tightened the introduction and novelty ceiling to "to the authors' knowledge" and defined ePC-SAFT, NCCC, BVP, and eNRTL in reviewer-facing terms.
- Revised Sections 3.1--3.3 so shooting, finite difference, and collocation are concise, equation-backed, and tied to the absorber benchmark instead of tutorial prose.
- Kept the ePC-SAFT model description to the five evaluated residual-Helmholtz contribution terms and referenced the detailed ePC-SAFT literature for the full expressions.
- Rebuilt Section 4 around distinct evidence roles: one-bed validation, thermodynamic driving-force comparison, solver behavior, full activity-coupled timing, and limitations.
- Added an attempted-row table for K20 and 7C so the accepted-row accuracy aggregate is conditional and clear.
- Updated Table 3 to include accuracy and timing for the routine campaign and the full activity-coupled path.
- Moved limitation language into the limitations section and removed defensive language from the abstract and main results narrative.

## Evidence Basis

- Routine accepted rows: K18, K19, and 1C--6C from `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`.
- Routine summary values: ePC-SAFT mean absolute capture error 3.73 percentage points and median runtime 9.86 s; Henry-law mean absolute capture error 3.78 percentage points and median runtime 8.62 s.
- Attempted rows: K20 rejected by mesh/domain-guard behavior; 7C rejected by accepted-row timeout.
- Full activity-coupled path: `analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv`, all seven 2017 C rows converged under relaxed feasibility settings, mean runtime 171.102 s, mean chemistry-solve time 127.764 s, mean absolute capture error 6.595 percentage points.
- Solver-method contrast: `analyses/nccc_validation/results/final/tables/method_case_contrast.csv`.

## Reproducibility Updates

- Added `REPRODUCE.md` with environment setup, artifact refresh, routine C-case campaign rerun, optional slow full-path rerun, validation, and LaTeX build commands.
- Added `docs/code_to_paper_traceability.md` mapping manuscript claims and figures/tables to source artifacts and scripts.
- Updated `README.md`, `analyses/README.md`, `analyses/nccc_validation/README.md`, `analyses/nccc_validation/analysis.yaml`, and `docs/workflow_map.md` to remove local machine paths and distinguish routine ePC-SAFT from full activity-coupled ePC-SAFT.
- Left `references.bib` untouched; software citation placeholder remains isolated in `docs/latex/software_references.bib`.

## Context / Voice / Metadata Sweep

- Replaced informal or internal phrases across the manuscript and active workflow docs, including branch-local wording and vague claims about model positioning.
- Removed SRP-style language from the manuscript narrative and described the evidence as a high-liquid-to-gas-ratio probe.
- Kept process-level framing around mass transfer, solver conditioning, local driving force, validation scope, runtime, and reproducibility.
- Confirmed the abstract has no bracketed citations.
- Confirmed local machine paths were removed from README and workflow docs.
- Confirmed figure/table source paths under `docs/latex` exist and the compiled PDF contains the new attempted-status table, solver table, timing table, and AUTHOR VERIFY placeholders.

## Remaining Author Verifications

- Replace the placeholder ePC-SAFT software citation with the final public package citation, release URL, version, commit hash, and archival identifier.
- Add the final absorber repository release URL, release commit, and archival identifier before submission.
- Decide whether the remaining bibliography underfull warning from `output.bbl` is acceptable; it is a minor bibliography line-break warning and does not block compilation.

## Verification

- `docs\latex\scripts\build_main.ps1` passed and produced a fresh 30-page PDF.
- `.\.venv\Scripts\python.exe docs\latex\scripts\check_main_pdf_fresh.py` passed.
- `.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py` passed.
- `git diff --check` passed with only line-ending normalization warnings.
- `docs\latex\scripts\sync_to_overleaf_mirror.ps1 -CleanBuildFiles -WhatIf` completed.
- `docs\latex\scripts\sync_to_overleaf_mirror.ps1 -CleanBuildFiles` completed.
- `docs\latex\scripts\test_overleaf_sync.ps1` passed.
