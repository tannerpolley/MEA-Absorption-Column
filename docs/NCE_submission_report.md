# Next Chemical Engineering Submission-Readiness Report

Date checked: 2026-05-08

## Executive Summary

The manuscript source uses Elsevier's CAS single-column class, `cas-sc`, from the Elsevier CAS template bundle. The working LaTeX project relies on the MiKTeX-installed CAS files for local compilation so the manuscript folder does not carry redundant class/style files.

The working manuscript source keeps figures in `docs/latex/figures` and table source files in `docs/latex/tables` for repository cleanliness. For Editorial Manager source upload, use `docs/latex/prepare_elsevier_submission.ps1` to create a flat copy because Elsevier's LaTeX instructions state that subfolders cannot be processed by EM.

The manuscript compiles successfully to a fresh PDF at:

`docs/latex/main.pdf`

## Standards Checked

- Next journals Guide for Authors, accessed 2026-05-08: `https://www.elsevier.com/en-gb/subject/next/guide-for-authors`
- Elsevier LaTeX instructions, accessed 2026-05-08: `https://www.elsevier.com/en-gb/researcher/author/policies-and-guidelines/latex-instructions`
- Elsevier generative-AI policies for journals, accessed 2026-05-08: `https://www.elsevier.com/en-au/about/policies-and-standards/generative-ai-policies-for-journals`
- Elsevier CAS template bundle, accessed 2026-05-08: `https://support.stmdocs.com/wiki/index.php?title=Elsarticle.cls`

## LaTeX Class And Bibliography

- Active class: `\documentclass[a4paper,fleqn]{cas-sc}`
- Target journal: Next Chemical Engineering
- Bibliography style: `cas-model2-names`
- Bibliography database: `references.bib`
- Local bibliography files present in `docs/latex`: `references.bib` only

The required CAS files `cas-sc.cls`, `cas-common.sty`, and `cas-model2-names.bst` are available through MiKTeX. The old `manual.bib` and `mendeley.bib` files were removed.

## Manuscript Metrics

- Article type: Original Research Article
- Abstract word count: 171
- Keyword count: 6
- PDF page count after CAS single-column restoration: 30

## Compile Status

Build command:

```powershell
.\docs\latex\build_main.ps1
```

Result: passed.

Freshness check:

```powershell
.\.venv\Scripts\python.exe docs\latex\check_main_pdf_fresh.py
```

Result: passed.

The LaTeX log scan found no fatal errors, undefined citations, undefined references, missing-figure errors, BibTeX missing-field warnings, rerun-required warnings, overfull-box warnings, or out-of-page annotation warnings after the final build.

Flat source-package compile command:

```powershell
Push-Location .\docs\latex\out\elsevier_submission_flat
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
Pop-Location
```

Result: passed.

## Results-Artifact Validation

Command:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py
```

Result: passed.

The validation script was updated to check only the live manuscript inputs and to reject stale `Figures/` or `figs/` figure paths.

## Figure And Source Layout

All manuscript figure references now use the manuscript-local `figures/` folder. The files currently in `docs/latex/figures` are all referenced by `main.tex` or `revised_benchmark_results.tex`.

All manuscript tables are in standard LaTeX `table` environments with `\caption{...}` and `\label{...}`. Each table is stored as its own `.tex` file under `docs/latex/tables/` and is inserted with `\input{tables/...}`. The included benchmark-results file uses normal subsection organization plus standard table and figure floats.

The Overleaf sync script now copies only the `figures/` directory and treats old `figs/`, `Figures/`, `thumbnails/`, and `benchmark_figures/` folders as stale mirror directories.

## Author-Owned TODO Items

The manuscript intentionally leaves the following unresolved author-owned items as `% TODO_AUTHOR:` comments:

- Confirm affiliations, corresponding-author designation, e-mail, and ORCID identifiers.
- Confirm the final CRediT role statement for each author.
- Add the final repository release URL or archival DOI.
- Confirm acknowledgements, funding statement, and funding-source role.

## Submission Packaging Notes

For initial review, Next journals allow flexible formatting and a single manuscript PDF, but source files may still be requested. Elsevier LaTeX instructions state that Editorial Manager processes LaTeX source only when files are flattened to one folder level. The working repository keeps `figures/` for clarity, while the submission-prep script creates a flat upload copy.

Flat package command:

```powershell
.\docs\latex\prepare_elsevier_submission.ps1 -Zip
```

This writes `docs\latex\out\elsevier_submission_flat\` and `docs\latex\out\elsevier_submission_flat.zip`, with figure paths and table inputs rewritten to bare filenames in the copied `.tex` files.
The script was verified directly on this machine after fixing its script-root path resolution and text encoding write path for Windows PowerShell compatibility.

Suggested source package contents:

- `main.tex`
- `revised_benchmark_results.tex`
- all table `.tex` files from `docs/latex/tables/`
- `references.bib`
- all files in `docs/latex/figures/`
- `main.pdf`

The local source package relies on the installed CAS bundle. If a submission portal or coauthor build environment requires every class/style file explicitly, add `cas-sc.cls`, `cas-common.sty`, and `cas-model2-names.bst` from the standard Elsevier CAS bundle to the upload package at submission time.
