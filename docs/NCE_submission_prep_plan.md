# Next Chemical Engineering Submission-Readiness Plan

> Handoff plan for preparing the MEA absorber manuscript for submission to Elsevier's Next Chemical Engineering. Follow this plan without changing scientific results, model code, generated benchmark data, or author-owned metadata unless explicitly instructed.

## Summary

Revise and submission-check the MEA absorber LaTeX paper for **Next Chemical Engineering** using the official Elsevier/Next Guide for Authors as the controlling standard, Elsevier LaTeX instructions as the secondary standard, and the pasted prompt guide as the implementation checklist.

The work should produce a polished, compilable manuscript PDF plus a submission-readiness report. Edits must preserve the paper's existing authorial voice: direct, practical, benchmark-focused, and transparent about limitations. Improve clarity and journal polish without making the manuscript sound generic, over-formal, or disconnected from the current writing style.

Do not invent scientific results, author metadata, affiliations, funding, declarations, citations, DOIs, acknowledgements, or conclusions. Missing author-owned information must be marked with `% TODO_AUTHOR:` in the LaTeX source and listed in the report.

No Python package API, model behavior, benchmark logic, or research data should be changed. This is manuscript, LaTeX, bibliography, figure-reference, and submission-package work only.

## Step-By-Step Plan

1. **Verify current journal standards**
   - Re-check the current official pages for:
     - Next Chemical Engineering Guide for Authors.
     - Elsevier LaTeX instructions.
     - Elsevier generative-AI policy.
   - Record in the report which standards were used and the access date.
   - If any conflict exists between the pasted guide and live journal instructions, follow the live official journal page.

2. **Inventory the manuscript repository**
   - Identify the main manuscript file, included `.tex` files, bibliography files, figure folders, class/style files, build script, generated PDF, and build output folder.
   - Confirm the active LaTeX class, currently expected to be Elsevier-compatible `elsarticle`.
   - Confirm bibliography workflow, currently expected to be BibTeX/natbib with `references.bib`.
   - Identify all figure paths and casing differences, with `figures/` as the manuscript-local figure folder.
   - Do not delete or reorganize files during inventory; only report unnecessary or stale submission-package files.

3. **Preserve paper voice while improving polish**
   - Before line editing, read the abstract, introduction, methods/model framing, results narrative, and conclusion to identify the manuscript's current voice.
   - Preserve the author's plainspoken, engineering-oriented style: concrete, transparent, reproducibility-focused, and honest about limitations.
   - Keep sentences clear and confident, but do not replace the paper's voice with generic journal filler.
   - Prefer tightening and clarifying existing phrasing over full rewrites.
   - Preserve first-person avoidance unless already used consistently.
   - Avoid promotional language, excessive hedging, and overly polished phrases that make the paper sound unlike the existing manuscript.
   - When a paragraph is awkward but scientifically clear, edit lightly first; only restructure heavily if flow, correctness, or journal compliance requires it.
   - Make terminology consistent while retaining phrases central to the paper's identity, such as "reproducible benchmark," "transparent Python framework," "controlled thermodynamic sensitivity," and "not a full electrolyte-thermodynamic replacement," where accurate.

4. **Clean front matter**
   - Keep the article type as **Original Research Article**.
   - Preserve the current author order.
   - Add or clean affiliations only where source information already exists.
   - If affiliations, corresponding-author designation, ORCID, or email are missing, add `% TODO_AUTHOR:` comments at the relevant source location.
   - Keep the title concise and technically accurate; do not change the claim scope.
   - Revise the abstract for journal tone, standalone clarity, and the official word limit. Use a stricter `<=250 words` cap unless live guidance says otherwise.
   - Reduce keywords to the official allowed count; if uncertain, use the stricter Elsevier-style maximum of 6.

5. **Tighten scientific framing**
   - Preserve the defensible claim: the paper is a reproducible benchmark for MEA absorber solvers and CO2 fugacity driving-force sensitivity.
   - Do not present the ePC-SAFT lane as a full electrolyte-reactive thermodynamic replacement.
   - Check abstract, introduction, results, figure captions, and conclusion for overclaims about industrial scale-up, thermodynamic fidelity, validation breadth, runtime, or predictive accuracy.
   - Where a claim is supported by existing benchmark artifacts, make the language precise.
   - Where support is missing, either qualify the claim or add `% TODO_AUTHOR:` if author input is required.

6. **Improve manuscript structure and writing**
   - Keep the discipline-appropriate structure rather than forcing a generic template.
   - Ensure the introduction clearly states context, gap, contribution, and conservative scope.
   - Ensure methods/modeling sections include enough reproducibility detail for equations, assumptions, boundary conditions, solver settings, units, tolerances, and validation data.
   - Ensure results describe key quantitative findings in text, not only in figures/tables.
   - Ensure discussion/conclusion interpret results without introducing new data or unsupported claims.
   - Add a short `What's Next` section after the conclusion if still consistent with current Next guidance.
   - Use consistent American English, consistent abbreviations, SI-style units, and chemical-engineering terminology.

7. **Repair LaTeX hygiene**
   - Remove visible draft markup, tracked-change artifacts, stale comments meant for authors, and any reviewer-response fragments that appear in the manuscript.
   - Keep only `% TODO_AUTHOR:` comments for unresolved author decisions.
   - Ensure labels are unique and consistently named with prefixes such as `fig:`, `tab:`, `eq:`, `sec:`, and `app:`.
   - Ensure equations are editable LaTeX, referenced equations are numbered, and symbols/units are clear.
   - Use the standard Elsevier `elsarticle` class for the submission manuscript.
   - Prefer simple, production-safe LaTeX over custom layout styling.

8. **Check figures and tables**
   - Confirm every figure and table is cited near first use and appears in logical order.
   - Ensure captions define panels, symbols, colors, markers, and abbreviations where needed.
   - Flag figures relying only on color or containing unreadable text.
   - Verify figure file existence, path casing, file format, and approximate size/resolution where possible.
   - Do not regenerate or alter research figures with generative AI.
   - Standardize figure paths only if it is needed for reliable build/source packaging.
   - Ensure tables are editable LaTeX, have units in headers, and avoid unnecessary vertical rules where safely fixable.

9. **Check references and citations**
   - Build or inspect citation output to confirm no unresolved citations.
   - Check for duplicate citation keys, malformed BibTeX fields, uncited references, missing required fields, and special-character escaping.
   - Do not fabricate DOIs or full author lists.
   - If full author lists or DOIs are missing and cannot be verified from reliable metadata, flag them in `NCE_submission_report.md`.
   - Preserve journal-appropriate natbib/Elsevier style unless live instructions require a change.

10. **Add structured declarations**
    - Replace the current broad `Disclaimers` section with journal-standard end matter:
      - `CRediT Author Statement`
      - `Declaration of Competing Interest`
      - `Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation Process`
      - `Data Availability`
      - `Acknowledgements`
      - `Funding`
    - Use known existing content only.
    - For unknown author-owned details, insert `% TODO_AUTHOR:` rather than guessing.
    - Include the Codex/OpenAI AI-use declaration as a draft for author review because this manuscript-editing workflow uses AI assistance.
    - Place declarations before references in the order expected by Elsevier/current journal guidance.

11. **Prepare reproducibility and submission packaging notes**
    - Ensure the Code Availability statement points to the repository and recommends a stable release/DOI before submission.
    - Ensure Data Availability accurately describes where input data, generated benchmark artifacts, and plotted data snapshots are available.
    - Flag whether Editorial Manager may require flattening figure/source files to one folder level.
    - Identify the exact files needed for upload: manuscript source, bibliography, class/style files if required, all figures, final PDF, declaration document, and optional highlights/cover letter.

12. **Build and validate**
    - Run the repository build command: `docs\latex\scripts\build_main.ps1`.
    - Use multiple LaTeX/BibTeX passes through the build workflow as needed.
    - Check for:
      - fatal errors
      - undefined citations
      - undefined references
      - multiply defined labels
      - missing figures
      - bibliography errors
      - serious overfull boxes
      - unresolved `% TODO_AUTHOR:` items
    - Run the existing paper-artifact validation check:
      - `.venv\Scripts\python.exe analyses\nccc_validation\scripts\validate_results.py`
    - Do not run long solver sweeps unless needed to verify a claim introduced by manuscript editing.

13. **Create final report**
    - Add `NCE_submission_report.md` with:
      - executive summary
      - compile status
      - final PDF path
      - abstract word count
      - keyword count
      - citation/reference status
      - files changed
      - journal compliance checklist
      - scientific-writing edits made
      - voice/style preservation notes
      - all `% TODO_AUTHOR:` items
      - build details and remaining warnings
      - submission packaging notes
    - Add `latex_build_notes.md` only if build behavior or warnings are complex enough to justify a separate file.

## Required Outputs

- Cleaned final LaTeX manuscript.
- Cleaned included `.tex` files if needed.
- Cleaned bibliography files if safe and necessary.
- Fresh compiled PDF at `docs\latex\builds\main.pdf`.
- `NCE_submission_report.md`.
- Optional `latex_build_notes.md` if the build has non-obvious warnings or workflow notes.

## Test And Acceptance Criteria

- Manuscript compiles without fatal errors.
- No unresolved citations or references remain unless explicitly justified in the report.
- Abstract is within the official limit, defaulting to `<=250 words`.
- Keywords comply with the official count, defaulting to no more than 6 if guidance conflicts.
- Required declarations are present or marked with `% TODO_AUTHOR:`.
- AI-use declaration is present for author review.
- No fabricated metadata, citations, funding, affiliations, acknowledgements, or results.
- All figures/tables are cited and have submission-appropriate captions.
- The final report clearly states whether the manuscript is submission-ready or what author decisions remain.
- The final prose still sounds like the existing manuscript: practical, transparent, engineering-focused, and conservative about claims.

## Assumptions And Defaults

- Use `Next Chemical Engineering` and `Original Research Article`.
- Use `elsarticle` as the manuscript class.
- Treat the current paper's claim boundary as conservative and correct.
- Preserve the author's current voice while improving readability and journal compliance.
- Do not modify research code, generated benchmark data, or scientific results.
- Use `% TODO_AUTHOR:` for missing author-owned facts.
- Do not create or modify research figures with generative AI.
