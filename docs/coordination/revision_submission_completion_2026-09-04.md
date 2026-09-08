# Manuscript revision package — 4 September 2026

The author authorized completing the language polish, reviewer-response reconciliation, cover letter and original-manuscript bundle, with a later Fable editorial check. Fable's language audit governed the polish; Astra's bounded wording corrections were included without adding scientific studies.

Canonical manuscript: `docs/latex/main.tex`; final PDF: `docs/latex/builds/main.pdf` (33 pages). Canonical response: `docs/fallback_reviewer_response.md`; canonical cover letter: `docs/revision_cover_letter.md`. The delivery package is `docs/latex/builds/revision_submission_2026-09-04.zip`.

The response reproduces all 20 numbered comments verbatim, gives direct responses without internal checklist labels, and identifies final manuscript pages. It distinguishes the completed operating study from a constrained optimum and the other-amine discussion from completed fitting or validation. The cover letter is addressed to Next Chemical Engineering; no manuscript ID was supplied.

The author confirmed the preserved May 10, 29-page manuscript as the original for the advisor's redline. The original PDF remains byte-identical (SHA-256 `09b8ce8226c7e9e083339deb2231adff9eca5b37171f1b9c03d178ea07c16f62`). Its 26 manuscript/source files match commit `c4726f69fdc034f5bec817ab720239967084ecdf`. Added Elsevier CAS build-support files allow an independent build; that build yields identical extracted text after whitespace normalization and the same page count. The unchanged preserved PDF is delivered.

Validation: canonical and independently exported revised source builds pass; final PDF freshness and immutable-wheel integration pass. All 59 snapshotted displayed equation blocks and existing parameter-table cells are byte-identical before/after polish. Snapshotted numerical figure CSVs are unchanged. Figure PDF text differences contain exactly the three approved label edits. Table S1 identifies grouped diagnostics, duplicate baselines, sampling grids and unavailable fields. Bibliographies and the sealed Astra report are unchanged. All 33 manuscript pages and both letter PDFs were visually inspected; cross-references/citations resolve and prohibited-word checks pass. One CAS title-area overfull-box warning remains with no visible clipping.

Figure rendering: the normal sensitivity renderer succeeded. The full transport renderer encountered a retired absolute input path and the operating renderer rejected the changed dependency-lock identity. Their exact plotting blocks were therefore executed against unchanged retained CSVs, and renderer/output hashes were updated without changing the original calculation identities. No model was rerun or identity check disabled.

No email, journal submission, Overleaf synchronization, tag creation or public release was performed. The author-approved `nce-revision-v3` reference remains a release step. Fable's final check has not yet been run.

## Final optional polish and Overleaf delivery

Fable found no blocking editorial issues. The author then approved the optional formatting changes. These are applied: consistent 10,001 formatting, finest-refinement wording with mesh/tolerance, removal of ambiguous “original,” matching abstract/conclusion loading precision, diagnostics numbered Table 11, and top-of-page table placement. Reviewer quotations remain verbatim. Letters and bundle now use Table 11; manuscript pagination remains 33 pages.

The final manuscript/source projection was committed and pushed on `codex/fallback-manuscript` at `01fa341`. The Overleaf mirror under Publications was committed and pushed on `master` at `794735d463f9f9a640829647b15bd2db34905bfc`. A fetch and strict remote audit confirmed identical source/mirror hashes and a clean mirror. The professor bundle's revised source matches all 33 mirror files byte-for-byte. Letters, original manuscript, QA report, build log, scripts and builds were excluded from Overleaf. The sync helper now refuses dirty mirrors; all six workflow tests pass. The approved tag name remains `nce-revision-v3`; tag creation remains the author's separate release step.
