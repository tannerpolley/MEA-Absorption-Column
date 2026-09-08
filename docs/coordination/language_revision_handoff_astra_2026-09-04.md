# Handoff: orchestrate the manuscript language revision (GPT Astra, orchestrator)

## Current assignment — independent review before execution

The author has superseded the immediate-execution instruction. A fresh Astra reviewer first performs and seals its own full audit of the manuscript without reading the earlier Fable/Claude review or the derived batch plan. It then reads that review and the complete updated batch plan, compares the findings and produces a consolidated execution plan. This is a language-and-story audit with bounded factual sanity checks against existing evidence. No new analyses, experiments or literature-research expansion are authorized. The parent agent remains responsible for later text-edit, build and verification executions. Manuscript execution is paused until this review and consolidation are complete; do not begin Batch 0 or edits under this handoff yet.

The intended review reports are `astra_independent_manuscript_audit_2026-09-04.md`, `astra_revision_plan_review_2026-09-04.md`, and `consolidated_revision_plan_2026-09-04.md` in this directory. Preserve the blind report and record its SHA-256 before reading prior findings. New findings are explicitly allowed in this review stage.

The execution rules below apply only to the subsequently approved consolidated plan. Do not silently expand its edits or change scientific results.

## Execution reading order (after the blind review)

1. `docs/coordination/language_revision_batches_2026-09-04.md` — the shared contract and Batches 0–7.
   It contains the original batches and the author-approved updates. The later approved consolidated plan will govern execution; other repository documents supply evidence.
2. `docs/latex/builds/main.pdf` — the current manuscript (32 pages). Skim so you recognize the sections.
3. `REPRODUCE.md` lines 1–20 — the build command chain.

## Fixed decisions (author-approved; do not reopen)

- Absorber code cited at tag `nce-revision-v3` (the author creates the tag; the text just names it).
- ePC-SAFT package: "not yet publicly released; available from the authors on request and will be
  released publicly when ready." No version or commit in the text.
- Per-run convergence diagnostics go into one supplementary table (Batch 2); the repeated
  diagnostic paragraphs in Results are deleted (Batch 3).
- "Reusable component descriptions" is motivation only; abstract and conclusions reworded (Batch 4).
- Enhancement divisor: 1.0454 in the text, full precision in Appendix B (Batch 5).
- Energy convergence: retain one concise Case 3C refinement statement and supporting supplementary-table values, as specified in the updated batch plan. Remove the unresolved-cause and repeated discrepancy discussion. Preserve original run values and distinguish their sampling from the two additional verification runs.
- Amine-ion parameters: "fitted by the authors to ion and speciation data in work to be reported
  separately" (Batch 6).
- The words local, history, historical, histories must not appear anywhere in the PDF.

## Execution rules

- Batch 0 yourself: `git add -A && git commit -m "Checkpoint before language-revision batches"`.
  The working tree is already dirty with the current manuscript; that is expected.
- Batches 1–7 run strictly in order, one at a time. Delegate each to your routine implementation
  lane (Codex `codex-implementer`, GPT-5.6 Luna, medium effort) with the spec's shared contract plus
  that batch's section pasted verbatim. If a batch fails twice on the same problem, apply that batch
  yourself; do not escalate to a further model.
- After each batch: read the implementer's report, run the four checks from the shared contract
  yourself (do not trust the report alone), then commit with message
  `Language revision batch N: <batch title>` and the standard Co-Authored-By trailer.
- An `OLD` string not found verbatim means the source drifted from the spec. Locate the current
  wording, confirm it is the same sentence, apply the `NEW` text, and note the drift in the report.
  Never skip an item silently.
- Batch 2 creates `tables/supplementary_run_diagnostics.tex`. Populate the original group rows from values already
  in `sections/results.tex` as listed in the spec, and the two additional Case 3C verification rows
  from the exact retained evidence listed in the updated plan. Write `--` for genuinely unavailable
  values and list those cells in the report. No new column calculations are part of manuscript editing;
  report any concrete wording limitation against existing evidence; do not propose or run new experiments during this pass.
- Batch 7 regenerates three figures. Read each figure directory's README before running its script,
  diff the SVG text and `provenance.json` to confirm only legend strings changed, then `sync-figures`.
- Numeric changes must map to specified rounding or approved additions/deletions, with layout and cross-reference changes accounted for separately. Never change equations, citation keys, existing labels, `references.bib`,
  `reviewer_checklist.json`, anything under `analyses/` other than the three render scripts in
  Batch 7, anything under `src/`.

## Close-out

1. Rebuild and read the final `builds/main.pdf` end to end. Confirm each removed-phrase grep across
   all seven batches prints nothing, and the prohibited-word grep prints nothing.
2. `git diff <checkpoint>..HEAD --stat` and a full `pdftotext` diff between
   `builds/main_before_batch1.pdf` and the final PDF; every changed line must map to a spec item.
3. Run `bash "$HOME/.codex/hooks/codex-cleanup.sh" --repo-root .`; remove only artifacts this work
   created (`builds/main_before_batch*.pdf` may stay; they are untracked scratch).
4. Final report to the author (this is your final response; do not message other agents): batches completed, commits, spec items not applied (with reason),
   source drift encountered, supplementary-table cells left as `--`, and the three checks' outputs.
   State plainly if anything could not be verified.

## Out of scope

Creating the git tag, releasing ePC-SAFT, updating the reviewer checklist, Overleaf sync, and any
scientific or numerical change. Flag these in the report if you think they are needed; do not do them.
