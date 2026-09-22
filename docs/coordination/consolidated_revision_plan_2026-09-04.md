# Consolidated manuscript language and story revision plan

**Date:** 2026-09-04  
**Checkout:** `/home/tnnrpolley21/Workspaces/Engineering/MEA-Absorption-Column`  
**Branch observed:** `codex/fallback-manuscript`  
**Owner of execution:** parent agent  
**Review owner:** Astra  
**Scope:** A bounded language, story, precision and reproducibility-presentation revision using existing manuscript and retained evidence. No new model runs, experiments, literature acquisition, numerical results or dependency changes are part of this plan.

This consolidated plan reconciles the sealed Astra audit, the prior Claude/Fable editorial audit and the updated seven-batch plan. The manuscript source and PDF were not changed while this plan was prepared.

## Evidence and sealed review identity

The sealed independent Stage-1 audit is:

`docs/coordination/astra_independent_manuscript_audit_2026-09-04.md`

Its SHA-256 is `7e82b293ca400c26767b40ef1fca68f3714236aad0aa10df8e8400826ab68ddd`. The hash was verified unchanged before this Stage-2 review and must be checked again before implementation.

The Stage-2 comparison used these final on-disk inputs:

| File | SHA-256 |
|---|---|
| `docs/claude_manuscript_language_audit_2026-09-04.md` | `894b972d4700871ee97f467dcb50e67ec4bc2d937606e2ad552ac649b84e6e33` |
| `docs/coordination/language_revision_batches_2026-09-04.md` | `274ba17804e822c295a6f4f51835a6f4b359141854bb0def1df6e92f59bc2184` |
| `docs/coordination/language_revision_handoff_astra_2026-09-04.md` | `4da222715569cb1dcdee10ea4e77a0d2757b5590f975e14d5aea16533c3d57d2` |

The current audited PDF is 32 pages. The pre-revision comparison is 29 pages. The current rendered manuscript has zero whole-word matches for `local`, `locally`, `history`, `historical` and `histories`; the final build must preserve this result.

## Editorial target

The article should tell one clean story:

1. It assembles a nine-species reactive absorber by coupling a molecular thermodynamic calculation, reaction equilibrium, a conventional enhancement closure, empirical caloric relations and transport correlations.
2. It evaluates that formulation on seven retained NCCC cases, with numerical completion separated from physical agreement.
3. It reports uneven capture agreement, a qualitative temperature comparison and bounded sensitivity/operating responses.
4. It shows a paired Case 3C refinement, including the additional sampling-grid energy diagnostic, without presenting that case as a general energy or thermodynamic validation.
5. It states the limits of the selected working parameters, conventional film closure, empirical calorics, transport applicability and public reproducibility route.
6. It presents predictive film development, other amines, alternative thermodynamic closures and optimization as future work.

The article must not imply that an accepted BVP residual validates the EOS, that a seven-case process comparison is an independent thermodynamic holdout, that a dense-grid Case 3C diagnostic applies to every run, or that a future public package already exists.

## Fixed author decisions retained

The following decisions from the handoff remain in force:

- use the tag `nce-revision-v3` for the eventual manuscript source route;
- state that the ePC-SAFT package is not yet public, is available from the authors on request, and will be publicly released when ready, without inventing a version or commit in article prose;
- put detailed run diagnostics in one supplementary table and remove repeated result paragraphs;
- describe reusable model components as motivation for future work;
- retain the divisor as 1.0454 in narrative text and give the full precision once in Appendix B;
- retain one concise Case 3C energy refinement statement with the original and new evidence distinguished;
- retain the author-approved amine-ion provenance statement as an author-supplied account, with the existing independent-validation limitation; do not present it as independently verified by this audit;
- preserve the ban on the five prohibited whole words in the rendered manuscript.

## Mandatory corrections before batch execution

These corrections make the existing batches internally coherent. They do not enlarge the scientific scope.

### 1. Add the missing limitations to the conclusion

The proposed Batch 4 conclusion still omits the context needed to interpret the headline result. Add one compact paragraph or sentence after the main numerical result and before future work. A suitable direction is:

> Across seven coarse campaign calculations, capture agreement is uneven (MAE 5.85 percentage points; signed errors +11.45 and -11.93 points), and the temperature comparison is qualitative because the source does not identify the measured phase. The conventional enhancement closure, empirical caloric properties and transport applicability limits constrain predictive interpretation; no optimum or other-amine validation is reported.

Use the journal's typography for minus signs and percentages. If the manuscript keeps the Case 3C energy diagnostic in the conclusion, state it as a bounded sampling result and do not imply that all cases satisfy the same energy criterion.

### 2. Separate the two Case 3C energy grids

The original displayed range is evaluated at 101 exported positions; the new additional refinements also report a 10,001-position dense-grid range. The Batch 3 phrase “reduces ... to 0.134 W” compares unlike sampling grids and implies a same-grid correction. Replace it with wording of this form:

> At 10,001 sampled positions, the second additional Case 3C refinement gives an axial net-enthalpy-flow range of 0.134 W; its capture differs from the paired refined result by less than 0.0001 percentage points. Table 5 retains the original 101-position range, and the supplementary table identifies both sampling grids.

Label the Table 5 row or note so that its 293.219/273.006 W values are visibly the original 101-position values. The new 10,001-position values belong in the supplementary record. Do not claim caloric accuracy or an all-run sub-watt invariant.

The retained additional evidence is bounded to Case 3C: the two attempts use 46 and 121 final nodes, with dense-grid ranges 30.329 W and 0.13384 W, respectively, and capture difference below 0.0001 percentage points for the second attempt. It supports a numerical sampling/interpolation diagnosis for this case only.

### 3. Define the supplementary table's row scope

Batch 2 must specify whether the table is per-run or grouped. The preferred implementation is one row per unique accepted run, de-duplicating shared baselines. The existing retained groups imply approximately 29 rows:

- 7 campaign cases;
- the original paired Case 3C refinement;
- 2 additional Case 3C energy refinements;
- 8 thermodynamic perturbations;
- 6 transport perturbations; and
- 5 accepted operating perturbations.

If the source records require repeated baseline rows, keep them but state the duplication rule. If grouped ranges are used, label the table and caption “grouped ranges” and retain exact per-run identities in the archive. Do not fabricate fields absent from retained records; use `--` with a note explaining why.

Recommended columns are study/run identifier, initial nodes, final nodes, requested tolerance, iteration count, maximum RMS residual, maximum scaled boundary residual, component discrepancy, charge discrepancy, range of signed net enthalpy flow at 101 positions, range of signed net enthalpy flow at 10,001 positions, acceptance/status and source-record reference. The failed lower-liquid/dry-gas operating attempt should have a clear not-accepted status if included in a status record; it must not appear as an accepted numerical row.

Use one consistent cross-reference. Recommended methods wording:

> Accepted-run diagnostics are listed in Table S1; the main numerical-verification table presents the original paired Case 3C refinement.

Use the actual `\cref{tab:supplementary-run-diagnostics}` label in the TeX source, and update Data Availability to point to that same table for detailed diagnostics. Do not alternate between “supplementary run table” and the main table without saying what each contains.

### 4. Qualify the low-loading mechanism

Retained baseline and low-loading profiles support higher free MEA and lower liquid CO2 fugacity at sampled positions. They do not support a claim that the driving force is higher everywhere in the bed: it is higher near the gas inlet and lower at other sampled positions. Replace the proposed broad sentence with a qualified direction:

> Lower lean loading raises free MEA and lowers liquid CO2 fugacity; near the gas inlet this increases the modeled driving force, and the full solved column gives the largest positive capture response among the tested changes.

Keep the response statement tied to the tested solved column. Do not generalize it to all operating regimes.

### 5. Correct abstract precision, timing and future-work strength

- Replace “up to 0.8 and 0.5” with “about 0.8 and 0.5 percentage points” or use 0.82 and 0.53; the rounded maxima are approximately 0.822 and 0.531.
- Identify the timing metric as “about 35 s of BVP wall time.” The retained timing also has distinct CPU and total-workflow values; do not call 35 s total time.
- Replace “provides a basis for predictive film modeling” with “motivates future predictive film-model development,” or equivalent conditional wording.
- Batch 4 says “three follow-on studies” but lists predictive film modeling, ePC-SAFT/eNRTL comparison, other amines and optimization. Say “four follow-on topics” or define three groups explicitly.

### 6. Preserve the author-approved provenance boundary

Retain the handoff's exact author-approved statement that the amine-ion values were fitted by the authors to ion and speciation data in work to be reported separately, together with its statement that those values have not been independently validated. The audit did not independently establish that account from the inspected row-level records; record that audit limitation here rather than silently replacing an explicit author decision. No new provenance investigation is part of this editorial pass. Keep literature citations attached only to the values they support.

### 7. Repair the NCCC footnote anchor

The current table has an unlettered introductory note followed by notes `b` and `a`. Reorder the labeled notes as `a` then `b`, but leave the introductory note unlettered unless a marker is added to a specific header or row. If it is labeled `c`, add the corresponding `\tnote{c}` anchor; do not leave an orphan note marker.

### 8. Make figure verification match the scripts' outputs

The sensitivity renderer writes `comparison.pdf` and `comparison.png`, not SVG. The transport and operating renderers write SVG, PDF and PNG. The execution plan must use output-specific checks:

- sensitivity: PDF text and PNG visual comparison, `summary.csv`, profile CSVs and provenance input fields;
- transport and operating: SVG/PDF/PNG comparison, `summary.csv`, profile CSVs and provenance input fields.

Each provenance JSON hashes its output files, so regenerated figures will necessarily change output-hash fields. Verify that numerical source files and plotted values remain unchanged and that the intended semantic change is limited to legend/note wording. A provenance hash comparison alone is insufficient.

### 9. Use a controlled checkpoint

Replace `git add -A` in Batch 0 with an explicit intended-file list or a reviewable patch. The checkout already contains many unrelated modified and untracked analysis, figure, dependency and report paths. Parent owns the implementation checkpoint and must not stage those paths accidentally. No Git mutation is part of the current audit. When manuscript execution is authorized, preserve the original checkpoint and per-batch commit discipline; this consolidation does not cancel it. No push is authorized here.

## Ordered execution batches

After applying the mandatory corrections above, execute the existing batches in this order. Each batch ends with a cheap source check before moving to the next.

### Batch 1 — availability, Appendix B and framework prose

Update Code Availability, Data Availability, Appendix B provenance/offset wording and the model-framework reference to use manuscript-facing cross-references. Remove the specified internal artifact names and development-status phrases. Preserve the statement that the exact package is not yet public. Confirm the tag sentence is true at the final release state; until then treat it as a pending release condition.

Check that detailed diagnostics point to Table S1 and that the five prohibited words do not enter the source prose/headings/captions/labels.

### Batch 2 — methods and Table S1

Remove internal run-log and runtime identifiers from Methods. Add the concise statement that accepted results satisfy the declared numerical criteria, with the final node counts and diagnostics listed in Table S1. Build Table S1 under the row-scope rule above, preserving source values and distinguishing accepted from not accepted.

Check every row against the retained result identities and group summaries. Use `--` for unavailable fields. Confirm that the main table remains the original paired Case 3C refinement and that Table S1 states its own run coverage.

### Batch 3 — results precision and diagnostic compression

Apply approved rounding to narrative values, replace repeated diagnostics with the Table S1 reference, keep the seven-case numerical-completion/uneven-physical-agreement distinction, and add the corrected Case 3C energy sentence. Label the original 101-position Table 5 values and add new dense-grid values only where retained evidence exists.

Retain the operating figure's transparent not-converged omission. Qualify the low-loading mechanism as specified above. Preserve “individual contributions to error are not resolved” or equivalent where causal attribution is not supported.

### Batch 4 — abstract, introduction, Results 4.4 and conclusion

Reshape the abstract and conclusion around the editorial story above. Apply the corrected capture rounding, BVP wall-time metric, future-work language and “four topics” count. Keep the reviewer-response fitting-count context concise and descriptive. Add the limitations sentence before future work. Do not claim an optimum, completed other-amine results, predictive-film validation or public release.

### Batch 5 — terminology and model caveats

Replace development-status language and clarify that the pressure-drop relation is evaluated to detect hydraulic operating-range conditions. Keep empirical transport-correlation applicability separate from a hydraulic-domain guard. Put the full divisor precision once in Appendix B, use 1.0454 in narrative text, and preserve the initialization-only solubility heading.

### Batch 6 — provenance and table notes

Apply the author-approved provenance wording with the independent-validation limitation. Reorder and anchor NCCC footnotes. Remove vocabulary that implies an undocumented public selection workflow. Keep citations attached to the exact values/claims they support.

### Batch 7 — figure labels and output checks

Change only the three approved script labels/notes. Regenerate the affected figure outputs under the existing input records. Run the format-specific checks above, compare numerical CSV/profile values and inspect the visual/text output. Record expected provenance output-hash changes.

## Final verification gate

Parent should complete these checks after all batches, without new numerical work:

1. Rebuild with `uv run ... sync-figures && bash build_main.sh` from the repository's documented LaTeX workflow.
2. Confirm the PDF is fresh with `uv run python docs/latex/scripts/check_main_pdf_fresh.py` or the repository's equivalent invocation, and verify page count, nonempty pages, resolved cross-references and absence of overfull/clipped layout.
3. Extract page-separated text and scan case-insensitively for whole words `local`, `locally`, `history`, `historical` and `histories`.
4. Run the removed-phrase scan from the batch plan, supplemented by a normalized text scan for variants that differ only by capitalization or TeX escaping.
5. Compare numerical tokens and table rows against retained source records. Explicitly permit only the approved rounded narrative values, new Case 3C dense-grid values, Table S1 additions and wording changes. Check that no source result or figure data changed.
6. Verify that all output-specific figure checks pass, including the sensitivity PDF/PNG route.
7. Run the repository final ePC-SAFT integration check (`scripts/check_epcsaft_integration.py --mode final`) under the documented immutable-wheel conditions.
8. Recheck the SHA-256 of the sealed Stage-1 report and verify it remains `7e82b293ca400c26767b40ef1fca68f3714236aad0aa10df8e8400826ab68ddd`.
9. Perform a final visual read of the complete rebuilt PDF, with special attention to the abstract, conclusion, Table 5, Table S1, Appendix B, footnotes and changed figure legends.

## Acceptance criteria

The revision is ready for external editorial review when:

- the article's abstract, results and conclusion agree on the seven-case scope, uneven capture agreement, qualitative temperature comparison and future-work boundary;
- Case 3C energy values identify their sampling grids and are not presented as an all-run conservation criterion;
- Table S1 has a declared row/status scope and all detailed citations are consistent;
- the low-loading mechanism is qualified to the observed profile and solved response;
- code/data availability distinguishes the current repository route from a future immutable release;
- all prior Fable blockers are resolved in reader-facing prose;
- the current PDF has no prohibited whole-word matches and no internal artifact/development language; and
- the final PDF, figures, numerical source records and immutable integration identity pass the verification gate.

This acceptance decision concerns language, story and traceability. It does not certify independent thermodynamic accuracy or convert the current process evaluation into a new validation study.

## Remaining coverage limit and parent requests

No post-edit PDF exists in this review, so layout fit, cross-reference resolution, final figure serialization and the final output hashes remain execution-time checks. The plan deliberately does not request new analyses or experiments. After the author resumes manuscript execution, parent should implement the bounded corrections, run the final verification gate and treat any missing retained evidence as a wording limitation rather than inventing a source claim.

## Parent consolidation check

The parent verified the sealed Stage-1 hash unchanged and corrected this execution plan against the existing author decisions and Table 5: on-request package access is current; author-supplied provenance remains distinguished from audit verification; Table 5 is the paired Case 3C refinement; and the prior execution commit discipline remains in force. The sealed audit and Stage-2 comparison are unchanged. Manuscript execution remains paused.
