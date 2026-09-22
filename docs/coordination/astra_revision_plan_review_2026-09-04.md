# Astra review of the language-revision plan

**Date:** 2026-09-04  
**Checkout:** `/home/tnnrpolley21/Workspaces/Engineering/MEA-Absorption-Column`  
**Branch observed:** `codex/fallback-manuscript`  
**Scope:** Comparison of the sealed independent manuscript audit with the prior editorial audit and the updated seven-batch language plan. This is an editorial, narrative and bounded factual sanity review. It does not add experiments, literature acquisition, model runs or manuscript edits.

## Decision

The batch plan is **conditionally ready for execution after the corrections below**. It addresses the prior review's major reader-facing defects: development-status prose, internal artifact references, the incomplete code-availability route, imperative appendix wording, repeated diagnostic paragraphs, excessive narrative precision, and weak future-work framing. The ordered batches are sensible and preserve the author's decision to make a tight article without adding new experiments.

The plan still has several concrete inconsistencies that could leave the final article misleading or internally incomplete. The most important are the missing limitations sentence in the proposed conclusion, an apples-to-oranges comparison in the new Case 3C energy sentence, an underspecified supplementary run table, an overbroad causal sentence for the low-loading result, and an output-format check that cannot run against one of the planned figure outputs. These are plan corrections, not requests for a new scientific program.

The sealed Stage-1 report remains unchanged. Its verdict was more cautious than the prior editorial review: the manuscript is readable and traceable, but the evidence does not justify a strong predictive-reactive-thermodynamics claim. The language plan can make that scope legible; it cannot turn a process evaluation into independent thermodynamic validation.

## Sealed Stage-1 identity and comparison inputs

The independent report was sealed before this review at SHA-256:

`7e82b293ca400c26767b40ef1fca68f3714236aad0aa10df8e8400826ab68ddd`

The hash was rechecked before Stage 2 and matched the recorded value in `/tmp/astra_manuscript_audit/stage1_report.sha256`. The comparison inputs were the following on-disk files and hashes:

| Input | SHA-256 |
|---|---|
| `docs/coordination/astra_independent_manuscript_audit_2026-09-04.md` | `7e82b293ca400c26767b40ef1fca68f3714236aad0aa10df8e8400826ab68ddd` |
| `docs/claude_manuscript_language_audit_2026-09-04.md` | `894b972d4700871ee97f467dcb50e67ec4bc2d937606e2ad552ac649b84e6e33` |
| `docs/coordination/language_revision_batches_2026-09-04.md` | `274ba17804e822c295a6f4f51835a6f4b359141854bb0def1df6e92f59bc2184` |
| `docs/coordination/language_revision_handoff_astra_2026-09-04.md` | `4da222715569cb1dcdee10ea4e77a0d2757b5590f975e14d5aea16533c3d57d2` |

The plan and handoff were treated as the final on-disk versions supplied for this comparison. The Stage-1 report was not edited after sealing. The manuscript PDF and manuscript source were not edited during this review.

## What the two reviews agree on

The prior Claude/Fable review and the independent report converge on the following points.

1. The current PDF is substantially cleaner and more readable than the pre-revision comparison. The model-responsibility table, flowchart, equations, parameter tables and figure captions make the nine-species reactive calculation understandable. The current rendered PDF has no whole-word occurrence of `local`, `locally`, `history`, `historical` or `histories`; the older comparison PDF still has eleven occurrences of `local`.
2. The article needs a strict distinction between a working model formulation, numerical convergence and physical validation. Seven accepted BVP solutions demonstrate numerical completion under the declared tests; they do not independently validate the reactive ePC-SAFT parameterization.
3. The article should retain the seven-case evaluation, the uneven capture errors, the single paired Case 3C refinement and the qualitative temperature comparison, while saying exactly what each supports.
4. The repeated energy, timing and convergence boilerplate obscures the story. One compact result plus a supplementary diagnostic table is clearer than repeating full checks in several results paragraphs.
5. Future predictive film modeling, other-amine extension, optimization and alternative thermodynamic closures must remain conditional proposals. A current result must not be written as an implementation or as an already available capability.
6. Reproducibility language must identify what is available now and what still depends on a future public release. The repository/tag wording in Batch 1 is an improvement, but it remains a release precondition rather than evidence that an immutable public package already exists.

## Differences from the prior review

The prior review called the manuscript close after language cleanup. The independent Stage-1 review assigns a major-revision risk because it weighs the evidence boundary explicitly. In particular, Stage 1 identified:

- no independent VLE, speciation or reaction-thermochemistry evaluation set for the selected reactive parameters;
- seven coarse capture cases with MAE 5.85 percentage points and signed extremes of +11.45 and -11.93 points, with only Case 3C paired refinement;
- a nonzero axial net-enthalpy-flow range in the original displayed results, including 273.006 W for the retained refined reference, without a general acceptance criterion; and
- an exact public calculation package not yet identified by an immutable archive.

These are not contradictions of the Fable language findings. They are evidence-scope findings that the language batches must expose rather than obscure. The revised plan's proposed limitation and future-work sentences help, but the conclusion as currently drafted still omits too much of this context.

One prior concern is partly superseded by new evidence. The retained Case 3C refinement records now support a bounded numerical-interpolation diagnosis for that case: the additional refinement reports 0.134 W over 10,001 sampled positions and capture differs from the paired refined result by less than 0.0001 percentage points. This does not establish caloric accuracy, an energy criterion for all runs, or a sub-watt invariant on the seven-case, sensitivity or operating studies. It should be described as an additional Case 3C numerical diagnostic, with its sampling grid stated.

## Batch-by-batch assessment

### Batch 0 — checkpoint and ownership

**Disposition: revise the execution instruction.** The proposed `git add -A` is broader than this task's intended ownership and can stage the many existing modified and untracked analysis, figure, report and environment files in the checkout. Parent-controlled implementation should create a deliberate checkpoint using an explicit file list or a reviewable patch. The plan may retain a checkpoint step, but it should say that only the intended manuscript, figure-script, figure-output and report files are staged. No Git mutation was made during this review.

### Batch 1 — availability, Appendix B and model-framework wording

**Disposition: conditionally ready.** The proposed replacements remove internal names such as `parameter document`, `reproduction guide`, JSON/runtime identifiers and the imperative offset instruction. They also correctly distinguish a repository tag from a future public package. Two cross-reference and release details need tightening:

- Data Availability currently points readers to `tab:reactive-numerical-verification` for timings and diagnostics, while Batch 2 creates the detailed `tab:supplementary-run-diagnostics`. Use the supplementary table for the detailed per-run record, or state clearly which values remain summarized in the main table and which are in the supplement.
- “Available in the project repository at tag `nce-revision-v3`” is accurate only after that tag and its matching files exist. Until the tag is created, the sentence is a release dependency. The final verification must confirm that the tag resolves to the exact source/figure state described by the PDF and that the package-availability sentence does not imply a public archive that has not been made.

The proposed model-framework wording is a good reader-facing replacement. It should continue to separate hydraulic-domain checks from empirical transport-correlation validity; a domain guard does not prove that a correlation is valid across the full state range.

### Batch 2 — supplementary diagnostics

**Disposition: revise before execution.** Moving detailed convergence information to one table is the right editorial decision, but the scope is currently ambiguous. The plan says that every calculation reported in Results satisfies the criteria, then proposes original group ranges and two new refinements. Group ranges alone cannot let a reader trace every accepted run or distinguish a missing operating condition from a reported accepted result.

Use one row per unique accepted run, with a stable study/run identifier and a source record. The existing groups imply approximately 29 unique accepted rows if the shared baseline is de-duplicated: seven campaign cases, the original paired Case 3C refinement, two new energy refinements, eight thermodynamic perturbations, six transport perturbations and five accepted operating perturbations. If the implementation chooses repeated group rows instead, say “grouped ranges” in the caption and preserve exact per-run identities in the archive. Do not invent values for fields that a retained record does not contain; use `--` and identify why.

The table should distinguish the 101-position export grid from the 10,001-position diagnostic grid. Suggested columns are study/run, initial and final node counts, requested tolerance, iterations, maximum RMS residual, maximum scaled boundary residual, component and charge discrepancies, signed energy range at 101 positions, signed energy range at 10,001 positions, and status/notes. The 10,001-position cells should be populated only for the two new refinement records. The failed lower-liquid/dry-gas operating attempt should be identified as not accepted rather than silently represented as a converged row.

The methods sentence is inconsistent across the plan: one instruction cites the main numerical-verification table and the supplementary table, while another says to cite the supplementary table alone. Choose one exact citation and use it consistently. A compact choice is: “Accepted-run diagnostics are listed in `tab:supplementary-run-diagnostics`; the main table summarizes the campaign-level ranges.”

### Batch 3 — result precision, energy, operating and repeated diagnostics

**Disposition: revise the energy and operating wording, then execute.** Deleting repeated diagnostic paragraphs and rounding narrative values is appropriate. The following corrections are needed.

- The proposed Case 3C sentence compares the original 273.006 W range evaluated on 101 positions with the new 0.134 W range evaluated on 10,001 positions. “Reduces ... to 0.134 W” implies a same-grid correction and is misleading. Say instead: “At 10,001 sampled positions, the second additional Case 3C refinement gives an axial net-enthalpy-flow range of 0.134 W; its capture differs from the paired refined result by less than 0.0001 percentage points. Table 5 retains the original 101-position range, and the supplementary table identifies both grids.”
- Table 5 still contains the original 293.219/273.006 W values, and the methods text identifies 101 exported positions. The table note and row label must explicitly mark those as the original 101-position calculation. Do not silently replace them with the new dense-grid values.
- “All seven calculations converge” should remain tied to the numerical criteria, followed immediately by “physical agreement is uneven.” Only Case 3C has paired refinement. This is a useful story sentence if it does not imply all-case physical validation.
- “Not converged (omitted): lower liquid/dry-gas ratio” is truthful for the figure, but retain the failed attempt in the supplementary/archive record and say that no zero response is implied. This avoids converting omission into a physical result.
- The proposed low-loading mechanism is too broad for a full-bed statement. Retained baseline/low-loading profiles show higher free MEA and lower liquid CO2 fugacity at sampled positions, but the driving force is higher near the gas inlet and lower at other sampled positions. Use a qualified sentence such as: “Lower lean loading raises free MEA and lowers liquid CO2 fugacity; near the gas inlet this increases the modeled driving force, and the full solved column gives the largest positive capture response among the tested changes.” Keep the magnitude as a response of this solved case, not a universal mechanism.
- The proposed sentence that individual error contributions are unresolved is a good replacement for causal speculation. Keep it adjacent to the uneven seven-case result.

### Batch 4 — abstract, introduction, Results 4.4 and conclusion

**Disposition: revise before execution.** This batch has the greatest effect on the article's story. The new structure is sound, but several phrases need exact correction.

- “Affect capture by up to 0.8 and 0.5” underbounds the retained maxima (approximately 0.82 and 0.53 percentage points). Use “about 0.8 and 0.5 percentage points” or retain two decimal places.
- “One coarse solution takes about 35 s” must identify the metric. The retained timing separates roughly 34.66 s BVP wall time, 34.23 s CPU time and 51.45 s total workflow time. Use “about 35 s of BVP wall time” wherever the abstract or conclusion cites it.
- “Provides a basis for predictive film modeling” is still stronger than the evidence boundary. Use “motivates future predictive film-model development” or state that predictive film validation is outside the present study.
- Results 4.4 says the formulation supports “three” follow-on studies but lists four topics: predictive film modeling, ePC-SAFT/eNRTL comparison, other amines and optimization. Say “four” or combine them into three explicitly defined groups.
- The conclusion must add a compact limitations sentence. A suitable direction, to be adapted to the journal's style, is: “Across seven coarse campaign calculations, capture agreement is uneven (MAE 5.85 percentage points; signed errors +11.45 and -11.93 points), and the temperature comparison is qualitative because the source does not identify the measured phase. The conventional enhancement closure, empirical caloric properties and transport applicability limits constrain predictive interpretation; no optimum or other-amine validation is reported.” This sentence uses only retained evidence and makes the conclusion match the abstract/results scope. The nonzero energy diagnostic can be stated as the bounded Case 3C sampling result if the table sentence already gives it; avoid implying that it is solved for every case.
- Future work should remain conditional and should not say that a capability “has begun,” “is implemented,” or is “ready.”

The conclusion's proposed sentence that neither the reusable component direction nor the enhancement change closes the observed capture gap is supported by the retained comparison. It should follow the limitations sentence so that the reader does not mistake a future component comparison for a completed result.

### Batch 5 — terminology and model caveats

**Disposition: ready with one wording guard.** Removing `implemented`, `implementation follows`, `domain checking` and similar development language improves the article. The new hydraulic sentence should retain the distinction between evaluating a pressure-drop relation to detect an out-of-range hydraulic state and claiming that the empirical correlation is physically valid there. The proposed statement that the calibration basis is not documented is appropriately cautious if it is tied to the cited correlation record.

The divisor text should appear once in the appendix with full precision, while narrative text uses the rounded value. The Appendix A heading “Physical Carbon Dioxide Solubility (Initialization Only)” correctly prevents an initialization device from being read as the reactive equilibrium closure.

### Batch 6 — provenance and table footnotes

**Disposition: revise the provenance sentence and footnote anchor.** The new table vocabulary is much better than “inherited,” “retained” or “adopted” when those words suggest a public or historical workflow. The exact sentence proposed for MEAH+ and MEACOO- values (“fitted by authors to ion and speciation data in work to be reported separately”) requires an evidence locator. The retained parameter bundle identifies a Phase-2 artifact, candidate/extrapolation qualifications, reaction/speciation selection records and a separate thermochemistry artifact, but the inspected row-level provenance does not directly state that those six segment/Born values were fitted to ion and speciation data. Before publication, either cite the exact retained record that supports that sentence or weaken it to a supported formulation such as “fit-derived inputs from the retained parameter bundle; independent validation is not reported here.” Do not add a new source claim solely to make the prose sound complete.

The NCCC table note order must be explicit. The current table has an unlabeled introductory note followed by notes `b` and `a`. Reordering `a` before `b` is correct, but labeling the introductory note without adding a matching superscript creates an orphan marker. Keep that note unlettered and order notes `a` then `b`, or assign it a new marker and attach that marker to a header/row. The plan must name the anchor.

### Batch 7 — figure labels and verification

**Disposition: revise the verification command.** The label replacements are precise and useful: “Reference (unperturbed),” “Mesh/tolerance refinement change” and “Not converged (omitted)” accurately describe the plotted categories. The generic output check is not executable for all three scripts. The sensitivity renderer writes `comparison.pdf` and `comparison.png`; it does not write an SVG. Transport and operating renderers write SVG, PDF and PNG outputs. Use output-format-specific checks:

- sensitivity: compare the PDF text and PNG rendering, plus `summary.csv`, profile CSVs and provenance fields that record inputs; allow expected output-hash changes after the legend edit;
- transport and operating: compare SVG/PDF/PNG outputs, summaries, profiles and provenance; inspect that the only plotted semantic change is the label/omission note.

Because each provenance file hashes its output files, a regenerated figure necessarily changes the output-hash fields. The verification should therefore compare numerical source CSVs/profiles and visual/text output content, while treating the expected output hash fields as changed. A provenance hash alone cannot prove that only a legend changed.

## Explicit execution and evidence requests for the parent

These are bounded plan corrections and verification requests; they do not request new model runs or literature research.

1. Add a conclusion limitation sentence containing the uneven seven-case agreement and qualitative temperature status, and preserve the constrained predictive scope for the conventional enhancement, empirical caloric and transport closures.
2. Rewrite the Case 3C energy statement so that 101-position original values and 10,001-position additional values are named separately. Preserve both records in the supplementary table.
3. Define the supplementary table's row scope and status fields. Prefer one row per unique accepted run with source identifiers; if grouped ranges are retained, label them as grouped and retain exact records in the archive.
4. Make the methods/Data Availability cross-reference to the supplementary table internally consistent.
5. Qualify the low-loading driving-force sentence to the sampled profile and solved-column response; do not state that driving force increases throughout the bed.
6. Correct the abstract's “up to” rounding, identify the timing as BVP wall time, change “basis for predictive film modeling” to a future motivation, and change “three” to “four” follow-on topics or group them explicitly.
7. Verify or weaken the proposed amine-ion provenance sentence and anchor the NCCC table's footnotes without an orphan marker.
8. Replace the generic Batch 7 SVG diff with format-specific figure checks and source-data comparisons. Include a full post-build PDF text scan for the five prohibited whole words, because the current PDF's zero-match result must survive the edits.
9. Use a controlled checkpoint file list rather than `git add -A`; parent retains ownership of all existing dirty files and does not commit this review's reports as part of manuscript execution unless separately intended.
10. After implementation, rebuild the PDF, verify the page count and visual layout, run the repository's freshness and final ePC-SAFT integration checks, and compare retained numerical tokens/rows against the pre-edit source records. These are verification actions only; they do not require a new analysis.

## Remaining coverage limit

This review did not rerender the manuscript or rerun the model after the proposed batches because no manuscript implementation was authorized in this task. The conclusions above apply to the on-disk current PDF and the plan text/hashes listed above. Whether the corrected prose fits the journal layout, whether all cross-references resolve and whether regenerated figures preserve their plotted values remain execution-time checks for the parent.

## Final plan verdict

Execute the batches after applying the nine concrete corrections above. With those corrections, the plan will produce a tighter article whose claims are aligned with the retained evidence: a readable coupled reactive-column formulation and bounded process evaluation, with uneven physical agreement, qualitative temperature comparison and future validation/extensions clearly identified. It should not present the language cleanup as independent thermodynamic validation or as a completed public reproducibility release.
