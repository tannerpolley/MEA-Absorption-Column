Review complete. Both PDFs were read page by page (current: 32 pages; pre-revision: 29 pages), along with the reviewer comments, response letter, independent assessment, and the checklist JSON and HTML. No files were edited or created inside the repository. The prohibited-word search was run on the PDF text, the LaTeX sources, the embedded figure PDFs, and the bibliography.

---

## A. Final editorial verdict

**Not yet ready for submission on language and narrative grounds, but close.** The revised paper is a different and much stronger article than the pre-revision benchmark manuscript. It has one scientific subject, a single evaluated model, and a defensible claim boundary. The remaining problems are concentrated, not diffuse.

**Blocking (must fix before submission):**
1. Development-status language in Section 4.4 and the Conclusions ("Implementation has begun on a liquid-film formulation", "the predictive liquid-film formulation already underway"). A journal article does not report the state of the authors' codebase.
2. Internal-artifact references in the main text and appendix ("the parameter document", "the retained calculation records", "the retained campaign Case 1C ... its exact run is identified in the reproduction guide", "runtime selection epcsaft_reactive_nine", file paths in Code Availability). These address the authors, not the reader.
3. Code Availability names no repository URL, tag, or archive for the absorber model, while stating "the repository URL alone does not identify the exact calculations." The pre-revision version cited a tag and a package version. The current statement leaves the reader with nothing to locate. Author verification required.
4. Imperative instructions to an implementer in Appendix B ("The R2/R4/R5 standard-state correlation offsets must not be added again").
5. The convergence-diagnostic template paragraph is repeated five times (pp. 15, 18, 18, 19, 19) with near-identical wording. Consolidate to one statement.

**Important (should fix):** the reviewer-response reasoning embedded in the Introduction on fitting effort (p. 1–2); the numerical over-precision in the abstract, prose, and conclusions; the "selected/adopted/retained/inherited/transferred" vocabulary; the future-work section written as a promissory research plan; the five separate reports of the net-enthalpy-flow diagnostic.

**Prohibited words:** zero occurrences of "local", "history", "historical", or "histories" in the current PDF (prose, headings, tables, captions, embedded figure text, appendices, back matter). Also none in the LaTeX sources or figure files. The only matches in the bibliography database are in abstracts of entries that are not cited in the paper and do not appear in the rendered reference list.

---

## B. Manuscript story

**Central question (as presented):** Can a molecular equation of state, ePC-SAFT, coupled to explicit five-reaction chemical equilibrium, supply the reactive bulk thermodynamics (nine-species speciation and CO2 fugacity driving force) of a rate-based MEA absorber, and how does the resulting coupled prediction compare with pilot data and respond to thermodynamic, transport, and operating inputs?

**Central contribution:** A steady nonisothermal column formulation in which ePC-SAFT activities and fugacities enter the reaction equilibrium and the interphase driving force, while enhancement, transport, hydraulics, and calorics remain conventional and empirical. Evaluation on seven NCCC one-bed cases, a paired mesh/tolerance refinement, one-at-a-time thermodynamic and transport sensitivities, discrete operating perturbations, and measured cost.

**Evidence sequence:** Introduction (motivation, literature table, model-family table) → Model (reactions, EOS, driving forces, transport, balances) → Evaluation methods (cases, numerics, perturbations, timing) → Results (Case 3C profiles, verification, seven-case comparison, sensitivities, operating response) → Interpretation and future work → Conclusions → Appendices A (properties) and B (parameters).

**Final conclusion:** The formulation connects component-level thermodynamic descriptions to capture and temperature; capture MAE is 5.85 points with case-dependent temperature discrepancies; the MEA–water interaction and the liquid-side transfer coefficient are the largest tested sensitivities; lean loading is the consequential operating variable; the work motivates a predictive film model, an eNRTL comparison, other-amine tests, and constrained optimization.

**Where the story loses momentum or consistency:**
- **The "reuse" premise is set up but not tested.** The abstract's first sentence and the Conclusions' "scientific value is a direct connection from reusable molecular and ionic descriptions" promote parameter reuse as the payoff. The Introduction (p. 1–2) then concedes that reuse gives no established fitting-effort advantage, and Appendix B discloses that the two amine ions, the CO2 dispersion energy, the MEA–water kij, and three reaction constants were all fitted or selected for this system. The results never test reuse. The framing is not contradictory, but the reader is told twice that reuse is the point and then given a paper about coupling. Recommend that the abstract and conclusions describe the contribution as the coupled reactive-EOS column and its evaluation, and mention reuse only as motivation for the proposed comparison.
- **Section 4.4 breaks the article's register.** Four paragraphs of future tense ("will examine", "will hold", "will be evaluated", "Implementation has begun") read as a proposal, not a discussion.
- **The results are interrupted by verification boilerplate.** Each results subsection ends with a paragraph of node counts, residuals, and discrepancy bounds. The argument of each subsection (what changed and why) is buried under the same sentence template.

---

## C. Reviewer-feedback integration

**Reviewer 1, comments 1–4 (theoretical basis, physical vs chemical thermodynamics, workflow, terminology).**
Addressed by re-scoping the paper: Section 2.2 opens with the five reactions and nine species, Eqs. 1–3 give the constraint structure, Table 3 assigns quantities to model parts, and Figure 1 shows the calculation flow. The term "fugacity benchmark" no longer exists, so comment 4 is moot. Integration is natural; this is the best-handled theme. Residual issue: the heading "Physical System and Model Responsibilities" (p. 3) and the "supplied quantities" framing of Table 3 carry a faint software-architecture flavor. Compared with the pre-revision manuscript, which explicitly kept "concentration-based chemistry with unit activity coefficients" and used ePC-SAFT only for the fugacity coefficient, this is a genuine scientific change, not a wording change.

**Reviewer 1, comments 5–7 (parameter table, kij, fitted or literature).**
Addressed by Tables 7–10 and the Appendix B provenance paragraphs. The content is complete. The language is where reviewer-response tone survives: "inherited fitted inputs", "transferred inputs", "provisionally calibrated", "source lineage", "retained Held/Uyan lineage", "not an independent holdout". These are the authors' internal provenance categories. The pre-revision appendix had one six-species table and a single kij of −0.052 with no provenance, so the improvement is large, but the vocabulary should be translated into ordinary scholarly terms (see D and E).

**Reviewer 1, comment 8 (thermodynamic sensitivity).**
Addressed with a real study (Section 3.3, Section 4.2, Figure 5). Integrated naturally. Slight over-defensiveness: "Every capture change exceeds the 0.005 10-point reference refinement difference" is a good argument, but the "reference refinement indicator" phrasing recurs four times across 4.2 and 4.3.

**Reviewer 1, comment 9 (transport uncertainty).**
Addressed with a ±10% study and Table 6. Well integrated. The Table 6 fourth column ("these errors do not quantify its accuracy", "concern the full general model, not the adopted explicit approximation") stacks disclaimers; one caption sentence already says this.

**Reviewer 1, comments 10–11 (iterations, Jacobians, residual records, cost).**
Addressed by Table 5 and the timing paragraph. The Table 5 footnote reads as a direct reply to the reviewer: "Memory and residuals at individual inner Newton steps were not measured for these refinement runs; the separate timing study reports memory." Comparison with pre-revision: the old paper's response letter said these could not be recovered; the new paper measured them. The evidence is now real, but the footnote should describe what is reported, not apologize for what is not.

**Reviewer 1, comments 12–13 (other amines, limitations).**
Addressed in 4.4 and the Conclusions. This is where process language is worst ("Implementation has begun", "already underway", "will examine", "will hold"). The limitations themselves are well placed in the first paragraph of 4.4.

**Reviewer 1, comment 14 (model-family comparison).**
Addressed by Table 2 and the Introduction's fourth paragraph. Table 2 is good. The paragraph's second half (p. 1–2: "In the selected MEA parameterization, the neutral-mixture stage refits one symmetric MEA–water coefficient to 25 binary VLE observations; published eNRTL work by Akula et al. adjusts four interaction quantities together with four formation and two heat-capacity quantities... do not establish a matched total-parameter or fitting-effort advantage") is reviewer-response reasoning lifted into the Introduction. It pre-empts an objection before the reader has met the model. Move the parameter-count comparison to 4.4 (where the eNRTL comparison is proposed) or Appendix B.

**Reviewer 2, comment 1 (numbering).** Done; references start at [1].

**Reviewer 2, comment 2 (literature shortcomings).** Addressed by Table 1 and the second Introduction paragraph. Integrated. Two phrases carry audit tone: "'Not reported' means not stated in the inspected article" and "exact code revision not pinned".

**Reviewer 2, comment 3 (parameters for repetition).** Same as R1.5–7 above.

**Reviewer 2, comment 4 (other reported results).** Addressed by the seven-case comparison with temperature profiles (Figures 3–4). Natural. The pre-revision paper had eight "accepted rows" with attempted-case accounting and gate language; all of that is gone, which is a clear improvement in tone.

**Reviewer 2, comment 5 (operating conditions).** Addressed by Section 4.3 and Figure 7. The science is bounded correctly (no optimum claimed). The language for the non-converged condition ("No accepted result", "no retained result", "failed lower-flow attempts") is run-log language.

**Reviewer 2, comment 6 (fitting a new amine).** Addressed in 4.4 as proposed work. Same future-tense problem as R1.12.

---

## D. Required language revisions

| # | Severity | Page / section | Excerpt | Category | Why it weakens the paper | Proposed replacement | Author verification |
|---|---|---|---|---|---|---|---|
| 1 | Blocking | p. 20, §4.4 | "Implementation has begun on a liquid-film formulation designed to calculate transfer..." | Development status | Reports the state of the authors' code, not science. | "A liquid-film formulation that calculates transfer from ePC-SAFT bulk and interfacial states, without an empirical enhancement factor, is the next step. The aim is to obtain the film response from the coupled thermodynamic and transport description rather than from a physical-transfer coefficient multiplied by an enhancement correlation." | No |
| 2 | Blocking | p. 21, §5 | "supports development of the predictive liquid-film formulation already underway" | Development status | Same as above. | "supports a predictive liquid-film formulation, a direct ePC-SAFT/eNRTL comparison, tests with other amines and constrained column optimization." | No |
| 3 | Blocking | p. 5, §2.2 | "R2, R4 and R5 use the selected fitted correlations in the parameter document, which replace the corresponding source coefficients in full" | Internal artifact | "The parameter document" is not a citable object. | "R2, R4 and R5 use the fitted correlations in Appendix B, which replace the corresponding source coefficients in full." | No |
| 4 | Blocking | p. 12, §3.2 | "with temperature scales of 400 K; the retained calculation records give the resulting scales" | Internal artifact | Points the reader to something they cannot open. | "with temperature scales of 400 K." (delete the rest, or "the resulting scales are listed in the supplementary numerical records" if such records are supplied). | Yes (whether records will be supplied) |
| 5 | Blocking | p. 12–13, §3.2 | "Chemical equilibrium normally starts with conservative species amounts generated using reaction-extent fraction 0.001, while the retained campaign Case 1C uses 0.0001 throughout. That exception changes the chemical initial guess, not reaction constants, inlet conditions or tolerances; its exact run is identified in the reproduction guide." | Run-log narration | Three sentences about one run's initial guess, ending with an internal pointer. | "The equilibrium solve is initialized from conservative species amounts generated with a reaction-extent fraction of 0.001 (0.0001 for Case 1C); this affects only the initial guess." | No |
| 6 | Blocking | p. 30, App. B | "The reproduction guide specifies the computational environment, commands and runtime selection epcsaft_reactive_nine." | Software configuration | A code identifier in appendix prose; "runtime selection" is a program option. | Delete from Appendix B. In Code Availability: "The computational environment, commands and model configuration used for every reported calculation are documented with the code." | No |
| 7 | Blocking | p. 30, App. B | "The R2/R4/R5 standard-state correlation offsets must not be added again; the subsequent EOS reference transformation remains required." | Instruction to implementer | Imperative addressed to whoever re-implements the code. | "These correlations already include the standard-state offsets; only the EOS reference transformation is applied afterwards." | No |
| 8 | Blocking | p. 30, App. B | "Thus the printed temperature correlations alone are not the final EOS-coordinate constants; the selected EOS and declared reference are also required, with each conversion applied once." | Instruction to implementer | Same. | "The effective constants therefore depend on the EOS and reference state as well as on these correlations." | No |
| 9 | Blocking | p. 31, Code Availability | "The Python absorber model is maintained in the project repository... (REPRODUCE.md) ... (docs/selected-reactive-parameters.md) ... the repository URL alone does not identify the exact calculations." | Repository description; missing locator | No URL, tag, or archive is given for the absorber model; file paths belong in a README; the last sentence is an internal caveat that tells the reader they cannot reproduce the work. | "The absorber model, parameter files and reproduction instructions are available at [URL], release/tag [X]; the ePC-SAFT package is available at [URL] [38], version [Y]. Full-precision parameters, software versions and the commands for every reported calculation are included. An archival DOI will be minted on acceptance." | Yes (URL, tag, version) |
| 10 | Blocking | p. 31, Data Availability | "the six retained operating conditions with failed lower-flow attempts, and the three timing repeats" | Run-log narration | "Failed attempts" is a log entry. | "the six converged operating conditions, and the three timing repeats." | No |
| 11 | Blocking | pp. 15, 18, 18, 19, 19 | "All eight calculations have 42 final nodes after two mesh iterations, maximum normalized RMS residuals below 0.044 and zero scaled boundary residual, with positive species and component/charge discrepancies below..." (and four near-identical siblings) | Repeated template; catalogue without argument | The same sentence structure closes every results subsection; it reads as machine output. | State once in §3.2: "Every calculation reported below satisfies the stopping, positivity, capture-range and boundary-error criteria; final node counts, residuals and discrepancy bounds for all runs are given in Table S1 [or in Table 5]." Then delete the per-subsection paragraphs, keeping only case-specific facts that matter to the argument (for example, that all six operating conditions converged on 22 nodes). | Yes (whether a summary table will be added) |
| 12 | Important | p. 1–2, §1 | "In the selected MEA parameterization, the neutral-mixture stage refits one symmetric MEA–water coefficient to 25 binary VLE observations; published eNRTL work by Akula et al. [14] adjusts four interaction quantities together with four formation and two heat-capacity quantities. The ePC-SAFT formulation also inherits ion and association parameters and adjusts reaction inputs, so these different calibration tasks do not establish a matched total-parameter or fitting-effort advantage." | Reviewer response in the Introduction; internal term ("neutral-mixture stage") | Pre-empts a reviewer objection before the model is introduced and uses a calibration-workflow term. | Replace with: "Whether this reduces total fitting effort for reactive MEA has not been tested on a common column model; the parameter counts for the present ePC-SAFT set and a published eNRTL set (Appendix B) do not by themselves establish an advantage." Move the counts to 4.4 or Appendix B. | No |
| 13 | Important | p. 1, Abstract | "joint mesh/tolerance changes alter capture by 0.005 10 points and sampled peak liquid temperature by 0.044 93 K" | Over-precision; digit grouping renders as a typo | "0.005 10" reads as two numbers. Five significant figures on a mesh-sensitivity number in an abstract. | "mesh and tolerance refinement changes capture by 0.005 points and peak liquid temperature by 0.04 K". Disable digit grouping for decimal fractions throughout, or keep full precision only in tables. | No |
| 14 | Important | p. 14, §4.1 | "It predicts 91.548 39 % capture against 89.50 % observed, with sampled peak liquid temperature 347.861 55 K." | Over-precision | Seven significant figures next to a 2-point error. | "It predicts 91.55 % capture against 89.50 % observed, with peak liquid temperature 347.86 K (Table 5 gives full precision)." Apply the same rule to 4.2 and 4.3 (e.g., "−0.82 and 0.76 points", "3.75 and −6.86 points"). | No |
| 15 | Important | p. 15, §4.1 | "The fastest run uses 34.23 s process CPU time and 51.45 s total wall time... Its capture is 91.543 29 %." | Redundancy | Capture of the coarse run is already in Table 5 and repeated on p. 15 line 17 and p. 19. | Delete "Its capture is 91.543 29 %." | No |
| 16 | Important | p. 15, Table 5 note | "no invalid-state penalties were recorded. Memory and residuals at individual inner Newton steps were not measured for these refinement runs; the separate timing study reports memory." | Checklist-response language | Explains to a reviewer what was not archived. | "Mesh iterations count outer refinement cycles, not inner Newton steps. Peak memory is reported with the timing study." | No |
| 17 | Important | p. 13, §3.2 | "Invalid reactive states propagate a failure rather than a finite penalty substitution, and failed conditions remain missing rather than receiving zero responses." | Software/diagnostic language | "Propagate a failure", "penalty substitution", "receiving zero responses" are code behaviors. | "A calculation in which the reactive equilibrium cannot be solved is reported as unconverged; no substitute value is assigned." | No |
| 18 | Important | p. 13, §3.3 | "the two flow perturbations and colder-inlet condition permit one additional attempt from a retained reactive Case 3C profile, without changing the equations or tolerances" | Run-log narration | "Permit one additional attempt" describes a job scheduler. | "Where the Henry-law initialization did not converge (the two flow perturbations and the colder inlet), the calculation was restarted from the reactive Case 3C profile with unchanged equations and tolerances." | No |
| 19 | Important | p. 20, Fig. 7 (in-plot and caption) | "No accepted result: lower liquid/dry-gas ratio. No zero response is implied." / "The lower liquid-to-gas condition has no retained result." | Workflow language | "Accepted/retained result" is gate vocabulary from the old paper. | In-plot: "Lower liquid/dry-gas ratio: not converged (omitted)". Caption: "The lower liquid-to-gas condition did not converge and is omitted." | No |
| 20 | Important | p. 19, §4.3 | "Six of the seven specified operating conditions meet the numerical inclusion checks" / "All six retained conditions" / "remain numerical conservation discrepancies without an established cause" | Workflow language; diagnostic admission | Same vocabulary; "without an established cause" belongs once, in limitations. | "Six of the seven operating conditions converged (Figure 7)"; "All six converged conditions"; delete "without an established cause" here and state once in 4.4: "The nonzero numerical variation of net enthalpy flow (Table 5) is a known limitation of the caloric closure whose cause has not been isolated." | Yes (wording of the limitation) |
| 21 | Important | pp. 15, 15, 18, 19, 19 | Net-enthalpy-flow range reported five times ("293.219 W to 273.006 W", "106.49–325.13 W", "272.08–273.15 W", "160.21–390.32 W", plus Table 5) | Repeated qualification | Each report repeats "with no asserted energy-error threshold" or "remain numerical conservation diagnostics". | Report once in 4.1 with Table 5, give the overall range across all runs in one sentence ("106–390 W over all reported calculations"), and remove the per-subsection restatements. | No |
| 22 | Important | pp. 20–21, §4.4 | "Comparative tests ... will examine", "Each test will combine", "A direct ePC-SAFT versus eNRTL comparison ... will hold", "will be evaluated", "will test", "will connect", "joint optimization will evaluate" | Promissory future tense; research plan | Reads as a proposal; commits the authors to work not in the paper. | Condense the last four paragraphs of 4.4 to one paragraph in conditional mood: "The formulation supports three follow-on studies: a predictive liquid-film model...; a matched ePC-SAFT/eNRTL comparison in the same column with fixed transport assumptions, which would separate inherited from newly fitted parameters and quantify calibration effort; and tests with DEA, MDEA, AMP, PZ and blends, which would show which component inputs transfer. The operating responses also motivate constrained optimization of circulation, lean loading and inlet temperature against a process energy or cost objective." | No |
| 23 | Important | p. 21, §5 | "The scientific value is a direct connection from reusable molecular and ionic descriptions to column-scale observables and operating response." | Self-assessment; claim exceeds evidence | The paper does not test reuse; several key parameters were fitted for this system. | "The formulation gives a direct connection from component-level molecular and ionic descriptions to column-scale observables and operating response." | Yes (claim boundary) |
| 24 | Important | p. 1, Abstract, first sentence | "offers a molecular thermodynamic basis for connecting reusable component descriptions to reactive absorber performance" | Framing exceeds evidence | Same as 23. | "provides a molecular thermodynamic description of the reactive liquid that can be coupled directly to absorber performance." | Yes |
| 25 | Important | p. 8, §2.3 | "This empirical correction applies only within the enhancement calculation... Its original calibration source is unavailable; it is retained as a specified model input rather than attributed to the equilibrium formulation." (with s_CO2 = 1.04542981654115) | Diagnostic disclosure; over-precision | A 15-digit constant whose origin is "unavailable" reads as an internal note. This is a genuine evidentiary gap; language can only frame it. | "The divisor s_CO2 = 1.0454 is an empirical constant of the enhancement correlation as used here; its calibration basis is not documented, and it is treated as a fixed model input (full precision in Appendix B)." | Yes |
| 26 | Important | pp. 28–29, Tables 7–10 captions and App. B text | "inherited fitted inputs", "transferred inputs", "inherited from an ion fit", "retained Held/Uyan lineage", "source lineage", "provisionally calibrated", "transfers and provisional fits", "not an independent holdout" | Internal provenance vocabulary | These are the authors' bookkeeping categories, not field terms. | "taken from ref. [x]" / "from a prior fit to ion data [x]" / "taken from the corresponding ion in ref. [x]" / "from Held et al. [33] and Uyan et al. [34]" / "fitted in this work to ... and not independently validated" / "not independent validation data". "Association edges/topology" → "association site pairs/scheme". | Yes (each source attribution) |
| 27 | Important | p. 2, Table 1 | "'Not reported' means not stated in the inspected article." / "exact code revision not pinned" | Audit language | "Inspected" and "pinned" are review-process words. | "'Not reported' means not stated in the cited article." / "exact code version not specified". | No |
| 28 | Important | p. 10, §2.4 | "the continuous implemented equations give...", "For the implemented additive component-enthalpy correlation", "constant in the retained correlation" | Software wording | "Implemented" and "retained" describe code, not equations. | Delete "implemented" and "retained": "the continuous equations give...", "For the additive component-enthalpy correlation", "constant in this correlation". Same for p. 24 ("this implemented correlation"), p. 25 ("the implemented functions"), p. 26 ("the implemented vapor-pressure correlation"), p. 27 ("retained with the property implementation"). | No |
| 29 | Important | p. 6, §2.2 | "the implementation follows the 2025 update of Figiel et al. [18]" / "The auxiliary EOS definitions are supplied by those formulations; the selected model choices and numerical inputs are specified in section B" | Software wording; inconsistent cross-reference | Same; also "section B" vs "Appendix B" used interchangeably (pp. 5, 6, 7, 28). | "the Born term follows Figiel et al. [18]"; standardize every cross-reference to "Appendix A" / "Appendix B". | No |
| 30 | Important | p. 3, §1 last paragraph | "Computational cost is reported as supporting information for coarse Case 3C." | Ambiguous; signposting | "Supporting information" suggests SI. | "Computational cost is measured for the coarse Case 3C calculation." | No |
| 31 | Polish | p. 3, §2.1 heading | "Physical System and Model Responsibilities" | Software-architecture term | "Responsibilities" is a design-pattern word. | "Physical System and Model Structure" | No |
| 32 | Polish | p. 7, §2.3 | "A pressure-drop correlation is evaluated for domain checking but is not coupled to the constant-pressure balance." | Software wording | "Domain checking" is a code guard. | "A pressure-drop correlation is evaluated only to confirm that each state lies within the hydraulic operating range; pressure is held constant in the balances." | No |
| 33 | Polish | p. 7, §2.2 | "The NCCC cases were not used to fit those thermodynamic parameters, although some thermodynamic measurements also used for model selection are not independent evaluation data." | Stacked qualification | Two hedges in one sentence; the second is unclear. | "No thermodynamic parameter was fitted to the NCCC cases. Some of the thermodynamic measurements used in parameter selection are not independent of the parameterization (Appendix B)." | No |
| 34 | Polish | p. 9, §2.3 | "The source comparison ranges and modifications in table 6 delimit the available evidence for these correlations; their relation to calculated states is examined in Results." | Assessment language | "Delimit the available evidence" is reviewer prose. | "Table 6 gives the source ranges of these correlations; Section 4.2 compares them with the calculated states." | No |
| 35 | Polish | p. 12, §3.2 | "finite differences serve derivative verification rather than a competing column solution method" | Defensive | Nobody proposed finite differences as a solution method in this paper (they were in the old one). | "finite differences are used only to verify the derivatives." | No |
| 36 | Polish | p. 13, §3.2 | "Recorded outer mesh iterations, residuals at each iteration, Jacobian evaluations and equilibrium solves describe the numerical work" | Awkward | "Describe the numerical work" | "...are reported as measures of numerical effort" | No |
| 37 | Polish | p. 14, §3.3 | "excludes imports, environment preparation, input loading and final diagnostic serialization" | Software-build language | | "excludes program start-up, input loading and output writing" | No |
| 38 | Polish | p. 18, §4.2 | "only partly separated from the reference refinement indicator" / "smaller than the 0.044 93 K reference indicator" | Jargon; repetition | "Indicator" recurs four times. | "only partly exceed the 0.045 K refinement difference" / "smaller than the 0.045 K refinement difference" | No |
| 39 | Polish | p. 19, §4.2 | "this is not a common limit for every diffusivity formula" | Unclear | | "the other diffusivity correlations have no documented upper temperature limit." | Yes |
| 40 | Polish | p. 19, §4.3 | "This is consistent with the capacity and driving-force dependence of the model." | Vague | | "Lower lean loading raises free MEA and lowers the liquid CO2 fugacity, increasing both capacity and driving force." | Yes |
| 41 | Polish | p. 20, §4.4 | "limit interpretation without establishing their individual shares of those errors" | Awkward | | "limit interpretation; their individual contributions to the error are not resolved." | No |
| 42 | Polish | p. 5, §2.2 | "Before this reference transformation, R4 has ln K4 = a4 + b4/T, while R5 has ln K5 = −ln(10)(a5/T + b5 + c5 T)." | Duplication | The symbolic forms add nothing; Appendix B gives the numbers. | Delete; keep "The effective coefficients, source conventions and calculation domains are given in Appendix B." | No |
| 43 | Polish | p. 5, §2.2 | "Reaction equilibrium is evaluated on a declared activity basis" | Spec language | | "on the molality activity basis" | No |
| 44 | Polish | p. 18, Fig. 5 legend; p. 19, Fig. 6 legend | "Selected-parameter reference" / "Prior mesh/tolerance change" | Internal label; chronology | "Prior" implies sequence. | "Reference (unperturbed)" / "Mesh/tolerance refinement change" | No |
| 45 | Polish | p. 12, Table 4 footnotes | Footnote b printed before footnote a; unlabeled first note | Formatting | | Order a, b; label the first note. | No |
| 46 | Polish | p. 21, §5 second and third paragraphs | Restates the abstract's numbers verbatim (5.85; 91.55 vs 89.50; 0.005 10; 0.044 93; 34.66 s; 0.82; 0.53; 3.75/−6.86) | Repeated conclusions | Abstract, results, 4.4 and conclusions carry the same list. | Keep the MAE and the loading result in the Conclusions; drop the refinement, timing and per-parameter numbers; replace with one interpretive sentence per study. | No |
| 47 | Polish | p. 7 and p. 9 | "numerical evaluation floors the raw value at 10^−12 ε" / "Numerical evaluation uses positive denominator floors and restricts E to 1 ≤ E ≤ 10^4" | Implementation guards | Fine to disclose; scattered. | Collect into one sentence in §3.2: "Holdup, enhancement denominators and E are bounded numerically (h_L ≥ 10^−12 ε, 1 ≤ E ≤ 10^4)." | No |

---

## E. Development and diagnostic language audit

**Prohibited words ("local", "history", "historical", "histories"):** none in the current PDF. Checked in the layout and raw text extractions, hyphenation splits, the LaTeX section and appendix sources, the TikZ flowchart, the seven embedded figure PDFs, and the rendered reference list. For comparison, the pre-revision PDF used "local" at least 14 times ("local temperature", "local chemistry solves", "local phase state", "MEA-local parameter files", "localized profile changes") and "histories" appears only in the reviewer comments, not in either PDF.

**Revision chronology / earlier versions / restoration / fallback / checkpoints / newly added:** none. "Revision" appears once in Table 1 ("exact code revision not pinned", meaning a prior study's code version) and once in the AI declaration ("language revision"). "Newly" appears once (p. 21, "newly adjusted component... quantities", meaning parameters fitted for a new solvent). None of these refer to this manuscript's revision, but "code revision not pinned" should still become "code version not specified".

**Development status / implementation in progress:** two occurrences, both blocking: p. 20 "Implementation has begun on a liquid-film formulation"; p. 21 "the predictive liquid-film formulation already underway".

**Internal artifacts, records, guides, files, identifiers:**
- p. 5 "the parameter document"
- p. 12 "the retained calculation records give the resulting scales"
- p. 13 "its exact run is identified in the reproduction guide"
- p. 28 "the accompanying machine-readable parameter and reaction files retain full precision"
- p. 30 "The accompanying parameter record distinguishes ... with source locators"
- p. 30 "Reproduction uses the retained parameter and reaction JSON files together"
- p. 30 "runtime selection epcsaft_reactive_nine"
- p. 31 "the accompanying numerical records", "REPRODUCE.md", "docs/selected-reactive-parameters.md", "the repository URL alone does not identify the exact calculations", "No public archival identifier has been assigned" (twice)
Recommendation: keep one Code/Data Availability statement with a URL, tag and version; remove artifact references from the body and appendix.

**Software / code wording ("implemented", "implementation", "domain checking", "floors", "propagate a failure", "penalty substitution", "receiving zero responses", "serialization", "pinned", "runtime selection"):** pp. 6, 7, 8, 9, 10 (three), 13, 14, 24, 25, 26, 27, 30. Table 1 "FORTRAN implementation described" and "exact code revision not pinned" are about prior work and acceptable with the "pinned" fix. "CppAD supplies automatic derivatives" (p. 12) is an acceptable tool citation.

**Test outcomes / diagnostic-report language:**
- p. 12 "Scaled directional comparisons with centered finite differences ... satisfy relative and absolute tolerances of 2 × 10^−5 and 2 × 10^−8" (acceptable as derivative verification; keep, shorten)
- p. 13 "one seven-state directional derivative check per multiplier uses the tolerances stated above" (test log; delete or fold into the previous)
- p. 15 Table 5 note "no invalid-state penalties were recorded. Memory and residuals at individual inner Newton steps were not measured"
- p. 15 "no invalid-state penalties occur"
- p. 18 "The unperturbed capture agrees with the thermodynamic reference within 3 × 10^−8 points, and the factor-one control changes it by only 1.01 × 10^−11 points" (replace with "the factor-one control reproduces the reference")
- p. 19 "remain numerical conservation discrepancies without an established cause"
- The five verification template paragraphs listed in D-11.

**Workflow gate vocabulary ("accepted", "retained", "inclusion checks", "attempt", "failed"):** p. 13 ("one additional attempt", "retained reactive Case 3C profile"), p. 15 ("reported numerical inclusion checks", "Only Case 3C has the paired refinement evidence"), p. 19 ("meet the numerical inclusion checks", "All six retained conditions"), p. 20 Figure 7 ("No accepted result", "no retained result"), p. 31 ("six retained operating conditions with failed lower-flow attempts"). Note that "retained" is also used legitimately as a physics word in several places (e.g., "an approximation retained in the column closure", p. 7) and those can stay.

**Provenance bookkeeping vocabulary ("inherited" ×11, "transferred" ×4 in this sense, "provisional(ly)" ×3, "lineage" ×2, "holdout" ×1):** Table 7 caption, Table 8 caption, Table 9 caption, Table 10 caption, p. 29 body (four sentences), p. 30 ("holdout"), p. 7 ("assembled from inherited, fitted, fixed and derived inputs"), p. 2 ("also inherits ion and association parameters"). See D-26.

**Statements about what the authors changed, removed, preserved or selected during development:** the "selected" family (27 occurrences: "selected MEA parameterization", "selected fitted correlations", "common selected temperature range", "selected model choices", "selected parameter set" in two captions, "Selected-parameter reference", "Selected component coefficients", "Selected ionic diameters", "selected topology", "selected relative permittivity", "selected model", "selected formulation", "selected MEA–water interaction coefficient", "selected R2, R4 and R5", "selected EOS", "The selected inputs"). In most cases "selected" can simply be deleted ("the MEA parameterization", "Component coefficients", "Ionic diameters"), or replaced by "adopted" where a choice among alternatives matters. "Adopted" itself appears 19 times and "retained" 15 times; after the deletions above, the residual density will be acceptable.

**Language addressed to the authors rather than the reader:** p. 30 "must not be added again", "remains required", "with each conversion applied once", "are also required".

**Categories with no occurrences:** commits, hashes, wheels, APIs, test suites, debugging, guard events, checkpoints, fallback decisions, restoration, earlier versions of this manuscript, reviewer names or "reviewer" as a word. "Wheel" matches only the author name Wheeldon in reference [25].

---

## F. Section-by-section cohesion check

**Title.** Works: accurate, specific, short; the running head matches. Nothing to change.

**Abstract.** Works: states the model, the evaluation, and the outcome in order. Unnatural: the third to sixth sentences are a number list with five-figure precision and a broken-looking "0.005 10". The first and last sentences promise "reusable component descriptions" that the paper does not test. Smallest fix: round every number to the precision that matters (5.85 points; 91.6 vs 89.5 %; 0.005 points and 0.04 K; about 35 s; up to 0.8 and 0.5 points; +3.8 and −6.9 points) and rewrite the first sentence per D-24.

**Introduction.** Works: paragraphs 1–3 and Tables 1–2 build a clean argument from the MEA problem to the thermodynamic choice. Unnatural: paragraph 4's second half (fitting-effort comparison) is reviewer rebuttal; the last paragraph's "supporting information" is ambiguous; "Model Responsibilities" leaks into §2.1. Smallest fix: D-12, D-30, D-27.

**Model (Section 2).** Works: the reaction set, constraint matrices, EOS terms and driving forces are presented as finished science; Table 3 and Figure 1 make the coupling clear. Unnatural: "the parameter document" (D-3), the symbolic R4/R5 forms (D-42), the s_CO2 disclosure (D-25), "implemented/retained" wording (D-28, D-29), scattered numerical floors (D-47), the "declared" basis (D-43), and "section B" vs "Appendix B". Smallest fix: apply those items; no restructuring needed.

**Evaluation methods (Section 3).** Works: cases, metrics, numerical method and perturbation designs are complete and reproducible. Unnatural: the Case 1C exception and the retained-records pointer (D-4, D-5), the failure-propagation sentence (D-17), the "additional attempt" narration (D-18), and the cost paragraph's software wording (D-37). Smallest fix: apply those; add the one-sentence verification statement from D-11 here so the Results can drop the template paragraphs.

**Results and discussion (Section 4).** Works: 4.1–4.3 each open with a real finding and the figures carry the argument; the point that every sensitivity exceeds the refinement difference but none closes the observed gap is exactly the right argument. Unnatural: the five verification paragraphs (D-11), five enthalpy-flow reports (D-21), over-precise numbers (D-14, D-15), "indicator" jargon (D-38), the operating-response workflow language (D-19, D-20), and 4.4's development status and future-tense plan (D-1, D-22). Smallest fix: consolidate diagnostics, round numbers, rewrite 4.4 as one limitations paragraph plus one future-work paragraph.

**Conclusions.** Works: the first paragraph is a correct one-paragraph summary. Unnatural: paragraphs 2–3 repeat the abstract's numbers; paragraph 4 self-assesses ("The scientific value is") and reports implementation status. Smallest fix: D-2, D-23, D-46.

**Appendix A.** Works: correlations and coefficients are complete; Table 6 is a useful applicability summary. Unnatural: "implemented" wording (D-28), "have not been recovered" phrasing for the vapor-pressure and conductivity sources (p. 25, 27). Smallest fix: replace "have not been recovered" with "are not documented in the sources available to the authors" and keep to one such statement per correlation.

**Appendix B.** Works: the parameter tables are complete and the standard-state conversion is explained. Unnatural: provenance bookkeeping vocabulary (D-26), implementer imperatives (D-7, D-8), the code identifier and JSON references (D-6), and duplication with §2.2 ("replace their source correlations in full", "No coefficient was fitted to the seven NCCC column observations" appears on p. 7 and p. 29). Smallest fix: apply D-6, D-7, D-8, D-26; keep the no-NCCC-fit statement in §2.2 and remove the appendix duplicate, or vice versa.

**Captions and tables.** Works: captions are informative and self-contained. Unnatural: "selected parameter set" (Figs. 2, 5; Table 5), "Selected-parameter reference", "Prior mesh/tolerance change", the Figure 7 in-plot note, Table 5 footnote, Table 1 "inspected/pinned", Table 4 footnote order. Smallest fix: D-16, D-19, D-27, D-44, D-45, and delete "selected" from captions.

**Back matter.** Works: CRediT, funding, competing interests and AI declaration are standard. Unnatural: Data and Code Availability read as internal notes, name no locator for the absorber model, contain file paths and "failed attempts". Smallest fix: D-9, D-10.

---

## G. Direct before-and-after assessment

**What became clearer.**
- The subject. The pre-revision paper was a solver benchmark with an ePC-SAFT fugacity closure bolted onto concentration-based chemistry; the revised paper is a reactive-thermodynamics column model. Reviewer 1's central objection (ePC-SAFT is not a reaction model) is resolved by the science, not by wording.
- The parameterization. Pre-revision Appendix A.3 gave six species and one kij with no provenance and a caveat that "the manuscript and executable benchmark stay aligned". The revised Appendix B gives nine species, all 36 pair coefficients, association pairs, reaction constants, standard states and the conversion procedure.
- The evaluation. "Accepted rows", "attempted-case accounting", "gates", "guard-penalty events" and "108 invalid-state events" are gone. All seven cases are solved and shown with temperature profiles.
- Numerical verification and cost are now measured rather than declared unrecoverable.
- Reference numbering, literature positioning (Table 1), and the model-family comparison (Table 2) are new and effective.

**What was lost.**
- The pre-revision paper had a plain-spoken Introduction voice ("A useful contribution must do more than reproduce a single pilot-plant case") that the revised Introduction replaces with denser, noun-heavy sentences. Some of that directness would help the current paragraph 4.
- A concrete code locator. The old Code Availability gave a repository URL, a submission tag and a package version; the new one gives none for the absorber model.
- Concision in the Results. The old Results were shorter per finding; the new ones are padded by repeated diagnostic paragraphs.

**Does the revised paper overexplain limitations or reviewer concerns?** Yes, in three places: the enthalpy-flow diagnostic (five mentions), the refinement-difference comparison (four mentions of the "indicator"), and the provenance caveats in Appendix B (the same "not independent", "provisional", "not all independently measured" point made in three consecutive sentences). Each concern is legitimate; each needs one clear statement.

**Old claims or framing that survived although the story changed.**
- "Retained/accepted/attempt" gate vocabulary survives in §3.3, §4.3, Figure 7 and Data Availability.
- "Finite differences serve derivative verification rather than a competing column solution method" (p. 12) answers a question only the old paper raised.
- The Henry-law initialization retains an entire Appendix A.1 with a sentence noting it "differs from the reactive bulk ratio in the main model". This is acceptable as an initialization description, but the appendix heading "Physical Carbon Dioxide Solubility" should say "(used for initialization only)" so a reader does not think two solubility models coexist.
- The abstract and conclusions' "reusable component descriptions" framing is a descendant of the old paper's "reusable benchmark" message and is not supported by the new evidence.

**Do abstract, introduction, results and conclusions describe the same contribution and scope?** Largely yes: all four describe the coupled nine-species ePC-SAFT column, seven NCCC cases, Case 3C refinement, sensitivities and operating response. The one drift is the reuse framing (abstract sentence 1, introduction paragraph 4, conclusions paragraph 4) versus the coupling contribution that the Results actually deliver.

**Where reviewer responses are still visible in the tone.** Introduction paragraph 4 (R1.14), Table 5 footnote (R1.10/11), Table 6 fourth column (R1.9), Appendix B provenance paragraphs (R1.5–7), Section 4.4 paragraphs 2–5 (R1.12, R1.13, R2.6), and the Figure 7 note (R2.5).

---

## H. Prioritized revision list

1. Rewrite Section 4.4 paragraphs 2–5 and Conclusions paragraph 4: remove "Implementation has begun" and "already underway", collapse the future-tense plan into one conditional paragraph (D-1, D-2, D-22, D-23).
2. Fix Code and Data Availability: add the repository URL, tag and package version; remove file paths, "failed attempts", and "the repository URL alone does not identify the exact calculations" (D-9, D-10).
3. Remove every internal-artifact pointer from the body and appendix: "the parameter document", "retained calculation records", "its exact run is identified in the reproduction guide", "runtime selection epcsaft_reactive_nine", "JSON files" (D-3, D-4, D-5, D-6).
4. Replace the five verification template paragraphs with one statement in §3.2 and, if needed, one summary table; report the enthalpy-flow diagnostic once (D-11, D-21).
5. Rewrite the implementer imperatives in Appendix B and translate the provenance vocabulary into source attributions (D-7, D-8, D-26).
6. Round numbers in the abstract, prose and conclusions to meaningful precision and fix the "0.005 10" rendering (D-13, D-14, D-15, D-46).
7. Move the fitting-effort comparison out of the Introduction and align the abstract's first sentence with the tested contribution (D-12, D-24).
8. Replace workflow and software wording in §3.2, §3.3, §4.3, Figure 7 and Table 1 (D-17, D-18, D-19, D-20, D-27, D-28, D-29, D-32).
9. Delete "selected" from captions and table titles; standardize "Appendix A/B" cross-references (D-29, D-44).
10. Apply the remaining polish items (D-30 through D-47) in a single pass.

A wholesale rewrite is not needed. Items 1–5 remove the development and response-letter register; items 6–9 restore the concision the Results lost; item 10 is ordinary copyediting.

One note on the cleanup audit required by the working instructions: it reported only pre-existing Python cache directories in the repository, none created by this review, and nothing was removed.
