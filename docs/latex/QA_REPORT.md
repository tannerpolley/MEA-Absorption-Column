# Fallback manuscript review

Reviewed 2026-09-03 in the isolated `codex/fallback-manuscript` worktree.

## Decision

### Seven-case reactive comparison and measured runtime — 2026-09-04

Integrated the investigator-supplied `analyses/nccc_validation/figures/reactive_parallel/README.md` snapshot (SHA-256 `95c7b82e676bc2374b2ae9704237f6d8958e7f24535800141dc37add36e1e568`). The three supplied PDFs replace the historical temperature views and add seven-case capture comparison. Abstract, Introduction, model scope, Methods, Results, Conclusion, data availability and reviewer responses now agree with this population. Historical six-species aggregate and method results remain distinct; refined sensitivity values are unchanged.

Copied and byte-verified 250 evidence files, including seven selected runs, source observations, provenance, the three runtime repeats and their reference/confirmation. Every figure-provenance input hash matches after mapping the source checkout prefix to this checkout. All absorber source hashes in the fastest-run identity match this checkout. Direct CSV checks give capture MAE 5.8548258457 points, runtime median 34.6569113731 s and maximum scaled profile difference 6.3884081358e-10. All seven source-change checks are clear. The 6C/7C errors and empirical energy ranges remain disclosed.

The optimized runtime result supersedes the old 309/517-second values as the manuscript's cost example; original refinement timings and earlier failed attempts below remain historical evidence, not current blockers. The fastest BVP repeat takes 34.4052 s wall, 34.2316 s CPU, 51.4535 s total and 203.5117 MiB peak RSS. Its exact wheel and separate failing temperature-boundary check are documented in REPRODUCE and the retained notebook; no all-Engine-tests-pass claim is made. The default environment pin is unchanged. No column was rerun for this integration.

The 40-page PDF builds successfully and passes the PDF-freshness check. Figures 3–5 show current temperature/capture comparisons; Table 6 retains refinement diagnostics without obsolete runtime rows; page 28 reports scoped timing repeats. Inspected the complete contact sheet and the new figures, numerical table, timing text and flagged equation pages. Color and grayscale renders are retained in `builds/seven-case-visual-qa/`. The conservative automated layout report returns 189 clearance warnings and two apparent math overlaps on pages 8 and 13; visual inspection identifies the latter as mathematical superscript/overbar extraction, not overlapping prose. The existing CAS front-matter box warning remains. No new clipping or unreadable figure labels were found.

The live checklist and its executable test report 19 of 20 reviewer points complete with zero changed-evidence warnings; R2.4 is addressed and R2.5 remains open. The separate operating-response package was not integrated in this seven-case task. Manuscript titles/headings, AI-use disclosure and bibliography remain unchanged. Banned-language and obsolete-runtime searches found no new vague terminology or old timing values presented as current in the active manuscript. No commit, push, mirror sync, source acquisition or parameter changes were performed.

### Full manuscript-to-calculation reconciliation — 2026-09-04

The investigator authorized correction of every finding in the manuscript audit. All identified reporting defects are addressed without rerunning a column or changing numerical inputs/results. The main title, short title, section/subsection headings, AI-use disclosure and bibliography are unchanged. No commit, push, mirror update, source acquisition or parameter fitting was performed.

| Finding | Implemented correction and retained evidence |
| --- | --- |
| Old reactions leading §2.1.1 | Five reactions and nine species now lead the subsection; the complete reduced six-species derivation is grouped with historical parameter tables in the appendix. |
| Fugacity-only wording applied to coupled chemistry | Abstract, Introduction, workflow, model descriptions, Results and Conclusion distinguish the coupled reactive calculation from the fixed-chemistry historical comparison. |
| Missing conservation and standard-state definitions | Explicit carbon/nitrogen matrix, mass conservation, charge, apparent initial amounts, molality activities and EOS concentration conversion match `reactive_bundle.py`. |
| Ambiguous reaction-constant precedence | The appendix prints effective R2/R4/R5 equations and states that typed `parameters.json` records supersede corresponding `reaction-system.json` correlations; conversion offsets and 293.15–393.15 K domain are explicit. |
| Interfacial-area mismatch | The implemented void fraction inside `(u_l epsilon/a_p)^(4/3)` and 1.42/0.12 coefficients are stated; source-versus-adopted differences remain in the applicability table. |
| Missing holdup transformation | Raw correlation coefficients, positive floor and bounded `epsilon*h_raw/(epsilon+h_raw)` transformation are explicit. |
| Undefined hydraulic/property bases | Hydraulic diameter, phase-average molar masses, mass densities and apparent mass fractions are defined. |
| Enthalpy-only state and boundary vectors | Current temperature-state vector, derivatives and mixed boundary conditions are shown; enthalpy coordinates are identified as the historical alternative. |
| Material-offset/energy-chain-rule mismatch | All four actual material derivatives include the retained numerical offset. Its cancellation in the signed countercurrent balance and its terms in the temperature chain rule are explicit. |
| Pressure and trial-handling overclaims | Pressure-drop calculation is diagnostic, not a coupled pressure balance; penalty handling is limited to historical calculations and current failures propagate. |
| Missing vapor-property closure | Vapor enthalpy/heat capacity and viscosity-weighted thermal-conductivity mixing equations and coefficients match the implementation. |
| Missing current profiles and mixed result populations | Retained Case 3C profile PDF is included. Its 91.54839% capture and 347.86155 K peak temperature are separate from historical 88.32%, aggregate statistics and method timings. Only the capture cross is an observation in the current figure. |
| Old parameter values presented without scope | Historical −0.052/169.21 K tables are explicitly separated from selected −0.07352749875/173.44025 K inputs. No full new parameter inventory is added. |
| Overstated calibration independence | CO2 dispersion selection jointly with R4 and Xu's earlier selection use are disclosed; exclusion from the final objective is not described as an untouched global holdout. |
| Dry/wet inlet and gallery ambiguity | Case 3C dry 0.093 versus reconstructed wet 0.085394 is identified; historical gallery is Cases 1C–6C, not 1C–7C. Generator wording preserves this distinction. |
| Misleading August-release availability | Availability sections identify the August revision as historical only. Current worktree results are not attributed to that public revision; public archiving remains a separate authorized action. |
| Obsolete tables and stale notebook/source mappings | Two unreferenced TeX tables were removed and the attempted-table writer was removed from the existing generator. Unique CSVs remain. Current profile and sensitivity copies have explicit source mappings in `latex_workflows.py`; `code_to_paper_traceability.md`, reviewer response and HTML are synchronized. Hash freshness is explicitly not scientific validation. |

Independent read-only thermodynamic and transport/energy reviewers accepted the corrections after checking the current implementation. A separate results reviewer verified current profiles and the historical six-case gallery against retained source files. The final 38-page PDF builds and passes freshness; all 39 citation keys resolve. All pages were inspected at full size in both color and grayscale: root reviewed pages 1–12, the results reviewer 13–26 and the thermodynamic reviewer 27–38. The last wording-only build changed only pages 6–7, which root re-inspected; all other rendered pages are byte-identical to those independently reviewed.

Visual QA returned 176 conservative findings, including three reported overlaps on pages 8, 12 and 13. Manual inspection identifies those as mathematical overbars, sum/derivative glyphs and vector subscripts, not collisions; tight-clearance warnings likewise do not obscure content. The pre-existing CAS front-matter overfull-box warning (117.0831 pt at `main.tex:100`) remains; the title/abstract render without clipping. Newly introduced long-expression overflows were corrected. This is a reviewed visual disposition, not an automated zero-warning claim. Scientific equations/retained plots: exact to the implemented/reported calculation; workflow: schematic; disposition: keep.

Validation: five LaTeX-workflow checks pass, including byte-for-byte copying of every mapped figure; the edited analysis generator parses and no longer emits the obsolete table. The reviewer notebook check and live endpoint report 18/20 complete and zero stale assessments. R2.4 multi-case integration and R2.5 controlled operating trends remain open. Selected parameter JSON, bibliography and both sensitivity-summary SHA-256 values are unchanged. Manuscript prose passes the plugin-wide check; manual banned-term inspection retains only concrete thermodynamic-model choices and statistical fitting uses. No vague banned term or development-note marker remains in active manuscript prose.

Final immutable-wheel integration also passes: Engine `9e1bef97fbea5c6f465612ae27b054192f91f19c`, wheel SHA-256 `b011d0f9d492e9db197f67cc0ae6781ac636fa3278805ddf1d6a05ecd167074b`. This smoke check is not a new column run. The repository cleanup audit found only Python/pytest cache candidates; pre-existing shared caches were not removed.

delete: `tables/full_ionic_speciation_timing.tex` and `tables/nccc_one_bed_attempted_status.tex` were unreferenced presentation duplicates. Their numerical records remain in `analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv` and `nccc_one_bed_all_attempted_results.csv`; tracked table history is recoverable through Git.

keep: current parameters and numerical evidence remain with their existing source/analysis owners; no second scientific-data owner or new audit machinery was created. The writing/build skills guided equation-to-code reconciliation and the scoped CSE audit removed obsolete presentation copies while preserving unique values. Change accounting against the August base includes the full earlier working overlay, not just this correction: 31 executable files +2451/−221 and six test files +362/−2 at this pass.

### Literature, model definitions and transport applicability — 2026-09-04

The investigator requested the coordinated R1.14/R2.2/R1.9/R1.4 revision and then narrowed the parameter presentation to a short cited count comparison, excluding the reduced local eNRTL fit. The new full parameter-inventory table was removed before delivery; the established selected-parameter appendix is retained. Titles, section headings, historical numerical labels, AI-use disclosure and model inputs are preserved. No column, thermodynamic sensitivity, operating-condition, full-film, CasADi, eNRTL or SRP calculation was launched.

The Introduction compares five published absorber studies and expands the existing model-family table. Its short count paragraph uses Akula 2023 Table 6 (four interaction, four formation and two heat-capacity quantities; inherited Hessen W2 additional) and Zhang 2011 Tables 8/10 (four neutral interactions; six electrolyte interactions and six ion formation/heat-capacity quantities). The model table defines each comparison's changing closure/chemistry, parameters, fixed physical formulas, caloric treatment and supported conclusion. Future Work coordinates complete published eNRTL verification, matched conventional-film column comparison and controlled calibration/held-out transferability evaluation. These are proposals, not launched campaigns.

Read-only source reviewers checked the published studies, parameter counts and transport applicability. Published-source corrections were applied: Akula's VLE/heat-of-absorption fitting is distinct from its speciation/heat-capacity comparisons; Zhang's electrolyte fit has four intercepts and two temperature slopes, not eight nonzero terms. Main-source locators: Akula 2023 pp. 7–11, Tables 4/6, Secs. 3.3/4.1 and Appendix B3–B4; Zhang 2011 pp. 68–73, Tables 1/6/8/10, with excluded-source VLE in Table 11. The selected ePC-SAFT JSON still matches the upstream selected file by SHA-256. Its reaction screen reports two retained SVD fitting directions; six changed coefficient fields do not mean six independent fitting coordinates. Full numerical fit-history claims were not imported into the paper.

Absorber-table locators: Tobiesen 2007 pp. 848–851, 857–858 and Appendices B/C/E; Zhang 2009 Tables 3–6 and 17–19; Chinen 2018 pp. 10454–10457, Tables 4–5 and Fig. 11; Akula 2021 pp. 5180, 5189–5190, Figs. 6–8/Table 5; Shahid 2019 pp. 12236–12244, Tables 1–4 and Figs. 4–9. Missing solver or executable reporting is not labeled as work not performed.

Transport-table locators: Morgan 2015 property article pp. 1826–1827/Table 1; Snijder 1993 p. 478/Eq. 8; Luo 2015 PDF pp. 61–62/Eqs. 21/32–33; Tsai 2010 dissertation pp. viii–ix; Chinen 2018 pp. 10454–10457; Gaspar 2015 absorption comparisons. The printed area exponent changes from 0.116 to the implemented 0.12, with the source's 1.34/0.116 distinguished from adopted 1.42/0.12. The code and numerical results are unchanged. Source-model fit errors are not presented as uncertainty bounds for modified correlations. Remaining source questions concern the modified viscosity vector, CO2 concentration correction, ion-diffusion expression and adopted transport calibration; no numerical validity conclusion is inferred solely from missing provenance.

Zotero and Better BibTeX became available after the investigator opened the app; source identity reads succeeded. All citations use existing records, with no library mutation, bibliography refresh or Overleaf synchronization. The frozen-draft build uses the retained bibliography; its existing local attachment metadata is not newly added or certified as a publication-ready sanitized export.

The final 34-page PDF builds and passes freshness. Rendered pages 2, 3, 6 and 31 were inspected: the literature, model-family, comparison-definition and transport tables and the short parameter-count paragraph fit without clipping. No undefined references or oversized floats are reported; the existing front-matter overfull-box warning remains. Prose checking and manual banned-language inspection pass with concrete statistical, interface and numerical-reporting uses retained. Both sensitivity summaries retain their previous hashes. The manuscript title, section headings, AI disclosure and retained bibliography are unchanged by this revision.

keep: the manuscript summarizes source counts and calibration ranges without duplicating a complete parameter inventory; adopted values remain in the selected parameter set and its supporting record. [sections/introduction.tex; tables/transport_applicability.tex; docs/selected-reactive-parameters.md]

net: the task-created parameter-inventory table was removed at the investigator's request; its relevant count comparison remains in the Introduction and no unique numerical inputs were deleted. No further deletion or ownership correction is proposed by this scoped CSE audit.

### Thermodynamic and transport sensitivity integration — 2026-09-04

The investigator requested complete manuscript coverage of the thermodynamic study followed by the supplied transport study. Abstract, Introduction, Methods, Results Figures 5–6, Conclusion and Future Work now distinguish eight ±5% thermodynamic perturbations from six ±10% transport perturbations. R1.8 and R1.9 are complete for these bounded Case 3C comparisons. The thermodynamic summary is unchanged (SHA-256 `af75939bc24179ae49745fd6202ad593c049f38b4c09e2bbf28f22d233c1db7e`). Transport capture changes are +0.28208/−0.29224 points for viscosity, +0.08248/−0.09333 for free CO2 diffusivity and −0.53137/+0.39813 for the liquid transfer coefficient. The paper explains enhancement recalculation, the conditional diffusivity response and the unresolved small temperature responses; it does not combine the perturbations into statistical uncertainty or rank unlike perturbation ranges together.

The supplied transport notebook and its numerical records were imported unchanged under `analyses/transport_sensitivity`, excluding caches. All 223 imported files match the source snapshot, and all 136 top-level provenance input hashes match. All seven summary rows were independently re-derived from the raw profiles and solver records (707 axial positions), including capture, peak temperature, material/charge discrepancies, energy-flow range and residuals. Six retained directional checks pass; the current transport value/gradient scaling and vapor-isolation check also passes. The factor-one initialization control and failed initializations remain available in the analysis, not as development narrative in the paper. The source handoff reports an independent review of that study; this integration check is not a new independent review. No column was rerun, no model equation changed, and no commit or push was made.

The 31-page PDF builds and passes freshness. Rendered pages 1, 16–17, 20–21 and 24 were inspected: both sensitivity figures, captions, Methods and concluding discussion are readable without clipping. The final LaTeX log has no undefined references. The existing CAS front-matter overfull-box warning remains. The immutable-wheel integration check passes. Manuscript title and section headings, AI-use disclosure and bibliography are unchanged by this integration. The plugin-wide prose check passes; manual inspection of the full banned list finds only concrete interface, model-choice and statistical uses plus exact identifiers and historical table labels. No scoped terminology-rule pass is claimed.

keep: the transport notebook owns its exact evaluated inputs, failed and successful calculations, source archives and derived tables; the article uses its figure as a presentation copy, verified byte-identical. No alternate adopted parameter set or new solver implementation was introduced. [analyses/transport_sensitivity; figures/transport-sensitivity.pdf]

keep: thermodynamic and transport comparisons retain different perturbation magnitudes and dependency assumptions; the manuscript reports their separate effects and leaves correlated uncertainty and controlled operating conditions to further study. [sections/methods.tex; sections/results.tex; sections/conclusion.tex]

net: no files or lines proposed for deletion or renaming in this scoped CSE audit; unique source evidence remains under its analysis owner. Detailed execution history is outside the manuscript. Multi-case integration and SRP recovery are separate work.

### R1.8 manuscript incorporation — 2026-09-04

The investigator authorized the completed interaction comparison and R4/R5 equilibrium-constant extension for manuscript use. All eight ±5% variants converge. Methods defines one-at-a-time coefficient multipliers and constant ln K offsets preserving logarithmic temperature derivatives. The Results comparison and Figure 5 are placed in the ePC-SAFT Driving-Force Comparison subsection before Numerical Solver Behavior; Conclusion reports the bounded capture response. R4 changes capture by +0.424773/−0.417079 points and R5 by +0.589370/−0.598872 points. MEA–water retains the largest response among the four inputs. Future Work concerns broader parameter uncertainty, reaction temperature dependence and controlled operating conditions. Titles, AI-use disclosure and historical aggregate results are unchanged. R1.8 is complete for this local sensitivity request, not a multi-case validation claim.

The combined retained table has nine rows and 909 profile positions. Scalar changes, positivity, residuals, material/charge discrepancies and all 116 top-level provenance input hashes were checked. The four reaction calculations preserve the reference EOS fingerprint; evaluated parameter and reaction document hashes match each run identity. Selected source files remain unchanged. All four reaction launchers exit zero. The implementation suite reports 87 passing checks and one expected legacy-domain warning; immutable-wheel integration passes. No independent review, commit or push is claimed.

The final 30-page PDF builds and passes freshness. Rendered pages 16, 19, 20 and 23 were inspected for equation, figure, caption and prose fit; the combined four-input plot is readable and has no clipped labels. No undefined references are reported. The pre-existing front-matter overfull-box warning remains; this is not a claim that all historical layout warnings disappeared.

keep: the four evaluated reaction documents preserve the actual sensitivity inputs; the existing figure renderer owns derived values and the manuscript uses its PDF, without creating a second adopted parameter owner. [analyses/nccc_validation/figures/reactive_column/output/sensitivity; figures/reactive-parameter-sensitivity.pdf]

keep: manuscript diagnostics explain numerical resolution and distinguish small temperature responses from capture sensitivity; detailed run histories remain in the working notebook. Manual banned-word inspection found only concrete interface, model-choice and statistical uses, plus exact identifiers and table inclusion labels; the plugin-wide prose check passes. [sections/methods.tex; sections/results.tex; sections/conclusion.tex]

net: no files or lines proposed for deletion or renaming in this scoped CSE audit; unique reaction inputs and results remain under the existing analysis owner. Change accounting against the August base includes the full earlier overlay, not just this reaction extension: 24 executable files +1849/−130 and five test files +360/−1. No new audit machinery was introduced.

### R1.8 calculation and retained comparison — 2026-09-04

Four ±5% interaction variants converge with 42 final nodes, two mesh iterations, RMS residuals below 0.05 and zero maximum scaled boundary residuals. The exact values, parameter documents, profiles and rejected/fresh initialization comparison are retained under the existing reactive-column figure owner. The working notebook distinguishes the capture response from the smaller CO2–water temperature response and does not label the chosen range as statistical uncertainty. Checks: 84 passed with one expected legacy warning; pinned-wheel integration passes. The adopted parameter file is unchanged. The R1.8 results are not yet inserted into the article, and its 30-page PDF remains current and unchanged.

keep: exact evaluated parameter documents and the rejected-state comparison preserve the scientific calculation and its initialization failure; they do not create a second adopted parameter owner. The shared renderer derives the displayed values from completed runs. No material deletion or ownership correction was identified in this scoped CSE audit.

The manuscript sections, appendices and tables pass the plugin-wide prose check and the scoped manual banned-word inspection. No personal terminology configuration or repository terminology table is present; no scoped-rule pass is claimed. Scientific uses of numerical diagnostics are retained. No manuscript title or AI-use disclosure was changed.

net: no files or lines proposed for deletion; retained values stay with the analysis. Independent review and investigator promotion of the sensitivity section remain pending.

### R1.10 and R1.11 manuscript coverage — 2026-09-04

The investigator requested manuscript incorporation and closure of these reporting comments. The numerical-verification table now reports outer mesh/residual histories, mesh iterations, 19/17 RHS batches, four Jacobian batches per run, 84/164 local state Jacobians, final residuals, CPU/wall timings, total initialization/output time, capture, peak temperature and energy-flow range. Methods explains native thermodynamic derivatives combined with differentiated transport/nonisothermal energy expressions, the lean/rich directional comparisons, state reuse, initialization and timing boundaries. The consolidated outcomes table adds same-machine temperature-state Henry CPU/wall results for all three tested methods, retaining rejected attempts as failures rather than successful timing comparisons. No column rerun or parameter change was needed.

The two reviewer responses are complete for available-diagnostics and cost reporting, not a claim that unavailable memory/inner-Newton histories were recovered or that all methods have accepted current-parameter results. Those limits are stated in the article; R2.4 and the broader result-set alignment remain separate. The user-authorized notebook snapshot and limited manuscript use are recorded in the existing result notebook. No commit, push, title change, declaration change or bibliography refresh was performed.

The 30-page PDF is current. The rendered Methods page 16 and numerical-results pages 20–22 were checked for legibility, table fit, units and scope distinctions. The new table and expanded outcomes table fit without clipping. Prose checks pass. Original front-matter and mathematical bounding-box warnings are not newly certified as absent.

keep: manuscript tables are presentation copies of the retained CSV/log values; their scalar values and outer-iteration precision were checked against those sources. No new model calculation or alternate parameter owner was introduced. [tables/reactive_numerical_verification.tex; tables/method_case_contrast.tex]

keep: Jacobian batches, local derivatives and liquid-equilibrium solve counts measure different work and remain explicitly distinguished; no one count substitutes for another or for inner Newton steps. [sections/methods.tex; sections/results.tex]

net: no files removed or renamed; four unique printed iteration rows retained with log hashes. No additional reporting machinery or new scientific assumptions were added.

### Current nine-species calculation and manuscript audit — 2026-09-04

The current-parameter nonisothermal Case 3C column converges with conventional enhancement-factor transport. The retained calculation and refinement, exact values, profiles and initialization/timing distinctions are in `analyses/nccc_validation/results/reviewer_energy/README.md`. This supersedes the earlier 300 s execution limit below. That limit did not establish an equilibrium failure: repeated fresh loading paths and finite-differenced column evaluations were expensive. Exact-state reuse, conservative native starts and native/AD derivatives reduce repeated equilibrium work without relaxing native physical checks.

The physical correction is separate: the coupled film resistance uses bulk EOS fugacity divided by free molecular CO2 concentration, assuming the bulk activity coefficient is constant across the film. The main model section states this approximation, its units, and its distinction from a reactive-loading derivative. The conventional enhancement formula, empirical kinetic inputs, concentration divisor and caloric equations are retained. No eNRTL or resolved reactive-film model was added. Eighty Jacobian, energy, transport-domain, reactive-chemistry, adapter and convergence checks pass in 48.80 s; the one warning comes from the deliberately nonraising legacy domain test. Final immutable-wheel integration passes. This is not an independent scientific review or a replacement of the historical multi-case campaign.

The 29-page PDF builds and is current. Automated geometry checks report 129 tight-clearance warnings and three overlap errors (pages 7, 12 and 13). Rendered inspection identifies those three as mathematical bounding boxes around overbars, stacked derivatives and boundary-vector subscripts, not unreadable collisions. The revised film equation on page 10 and neighboring pages 11–13 are readable, as are the inspected title, workflow and property equations. The automated geometry report is therefore retained as a failed automatic check with manual disposition, not reported as a clean automatic pass. The existing CAS front-matter warning remains. The prose checker passes; manual inspection of the complete banned-word list found only exact identifiers and concrete modeling/statistical terms in active prose. Title and section headings are preserved; the AI-use declaration is unchanged at the investigator's request.

keep: the selected parameters and reaction data are owned by MEA-Thermodynamics and retained downstream as exact files; the native wrapper and final integration check preserve identities. No coefficient copy or new fitting implementation is needed. [selected-reactive-parameters.md; integration/epcsaft_contract.json]

dedupe: numeric and differentiated column evaluations now share transport and empirical-energy expressions; retain the independent directional and two-film balance checks to expose divergence, rather than copying those equations into another solver. Superseded inline transport expressions were removed. [Transport/; Properties/; BVP/reactive_jacobian.py; tests/test_reactive_column_jacobian.py]

keep: curated current-column values and their figure have one owner under `analyses/nccc_validation/figures/reactive_column/output`; the renderer reads raw calculations and the reviewer notebook cites that owner. Exact input/source hashes and Engine identity accompany the values; a narrow ignore-rule exception prevents local output exclusions from hiding this evidence.

investigate: the historical abstract, aggregate tables and quantitative figures remain one six-species result set; a single current nine-species case cannot replace their case count, aggregate error or method ranking. The next manuscript integration must replace those mutually dependent values together. The unknown calibration source of the fixed enhancement divisor is explicitly disclosed, not assigned a fabricated citation. [main.tex; sections/results.tex; tables/epcsaft_parameter_summary.tex]

net: no further files proposed for deletion; unique numerical evidence retained. Shared expressions replace duplicated equations, and new figure data remain with their renderer. Two unresolved questions are full current-case manuscript alignment and the historical concentration-divisor source. Change accounting against the August base includes the entire inherited uncommitted overlay, including selected data and CAS files; it is not a count of this numerical repair alone.

### Equation-to-implementation alignment — 2026-09-04

The investigator requested the supplied whole-manuscript corrections. Verified and corrected the printed gas mass-transfer velocity group, heat-transfer density/diffusivity denominator, concentration-weighted enhancement kinetics and fixed CO2 concentration divisor. Corrected the liquid heat-capacity polynomial, CO2 molar volume, liquid viscosity correction, nitrogen/oxygen vapor viscosity, apparent-MEA diffusivity basis, reciprocal molar-mass conversion and solvent-basis surface-tension fraction. Twenty-four numerical transcription comparisons at 298.15, 325 and 353.15 K agree with the implementation; these are not physical-validation tests. No transport or caloric code was changed.

The main model now distinguishes the reported six-species reference from the implemented coupled nine-species option, with five reactions, consistent activity standard states, EOS true-species concentrations, and separate empirical hydraulic density and enthalpies. The responsibility table and flowchart distinguish fixed formulas from state-dependent values. Methods defines the six-percentage-error boundary norm, distinguishes it from internal residuals, includes one-sided finite-difference endpoints and qualifies initialization costs and unavailable historical diagnostics. Future Work requests further validation rather than reimplementation of the nine-species calculation. Historical Case 3C configurations are explicitly distinguished. The manuscript title and AI-use declaration remain unchanged, the latter at the investigator's explicit direction.

The original source of the fixed enhancement concentration divisor 1.04542981654115 remains unresolved; it is disclosed as an empirical input, not silently removed or assigned a fabricated citation. Historical parameters, quantitative figures, case counts, abstract/conclusion values and archival references await accepted replacement calculations. The current-wheel column attempt timed out at 300 s without a numerical result; no parameter failure is inferred. Exact inputs and retained outcome are in `analyses/nccc_validation/results/reviewer_energy/README.md`.

### Reader-facing editorial revision — 2026-09-04

Applied the investigator's editorial pass without changing the manuscript title or retained scientific values. Removed the activity-rebased timing and exploratory reactive-film subsections, their abstract/conclusion summaries, and their remaining model-description references. Their detailed text is retained in the fallback reviewer response outside the article. Future Work now summarizes independent activity-based equilibrium and finite-rate film extensions without reporting development experiments.

The Introduction states the benchmark contribution directly; the Conclusion retains the quantitative comparison and principal interpretation. Methods distinguishes state-variable checks from property-domain validity and describes finite penalty residuals for inadmissible benchmark trials. Temperature profiles remain useful spatial comparison evidence without a predefined pass/fail threshold. One numerical-results table now groups method outcomes, guard counts and excluded cases. No new equilibrium or column results were promoted into the article, and no solver was rerun for this editorial revision.

The 27-page PDF builds and passes freshness checks. Pages 1–2, 14 and 18–20 were inspected as rendered pages: the introduction, revised methods, consolidated table, Future Work and shorter conclusion have no new clipping or overlap. The existing appendix page break leaves whitespace below the conclusion. The CAS front-matter overfull warning remains; no undefined references were reported.

move: the two development-only Results subsections now reside in `docs/fallback_reviewer_response.md`; their unique numerical values and the original timing-table source are preserved outside the compiled article.

dedupe: case exclusions and method failures now share `tables/method_case_contrast.tex`; the old attempted-case table is no longer included, and the retained numerical values were checked against its rows.

keep: exact callable names, statistical regression, and thermodynamic modeling uses of “route” have concrete meanings; no vague CSE-banned terms remain in the edited active prose.

net: two development subsections and repeated summaries removed from the article; no unique numerical evidence deleted. Historical equation-to-result consistency remains the separately recorded scientific question, not resolved by this editorial pass.

### Corrected energy and bounded column attempts — 2026-09-04

The liquid-energy sign, full composition chain rule, empirical liquid enthalpy derivative and temperature-state initialization are repaired. Explicit enthalpy-profile labels prevent equal-valued derivatives from acquiring the wrong names. Fifty focused energy/chemistry/convergence checks pass; independent review accepts this bounded correction, not submission readiness.

Four source-backed 2017 Henry cases (1C–4C) converge. Refined Case 3C predicts 89.52775% capture versus 89.50% observed; refinement changes capture by 0.004206 percentage points and sampled peak liquid temperature by 0.003980 K. Native one-bed collocation diagnostics and process CPU/wall time are retained. Shooting and finite-difference attempts fail existing acceptance checks and are not ranked as successful method costs. Temperature-tap RMSE is unclaimed pending source coordinate/phase mapping. Staged solver counts are outside this one-bed study and remain unavailable.

The exact previously rejected nine-species state (325 K, 109500 Pa, loading 0.420056 mol/mol) now passes with the Orchestrator-supplied immutable wheel and conservative positive loading path: density 51179.28088199359 mol/m3, CO2 fugacity 1622.949444608249 Pa, balance residual 6.38e-16 and reaction-affinity residual 1.71e-13. The independent 318.15 K reference also reproduces. The earlier old-wheel/unseeded invocation was inadequate; attributing it to an Engine solve-coverage problem was premature. Parameters and effective reaction constants are unchanged. Sixteen focused reactive/energy/RHS checks and the final immutable-wheel integration gate pass. These local equilibrium checks do not establish a full axial column solution. Exact inputs and runtime identity are retained in `analyses/nccc_validation/results/reviewer_energy/README.md` and `nine_species_replay.csv`.

R1.10, R1.11 and R2.4 have improved working evidence but remain partial. R1.8 and R2.5 are explicitly the next scientific analyses in Future Work. R1.1, R1.7 and R2.3 remain closed explanations/disclosures; 15 of 20 reviewer comments remain complete.

Independent read-only review accepted the bounded runtime behavior and requested provenance/API and direct-input hardening. The integration contract now pairs the installed wheel with SHA-256 pins for all five retained bundle files; final mode always executes the canonical reactive smoke, including when `--self-only` is supplied. Direct fugacity evaluation rejects unnormalized or inconsistent composition/concentration pairs. The existing actual-RHS check covers the valid caller. These small follow-up changes were checked locally, not re-reviewed by the other agent.

The manuscript equations and caloric appendix reflect the energy correction; historical figures and quantitative conclusions have not been replaced by the new runs. At the investigator's explicit direction, development history, repair status, working-tree details and numerical blockers are recorded here and in the reviewer notebook, not narrated in the article. All manuscript titles and headings are preserved. This working PDF is not submission-ready even though development narration is absent from its reader-facing text. The older dated reviews below describe preceding revisions and are superseded where this update differs.

The final reader-facing PDF has 29 pages and passes freshness checks. The bibliography remains byte-identical to the retained snapshot. All pages were rasterized and checked at contact-sheet scale; energy equations, caloric appendix, Future Work, parameter disclosure and availability pages were inspected at full-page resolution. No new clipping was found. Automated QA retains 141 findings (138 warnings and three error-level mathematical bounding-box overlaps on pages 7, 11 and 12); the new page-11 flag concerns numerator subscripts in the chain rule and does not show printed overlap. This is a manual layout check, not a clean automated pass. The previously reviewed CAS front-matter overfull warning remains.

### Closure of R1.1, R1.7 and R2.3 — 2026-09-04

At the investigator's request, these three comments are addressed as theoretical explanation, source/fitting disclosure and parameter disclosure for peer repetition. The selected-input appendix now explicitly presents the coupled EOS/reaction basis and standard state, the fitted versus retained/transferred inputs and calibration/evaluation split, and the exact-file/environment route for repetition. These responses do not depend on completing the separately tracked energy repair or column-validation work.

The reviewer notebook now records 15 of 20 reviewer comments complete, with unchanged original feedback and Before scores. Its focused checks pass with no stale evidence. No model equations, parameter values, column results or figures were changed in this closure; only explanatory prose, assessment records and the expected completion count were updated. Earlier dated counts and PDF hashes below identify preceding revisions, not this closure.

The rebuilt PDF is 31 pages and passes freshness checks. The new appendix text on pages 28–29 was visually inspected: equations, units and paths render without new overflow. The existing appendix-end page break leaves whitespace on page 29. Only the previously reviewed CAS front-matter overfull warning remains in the build log.

### No-column update — 2026-09-04

verified: Added `docs/selected-reactive-parameters.md` as a transcription of the exact selected input JSON, with component/fixed, pair, association, correlation and reaction tables, standard states, source/fitting distinctions and candidate domains. The appendix links this supporting record without replacing historical parameters. Independent read-only comparison confirmed the selected parameter SHA-256 matches the upstream selected file. Upstream full-replay metrics with different R1/R3 shifts were not promoted.

verified: Removed repeated inlet-temperature warnings from Results and its representative caption; the accepted 45 C assumption remains in the input table. Replaced stale future-rebasing language with the selected coupled-chemistry description. The energy sign and composition chain-rule correction is now explicit at the end of Results, with a brief reference in Conclusion. Code and Data Availability now name the correct nine-case August base and identifies the uncommitted overlay.

verified: Rebuilt the 30-page PDF, SHA-256 `d60183fadbbb3e7524c65d66f9efc82716a429b686852ece983de46dc80d0bbd`; numerical figures and reported results were not regenerated. The bibliography hash is unchanged. PDF freshness and reviewer-notebook checks pass: 12 of 20 reviewer points complete, no stale evidence hashes. New source-path overflow was fixed with breakable paths; only the previously reviewed CAS front-matter overfull warning remains. Full-page render checks are retained under `builds/visual-qa/no-column-final`, with final availability-page renders after correcting both availability sections. The automated check reports 137 findings, including the same two previously reviewed math-bound overlap flags on pages 7 and 12; this is a manual layout review, not an automated clean pass.

No column run, parameter fitting, energy-code change, commit, push or mirror synchronization was performed. This pass adds no model executable code; the existing notebook check's completion count changes from 11 to 12. Change accounting was run against `0d4e552`; the larger runtime additions in that report predate this documentation pass. Prose-check hits in the parameter tables are exact retained source identifiers/citations, not newly coined prose.

The earlier review below describes the retained historical draft and prior runtime work; this dated update supersedes its statements about the location of the inlet assumption and absence of the selected-input supporting record.

**Complete historical review draft; not submission-ready.** The current 30-page PDF preserves the substantial revised article, including its literature review, governing model, methods, five figures, nine tables, appendix, and 33 references. The historical numerical results are preserved for inspection, not approved for submission. The material energy-balance error below cannot be repaired by editorial changes alone.

The manuscript source is `docs/latex/main.tex`; the output is `docs/latex/builds/main.pdf`. The exact 20-point response and coverage record is `docs/fallback_reviewer_response.md`. Neither record is included in the article.

## Source identity and bounded changes

- verified (fallback checkout update): This branch retains HEAD `0d4e552e15fdf9ee17edc64908c3a04a6525f98b`, the 2026-08-25 merged MEA-only freeze. The manuscript tree was preserved byte-for-byte during the move, including its PDF and bibliography. The later user-authorized nine-species runtime, dependency lock, and parameter update are an uncommitted overlay alongside the manuscript edits and reviewer records. The replacement runtime is not evidence of historical numerical reproduction. Exact identities and commands are recorded in `REPRODUCE.md`.
- verified: The base is the top-level `latex/` tree from the authoritative archive `legacy/manuscript-pre-reactive-film-2026-09-03/latex-source-and-build.tar.gz` in the original checkout. Archive SHA-256: `3dc5903c233935093bfb61a29c7f8a18a9aad6bfb0e1d63900064e20c1cfe633`. Its README identifies capture at source commit `4cd362c83e0a6b439655a18819a862da25e782cf`; the extracted manuscript sources also match this worktree's starting commit `32d5a3baeb1454e1b5eb9237515e0d26b3381690` before these edits.
- verified: No nested August submission package or historical built PDF was used as the editable base. The exact end-of-March submitted PDF was not established.
- verified: Added the Henry-law comparison row, explicit capture/MAE/temperature-RMSE definitions, mixed vapor/liquid boundary-condition notation, a new-solvent fitting/evaluation procedure, and industrial-gas/finite-rate-film limitations. New prose follows the CSE sentence-per-line and paragraph-comment convention.
- verified: Private bibliography notes are suppressed at rendering. The Zotero snapshot remains byte-identical to the archive and the canonical export, SHA-256 `217b58678d85d584d9d52a10200af93c67f1df369a67fc95e1d2d87b0736e870`. No bibliography refresh or metadata edit was performed.
- verified: All five figure files and all numerical table values are preserved. Three table wrappers and four result-figure wrappers now use native CAS floats, with actual float barriers, to keep captions attached. No manuscript parameter, reaction constant, calculated value, or source observation was substituted.
- verified: The original checkout, its current manuscript, checklists, and server were not edited. A non-editable replacement Engine wheel was installed only in this worktree's environment. No full solver campaign, commit, push, or submission was performed.

## Blocking scientific findings

### Current revision priorities — 2026-09-04

verified: The selected bundle and downstream `compile_reaction_constants` use retained temperature-dependent constants and explicit source-standard-state conversion, not initial-composition activity rebasing.
The historical rebasing objection is resolved for the new runtime; the older manuscript results have not yet been replaced with matching runs.
No additional parameter-fitting project is required simply to remove that objection.
The investigator accepts the assumed inlet temperatures and does not require exhaustive mesh histories.
Keep the 45 °C assumption in the input-table note, collect available final diagnostics and one useful numerical-accuracy comparison, and avoid repeated generic caveats in the article.
Consolidate substantive scope limits near the end, including additional gas species, finite-rate film chemistry and other solvents.
The primary correctness repair remains the energy formulation and matching column calculations.
The historical findings below retain their evidence; they are not a list of equally mandatory new studies.
Current point-by-point actions and readiness assessments are in the reviewer notebook and `docs/fallback_reviewer_response.md`.

### 1. Countercurrent energy balance and temperature chain rule

verified: The archived article defines opposed interphase energy sources in `sections/model_framework.tex:293-296`, but gives the liquid enthalpy-flow derivative the positive source sign at lines 322-337. The material balances at lines 307-310 already distinguish the countercurrent liquid direction. The temperature-gradient expressions also use mean mixture enthalpies where composition changes require the corresponding partial component enthalpies.

The retained implementation has the same problem: the original checkout's `src/mea_absorption_column/BVP/ABS_Column.py:301-310` uses `dHlf_dz = Hl_flux` and mean enthalpies `Hl_T` and `Hv_T`. Its `Transport/Flux.py:22-31` defines `Hl_flux = -Hv_flux`. The separately corrected implementation's handoff, `/home/tnnrpolley21/.codex/worktrees/6c99/MEA-Absorption-Column/docs/coordination/casadi_implementation_handoff.md:654-680`, explicitly records the countercurrent sign and partial-component-enthalpy correction and requires historical thermal results to be rerun.

inference: Because temperature changes equilibrium, properties, and transfer rates, this is not confined to a temperature caption. The historical Case 3C capture of 88.32%, the 3.32/3.14 percentage-point MAEs, 7.70/5.25 s runtime medians, solver contrasts, and activity-rebased column results cannot establish the corrected model's performance. Their associated profiles and temperature errors are also affected. The historical equations and results have not been silently changed into an unexecuted corrected-model claim.

The smallest next scientific step is to agree a bounded repair/reproduction scope for the fallback formulation, then establish corrected energy conservation and matching capture/temperature outputs before reusing the quantitative claims. A corrected manuscript equation alone is insufficient. The new runtime parameters do not repair this issue, and no reactive-film formulation was introduced.

### 2. Parameters and execution identity

verified: `tables/epcsaft_parameter_summary.tex:34-51` discloses missing source/estimation details, assumed extra-ion parameters, and an association-scheme mismatch in the lineage of MEA/water `k_ij = -0.052`. These historical manuscript values have not been replaced with the current reactive bundle. The historical six-species, unit-activity chemistry and locally activity-rebased nine-species results remain distinct from the new runtime.

unknown: The complete run-by-run identity linking every historical result to its original source commit, parameter document, reaction constants, and Engine wheel has not been independently reconstructed. The code-availability statement's old Engine identity is preserved, not newly certified. The current worktree's dependency specification is not evidence of the environment that produced those historical numbers.

The user selected the current MEA-Thermodynamics notebook parameters for the new runtime: parameter SHA-256 `a9186c93759f2e2c02a6c913350ad06a244fff3f82503820c9962b3df8dd40d9`, wheel SHA-256 `40fba7cfb9c8414152f3e49636c49ae2e3f7099e30040d54d464ccb38355f805`, Engine commit `8438ce5f94a547189c91c4ec180a7782d60879d6`. The unused polar declaration was removed. The canonical `epcsaft_reactive_nine` mode couples the nine-species EOS equilibrium to the unchanged legacy flux and enhancement formulas, with EOS-density concentrations and explicit nine-species profile fields. The selected candidate's applicability and column predictive accuracy are not established by this integration.

The final integration gate and focused checks are required for this replacement runtime. Passing them does not validate the old outputs or establish corrected energy balances. The source notebook owns parameter adoption; its full publication refresh remains incomplete. See `REPRODUCE.md` for the exact retained inputs, domain precedence, checks, and limitations.

### 3. Experimental inputs and numerical evidence

verified: The article states that the source liquid-inlet temperatures for 1C–3C are missing and that 45 °C is assumed (`sections/results.tex`, NCCC comparison). It does not claim this temperature was measured. The handoff identifies Morgan2020 Table A1 Case 3C dry gas values CO2 = 0.093 and O2 = 0.090; these must be reconciled with each actual historical input, not silently assigned to old output files. The source-coordinate orientation and height mapping of temperature taps also remain to be confirmed for validation use.

verified: Methods explicitly states that final mesh/refinement histories, method-specific iterations/evaluations, controlled hardware identity, CPU time, and memory were not retained. A success flag and a small boundary residual do not establish solution accuracy, physical validity, or matched-accuracy computational cost. Thermodynamic sensitivity, independent validation, and a controlled operating/optimization study remain open reviewer requests.

## Visual pilot integration — 2026-09-03

verified: Integrated the CAS presentation work from “Quarto MEA manuscript visual pilot,” commit `63678e33557c9f775cdb4fe34f7aa00e72275904` and later archive snapshot `2fdd4fd53f9cc54e3aaf208de4f41229387f4fea`.
The pilot branch name still resolves to the August base and is not the source of this port.
The workflow diagram is now native TikZ; table alignment, equation notation, chemical and SI units, cross-references, and appendix spacing follow the pilot.
The fallback's added Henry-law row, mixed phase boundary conditions, metric definitions, typed-bundle diagnostic, solvent-extension procedure, and industrial-gas/finite-rate-film limitations remain.
No equation right-hand side, numerical table value, quantitative figure file, bibliography byte, or runtime implementation was changed by this visual integration.
The old workflow PNG remains retained but is no longer the active figure.

verified: The frozen-bibliography XeLaTeX build and PDF freshness check pass.
Current PDF: 30 pages, SHA-256 `1ee2420ff2202ef5e25478de6a3505c6890a99924367c08c75b13009e2e70fce`.
All 30 pages were inspected at full-page resolution in color and grayscale, with independent inspection of pages 11–30.
The Celsius glyph, chemical-word spacing, table-footnote allowance, and placement of parameter units with their table were corrected during this port.
There are no undefined citations/references or LaTeX errors; the existing 117.0831 pt CAS front-matter warning remains, without title or abstract overflow.
The CAS front-matter line numbers have a small overlap near the article-info heading; this does not obscure manuscript text.
The deliberate appendix page break leaves page 23 sparse.

verified: Automated visual QA reports 138 conservative findings, including two error-level text-overlap flags on pages 7 and 12.
Manual inspection identifies those two flags as mathematical accent/subscript bounding-box intersections, not overlapping printed content.
The remaining clearance flags were checked against the rendered pages; dense tables and the six-panel plot remain readable.
The automated report is retained unchanged at `builds/visual-qa/visual-port/report.json`; it is not a clean automated pass.
This visual review does not remove any scientific blocker below or upgrade reviewer scores.

verified: Independent final source review found no scientific regressions or dropped fallback content.
It checked the phase-specific BVP, shooting and finite-difference boundaries, Henry-law row, reported numbers, reviewer metric definitions, typed-bundle diagnostic, and solvent/industrial-gas limitations against the pre-port files and pilot snapshot.
No runtime or transport code was edited in this visual stage; change accounting against the August base retains the earlier executable +509/-13 and test +139/-1 totals.

## Earlier checks completed before the visual port

- Independent final runtime review passed after correcting selector-mismatch failure handling and case-insensitive driver metadata. The review covered the final uncommitted runtime tree against the August base, including scientific ownership, species/density bases, output provenance, and failure preservation. This scoped pass does not approve the historical manuscript results.
- Replacement-runtime checks: 55 focused tests passed in 6.70 s; the final integration gate passed with the selected wheel and a coupled nine-species state at 313.15 K and 109500 Pa. Its density was 48710.836648 mol/m3 and parameter fingerprint `sha256:07e3f93b209d8e117af4b69e2d0ab3b5bc0b2e94568643c28629e4799d8aa062`. The Case 3C inlet check also exercised actual RHS and nine-species output evaluation at 318.15 K. The stopped-solver driver check verifies the selected dataset identity, unavailable legacy residuals, and failure status for lowercase and uppercase model names without claiming convergence. These are integration/conservation checks, not a full-column validation.
- `Transport/Flux.py` and `Transport/Enhancement_Factor.py` are byte-identical to the August base. The historical PDF and bibliography hashes below are unchanged after the runtime update. The reviewer notebook's exact-comment, scoring, changed-source, and escaping checks passed.
- CSE change accounting against `0d4e552e15fdf9ee17edc64908c3a04a6525f98b` reported executable +509/-13 and tests +139/-1, including the earlier reviewer notebook. The new runtime module maps retained parameter/reaction inputs and the public coupled Engine solve; column changes preserve EOS concentrations, explicit profile bases, native evidence, and visible failures. The six-line unused polar declaration was removed. The low removal/addition ratio (0.026) reflects the newly authorized integration and notebook, not replacement of the historical transport model.
- The existing `TEXMFHOME=/home/tnnrpolley21/texmf bash docs/latex/scripts/build_main.sh` completed the LaTeX/BibTeX loop. `python3 docs/latex/scripts/check_main_pdf_fresh.py` passed. No root-level `docs/latex/main.pdf` was created.
- Final PDF: 33 pages, SHA-256 `93fa8bd4ccc2c86525195dcf76385437e9a0b63fe0b2e73c4cb1e830bcf8d7d9`.
- All 33 final pages were rendered and visually inspected. The detached captions found in the first build were corrected. Figures, tables, equation numbers, and references render without clipping or missing content. The existing deliberate appendix page breaks leave some sparse pages.
- The first in-text citation is `[1, 2, 3]`, the reference list begins at `[1]`, and Bülow2021a is reference `[18]` without its private exported note.
- No undefined citations/references or LaTeX errors remain. One existing 117.0831 pt CAS front-matter overfull-box warning remains at `maketitle`; the title/abstract have no visible overflow. Underfull table/bibliography spacing is nonblocking.
- A read-only CSV arithmetic check reproduced 20 attempted rows, 18 included rows, nine cases per closure, the stated rounded MAEs, and the stated median wall times. This checks transcription and the stated inclusion rule only, not the physical correctness of the results.
- All referenced figure paths exist. An archive comparison confirmed unchanged figure/bibliography bytes and table values apart from placement commands. All 20 quoted reviewer points were compared verbatim with `docs/reviewer_comments.txt` in the source checkout.
- `git diff --check` and the CSE prose checker passed for edited prose. Repository guidance contains no scoped terminology table; no scoped replacements were invented.
- The repository cleanup audit reported `cleanup_state: clean`. Final build files and rendered review sheets remain under the ignored `docs/latex/builds/` directory.
