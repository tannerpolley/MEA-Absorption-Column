# Independent manuscript audit — Astra

**Audit date:** 2026-09-04  
**Checkout:** `/home/tnnrpolley21/Workspaces/Engineering/MEA-Absorption-Column`  
**Branch observed:** `codex/fallback-manuscript`  
**Review mode:** blind Stage 1; no prior editorial review or revision-plan document was read before this report was written and sealed.

## Verdict

The current 32-page PDF is readable and its reported calculations are internally traceable to the retained Case 3C, seven-case, sensitivity and operating records. The manuscript now makes the important model separation visible: ePC-SAFT supplies bulk liquid and neutral-vapor thermodynamic quantities, reaction equilibrium supplies the nine-species distribution, conventional enhancement supplies the film approximation, and empirical calorics/transport close the column. The equations, units and displayed numerical values checked here agree with that stated implementation.

The article is not yet ready to support a strong predictive-reactive-thermodynamics claim. The central unresolved issue is evidence, rather than the displayed coupling: the selected reactive parameterization is identified as a working set, but there is no independent VLE/speciation/thermochemistry holdout demonstrating that the EOS-plus-reaction construction is accurate for chemical absorption. At the process level, seven coarse cases give 5.85 percentage-point capture MAE with +11.45 and -11.93 point case errors; only Case 3C has a paired refinement. Every accepted run also retains a nonzero axial net-enthalpy-flow range, including 273.006 W for the refined reference, with no declared energy acceptance limit. These facts are disclosed, but they limit the strength of the temperature and predictive claims.

The reviewer requests on format, model-family context, equations, parameter listing, bounded sensitivity, transport discussion and numerical reporting are substantially represented in the PDF. Reviewer requests for an optimum and for additional amine results remain unperformed by stated scope. The conclusion does not repeat the limitations that are essential for interpreting its headline results, and the public code/data route does not identify an immutable calculation package. The recommended decision is **major revision focused on claim scope, independent thermodynamic evidence or an explicit limitation, energy-closure treatment, and repeatable release records**. No manuscript edits are made by this audit.

Evidence labels used below: **verified** means directly checked in the PDF/source or retained record; **reference-backed** means tied to the cited NCCC/source record; **inference** means an interpretation of verified values; **assumption** means an imposed or reconstructed input; **unknown** means the inspected material does not establish the claim.

## Read scope and identity

I read the current PDF end to end, checked its 32 nonempty pages visually after rendering, extracted each page separately, and checked the associated source equations, tables, captions, appendices, embedded figures, retained records and the pre-revision PDF. I also read `docs/reviewer_comments.txt`, `docs/fallback_reviewer_response.md`, `REPRODUCE.md`, the seven-case figure README/CSV records, the retained Case 3C refinement confirmation and its profile outputs. I inspected the relevant implementation paths for reactive composition, fugacity, enhancement, fluxes, balances, temperature state and NCCC input reconstruction; no model run was launched.

The current PDF has 32 pages and the pre-revision comparison has 29 pages. All current pages contain text or a figure, all 32 footer page tags are present, and the rendered page review found no clipping, overlapping equations, unreadable table cells or detached captions. The current PDF scan has zero whole-word matches for `local`, `locally`, `history`, `historical` or `histories`. The pre-revision PDF has eleven `local` matches (pages 3, 5, 6, 7, 13 and 27) and no matches for the other four words; this confirms that the current wording cleanup is present in the rendered article.

The pre-revision PDF is a materially different article, not merely a shorter build: it presents a Henry-law/ePC-SAFT fugacity and numerical-method benchmark, eight solver-accepted rows, six-species/concentration chemistry and method timings, whereas the current PDF presents a five-reaction/nine-species reactive model, seven NCCC cases, a single collocation route, paired Case 3C refinement, sensitivity and operating studies. The current article correctly does not reuse the pre-revision headline values as current results.

### PDF and source hashes

Hashes are SHA-256 and were recorded before this report was written. The first two are the PDFs audited; the remaining entries are the source files, embedded figure PDFs and retained records used for the checks.

| File | SHA-256 |
|---|---|
| `docs/latex/builds/main.pdf` | `a2d6f00c9cce124d47fb579cd20a4a6b61d5f2b70a4bddfe3dfc0bb0c949b532` |
| `docs/latex/builds/main_before_reviewer_quick_revisions_2026-08-12.pdf` | `09b8ce8226c7e9e083339deb2231adff9eca5b37171f1b9c03d178ea07c16f62` |
| `docs/latex/main.tex` | `3a472827cd629156a00ebb500bc9f5a5e04069a75ec81f179177a185165bfd5e` |
| `docs/latex/sections/introduction.tex` | `87c49c66d50891341970ea7efe19e4cbdccebdc92301d9170136acf5043d6ca2` |
| `docs/latex/sections/model_framework.tex` | `5d90313b2670cb309f7367d2d4628972b43aeb4dd7690bb99f70df445ecf9b0e` |
| `docs/latex/sections/methods.tex` | `153e898c8980c064246ef81996b030ced7e548d0a23da4724963090084f93389` |
| `docs/latex/sections/results.tex` | `82e626064a632d1dd980bbba1c119770fd8e4ada985df3c7a4a4e354a1d4fea7` |
| `docs/latex/sections/conclusion.tex` | `be612db3153e5c0070ce4e6b64c789e64f822595d6cb0e83133dbbf4861042cc` |
| `docs/latex/sections/data_availability.tex` | `df25983acc5bfeac607e6c1cfac5eb66a4b408e1db0e53d0417a208bfa380e46` |
| `docs/latex/sections/code_availability.tex` | `664f3cf4882121452757ece5c4fcb683f8c75f031d7b88333a6caa04efe49048` |
| `docs/latex/sections/declaration_competing_interest.tex` | `12aabfb34902d7e719d820b573ed0eb3ca0ec0d836c22963c38c91bb98123b6a` |
| `docs/latex/sections/generative_ai_disclosure.tex` | `60a7aea8e4835b6db1209617d8ba2f3b9b1ea350d7e244767c5de59c0004d251` |
| `docs/latex/appendices/appendix_properties.tex` | `1c450d81c0c0f72ad0d0e374e22196a635db3c147bdfb839e16b1a984f46eb4b` |
| `docs/latex/appendices/appendix_reactive_inputs.tex` | `3f03e44a497ddf3f47b21042425d5b3f0b5962f7931344a5b85c87ccd4df1771` |
| `docs/latex/tables/absorber_literature_comparison.tex` | `e776c522a2ac50d9034e391ab322227db4b8c9627c11c3e021b830d505cb60c4` |
| `docs/latex/tables/epcsaft_parameter_summary.tex` | `ba8868500e49f287b07d90212eda9870c9874995bd8f1ad86703b1a3ff33099c` |
| `docs/latex/tables/nccc_one_bed_case_scope.tex` | `010172a76c6c93f77b2cc42d56bd7927b7acd65eaf52641840b66b624ce2b48a` |
| `docs/latex/tables/reactive_numerical_verification.tex` | `ed08df5e4f9623ff50239cf9b3c0e55b95430560197fcb67dda1cf4faf609391` |
| `docs/latex/tables/transport_applicability.tex` | `32708c2c74a911a63f21d8d2e9f82c6936fe966646665505c7ff0e9601e144e5` |
| `docs/latex/figures/tikz/model-framework-flowchart.tex` | `4b8142097e5533328f129663bd907baa6b3c6cb415b73bfdacc4ca1efb98168c` |
| `docs/latex/figures/reactive-case3c-profiles.pdf` | `833770d769759da4a920811ddafc2b79ca7f06ca379eceae6a698f9627f803f5` |
| `docs/latex/figures/reactive-seven-case-capture.pdf` | `1c098e07e7fda15094a6b0b36dda31d8d92d7b5efbd5be637b0feb5e4fffad62` |
| `docs/latex/figures/reactive-seven-case-temperatures.pdf` | `59d58450b72de11f6e34c4679dbbacee6703362cded9f6c9076f6bb61f8c52ed` |
| `docs/latex/figures/reactive-parameter-sensitivity.pdf` | `b5f7f0ff69cbad50ba97f000e3ea29b7aa16bdd91f447edbe2d0ed6359027343` |
| `docs/latex/figures/transport-sensitivity.pdf` | `9c1ed0c7c67be8825cc30aefe840c1a18ac58967a90f6e6c7c38989ddd24bd1c` |
| `docs/latex/figures/reactive-operating-response.pdf` | `a6ae0054ae5ef0db81758d95344daa192f9b3f1f8e02dbb17913d105889051bd` |
| `docs/latex/references.bib` | `217b58678d85d584d9d52a10200af93c67f1df369a67fc95e1d2d87b0736e870` |
| `docs/latex/software_references.bib` | `08efded58037b202f5d2aafddce29185f9724bbeb4b50d338d5fba1ba7e4961e` |
| `REPRODUCE.md` | `d425236972bfdd2102af22f116810f0ab794343072b9c36690166620d74b75e` |
| `docs/reviewer_comments.txt` | `dde6c21b58cc4f052153cf9940cb2b71b6a5f78f8ca7fd465abf1e63a5897c5d` |
| `docs/fallback_reviewer_response.md` | `a3cb7f6ae4ba1943c46e9556be868d25892121bd8f4473b9bc5ff53e0a8faa96` |
| `analyses/nccc_validation/figures/reactive_parallel/README.md` | `a10b36305f762100bcc509bf50f9458954d1a57d1da008fcb2c6d6d019043934` |
| `analyses/nccc_validation/figures/reactive_parallel/output/summary.csv` | `de4d4a1e581ff3d1697ae420fcc82fd150d9abe11b571f2467f34673b78d8acb` |
| `analyses/nccc_validation/figures/reactive_parallel/output/temperature_observations.csv` | `14dc8c613e13d6bdd96f08bb89616e5aa3648b61d4269c2befd996e83ad3d354` |
| `analyses/nccc_validation/figures/reactive_column/output/refinement_confirmation.json` | `422621d4ed1ded59347766744e42070b13fd5a9dfc795bf991462580ad73e93e` |
| `analyses/nccc_validation/figures/reactive_column/output/summary.csv` | `305ab1f8d91ee4594eadd2292c276c5b957957479ede45b13a72c7b19bfc8583` |
| `analyses/nccc_validation/figures/reactive_column/output/profiles.csv` | `fc6f17b79107c91d8e2f7651de9ce2a425aa25fcffee9848a33cadbce725f10d` |
| `analyses/nccc_validation/results/runs/reactive_refinement_confirmation_20260904/enhance_factor.csv` | `26011293d72fe9e593658d7e625b63317fd9537a8de642f14e5336c79ee656f1` |
| `analyses/nccc_validation/results/runs/reactive_refinement_confirmation_20260904/identity.json` | `8010e02f888c59200ed58901bd80e545f25516b40eedf9dd685be7a9d4c3c456` |
| `analyses/nccc_validation/results/runs/reactive_refinement_confirmation_20260904/result.json` | `cefc1ee386c39d061123e2d3d048f1046083142d150b61cfdba6392899c25863` |

## Findings

### A1 — Major: the reactive ePC-SAFT theory is described, but its chemical-absorption accuracy is not evidenced

**Locations:** PDF pp. 1, 3–7; `docs/latex/sections/introduction.tex:17–20,23–29,49–52`; `docs/latex/sections/model_framework.tex:39–90,93–143`; `docs/latex/appendices/appendix_reactive_inputs.tex:7–38`; `REPRODUCE.md:74–85`.

**Observed evidence:** The equations and responsibility table correctly separate molecular nonideality, reaction equilibrium, fugacity, enhancement and empirical calorics (**verified**). The five reactions and nine species are explicit, the activity convention and EOS conversion are stated, and the appendix supplies the selected coefficients (**verified**). The NCCC cases were not used to fit the thermodynamic inputs, but the appendix says that some thermodynamic measurements used for model selection are not independent evaluation data (**verified**). The reproduction record calls the selected input a working parameter set and says it is not an independently validated column model (**verified**).

**Concern:** Reviewer 1's fundamental question is why this physical EOS plus reaction-constant construction should be accurate for reactive absorption. The current text establishes how the calculation is assembled; it does not establish that its activities, speciation, liquid CO2 fugacity or reaction thermochemistry are accurate in the relevant MEA state. The 5.85-point capture MAE is a process-level result from the same unvalidated parameterization and cannot substitute for a thermodynamic holdout (**unknown**, with the process comparison serving only as **reference-backed** observation comparison).

**Required direction:** Either add source-backed independent validation of the relevant liquid VLE/speciation/heat-of-absorption quantities, clearly separated from data used for selection, or narrow the manuscript and response language to a working formulation and a process-level evaluation. State that formal EOS/reaction coupling is a model construction, not evidence of predictive accuracy. Do not imply that the existing NCCC comparison validates the EOS itself.

**Missing evidence request:** An immutable table identifying each thermodynamic datum used for fitting or model selection, each independent datum withheld for evaluation, the predicted quantity and unit, and the resulting error is needed to support an accuracy claim. If no independent set exists, record that absence as a limitation and keep future validation as a proposal.

### A2 — Major: energy conservation is a stated continuous property but not an accepted numerical property

**Locations:** PDF pp. 7, 9–11; `docs/latex/sections/model_framework.tex:312–373`; `docs/latex/sections/methods.tex:62–71`; `docs/latex/sections/results.tex:19–24,62–68,109–114,137–146`; `docs/latex/tables/reactive_numerical_verification.tex:21–28`; retained `reactive_refinement_confirmation_20260904/result.json` and `figures/reactive_column/output/summary.csv`.

**Observed evidence:** The displayed balance signs and temperature chain rule are consistent with the implementation, and the continuous equations imply zero variation in signed net enthalpy flow (**verified**). The refined retained run reports 273.006483 W axial net-enthalpy-flow range; the seven-case runs report 106.49–325.13 W, and operating runs report 160.21–390.32 W (**verified**). The text explicitly says no energy-error threshold is asserted (**verified**).

**Concern:** Small differential residuals and near-zero boundary residuals certify the declared BVP stopping tests, but they do not certify the separately sampled energy invariant. Nonzero energy-flow variation is especially material because axial temperature is a headline observable and the model uses empirical enthalpy and heat-transfer expressions. The manuscript discloses the discrepancy but still presents temperature profiles, peak-temperature responses and process-optimization directions without an accepted energy-closure criterion (**inference**).

**Required direction:** Diagnose whether the range is due to interpolation/postprocessing, the temperature chain rule, composition-dependent enthalpy, or collocation residuals; then report an absolute and relative closure criterion with a stated acceptance rule. If no correction or criterion is authorized, explicitly make every temperature and energy interpretation conditional on this unresolved numerical diagnostic and avoid treating the BVP residual as a full conservation certificate.

**Missing evidence request:** For the refined reference and one representative sensitivity/operating case, retain the signed net-enthalpy-flow at collocation nodes and exported points, the independent recomputation from each phase flow and enthalpy field, and the maximum local energy-balance residual under the proposed acceptance threshold.

### A3 — Major: the seven-case process comparison is numerically complete but physically uneven and only one case is refined

**Locations:** PDF pp. 1, 7–9, 13; `docs/latex/sections/methods.tex:4–22,24–71`; `docs/latex/sections/results.tex:33–68,149–153`; `docs/latex/tables/nccc_one_bed_case_scope.tex:30–43`; retained `analyses/nccc_validation/figures/reactive_parallel/README.md:24–38,74–86` and `output/summary.csv`.

**Observed evidence:** The seven retained captures are 99.1020, 97.3889, 91.5433, 91.4949, 92.2754, 71.6467 and 64.4674 percent, giving 5.85483 percentage-point MAE; the largest signed errors are +11.44669 and -11.93263 points (**reference-backed/verified**). All seven pass the stated numerical inclusion checks, but only Case 3C has a paired 21/41-node refinement (**verified**). Cases 1C–3C use an imposed 318.15 K lean-liquid inlet because the source entry is blank, and all cases reconstruct wet feed from dry gas data (**assumption**). The 35 temperature taps have no source phase designation, and no quantitative temperature error metric is reported (**reference-backed/verified**).

**Concern:** Numerical success is being used appropriately as a separate gate, but a reader can still read the seven-case figure and 5.85-point MAE as broad model validation. The two large opposite-sign errors, coarse meshes and imputed/reconstructed inputs show that physical predictive accuracy remains unresolved. The temperature gallery is useful for exposing discrepancies, but comparing both predicted phases to an unspecified sensor cannot establish temperature accuracy (**inference**).

**Required direction:** Keep the seven-case scope, but label it consistently as a coarse process evaluation/numerical completion. Put the two large errors and the no-refinement status adjacent to the headline MAE, and state that the temperature comparison is qualitative unless a source-backed phase mapping and error metric are available. If stronger validation is intended, provide all-case refinement and quantitative temperature errors; otherwise do not imply it.

**Missing evidence request:** Provide per-case mesh/tolerance refinement or state explicitly that it is unavailable, and provide a source-backed measurement uncertainty or per-tap error table if temperature accuracy is to remain a conclusion.

### A4 — Major: the public reproducibility route does not identify the exact numerical package

**Locations:** PDF p. 23; `docs/latex/sections/data_availability.tex:1–5`; `docs/latex/sections/code_availability.tex:1–6`; `REPRODUCE.md:5–48,63–85,104–126`.

**Observed evidence:** The PDF points to the project repository and the reproduction guide, while explicitly stating that the numerical records have no public archival identifier (**verified**). The guide identifies selected parameter JSON hashes, run settings, result identities and several wheel hashes (**verified**). It also says the measured fast timing wheel came from a source with uncommitted Engine optimizations and is distinct from the restored pinned wheel (**verified**). The current article source and retained result package are uncommitted on the observed branch (**verified from checkout state**).

**Concern:** A repository URL plus a branch-local guide does not give a reader an immutable public copy of the exact source, figure values, input records, Engine bytes and run identities used for the article. This is a direct gap for Reviewer 2's peer-repetition request, even though the manuscript is candid about it. The timing number is particularly difficult to repeat unless the exact measured wheel and source state are released together (**inference**).

**Required direction:** Release an immutable archive containing the exact manuscript source, embedded figures, processed observations, seven-case and Case 3C records, parameter/reaction JSON, lock/environment files, Engine wheel bytes or a public source commit that deterministically produces them, and the run identity hashes. Replace the temporary branch/path references with a stable archive identifier. Until then, describe peer repetition as source-assisted rather than public reproduction.

**Missing evidence request:** A stable public identifier and a clean archive manifest linking every displayed result to source/input/Engine hashes are required. Do not substitute a later default wheel for the wheel used in the measured timing.

### A5 — Medium: the conclusion does not carry the limitations needed to read its headline claims

**Locations:** PDF p. 13; `docs/latex/sections/conclusion.tex:1–21`; contrast with `docs/latex/sections/results.tex:117–153` and `docs/latex/appendices/appendix_properties.tex:227–279`.

**Observed evidence:** The Results and Appendix disclose the +11.45/-11.93 point case errors, coarse campaign settings, imputed temperatures, one missing operating condition, nonzero energy ranges and transport ranges exceeded by the calculated state (**verified**). The conclusion repeats the 5.85-point MAE, 91.55% Case 3C result, sensitivity magnitudes and optimization directions but contains no corresponding limitation sentence (**verified**).

**Concern:** Reviewer 1 explicitly asked for limitations in the conclusion. A reader who sees only the conclusion receives the positive process interpretation without the conditions that bound it (**inference**).

**Required direction:** Add one compact sentence or paragraph stating that the current model uses a conventional enhancement closure and empirical calorics, has uneven seven-case agreement, nonzero sampled energy diagnostics, out-of-range transport conditions and no completed optimum or other-amine validation. Keep future work as future work.

### A6 — Medium: the abstract's final sentence reaches beyond the completed study

**Locations:** PDF p. 1; `docs/latex/main.tex:92–100`; `docs/latex/sections/results.tex:155–173`; `docs/latex/sections/conclusion.tex:18–21`.

**Observed evidence:** The completed work is a selected MEA parameterization with conventional reaction enhancement, empirical calorics, seven-case evaluation, Case 3C refinement, bounded perturbations and discrete operating changes (**verified**). Predictive liquid-film development, a matched ePC-SAFT/eNRTL comparison, other-amine testing and constrained optimization are described as programs or work already underway, without completed results (**verified**).

**Concern:** “Provides a basis for” is weaker than claiming completion, but the abstract's final list can still be read as demonstrated capability. The current evidence supports motivation for those studies, not their performance or feasibility (**inference**).

**Required direction:** Keep the forward program if desired, but explicitly label it as motivation/future work and state in the abstract or conclusion that no predictive-film, matched-model, other-amine or constrained-optimum result is reported here.

### A7 — Medium: two reviewer requests are integrated as proposals, but their response status is not uniformly represented

**Locations:** `docs/reviewer_comments.txt:14–16,19–24`; `docs/fallback_reviewer_response.md:99–121,147–169`; PDF pp. 12–13; `docs/latex/sections/results.tex:160–173`.

**Observed evidence:** The response correctly marks Reviewer 2 item 5 as partial because the lower-flow condition has no result and no optimum is established (**verified**). Item 6 is marked complete, although the article only proposes DEA/MDEA/AMP/PZ/blend tests and does not report a new-amine fit or validation (**verified**). The conclusion also presents these as future studies (**verified**).

**Concern:** The actual evidence supports a partial roadmap response for both “optimal operation” and “how to fit a new amine.” Calling the second complete may overstate what the manuscript gives the reviewer (**inference**).

**Required direction:** Preserve the author's scope constraints, but classify the new-amine item as a documented extension plan rather than a completed fitting demonstration. State the concrete reusable inputs, solvent-specific inputs and independent measurements still required; do not claim cross-solvent transfer has been shown.

### A8 — Low/Medium: reader-facing numerical precision is higher than the physical evidence warrants

**Locations:** PDF pp. 1, 7–13; `docs/latex/main.tex:95–100`; `docs/latex/sections/results.tex:19–31,73–114,124–146`; `docs/latex/tables/reactive_numerical_verification.tex:14–23`.

**Observed evidence:** The article reports calculated capture and temperature values to five decimal places, refinement changes of 0.00510 percentage points and 0.04493 K, and energy ranges to 0.001 W, while the reported capture is 89.50% and source temperatures are tabulated at much coarser precision (**verified**). The retained CSVs support the printed numerical digits (**verified**).

**Concern:** These digits describe a particular numerical run, not parameter, property, measurement or model uncertainty. The text does not always mark that distinction at the point of use, so numerical repeatability can be mistaken for physical precision (**inference**).

**Required direction:** Keep full precision in machine-readable records and diagnostic tables, but round headline physical results or add a clear statement that extra digits are numerical identifiers/diagnostics and carry no uncertainty claim.

## Positive checks and bounded claims

- The current PDF's page geometry, equations, table cells, figure labels and references were visually readable across all pages. No layout defect was found that would block review.
- The model responsibility table, flowchart and equations distinguish the nine-species liquid equilibrium from the reduced neutral vapor EOS and conventional film closure. This directly addresses the ambiguity in Reviewer 1 items 2–4 (**verified**).
- Appendix tables disclose all 36 off-diagonal pair coefficients, ionic/association values, reaction correlations, units, standard-state conversion and provenance classes. The appendix also honestly identifies coefficients and ranges that remain unrecovered (**verified**).
- The Case 3C refinement record is internally consistent: the retained confirmation gives 91.5483866223% capture, 347.8615451766 K peak liquid temperature, 0.0431383944 maximum normalized RMS residual, zero scaled boundary residual and a 273.006483 W net-enthalpy-flow range. The displayed changes agree with the retained coarse/refined pair (**verified**).
- The transport and thermodynamic perturbations are presented as bounded one-at-a-time changes, not statistical intervals. The article's interpretation appropriately avoids claiming a general uncertainty ranking (**verified**).
- The operating figure is transparent about the missing lower-flow condition and does not claim an optimum. This is a sound scope boundary, provided the reviewer response keeps its partial status (**verified**).

## Stage-1 action requests for the parent

1. Treat A1 and A2 as release-blocking evidence/claim decisions: either obtain the narrowly specified thermodynamic and energy evidence or narrow the manuscript's predictive and temperature language.
2. Keep the seven-case data and figures, but make coarse numerical completion, uneven physical agreement and qualitative temperature comparison visible beside the headline MAE.
3. Add the limitation sentence requested in A5 and mark all future film/model/amine/optimization work as proposals.
4. Package the exact source, wheel, inputs, figures and result identities under a stable public identifier before claiming peer repetition; preserve the measured timing wheel identity.
5. Retain the current rendered-word scan result: the current article has zero prohibited whole-word matches, while the pre-revision comparison does not.

This report is sealed at the hash recorded immediately after writing it. It must not be changed after Stage 2 begins.
