## Carbon-capture motivation restored — 2026-09-03

The abstract and Introduction again open with the emissions-reduction purpose of post-combustion capture before introducing MEA and the absorber-model question. The earlier flattened manuscript abstract supplied the power/industrial capture framing; the revised archive supplied MEA's reference-process role. Retained bibliographic abstracts for Harun2012a, Freguia2003, Mores2012 and Rodriguez2014 support combustion-emission reduction, MEA absorption, regeneration energy, and operating-condition effects. No contemporary deployment statistics or new literature claims were added.

The argument now proceeds from emissions reduction to solvent-based capture, separation/energy/operating needs, coupled absorber physics, the named literature review, and the controlled thermodynamic/film/numerical studies. The abstract closing sentence and Introduction closing bridge return those comparisons to capture-model selection and design. All prior citations and technical study scope are preserved; no results or completion scores change.

---

## Scoped supporting result insertion — 2026-09-03

The investigator authorized the two specifically recommended additions: the retained constructed isothermal Henry/eNRTL comparison and the local derivative-assembly verification. This selection uses the result files listed below directly; it does not adopt the producer notebook as a whole, its evolving ePC-SAFT attempts, or a new runtime/parameter set.

The isothermal result uses the 33-point initial-mesh entries (Henry 98.86120750035371%, eNRTL 96.73351866554502%) and the unchanged retained vector figure. Methods records both complete feeds, geometry, temperature and numerical settings. Figure labels and the primary missing-result markers are preserved. The plotted legacy-C3 label denotes this constructed control, not published Case 3C validation. The eNRTL parameter SHA-256 remains f8ed209c4e69da4a2da124d12238700284b7fdd7a80a0ceed4c0e1de8836ffdb; the JSON retains source-code and dependency identities.

The node result comes from test_full_native_column_node_jacobian, reported in the H² verification checkpoint: Engine 3375eadf58e32130d1556873879c18c73b36551c, wheel SHA-256 e1ccaf93b9e96b3e980b3db72535376dd4eece008f6015147d67ff6303b1f6a7. It is a synthetic local check, not the current installed Engine or a solved axial column. The inspected test defines the state, direction, step and denominator scale; no test or model was rerun.

The local submission supplement `supplementary/supporting-results.zip` preserves the supporting inputs/results, hash-verified eNRTL source model and calibrated parameters, and the synthetic derivative-check definition and errors. Its SHA-256 is `8ca8d87007347d518e98560fde59c1fa11b9cecc4218f17ddb2a2e74bfac8b80`. The broader primary-comparison input obligations remain open.

Retained source identities at insertion:

- `/home/tnnrpolley21/.codex/worktrees/b6f5/MEA-Absorption-Column/analyses/enrtl_comparison/inputs/comparison_inputs.json` — SHA-256 `6b560f077608c3b8e335df32fa3ebf63dd20966f560c9f6d3602117268730668`.
- `/home/tnnrpolley21/.codex/worktrees/b6f5/MEA-Absorption-Column/analyses/enrtl_comparison/results/henry/column_results.json` — SHA-256 `891cf9a37a901a3e44dd7744505c0a6172842a024d72d7197674f4104d5301cb`.
- `/home/tnnrpolley21/.codex/worktrees/b6f5/MEA-Absorption-Column/analyses/enrtl_comparison/results/enrtl_baseline/column_results.json` — SHA-256 `5874b3c97873d03b9e8314053d7ba06ab4a4e3e55cad42890dc8336c85b81b77`.
- `/home/tnnrpolley21/.codex/worktrees/b6f5/MEA-Absorption-Column/analyses/enrtl_comparison/figures/column_profiles/output/column_profiles.pdf` — SHA-256 `51a0c2e8b56592d21c3e96e4711ed00fc1c32e8864c4aa8ed43677c759b6a574`.
- `/home/tnnrpolley21/.codex/worktrees/b6f5/MEA-Absorption-Column/analyses/enrtl_comparison/figures/column_profiles/output/column_profiles.csv` — SHA-256 `9a9dda9989bc3febe6fac05df68c515d011552d0eeab181f44ef5b37ee16e375`.
- `/home/tnnrpolley21/.codex/worktrees/b6f5/MEA-Absorption-Column/analyses/enrtl_comparison/figures/column_profiles/output/sources.json` — SHA-256 `34b9dafde507043d4ec236727d71d8a88f7b027bc215a0cb11440813121d92cf`.
- `/home/tnnrpolley21/.codex/worktrees/6c99/MEA-Absorption-Column/docs/coordination/casadi_implementation_handoff.md` — SHA-256 `0f5ef3dd5d0142eac0c73660009e91377f8aa9ef753ba20c92d71c4552eca4ae`.
- `/home/tnnrpolley21/.codex/worktrees/6c99/MEA-Absorption-Column/tests/test_coupled_column.py` — SHA-256 `76459f6f927652b2eea655c45afca55d8ea1e10519f1b8ba42b466dc87d397cf`.

---

## Literature synthesis and comparison scope — 2026-09-03

The Introduction preserves all 21 citation keys from the archived post-March revised Introduction and retains the Gaspar2015 addition. Morgan2018a now explicitly supports NCCC multi-case evaluation alongside Morgan2020; Morgan2018 remains identified as the MEA-4 experimental-design report. The named Literature review separates rate-based evaluation, thermodynamic descriptions, and reactive-transfer/numerical comparisons. Source-specific synthesis reuses the archived literature and the abstracts retained in the read-only bibliography: Zhang2011 combines VLE, caloric and speciation inputs; Baygi2015 concerns PC-SAFT/MEA association choices; Najafloo2018 concerns SAFT-HR/MEA; Cleeton2020 and Bulow2021a concern MDEA/aqueous sour-gas solvent systems; Gaspar2015 concerns a reversible enhancement-factor treatment. These are thermodynamic or method studies, not evidence that this column has been validated.

NCCC capture and axial-temperature evaluation is explicit in the study scope, without reusing historical result counts or numerical outcomes. The original controlled absorber-comparison question connects the added parameter and film studies to the operating response. Results order is unchanged.

The nine-species statement now applies specifically to the ePC-SAFT bulk liquid and equilibrium-manifold film. Comparators retain their own declared species and reaction descriptions on common analytical feed bases. The current six-species eNRTL input record was inspected directly at b6f5/data/reference/enrtl/six_species_idaes.json; no notebook result was promoted or parameter set adopted. The six species and two combined formation reactions are documented explicitly in the formulation and input table from the reference record's species order, reaction names and conserved-amount balances. The abstract and study design identify the ePC-SAFT/eNRTL arm as a nine-/six-species complete-model comparison; closure-only attribution remains restricted to common chemistry. Abstract, Introduction, formulation, comparison-design table, Methods, Results attribution and conclusion consistently distinguish a complete thermodynamic-model effect from an isolated activity/fugacity closure effect, which additionally requires common species and reaction thermodynamics. Final model-specific input tables and result conditions remain their existing insertion obligations.

---

## Current article order and cohesion — 2026-09-03

The investigator selected Results 4.1 thermodynamic model comparison, 4.2 thermodynamic parameter sensitivity, 4.3 film transport and mobility response, 4.4 numerical accuracy/cost, 4.5 LHC operating response, and 4.6 engineering interpretation. This order supersedes the section numbers in the historical preparation notes below. Thermodynamic sensitivity remains a results contribution, immediately following the model comparison.

Methods 3.3 contains thermodynamic/LHC design; 3.4 contains the relocated film-group/thickness perturbations, input table and sensitivity measure. Numerical formulations and settings are 3.5 and 3.6; performance definitions and refinement/timing are 3.7 and 3.8. The accuracy/runtime result table is in Results 4.4.1. Stable paragraph IDs and figure/table/checklist labels survive relocation. The introduction, Results transitions, synthesis and conclusion follow this same argument; the named literature review and property appendix are retained.

The generated `builds/manuscript-story-map.md` indexes paragraph IDs and sentence source lines against each section's scientific purpose. It is a derived review view; source paragraphs remain authoritative. The grouped conclusion checklist now includes the local operating/engineering finding. The checklist remains 33 rows: no empirical obligation was closed by rearranging prose. Reviewer scores and original quotations are unchanged; changed source evidence was reviewed for preservation of the previously addressed scope.

---

## Sensitivity design and Case 3C manuscript preparation — 2026-09-03

Methods now contains the agreed nine-calculation thermodynamic design, two five-reaction enthalpy directions, the pivot-preserving ln K perturbation, and the 48-point Latin hypercube design (15 factors, seed 1803). The two new checklist rows distinguish completed method/range definitions from unevaluated result figures. Results 4.4 separately reserves thermodynamic sensitivity and operating-response figures and numerical interpretations; integrated interpretation moves to 4.5. No new numerical result is asserted.

The design authority is the user-selected study in `/home/tnnrpolley21/.codex/worktrees/1b26/MEA-Absorption-Column/analyses/nccc_validation/r18_sensitivity.md` and its `inputs/r18_sensitivity/{factors.csv,thermodynamic_runs.csv,sources_and_design.json}`. Printed reaction directions retain approximately 12 significant digits; source JSON preserves full precision. Exploratory ranges are not covariance, measurement error, or probability distributions. The independent species-group/thickness design remains separate and still needs its own selected ranges.

Case 3C Methods uses the source audit in that design record: Morgan2020 Table A1 pp.18–19, Table6 p.15, and TableC2 p.27. Dry CO2/O2 are .093/.090; 318.15 K lean inlet is assumed because the measurement is NA. Gas mass flow is treated as total wet mass by explicit model convention. Source temperature coordinate ζ is transformed to the manuscript gas-inlet-origin coordinate as z/H=1−ζ, keeping the temperature pairs intact. The source record identifies Zotero parent 5D3TJVWU / PDF child HX2358GV and its PDF hash. These definitions supersede the unresolved Case3C coordinate/input note below, not other campaign cases. Final run exports must match these inputs.

Reviewer R1.8 and R2.5 now have the design and analysis prose written; their internal scores rise to 5/10 while both remain In progress. Completion still requires their evaluated responses. Current eNRTL input-table and final implementation/data identities remain separate outstanding obligations.

The supervising task subsequently reported independent eNRTL re-review passed. The retained b6f5 handoff confirms the existing imposed-313.15 K, conventional-film Henry/eNRTL comparison and its unchanged column results. Its inputs are a constructed legacy-C3 case, not the corrected published Case3C used by the new sensitivity design. This supports a bounded supporting comparison only; it does not fill the current three-arm/nonisothermal plots, R1.8 sensitivity, or R2.4 measured-case validation. The supervising task subsequently confirmed the missing primary ePC-SAFT/eNRTL comparison and resumed the eNRTL owner to add the current nine-species ePC-SAFT arm under the same imposed-temperature and conventional-film conditions. The main thermodynamic comparison therefore remains open. No numerical result from that notebook was inserted in this prose pass. Exact notebook promotion and the final selected eNRTL distribution/species mapping remain separate from its result review.

# Manuscript sources and remaining insertions

Updated 2026-09-03 for the full three-study manuscript. This author working document is not included in the paper. `main.tex` is the sole manuscript entry point; `builds/main.pdf` is its output. There is no Quarto manuscript or Quarto helper in this transfer.

## Current scientific intake — 2026-09-03

This snapshot includes the orchestrator handoff, the active absorber successor, and the eNRTL notebook. The original ac57 input snapshots below retain their historical identities. It supersedes older runtime descriptions below. Manuscript writing remains in the local `codex/reactive-film-overhaul` checkout; no solver work or result promotion was performed.

| Owner / current location | Evidence available | Still expected for this article |
|---|---|---|
| Coupled absorber, **Takeover: coupled absorber and A2 integration**, task `01a069ef-d021-7050-abc7-d38ddb17425a`; `/home/tnnrpolley21/.codex/worktrees/6c99/MEA-Absorption-Column` | Twelve-state native value assembly and signed film quadrature; orchestrator reports focused checks and independent re-review. Successor reports installation of the immutable A2 wheel and a passing final integration check; handoff: `docs/coordination/casadi_implementation_handoff.md` | Complete assembled outer derivatives and coherent column outputs; paired thermodynamic/film, refinement, sensitivity, and cost exports |
| A2 derivative actions, task `01a0695f-67e5-7212-9012-87dd6651f3e9`; `/home/tnnrpolley21/.codex/worktrees/36be/ePC-SAFT-project` | Orchestrator reports reviewed delivery, 104 tests and the 178-action study passed: Engine `22b210bb96b0bf7f125fc4245db0ad6e9bceabed`, wheel SHA-256 `c71c2bd701904bc58390f36fc5a18d9e133ec4a2e72c9055ea78482fc06164f4`; draft PR 210 stacked on 209, no merge reported | Absorber consumption and final run identity; capability checks are not column accuracy/cost results |
| Engine calorics/orchestrator, task `01a06974-7345-75a2-95b3-8e3ca066d7c7`; `/home/tnnrpolley21/Workspaces/Engineering/ePC-SAFT-project` | Total reference-plus-residual enthalpy and fixed-composition Cp in `engine/src/epcsaft/eos.py`; polynomial reference functions in `thermochemistry.py`; source transfer in `reference.py` | Second total-H actions for outer derivatives of vapor Cp and vapor partial molar enthalpies, plus the immutable identity used by the completed run; no nine measured ionic Cp values or new caloric solver are required |
| Thermodynamic fit, task `01a06835-c702-7d72-8e10-792227fee3d7`; `/home/tnnrpolley21/Workspaces/Engineering/MEA-Thermodynamics` | Selected parameters and continuous neutral-anchor caloric reconstruction; retained files include work after the old task messages | Any replacement joint-fit parameters, their retained validation, and matching thermal reconstruction; no adoption inferred from a candidate file |

### Selected reaction and caloric tables — updated together

The manuscript now uses the absorber's retained `MEA_reactive_epcsaft_2026_09_03` bundle under `/home/tnnrpolley21/.codex/worktrees/ac57/MEA-Absorption-Column/src/mea_absorption_column/data/epcsaft_datasets/`. Direct inspection verifies that `Thermodynamics/thermo_models.py` selects this dataset by default and `reactive_bundle.py` substitutes typed fitted correlations as complete records with zero additional offset. This is parameter reporting, not a completed column result.

- `parameters.json` SHA-256: `a9186c93759f2e2c02a6c913350ad06a244fff3f82503820c9962b3df8dd40d9`.
- `reaction-system.json` SHA-256: `810dfec15760cf74451df91743d6e63684cee93ddaf3e1ff4e42bf4a686afe29`. R1/R3 retain their source coefficients and offsets. Typed parameter records replace R2/R4/R5; R2 a already includes the molality conversion, so its separate offset is zero.
- `anchored-reference-thermochemistry.json` SHA-256: `a24a6b3c8b506fc659fc1bbd8a470b55919ba93da23eea27ffdf882645706185`; scientific fingerprint: `sha256:6919acbc3f1125b89363dd6997fcfd936036854768cd4c47d845d0133c8b0648`.
- Reference-generation Engine commit: `8438ce5f94a547189c91c4ec180a7782d60879d6`; wheel SHA-256: `40fba7cfb9c8414152f3e49636c49ae2e3f7099e30040d54d464ccb38355f805`. This identifies reference construction, not necessarily the later column execution wheel.
- `adoption-receipt.json` SHA-256: `bf178ed7b043e90dcca297ba9b80f1b262ef41bcdfe5e53af9211a1fe8b237e6`; decision `adopted_as_exploratory_incumbent`. Its `adopted_parameter_sha256` names the new document; incumbent fields name the previous one.

All nine reference enthalpies and 72 Cp coefficients were replaced together with the selected reaction coefficients. Printed coefficients use theta = (T − 353.15 K)/(100 K), so coefficient k is the retained coefficient multiplied by 100^k. Cp is integrated analytically for h. The reference interval is 293.15–393.15 K, anchor pressure 101325 Pa, construction grid 2.5 K, and representation degree-8 enthalpy/degree-7 Cp. EOS pure/pair parameters are unchanged from the preceding snapshot.

The upstream producer is `/home/tnnrpolley21/Workspaces/Engineering/MEA-Thermodynamics/analyses/mea_parameter_bundle/results`. Its selected parameter and caloric files match the retained copies above. Matching `calorimetry/thermal-reference-validation.json` SHA-256 is `a8089c3acbce93ca63d0f12d5e2f4f660615ea12b8ee4a19067401485dcfbca8`. Construction is owned by `analyses/mea_parameter_bundle/scripts/evaluate_direct_absorption_heat.py`; transformed reaction enthalpy includes source-reference and activity-basis transfer derivatives, not just the raw source-correlation derivative. Source-domain review remains for Hilliard anchors and primary NIST formation records.

### Prior table and result identities

The superseded printed table used parameter `00049473d53c7e8088ef3e2dbbc6a1bab058f6dc4de963ee98936b4cd9bda25e` and caloric fingerprint `sha256:a0440b08f0c3fad43f753f973568923e88a6ee6f94f523f6c3838c86b7be3b29`. That parameter remains in this checkout's older `MEA_reactive_epcsaft_bundle`; no local solver dataset was changed during this manuscript-only update. Existing figures and analysis records retain their original run identities. They are not relabeled as new-bundle results. New result deliveries must identify their actual parameter, reaction, reference, and execution Engine identities together.

### Specific input questions, not missing prose

- **Temperature taps:** `src/mea_absorption_column/data/C_cases_campaign_inputs.csv` stores legacy kelvin profiles at labels 0, 0.2, …, 0.8 treated as normalized coordinates. `NCCC_2017_absorber_temperature_profiles.csv` stores six Celsius profiles at labels 0.20, …, 1.00 without retained measurement-height metadata. Neither label set alone establishes physical height or orientation. The result owner must identify the source coordinate convention before temperature RMSE/peak comparisons. Do not combine these two lanes. Per-tap uncertainty is not retained; do not manufacture error bars.
- **Case/feed basis:** one-bed scope is 2014 K18–K20 and 2017 1C–7C; D cases are two-bed. The historical accepted table excludes K20. Corrected 2017 inputs are `NCCC_2017_model_inputs_mass.csv`; 1C–3C use the recorded 45 °C lean-temperature imputation. Preserve `reported_total_wet` mass flow and the selected dry/saturated composition conversion. `NCCC_2017_cases.csv` gives capture standard deviations 0.7, 0.5, 1.2, 0.8, 0.6, 0.5, 2.2 percentage points for 1C–7C; no equivalent 2014 capture SD is retained.
- **eNRTL:** Task `01a06992-3aa7-70b3-8471-6126cba14ef3` supplies `docs/coordination/enrtl_comparison_handoff.md`, `data/reference/enrtl/six_species_idaes.json`, and `analyses/enrtl_comparison/notebook.qmd` with its HTML companion under `/home/tnnrpolley21/.codex/worktrees/b6f5/MEA-Absorption-Column`. The calibrated source equilibrium solver and eight six-species reference states exist. The experimental source adapter now also supplies this repository's matched C3 isothermal comparison at imposed 313.15 K on two meshes: `analyses/enrtl_comparison/inputs/comparison_inputs.json`, `results/henry/column_results.json`, `results/enrtl_baseline/column_results.json`, and `figures/column_profiles/output/column_profiles.csv` (result paths relative to that analysis). This is thermodynamic-package sensitivity with common conventional transport and the retained `legacy_ratio` feed conversion; it is not EOS-only attribution or validation against nonisothermal C3 measurements. Independent re-review supported the retained control; the investigator subsequently authorized its scoped quantitative insertion, as recorded at the top of this source map. Baseline `Parameters_fit.csv` SHA-256 is `f8ed209c4e69da4a2da124d12238700284b7fdd7a80a0ceed4c0e1de8836ffdb`; sensitivity `Parameters_fixed_bicarbonate.csv` is `218857a37c48c819d11ef4b0778ffa54500138ec54254f63d6d2235b35276933`. Portable interaction/reference records and the adopted six-/nine-species reaction mapping remain to document. Zero standard-enthalpy arrays in the older fixed-composition evaluator are not physical caloric inputs; that limitation does not imply the calibrated source equilibrium solver is absent. The b6f5 manuscript remains historical and does not replace this local three-study paper.
- **Enhancement:** `Transport/Enhancement_Factor.py` actively uses the Luo-labeled 2.003e4 exp(−4742/T), 4.147 exp(−3110/T) pair after overwriting Putta coefficients. E alone is limited to [1, 10000]. The divisor 1.04542981654115 has no retained derivation or source locator; its meaning/replacement is an absorber-owner scientific question. Keep the written implementation explicit and update it if the owner changes it.
- **Mobility:** `analyses/reactive_film_evidence/inputs/diffusion_anchor_assumptions.csv` and `tables/reactive_film_parameters.tex` retain source anchors and estimates. Polat covers 293–353 K, 10–50 wt% MEA; Melnikov's loading correction is anchored near 313 K/30 wt%; Jerng values concern dilute MEA/D2O. These do not constitute a measured nine-species Onsager matrix. Range choices in the sensitivity table must be tied to the executed variations.
- **Pressure/gas:** the article's constant-pressure, four-component ideal-gas/common-gas description agrees with the corrected absorber direction. Keep it for controlled liquid-model comparisons. Any full gas-EOS extension requires explicit gas parameters and phase reference consistency; no complete O2/N2/CO2/H2O EOS/caloric delivery was identified.

### Incremental result intake

The first completed representative coupled case can fill the relevant profile panel and its scoped finding immediately. Restrict its caption to the delivered case/arm. Add paired arms and the case sweep as they arrive. Each delivery needs case inputs and basis, calculated profiles/capture from the **same run**, parameter/reaction/Engine identities, numerical residuals/refinement, figure-ready values, and a concise supported finding. A branch merge is not required to read these files. Copy final figure assets into `docs/latex/figures/`.

Every figure/table row now carries its question, quantities/units, cases/arms, observation/calculation distinction, numerical evidence, owner, input location, and interpretation in the existing checklist Details. No new tracking service exists. Exact filenames of the new coupled exports have not yet been delivered; replace the explicitly requested input descriptions with their real paths when received.

Existing filenames are useful retrieval anchors, **not evidence that their values are corrected**: `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`, `method_case_contrast.csv`, and `analyses/reactive_film_evidence/results/final/tables/column_film_{capture_comparison,axial_profiles,temperature_metrics,run_provenance}.*`. The local and ac57 film capture files differ (SHA prefixes `4131aac3` and `33c1e7df`), and their provenance/status differs. Do not mix them, or promote either set as the completed coupled study. Historical energy-sign-affected outputs remain excluded.

## Reviewer restoration and input documentation — 2026-09-03

The investigator authorized restoration from the superseded revised manuscript and completion of supported input documentation. The historical baseline order is the March submission (investigator-identified), the superseded revised manuscript preserved in `legacy/manuscript-pre-reactive-film-2026-09-03/latex-source-and-build.tar.gz`, and the current article. The August submission-package filename is not evidence that it was the March submission. The existing Before scores retain their original assessment source; no fresh March PDF score is inferred.

Carried forward and adapted: the named Literature review, rate-based/model-selection synthesis, practical model strengths and limitations, definition of the ePC-SAFT driving-force comparison, qualitative transport uncertainty, and solvent-extension/applicability discussion. Archive source locations are `latex/sections/introduction.tex`, the Transport subsection of `latex/sections/model_framework.tex`, and `latex/sections/conclusion.tex`. Current text retains nine species, five reactions, the equilibrium-manifold film, and the three controlled studies. No old capture, timing, or validation values were copied.

Input additions use the selected `a9186c93` parameter document named above. Its `pairs[].coefficients[].provenance` distinguishes the MEA/water neutral-VLE fit, the fixed CO2/water and ion/water assignments, and the historically retained MEAH+/MEACOO− value. Source IDs `cai-1996-neutral-refit`, `mea-best-in-slot-campaign-2026-09-02`, and `mea-reaction-temperature-fit-2026-09-03` identify the neutral-mixture fit, CO2 dispersion selection, and R2/R4/R5 temperature adjustment. The reaction-temperature objective uses five pressure targets (including Jou 1995 at 120 °C), 22 speciation targets, and six Kim–Svendsen 2007 calorimetric intervals. Xu 2011 pressure and 120 °C calorimetry are excluded from that objective; Kim 2014 is a model-selection comparison. Full-replay evaluation cohorts are not the fitting objective. These are fitting roles, not asserted validation outcomes.

The explicit solvent dielectric equation follows `engine/src/epcsaft/_native/electrolyte.hpp`, `dielectric_solvent_component` and `mixture_relative_permittivity`: neutral components with supplied permittivity contribute `x_i M_i`; the solvent-only branch returns dielectric mass divided by solvent mass. The selected document supplies permittivity for neutral MEA and water, giving their normalized free-species mass fractions. CO2 and ions are excluded. This input description does not change an installed Engine or select a different parameter set.

The remaining reproduction inputs are the selected portable eNRTL interaction/reference records and species/reaction mapping, final numerical settings, complete primary source locators, and final software/data identities. Those obligations remain in the existing input/checklist rows. Source dates and file identities remain author notes; development status does not enter the paper.

Reviewer closure follows each original request: qualitative transport discussion closes R1.9 without requiring the optional sensitivity figure; a described solvent-fitting/evaluation procedure closes R1.12/R2.6 without requiring another-solvent results. Numerical and operating studies remain open. Reporting methods now explicitly request CPU time, peak process memory, iterations/evaluations, residual/refinement histories, mesh size, and hardware alongside wall time.

Bibliography rendering suppresses exported research-note fields through `main.tex`; the Zotero-owned bibliography remains unchanged. This prevents the private Bülow reference note from appearing in the article.

## Live checklist for authors and agents

Run `python3 docs/latex/scripts/manuscript_checklist.py --serve` from the repository root and open the printed local address. The page rescans every three seconds. An offline snapshot is generated by the same command with `--html`; `--json` returns the live item states for any agent without a browser.

The 29 content rows are in `scripts/manuscript_checklist.json`. They group the original eight figure/result-paragraph pairs and the three conclusion findings, and include the already-written Introduction. All original result obligations are retained in the grouped rows. This is a missing-content checklist, not a submission audit: a complete abstract or methods passage does not wait for final result numbers or source reconciliation.

Stable `% CHECKLIST-BEGIN: id` / `% CHECKLIST-END: id` comments identify editable passages and table locations. Preserve those markers and figure labels. Replace framed figure panels and bracketed table cells with retained values. A figure's box checks only after its linked result paragraph is also filled; the conclusion box covers all three findings. A table must be present with its assigned label and contain no bracketed value placeholders. Final source/parameter checks are stored in each row's `review` field and in nonprinting `CHECKLIST-REVIEW` comments; they remain visible in Details without holding completed writing open. `CHECKLIST-REMAINING` is reserved for genuinely absent content.

The page can filter to **Figures & tables**. Checked means the scoped content is written or inserted; it does not certify scientific validity. No unknown outcome, parameter, or solver setting should be invented to close a row.

## Figures: what will replace the eight panels

All eight result figures and their captions are owned by `sections/results.tex`. Replace each framed panel with its final figure under `figures/`, retaining the label. The exact final plotted-value file, renderer command, and input identity must be recorded here when supplied; no older result file has been designated as the source of these new comparisons.

| Current figure / label | Final display to add | Associated finding to insert |
|---|---|---|
| 2 / `fig:thermo-capture` | Henry, eNRTL, and ePC-SAFT predicted versus measured capture; signed error by case, on the same case set with common properties | Capture MAE and signed bias for each model, largest/smallest case differences, and supported ordering |
| 3 / `fig:thermo-profiles` | Axial apparent loading, molecular CO2, carbamate/bicarbonate distribution, and gas/liquid CO2 fugacities | Species and axial regions explaining the thermodynamic difference, with fugacity/loading changes |
| 4 / `fig:full-property` | Paired common-property/full-property capture changes; liquid temperature and volumetric flow; enthalpy and absorption distribution | Additional capture change, temperature-maximum change and position, and temperature-tap RMSE |
| 5 / `fig:film-capture` | Enhancement-factor/reactive-film capture, casewise capture change, and gas/liquid resistance | Capture-change range, aggregate errors, and conditions giving the largest transfer-model effect |
| 6 / `fig:film-profiles` | Axial CO2 flux, interfacial fugacity, cumulative uptake, and gas/liquid temperatures | Dominant resistance, location of the largest flux change, and redistribution of absorption |
| 7 / `fig:film-sensitivity` | Capture response to diffusivity groups; capture/temperature response to film thickness | Perturbation ranges, influential inputs, effect sizes, and persistence of the transfer-model comparison |
| 8 / `fig:solver-accuracy` | Conservative simultaneous/shooting/finite-difference/collocation profiles, mesh and film-quadrature refinement, integral material/energy balance | Agreement in capture and temperature, refinement changes, and material/energy/charge conservation at the reported settings |
| 9 / `fig:solver-cost` | Runtime versus achieved accuracy and measured equilibrium/film/column-solve cost contributions | Median runtime and spread, evaluation counts, final mesh sizes, and operating-point dependence |

Figure 1 is already a native TikZ diagram: `figures/tikz/model-framework-flowchart.tex`, styled by `figures/tikz/styles.tex` and included by `sections/model_framework.tex`. It describes the revised scientific formulation; the pilot's older chemistry and iteration sequence were not copied back.

Final result figures require quantities and units, explicit case/series identities, observation markers distinguished from calculated curves, readable legends, and differences distinguishable in grayscale. Captions will identify the exact comparison conditions and case subset. The current numbered panels are author placeholders, not calculated plots.

Result selection note for the author: the absorber-repair task reported an inherited countercurrent energy-sign error on 2026-09-03 and is correcting the runtime. Use rerun and reviewed column/temperature results from the corrected energy equations for these panels. Historical affected calculations remain preserved and must not be relabeled as refreshed results. This note is outside the manuscript.

## Remaining prose

The eight findings above correspond to eight bracketed passages. One additional bracketed passage in `sections/results.tex` needs the cross-study synthesis: relative thermodynamic, transfer, and numerical effects; the supported model/method choice; and the capture/temperature evidence for that choice.

Three bracketed conclusion passages in `sections/conclusion.tex` share one checklist row and need the principal thermodynamic, transfer/sensitivity, and method/runtime/accuracy findings. The abstract is complete as a description of the problem, formulation, study design, and methodological contribution; its final editorial pass can incorporate the principal numerical findings. That enrichment is recorded as a review note, not missing abstract content. These statements reuse the final results, not additional studies.

## Exact method, parameter, and reproducibility additions

The table below retains the original detailed obligations as a reference for final reconciliation. The live JSON distinguishes completed writing from the four tables still requiring values and from source-review notes. This historical inventory is not a list of wholly unwritten sections.

| Item | What will be added or completed | Manuscript source |
|---|---|---|
| Conserved basis | Written: two material rows B, the Engine-enforced molar-mass row, fixed-zero charge, and normalized feed mapping; reconcile final species/mass identity | `sections/model_framework.tex`, equilibrium subsection |
| eNRTL comparison | Selected activity equations, solute/solvent reference transformation, parameters and parameter identity; explicit common gas fugacity convention | `sections/model_framework.tex`, thermodynamics; `appendices/appendix_properties.tex` |
| Interfacial energy | Selected phase and temperature for transferred-material partial molar enthalpy, its exact definition, and the matching sensible-transfer closure | `sections/model_framework.tex`, heat transfer |
| Enhancement reference | Exact enhancement function, reaction-rate coefficients, concentration units, source and temperature/composition domain | `sections/model_framework.tex`, enhancement subsection |
| Case and observation definition | Final case set for each study, wet/dry gas composition and four-component conversion, observed-temperature tap positions and available measurement uncertainties | `sections/methods.tex`, operating cases; result captions |
| Common model inputs | Packing height/area/voidage/hydraulic diameter, packing and transfer coefficients, fixed operating/model parameters, and the parameters changed in each study | `sections/methods.tex`, common model-parameter table |
| Numerical formulations | Exact finite-difference endpoint stencil, selected shooting integration/root and nonlinear algorithms, collocation implementation, initial meshes and physical scaling values | `sections/methods.tex`, BVP formulations and scaling |
| Solver settings | Equilibrium, interface-root, quadrature, integration, boundary and column tolerances; mesh/node limits; repeat count; timing scope, hardware and thread configuration | `sections/methods.tex`, settings/refinement/timing table |
| Numerical comparison table | Final settings alongside refinement changes, conservation/accuracy measures and timings, all on identical equations | `sections/methods.tex` and the solver results |
| Thermodynamic parameter record | Complete selected ePC-SAFT and eNRTL pure/pair parameters, association topology/interactions, options, sources and applicable ranges; verify the existing nine-species table against final inputs | `appendices/appendix_properties.tex`; `tables/reactive_film_parameters.tex` |
| Reaction constants | Verify the existing five-reaction coefficient table against the final reaction system; complete standard-state offsets/conversions, sources and each source temperature domain | Same appendix and parameter table |
| Nine-species calorics | Inserted: three neutral anchors, five transformed reaction constraints, ionic reference convention, nine reference enthalpies and Cp polynomials; reconcile the final runtime identity and source ranges | `appendices/appendix_properties.tex`, caloric subsection |
| Physical-property coefficients | Complete the surface-tension, viscosity, diffusivity, sensible-heat-capacity, thermal-conductivity, holdup and gas/heat-transfer definitions and coefficient sets used in the final common-property calculation, with sources, units and domains | `appendices/appendix_properties.tex`; transport subsection in `sections/model_framework.tex` |
| Mobility inputs | Verify each existing species diffusivity against final inputs; add its measurement/estimation source and range, and the distinct physical diffusivity defining film thickness | `tables/reactive_film_parameters.tex`; appendix |
| Sensitivity specification | Exact perturbed input groups, multipliers/ranges and rationale, fixed conditions, and whether the variation affects mobility or the thickness-defining transfer coefficient | Film sensitivity subsection and its figure caption |
| Code identity | Archived absorber revision, immutable Engine commit and wheel SHA-256, and exact parameter identities used for the figures | `sections/code_availability.tex` |
| Data identity | Final deposit/revision containing exact plotted values, case inputs, casewise analyses and figure provenance | `sections/data_availability.tex` |

After table and figure insertion, reconcile the case captions and parameter identities, replace the linked bracketed findings, enrich the abstract/conclusions with the supported results, and preserve the CHECKLIST-BEGIN/END marker comments. Scientific validity ranges remain scientific definitions; development-state descriptions do not enter the manuscript.

The activity subsection now defines the excess-Gibbs chemical potentials and reference conversion without substituting the repository's neutral three-component NRTL routine for electrolyte NRTL. The electrolyte-specific interaction expression/records still belong to the additional thermodynamic-input table. The gas-side transferred-enthalpy convention follows `src/mea_absorption_column/Transport/Flux.py` and the gas reference integral in `Properties/Thermophysical_Properties.py`. Neither addition asserts a new computed comparison result.

## Tables and their present owners

| Tables | Editable source | Source of content |
|---|---|---|
| Thermodynamic descriptions | `sections/introduction.tex` | Cited literature comparison |
| Controlled study design and common column inputs | `sections/methods.tex` | Paired comparison definitions and current packing/geometry constants |
| NCCC operating inputs | `tables/nccc_one_bed_case_scope.tex` | `analyses/nccc_validation/results/final/tables/nccc_one_bed_case_scope.csv`; measured input/capture record |
| Species basis, reaction coefficients/ranges, pair and association parameters | `appendices/appendix_properties.tex` | Current reactive bundle; primary-source gaps remain in the checklist |
| Apparent-volume, surface-tension, viscosity, heat-capacity and conductivity coefficients | `appendices/appendix_properties.tex` | Existing manuscript equations reconciled against current property modules |
| Nine-species pure parameters and species diffusivities | `tables/reactive_film_parameters.tex` | Current parameter document and explicit mobility estimates |

| Numerical settings and accuracy/runtime comparison | `sections/methods.tex` | Prepared table layouts; populate from the paired run settings and numerical results |
| Sensitivity multipliers and range rationale | `sections/results.tex` | Prepared input table; populate from the executed sensitivity design |
| Additional eNRTL/dielectric inputs | `appendices/appendix_properties.tex` | Prepared parameter table; populate from the selected electrolyte model records |
| Nine-species calorics | `appendices/appendix_properties.tex` | Inserted selected v2 neutral-anchor reconstruction; exact identity and remaining runtime/source reconciliation above |

Table numbering is generated by LaTeX and follows the current PDF.

Paths beginning `analyses/` or `src/` are relative to the repository root; other paths in this document are relative to `docs/latex`. No scientific generator is invoked by the manuscript build. Older result plots and six-species/solver-status tables remain outside the active input tree and do not supply the new figures.

## Visual-pilot carryover

Compared against the referenced task “Quarto MEA manuscript visual pilot” and the actual working tree on `codex/quarto-latex-pilot`, whose committed tip is `0d4e552` and whose later presentation work is uncommitted.

Already retained: CAS class/title/page furniture, native TikZ geometry and shared color/arrow styles, booktabs, siunitx quantities and numeric columns, threeparttable operating-input notes, portrait layout, real caption-owning floats, microtype, xurl, and bibliography wrapping. Older redesigned tables remain on disk; their obsolete scientific contents are not reintroduced.

Added in this follow-up: semantic ordinary/partial derivative commands, an upright integration differential, semantic linked references through cleveref, and this updated source-and-insertion record. Existing equation operands, fixed-variable subscripts, chemical species, units, values and labels are preserved. The standalone diagram build can use the existing embedded TikZ source without a second drawing.

Quarto sources, renderers, configuration, boundary-check scripts and a second manuscript edition are excluded by the investigator's instruction. PGFPlots and algorithmicx were not used by the pilot and are not missing ports. The pilot's old state-symbol aliases and shaded-region style are not copied where the revised equations/diagram do not use them. CSE visual QA is run on the actual CAS PDF; its result and manual disposition belong in `QA_REPORT.md`.

## Targeted methods completion — 2026-09-03

The ten-item writing pass preserved the current manuscript structure and reused applicable material from the archived `latex/sections/model_framework.tex` (lines 196–277), `latex/sections/methods.tex` (lines 17–113), and `latex/appendices/appendix_properties.tex` (lines 6–173). The archive remains unchanged. Numerical-method and property prose already transferred in the full-paper revision was retained. The older kinetic coefficient, density expression, heat-capacity typo, gas-viscosity forms, and run-specific settings were not reinstated.

The conserved matrix and common one-bed inputs are filled. Enhancement, case conversion, endpoint stencils, reaction ranges, ePC-SAFT pair/site data, property coefficients, mobility assignments, and sensitivity procedure are drafted. That earlier pass treated final selections and source checks as drafting blockers. The content-focused pass above supersedes that counting rule; it preserves those checks as review notes and places missing numerical inputs in named tables.

Evidence owners: `src/mea_absorption_column/config/Constants.py`; `misc/Convert_Data.py`; `Transport/Enhancement_Factor.py`, `Hydraulic_Variables_Correlations.py`, and `Transfer_Coefficients.py`; `BVP/Methods/Finite_Difference_Solve.py`; `Properties/Transport_Properties.py` and `Thermophysical_Properties.py`; and `data/epcsaft_datasets/MEA_reactive_epcsaft_bundle/{reaction-system,parameters,bundle}.json` under the package. The wetting-area equation now matches the implemented exponent 0.12 and the factor of void fraction in the packing-length definition. The previously written viscosity expression was rechecked and retained.

Mobility source anchors remain in `analyses/reactive_film_evidence/inputs/diffusion_anchor_assumptions.csv`; complete primary citations and source domains remain open. The sensitivity group description is an author-defined procedure; final group membership and multipliers must match the executed campaign. The current ac57 callback maps normalized apparent feed through the complete Engine invariant matrix; the older trace-seeded adapter is not the intended final derivative formulation. No result numbers, final run settings, or final software/data identities were inferred from historical calculations.
