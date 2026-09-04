# Fallback manuscript: reviewer response and coverage

Updated 2026-09-04. Author working record; not part of the article or a submission authorization.

## Edition and decision

This edition restores the superseded post-March revision from the top-level `latex/` tree in `legacy/manuscript-pre-reactive-film-2026-09-03/latex-source-and-build.tar.gz` in the read-only source checkout. The archive SHA-256 is `3dc5903c233935093bfb61a29c7f8a18a9aad6bfb0e1d63900064e20c1cfe633`. Its manuscript sources matched this worktree's starting commit `32d5a3baeb1454e1b5eb9237515e0d26b3381690` before editorial changes. Nested August submission packages were not used. The investigator identifies the original submission as end of March; its exact submitted PDF has not been verified here.

The editable fallback is `docs/latex/main.tex`, with the complete sections, appendix, figures, and tables. Its PDF is `docs/latex/builds/main.pdf` in this isolated worktree on `codex/fallback-manuscript`. The original checkout, current reactive-film manuscript, live checklists, and server remain untouched.

**Not submission-ready.** The locally retained selected nine-species Case 3C has a refinement check, eight ±5% thermodynamic sensitivity calculations and six ±10% transport perturbations with a matched reference. R1.8, R1.9 and R1.10/R1.11 numerical reporting are included in the manuscript. The seven-case selected-parameter NCCC package is now integrated, addressing R2.4 with capture and packing-temperature comparisons; its substantial 6C/7C errors remain visible. Historical aggregate figures remain separate. No parameter fit, commit, push or submission was performed.

Scores assess current response readiness, including verified supporting runtime work, not just text present in the PDF. Completed explanations remain completed; parameter disclosure can be complete independently of the still-open numerical validation. Exact reviewer wording is preserved. Historical Before scores are not reinterpreted as grades on the exact March submission.

## Manuscript-to-calculation reconciliation — 2026-09-04

The main reaction subsection now presents all five reactions and nine species; reduced chemistry is grouped with historical parameter tables. Equations state the implemented void-fraction dependence, bounded holdup, hydraulic diameter, phase-average molar masses, vapor enthalpy/conductivity mixtures, actual material derivatives, temperature-state chain rule and boundary conditions. Pressure drop is identified as a diagnostic, and legacy penalty handling is not attributed to the coupled calculation.

The selected Case 3C profile figure accompanies its refinement and sensitivity evidence. The seven-case reactive temperature gallery replaces the historical six-case gallery. Historical aggregate statistics and solver comparisons remain separately scoped, including the dry-to-wet Case 3C inlet conversion. Selected reaction constants, parameter provenance and the historical August release boundary are explicit. Traceability maps each figure to its retained source; freshness is not a scientific check. The title, headings, AI disclosure, bibliography and numerical data are unchanged. R2.4 multi-case integration is complete; R2.5 operating trends remain open.

## Revision priorities

1. R1.8 and R1.9 are complete for the bounded Case 3C studies: four thermodynamic inputs at ±5% and three transport inputs at ±10% are reported in the manuscript. Wider or joint studies would extend these local comparisons, not supply parameter uncertainty without measured or fitted input uncertainty.
2. Seven-case NCCC capture and temperature comparisons are integrated. Reconstruct source-linked SRP cases separately if extending facility coverage. Intercooled NCCC cases remain outside this immediate work.
3. R2.5 controlled operating-condition analysis remains the next calculation; broader interaction/reaction uncertainty is a subsequent extension of the R1.8 screen.
4. The selected Case 3C reference, refinement and sensitivities are integrated with their parameters and equations. The seven-case physical figures use matching nine-species results; the historical fixed-chemistry aggregate remains a separate comparison.

The 45 °C inlet assumption is accepted and needs only an input-table note, not a standalone blocker or repeated warning.
Exhaustive mesh histories, a new facility campaign, full uncertainty propagation, another amine implementation and a new parameter fit are not prerequisites for this revision.
The explicit requests for numerical evidence and operating trends remain open where no calculation has been done; modest checks can answer them without turning each into a separate project.
The selected input tables remain in [the supporting parameter record](selected-reactive-parameters.md), separate from the historical parameter tables. New working numerical evidence is retained in [the energy/column check record](../analyses/nccc_validation/results/reviewer_energy/README.md); no historical manuscript figure or numerical result was overwritten.

## Corrected-column progress

![Case 3C refinement and four-case capture comparison](../analyses/nccc_validation/results/reviewer_energy/comparison.png)

The left panel compares retained Henry liquid-temperature profiles, not observations. The right panel compares calculated capture with retained NCCC observations; no measurement uncertainty was supplied. These are working checks, not validation of the selected reactive chemistry.
Case 3C capture changes from 89.52354% to 89.52775% with refinement, versus 89.50% reported capture. Peak liquid temperature changes by 0.00398 K. Across Cases 1C–4C, capture errors range from +0.02775 to +3.67282 percentage points using refined 3C. Temperature-tap error is not claimed until the source coordinate/phase mapping is confirmed.
Collocation final iterations, nodes, residuals and process CPU time are retained. The shooting and finite-difference attempts failed the existing acceptance checks, so their elapsed times do not establish a speed advantage. The selected nine-species Case 3C reference predicts 91.548387% capture; its refinement and four local interaction perturbations are retained separately in the linked energy/column record.

## Reviewer 1

### R1.1 — Addressed in manuscript

> 1. The theoretical justification for applying ePC-SAFT to chemical absorption is insufficient. This is the largest weakness of the manuscript. Throughout the manuscript, the authors present ePC-SAFT as the thermodynamic engine for predicting CO2 fugacity and absorber performance. However, they never explain why a molecular equation of state developed for physical interactions can accurately describe a strongly reactive absorption system.

Response/coverage: Addressed in the manuscript: §2.1.1 and the selected-input appendix explain why reactive absorption requires EOS activities coupled to five reaction equilibria, carbon/nitrogen and mass balances, and charge, with explicit standard-state conversion and no initial-composition rebasing.

Next: Complete — The explanation and current converged Case 3C reference are in the manuscript. Additional multi-case validation remains separate.



### R1.2 — Addressed in prose

> 2. ePC-SAFT is essentially a physical equation of state rather than a reaction model. The manuscript currently blurs the distinction between molecular thermodynamics and chemical thermodynamics. The original PC-SAFT framework accounts for hard-chain repulsion, dispersion interactions, association (hydrogen bonding), dipolar interactions, ionic electrostatics (extended versions). These terms describe physical intermolecular interactions.

Response/coverage: The responsibility table distinguishes molecular nonideality from chemical equilibrium. The new runtime couples EOS activities to declared reaction constants rather than treating the EOS alone as a reaction model.

Next: Complete — The comparison-definition table distinguishes the six-species fugacity experiment from the coupled nine-species calculation without changing historical result labels or the manuscript title.


### R1.3 — Addressed in prose

> 3. The manuscript should explicitly identify which quantities are predicted by ePC-SAFT and which are obtained from reaction equilibrium. The workflow is currently ambiguous. A schematic workflow should be provided illustrating

Response/coverage: The native TikZ workflow and responsibility table identify chemistry, phase equilibrium, transport, fluxes, and balances.

Next: Complete — The main diagram leads with the coupled nine-species step and distinguishes historical six-species calculations. Empirical transport formulas are fixed, while their state-dependent inputs change.


### R1.4 — Addressed in prose

> 4. The manuscript repeatedly uses the term "ePC-SAFT fugacity benchmark", but its meaning is not sufficiently defined. The authors should clearly define this terminology in the Introduction to avoid ambiguity.

Response/coverage: The Introduction defines the controlled fugacity comparison. The model's comparison-definition table identifies changing chemistry/closure, parameter set, fixed transport/enhancement formulas, caloric treatment and supported conclusion for the six-species closure comparison, numerical-method comparisons, coupled reactive verification and local sensitivities. Simultaneous changes in speciation and fugacity are explicitly not attributed to an isolated fugacity effect.

Next: Complete — Preserve titles and historical data labels; extend the comparison definition when an additional verified result set is incorporated.


### R1.5 — Parameter documentation complete

> 5. Provide more information on ePC-SAFT parameters, since these parameters determine prediction accuracy, a complete parameter table should be included either in the main text or Supporting Information.

Response/coverage: Historical parameter tables are preserved. The appendix now identifies a supporting record transcribed from the exact selected JSON: all component/fixed and pair coefficients, association sites/edges, temperature correlations, model coefficients, five reactions, standard state and domains.

Next: Documentation complete — §2.1.1 and the selected-input appendix describe the current reaction system and effective R2/R4/R5 constants. The six-species derivation and old parameter tables are grouped as historical evidence. The retained JSON remains executable authority.


### R1.6 — Addressed as disclosure

> 6. The manuscript does not clearly state which binary interaction parameters (kij) were adopted,

Response/coverage: Both the historical binary coefficient and every selected pair coefficient (including zeros) are disclosed. The supporting record also lists association sites, explicit edges and combining-rule choices.

Next: Documentation complete — No additional pair fitting is required for this response. Associate the selected values with future corrected results.


### R1.7 — Addressed in manuscript

> 7. whether they were fitted, or whether literature values were directly used. The predictive capability of the model strongly depends on these parameters, and additional clarification is necessary.

Response/coverage: Addressed in the manuscript and supporting parameter record: retained literature, transferred diagnostic, historical local-fit and newly fitted coefficients are distinguished, with selected interaction/dispersion values and calibration versus held-out data identified.

Next: Complete — Source and fitting disclosure is addressed. Preserve the recorded source/domain qualifications; no additional fit is required to close this comment.



### R1.8 — Addressed in manuscript

> 8. The authors should evaluate the sensitivity of the simulation results to key thermodynamic parameters. This analysis would improve confidence in the robustness of the proposed benchmark.

Response/coverage: Eight one-at-a-time ±5% perturbations converge at Case 3C. MEA–water changes capture by −0.822437/+0.758185 percentage points; CO2–water by +0.162723/−0.165758 points. Multipliers of 0.95/1.05 on R4 and R5 equilibrium constants give +0.424773/−0.417079 and +0.589370/−0.598872 points, respectively. Every capture shift exceeds the earlier 0.005100-point refinement difference. Reaction peak-temperature shifts are about 0.060–0.067 K, only modestly larger than the 0.044926 K refinement difference; the CO2–water shifts are smaller. All variants have 42 final nodes, two mesh iterations, RMS residuals below 0.044, zero maximum scaled boundary residual and positive nine-species concentrations. The multipliers define local sensitivity, not fitted uncertainty. R4/R5 use constant ln K offsets preserving logarithmic temperature derivatives; selected source files and all unperturbed inputs remain unchanged. The implementation checks pass 87 tests with one expected legacy-domain warning.

Next: Complete — The interaction and R4/R5 equilibrium-constant comparison is in manuscript Methods, Results the thermodynamic-sensitivity figure and Conclusion. Broader uncertainty and controlled operating-condition analysis remain in Future Work; no general parameter ranking or correlated uncertainty is inferred from one condition.


### R1.9 — Addressed in manuscript

> 9. The manuscript focuses on thermodynamic uncertainty but provides little discussion regarding transport-property uncertainty. Mass-transfer coefficients, diffusivity, and liquid viscosity may significantly influence absorber performance and should at least be discussed qualitatively.

Response/coverage: The manuscript retains the qualitative correlation discussion and adds six one-at-a-time ±10% column perturbations with the selected nine-species Case 3C model. At multipliers 0.9/1.1, liquid viscosity changes capture by +0.282078/−0.292240 percentage points, free CO2 diffusivity by +0.082479/−0.093328 and the liquid-side coefficient by −0.531371/+0.398125. Dependent transport quantities and enhancement are recalculated in both the equations and their Jacobian. All seven study points converge with 42 nodes, two mesh iterations, RMS residuals below 0.044 and zero maximum scaled boundary residual. Capture responses exceed the earlier refinement change; all temperature responses are smaller than the 0.04493 K refinement change and are not interpreted as resolved. These exploratory multipliers are not independent statistical uncertainties.

Next: Complete — The transport study is integrated in Abstract, Methods, Results the transport-sensitivity figure and Conclusion, alongside the thermodynamic sensitivity comparison. Joint uncertainty, additional inputs and operating cases remain future work. The retained notebook preserves failed initialization attempts, exact-state checks and the factor-one control without placing development history in the manuscript.

The supporting transport-applicability table now gives source systems, documented ranges, dependencies and reported errors for viscosity, diffusivity, interfacial area, transfer coefficients and enhancement. Snijder's MEA diffusion correlation and Luo's base CO2 diffusion/kinetic inputs are traceable. Published Morgan, Tsai and Gaspar errors are not assigned to modified coefficient sets or reduced formulas. The printed area exponent is corrected to the implemented 0.12; no model equation or retained result changed. The provenance of the additional CO2-diffusion concentration correction, ion diffusion, modified viscosity vector and adopted area calibration remains a source question, not an invented uncertainty interval.


### R1.10 — Addressed in manuscript

> 10. Three numerical methods are compared. However, the manuscript mainly reports convergence time. The comparison would be more convincing if the authors additionally reported iteration numbers, Jacobian evaluations, nonlinear residual histories, mesh refinement histories.

Response/coverage: Methods and the numerical-verification table now report two mesh iterations, 21→22 and 41→42 nodes, outer RMS residual histories 1.86→0.125 and 0.0566→0.0431, exact final residuals, zero scaled boundary error, 19/17 RHS batches and four Jacobian batches covering 84/164 local state Jacobians. Native thermodynamic derivatives, automatic transport/nonisothermal energy derivatives, lean/rich directional checks, and the refinement changes in capture and temperature are described. Mesh iterations, local evaluations and inner Newton steps are distinguished.

Next: Complete — The requested numerical-diagnostics reporting is included in the manuscript. Inner-Newton residual traces and counts for rejected competing-method attempts were not retained and are explicitly unavailable; no general method ranking is claimed.


### R1.11 — Addressed in manuscript

> 11. Since one objective is benchmarking numerical methods, the computational expense should be quantified. For example, CPU time, memory usage, mesh size, nonlinear iterations, under identical hardware conditions.

Response/coverage: Three fresh-start Case 3C repeats on an AMD Ryzen 5 5500 with single-thread libraries give median BVP wall time 34.66 s (34.41–34.89 s). The fastest BVP run takes 34.41 s wall, 34.23 s CPU and 51.45 s including initialization and profile export, with peak RSS 203.51 MiB. It uses 21→22 nodes, two mesh cycles, tolerance 0.5 and boundary tolerance 0.001. Methods describes exact derivative-work reductions and timing boundaries. Original refinement counts remain separately identified.

Next: Complete — CPU/wall time, repeated measurements, memory, hardware, mesh and iterations are reported. These coarse single-case timings are not an accuracy-matched ranking against historical methods or a refined-mesh timing result.


### R1.12 — Addressed in prose

> 12. The benchmark is demonstrated only for MEA. The manuscript should briefly discuss whether the proposed framework can be directly extended to DEA, MDEA, AMP, PZ, blended amines, or whether additional parameterization would be required.

Response/coverage: The solvent-extension discussion covers DEA, MDEA, AMP, PZ and blends through species/conserved components, reactions, standard states, thermodynamic parameters, kinetics, transport and held-out thermodynamic/absorber evaluation. It is coordinated with the model-family comparison and the bounded Akula/Zhang verification proposal rather than repeating a claim of automatic transferability.

Next: Complete — Further work is to verify complete published configurations, then compare them using the same conventional enhancement-factor column and controlled calibration/held-out data. No additional solvent campaign is launched by this revision.


### R1.13 — Addressed in prose

> 13. The Conclusion primarily emphasizes the advantages of the benchmark. It would be beneficial to briefly discuss the current limitations, such as dependence on reliable thermodynamic parameters, applicability to reactive electrolyte systems, extension to multicomponent industrial flue gases, future incorporation of reaction kinetics.

Response/coverage: Results and Conclusion discuss scientific scope, parameter/transport sensitivity, operating trends and solvent extensions. At the investigator's request, implementation changes and diagnostic history are kept in the reviewer notebook and QA record, not narrated in the article. The 45 C assumption remains in the input table.

Next: Editorial cleanup complete — Preserve reader-facing scientific discussion and unchanged titles. Keep numerical-development status in the working review records; historical results still require replacement before submission.


### R1.14 — Addressed in prose

> 14. The manuscript would benefit from adding a concise comparison between electrolyte-NRTL, Kent-Eisenberg, CPA, and ePC-SAFT. Although the focus is on ePC-SAFT, discussing the advantages and limitations relative to other widely used thermodynamic models would better demonstrate the novelty and necessity of the proposed benchmark. Such a comparison would also help readers identify the situations in which ePC-SAFT provides clear advantages over traditional electrolyte models.

Response/coverage: The model-family comparison covers Henry-law, Kent–Eisenberg, eNRTL, CPA and ePC-SAFT, separating thermodynamic quantities, reaction treatment, reusable component/mixture inputs, calibration needs and extension limits. Component-based ePC-SAFT and combining rules may reduce additional mixture fitting when applicable inputs exist; CPA component/association reuse and eNRTL parameter/default reuse are equally acknowledged. Exact derivatives are not exclusive to ePC-SAFT. No universal parameter-count, regression-ease, accuracy or extrapolation ranking is claimed.

Next: Complete — A short cited parameter-count comparison uses Akula Table 6 (four interaction + four formation + two heat-capacity quantities) and Zhang Tables 8/10 (four neutral-interaction + six electrolyte-interaction + six formation/heat-capacity quantities). It distinguishes selected ePC-SAFT fitting choices from inherited/fixed inputs without a full new parameter inventory table. The local reduced eNRTL fit is excluded. Complete published-configuration verification and a matched column comparison remain bounded future work.


## Reviewer 2

### R2.1 — Addressed and checked

> 1. Formats of this paper need rearranged. For example, reference number should start from [1].

Response/coverage: The rebuilt CAS manuscript starts references at [1], uses native captioned floats, and incorporates the reviewed visual pilot.

Next: Final check only — Rebuild and check references, captions, units and table fit after replacing the results.


### R2.2 — Addressed in prose

> 2. Shortages of MEA absorber modeling should be clarified in literature review section.

Response/coverage: The literature review now compares five published absorber studies: Tobiesen 2007, Zhang 2009, Chinen 2018, Akula 2021 and Shahid 2019. The compact table identifies thermodynamics/reactions, resolved-film versus enhancement treatment, experimental coverage, numerical resolution/convergence reporting and availability of parameters or executable inputs. Existing numerical-refinement and validation work is acknowledged rather than claimed as new. Missing reporting is not described as an unperformed calculation.

Next: Complete — The literature comparison supports the bounded gap: matched closure and numerical-method comparisons under a declared common absorber model. No unsupported first claim, new absorber campaign or SRP calculation is introduced.


### R2.3 — Addressed in manuscript

> 3. Parameters for the modeling should be presented for peer repetition.

Response/coverage: Addressed in the manuscript: parameter and reaction files, species/units/standard states, selected runtime, immutable wheel identity and locked-environment instructions are identified through the appendix and REPRODUCE.md.

Next: Complete — Parameter disclosure for peer repetition is addressed. Archiving the working revision and replacing historical numerical outputs are separate closeout tasks.



### R2.4 — Addressed in manuscript

> 4. There are more reported MEA absorber results. Modeling validations should also be conducted for the other results.

Response/coverage: Seven coupled nine-species NCCC cases (1C–7C) are included in Results with capture and temperature figures. Capture MAE is 5.85483 percentage points; 6C/7C errors are +11.44669/−11.93263 points. All runs converge with positive species, component/charge checks and no source-change warnings. Morgan Table C2 supplies all 35 packing-temperature observations; z=1−x and Celsius-to-kelvin conversion are explicit, without assuming a phase-specific sensor. Coarse campaign settings remain separate from the refined Case 3C sensitivity reference.

Next: Complete — Additional reported NCCC cases are compared in the manuscript. The remaining prediction discrepancies are visible; seven successful solves do not establish uniform accuracy or seven-case mesh convergence. SRP would extend facility coverage.


### R2.5 — Open

> 5. Based on the model proposed in this paper, MEA absorber performance under various run conditions should be obtained and discussed for optimal operating.

Response/coverage: The accepted Case 3C reference and completed R1.8 interaction screen are available to initialize a controlled operating-condition analysis. The validation observations vary several inputs together; no controlled operating sweep or optimum has been calculated.

Next: Next scientific addition — Sweep liquid/gas ratio, lean loading and inlet temperature within the selected domain. Report capture and temperature trends; define an energy/cost objective and hydraulic constraints before claiming an optimum.


### R2.6 — Addressed in prose

> 6. Besides MEA, other amines for CO2 capture process are more and more used. How to fit new amine to this model?

Response/coverage: The article explains how another amine would be parameterized and evaluated, including thermodynamic and absorber measurements, internally consistent reactions and standard states, transport and kinetics. The coordinated future-work sequence first reproduces complete published thermodynamic configurations, then compares the same column formulation, and finally tests data efficiency and transfer using controlled calibration and held-out observations.

Next: Complete — An ePC-SAFT parameter file alone does not provide another solvent's reaction network, kinetics, transport correlations or validation. No other-solvent implementation or new numerical campaign is claimed.


## Primary correctness repair and accepted assumptions

The countercurrent liquid-energy sign and composition chain rule are repaired in the working implementation and manuscript equations. The liquid enthalpy derivative now matches the retained empirical caloric correlation, and the temperature initial guess no longer applies enthalpy-polynomial coefficients to kelvin. Fifty focused checks pass. The original historical column outputs remain unreplaced.

The exact previously rejected equilibrium state now passes with the supplied immutable wheel and conservative positive loading path, as does the independent Orchestrator reference. Sixteen reactive/energy/RHS checks and the final integration gate pass. The old-wheel/unseeded attempt was inadequate, not evidence that the selected parameter set fails.

Next: The selected-reactive Case 3C profiles, refinement and local sensitivities are incorporated. Review the separately reported multi-case package before extending that conclusion. Retain conventional enhancement with corrected free-species coupling and energy balance.

The old parameter tables are explicitly historical. Locally rebased calculations remain outside the article; selected Case 3C profiles and parameters are reported separately.
The new bundle's reaction constants and standard-state conversion are already retained in the downstream runtime; they are not inferred from the initial composition.
Keep the selected bundle's actual calibration domains and source distinctions in the supporting parameter record, without repeating generic predictive-chemistry disclaimers throughout the article.

Use the accepted 45 °C lean-inlet assumption for the missing rows and label it as assumed once.
Match temperature-tap coordinates when calculating temperature error, as an ordinary data-preparation step.
Retain readily available final solver diagnostics and one useful numerical-accuracy check; do not reconstruct perfect mesh histories.
Keep the distinction between repaired equations and unreplaced historical numerical results explicit in this working record and QA notes. The investigator specifically requests no change-history or development-status narration in the article; this does not make the working draft submission-ready.


## Development calculations excluded from the article

The historical pure-component table also contained three species unused by the six-species fugacity benchmark. Those rows are retained here rather than in its reader-facing parameter table: carbonate, hydronium and hydroxide each had segment number 1, segment and Born diameters 3 angstrom, dispersion energy/k 300 K, relative permittivity 8 and no association. Their molar masses were respectively 0.060010, 0.019020 and 0.017010 kg/mol, with charges −2, +1 and −1. These are the old numerical-test assumptions, not the current selected nine-species coefficients.

The following historical text is retained outside the reader-facing manuscript. It does not describe the current independently parameterized nine-species replay. The numerical record remains in `analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv`; the obsolete manuscript table was removed.

    \subsection{Activity-Rebased Ionic Speciation Timing Test}
        The routine one-bed campaign keeps the concentration-based chemical-equilibrium calculation fixed and uses ePC-SAFT for the \COtwo{} fugacity coefficients. A separate ionic activity-speciation calculation tested numerical coupling between ePC-SAFT activity-related quantities and chemical equilibrium. The calculation used a nine-species ionic state for \COtwo, MEA, \HtwoO, \MEAH, \MEACOO, \HCOthree, $\ce{CO3^{2-}}$, $\ce{H3O+}$, and $\ce{OH-}$. For each local chemistry call, it evaluated the ePC-SAFT activity quotient at the concentration-based initial composition and used that quotient as the local reaction constant. It did not use an independent set of standard-state activity-basis $K_r(T)$ values. The solver returned all seven 2017 C-case rows with no invalid states, no domain guards, and chemistry residuals below \num{1e-8} under the feasibility settings reported below.

The retained CSV compares this calculation with the historical six-species rows. The activity-rebased calculation uses feasibility settings of 7 starting mesh points, a solver tolerance of \num{10}, a boundary-condition tolerance of \num{0.5}, and a maximum of 80 mesh nodes. The nine-species Case 3C row predicts \qty{88.736}{\percent} capture, \num{-0.764} percentage points from the measured value, and requires \qty{139.420}{s}; \qty{102.267}{s} is spent in the ePC-SAFT chemistry solves. Across all seven C rows, the mean runtime is \qty{171.102}{s}, the mean chemistry-solve time is \qty{127.764}{s}, and the capture mean absolute error is \num{6.595} percentage points. These values establish only that the numerical activity calculation returned solutions under the stated feasibility settings. They do not establish predictive reactive ePC-SAFT chemistry because the reaction constants were not obtained independently of the initial state.

        \FloatBarrier

    \subsection{Reactive Thermodynamic Formulation Diagnostic}
        A separate diagnostic checked the internal thermodynamic and transport structure of the vendored nine-species reactive bundle without treating it as column-validation evidence. At \qty{313.15}{K} and \qty{101325}{Pa}, a fresh homogeneous equilibrium solve satisfied the retained balance and reaction-affinity checks; the reaction-affinity infinity norm was $1.7\times10^{-13}$. The exact composition-tangent construction had a symmetry residual of $4.7\times10^{-17}$ and a positive constrained Hessian in the retained evaluation, but its condition number was approximately $5.3\times10^{10}$ because hydronium is a trace species.

        The exploratory transport closure forms a symmetric pair-friction mobility from labeled diffusivity anchors, then removes total-molar-flow and electrical-current modes. It recovered the ideal binary Fick limit, produced nonnegative entropy production, and gave zero total-flux and zero-current residuals at approximately $10^{-26}$ in the local check. These are algebraic and limiting-behavior checks on the formulation, not measurements of the nine-species mobility matrix. The inorganic-ion and hydronium diffusivities remain bounded assumptions, and the forward rate prefactors retain the source concentration basis. With a 100-micrometre film thickness, the local \COtwo{} diffusion time was \qty{11.1}{s} while the detailed-balance reaction-time estimate was \qty{1.45e-3}{s}, giving a local Damkoehler estimate of $7.6\times10^{3}$. The scale separation indicates a stiff fast-reaction manifold and motivates reduction or continuation before any column coupling.

        The diagnostic therefore supports internal thermodynamic consistency and identifies the numerical stiffness expected from the proposed formulation. It does not establish predictive reactive chemistry, a measured mobility law, physical film flux, or packed-column capture; those claims require source-complete transport and kinetics plus held-out film observations.

        \FloatBarrier
