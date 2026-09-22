# Manuscript story and paragraph plan

Prepared 2026-09-04 for the current nine-species MEA absorber manuscript.
Revised 2026-09-04 following three independent GPT-5.6 Sol reviews and the main chat's implementation checks.
This is an editorial plan, not a new scientific result or a manuscript rewrite.
The current LaTeX manuscript and retained numerical records supply the content; the paragraph assignments below specify how to build the argument from that content.
Numerical values are a planning snapshot and must be checked against the final retained tables when inserted.
Author directions for the final article: use the title ending at “Aqueous MEA,” without a colon or subtitle; exclude software-build discussion and development chronology throughout the article. Technical implementation identities belong in the reproduction records. These directions govern the final editorial pass below.

## 1. The argument the paper should make

**Central question:** How does the specified coupling of reactive equilibrium, ePC-SAFT and conventional transport describe a steady MEA absorber, and what do experimental comparison and controlled perturbations reveal about its predictions?

**Central argument:** Explicit chemical equilibrium makes ePC-SAFT activities and CO₂ fugacities usable within a reactive absorber calculation.
The resulting model connects bulk speciation and driving force to capture and axial temperature profiles through conventional transport and empirical caloric closures.
Seven NCCC cases establish the extent and limitations of the physical agreement.
Reference-case refinement gives small observed changes in capture and peak temperature that can be compared with physical discrepancies and responses, without certifying solution accuracy.
Thermodynamic, transport and operating studies then identify which tested changes affect the predictions, without establishing statistical uncertainty, general transferability or optimal operation.

**Reader's progression:** Why this coupling is needed → what the model actually calculates → how it is evaluated → how it agrees with observations → what changes its predictions → what can reasonably be concluded.

The main scientific draw is the connection between explicit reactive thermodynamics and observable column behavior, with enough numerical and parameter detail to evaluate that connection.
Convergence alone is not evidence of physical accuracy, and the results do not establish superiority over other thermodynamic models.
The study evaluates the coupled formulation; it does not isolate the incremental benefit of reactive coupling against an otherwise matched uncoupled model.
Keep three limitations distinct: the reduced vapor composition used for the EOS, correlation extrapolation at calculated states, and the observed numerical variation of signed net enthalpy flow.
The physical adequacy of the empirical caloric properties is a separate question from whether the numerical solution conserves their adopted enthalpy definition.

### Scope decisions

| Keep central | Keep as supporting evidence | Remove from the article's objectives |
|---|---|---|
| Five-reaction, nine-species equilibrium coupled to ePC-SAFT | Reference-case refinement, residuals and conservation checks | Shooting versus finite-difference or collocation comparisons |
| Seven-case capture and packing-temperature comparison | Computational cost for the identified coarse Case 3C implementation | Solver speed rankings and a separate solver-performance section |
| Thermodynamic and transport sensitivity | Complete parameters, conventions and correlation applicability | A new eNRTL/Henry/ePC-SAFT performance comparison |
| Discrete operating response | Failed lower-flow condition and initialization procedure | A resolved reactive-film comparison or claimed plant optimum |

Retain centered finite differences only as a derivative-verification procedure.
Use a short computational-cost paragraph within the reference-case evidence.
Identify sections by purpose when editing: the earlier request to remove a numbered section concerns the solver-comparison content; numbering changes must not accidentally remove the operating study.

## 2. Complete section skeleton

| Part | Proposed heading | Question answered | Planned content |
|---|---|---|---|
| Title | Reactive ePC-SAFT Modeling of Carbon Dioxide Absorption in Aqueous MEA | What system, approach and evidence does the paper contain? | A bounded description; no benchmark or optimization promise |
| Abstract | Unheaded | What was done, found and learned? | Eight sentence roles, approximately 200–250 words |
| 1 | Introduction | Why is this scientific question useful and unresolved? | Six paragraphs; two concise literature tables |
| 2 | Absorber and Thermodynamic Model | What equations and closures connect inlet conditions to predictions? | Physical basis, reactive thermodynamics, transport, balances |
| 3 | Evaluation Methods | How are comparisons and controlled studies performed? | Experimental basis, numerical method, diagnostics, perturbations and timing |
| 4 | Results and Discussion | What does the evidence show and what does it mean? | Reference and seven-case comparison, sensitivity, operating response, synthesis |
| 5 | Conclusions | What has been established and why does it matter? | Four connected paragraphs; no new evidence |
| Appendix A | Property and Transport Correlations | How are auxiliary properties evaluated? | Equations, coefficients, units, bases and applicability |
| Appendix B | Reactive Thermodynamic Inputs and Parameter Provenance | Which exact thermodynamic description was used? | Components, interactions, association, reactions, conventions and fitting provenance |
| Back matter | Existing journal-required headings | Who contributed and how can the work be inspected? | Declarations, data, code and AI disclosure, followed by references |

Suggested main-text balance is approximately 12–15% Introduction, 25–30% Model, 15–20% Methods, 30–35% Results and Discussion, and 5–8% Conclusions.
These are editing guides, not quotas; equations and floats determine the final length.
Do not add introductory or concluding paragraphs merely to fill a subsection.

### Reading the paragraph assignments

Each ID identifies one intended prose paragraph, not a section to print in the manuscript.
“Lead” is a proposed topic sentence or its precise meaning.
“Develop” gives the order of the remaining sentences and the evidence to insert.
“Exit” states the connection to the next paragraph.
Equations, figures and tables interrupt the typography where needed but remain attached to the paragraph that explains their scientific purpose.
Generally use three to five sentences per paragraph; split a paragraph only when its scientific purpose changes.

## 3. Title and abstract

### Title

The author-selected title is “Reactive ePC-SAFT Modeling of Carbon Dioxide Absorption in Aqueous MEA.” It has no subtitle; the abstract and body describe the evaluation and operating studies.
Avoid “predictive benchmark,” “optimization,” “general framework,” and claims of comparative accuracy unless the paper actually establishes them.

### Abstract sentence plan

| Sentence | Role | Content to write | Evidence or qualification |
|---|---|---|---|
| AB1 | Problem | Explain that reactive absorber predictions depend on the coupling of chemical equilibrium, molecular nonideality and transport. | One specific problem sentence; no broad climate preamble |
| AB2 | Approach | State the five-reaction, nine-species ePC-SAFT coupling, conventional enhancement-factor transport and nonisothermal balances. | Identify the empirical caloric closure if needed to prevent a complete-EOS interpretation |
| AB3 | Evaluation | State seven NCCC cases and the Case 3C refinement and studies. | Observed packing temperatures have no established phase-specific designation |
| AB4 | Physical result | Give capture MAE 5.85 percentage points and the largest signed discrepancies, +11.45 and −11.93 points. | The result should convey uneven agreement, not merely successful solution |
| AB5 | Numerical distinction | Give refined Case 3C capture 91.55% versus 89.50% observed, with refinement changes of 0.00510 points and 0.04493 K. | Small observed changes under the tested joint refinement; no certified error bound |
| AB6 | Input sensitivity | Report capture changes up to 0.82 points for the tested ±5% thermodynamic variations and 0.53 points for the tested ±10% transport variations. | Different perturbation sizes; no common ranking or confidence interval |
| AB7 | Operating and practical result | State that lean loadings 0.225/0.275 around 0.25 mol CO₂/mol MEA change capture by +3.75/−6.86 points, respectively, and identify incomplete lower-flow coverage; include median BVP cost 34.66 s only if space permits. | Six accepted conditions including the baseline; timing belongs to the identified coarse calculation |
| AB8 | Main interpretation | State that explicit reactive thermodynamics supports coupled profile and response analysis, while physical discrepancies and closure limitations constrain quantitative operating conclusions. | No optimum, universal accuracy or transferability claim |

Write the abstract last.
It must represent the operating study, physical discrepancies and interpretation, not spend its limited space listing every solver diagnostic.
If the word limit forces a cut, omit runtime before omitting the operating finding or main conclusion.

## 4. Section 1 — Introduction

Use a continuous six-paragraph introduction unless the journal strongly favors subheadings.
The current “Literature Review” and “Scope and Contributions” headings may be retained, but they must not repeat the opening overview.

### I1 — Establish the engineering question

**Lead:** Predicting CO₂ capture in an aqueous MEA absorber requires the coupled effects of reaction, interphase transfer and heat release to be represented.
**Develop:** Establish MEA as an experimentally documented reference solvent; identify capture and axial temperature as complementary observables; explain why their prediction matters for interpreting changes in solvent and gas conditions.
Use the existing absorber literature citations.
**Exit:** The prediction depends on the thermodynamic and transport descriptions used to connect these processes.
**Revision:** Replace the current opening emphasis on competing numerical methods; avoid a long general emissions discussion.

### I2 — Establish what prior absorber models already accomplish

**Lead:** Rate-based MEA models already couple chemistry, equilibrium, transport and energy balances to pilot-scale observations.
**Develop:** Synthesize the existing studies by experimental coverage, reaction/film treatment and calibration basis; distinguish fitting measurements from evaluation measurements; use Table T1 to carry study-specific details.
**Exit:** Their differing assumptions make a clearly specified coupling and evaluation basis necessary.
**Revision:** Remove promises of all-case runtime or temperature-error statistics that are not reported.

### I3 — Explain the thermodynamic distinction

**Lead:** A molecular equation of state supplies nonideality, whereas reaction equilibrium determines the distribution among molecular and ionic species.
**Develop:** Explain why these descriptions must be coupled for MEA; introduce ePC-SAFT activities and fugacities; distinguish this role from the conventional enhancement model and empirical calorics used here.
**Exit:** This distinction provides a fair basis for comparing the scope of available thermodynamic approaches.
**Revision:** Use scientific responsibilities rather than an unexplained “fugacity benchmark” label.

### I4 — Explain the attraction of component-level ePC-SAFT inputs

**Lead:** Physical component descriptions and combining rules offer reusable thermodynamic inputs for new solvent systems.
**Develop:** Compare eNRTL, Kent–Eisenberg, CPA and ePC-SAFT in T2. Explain the potential reduction in additional mixture-specific binary fitting. Give the source-backed one-scalar MEA–water refit and the named eNRTL fitting example, distinguishing binary quantities from pure-component, ion, association, reaction and caloric inputs and inherited/fixed values.
**Exit:** A matched comparison in this column can test the component-reuse advantage in data requirements, fitting effort and process accuracy.
**Qualification:** Different published calibration tasks do not establish universally fewer total parameters or easier fitting.

### I5 — Connect thermodynamic reuse to process evaluation

**Lead:** Column evaluation connects thermodynamic choices to capture, axial temperature and operating response.
**Develop:** Explain the joint value of speciation/fugacity profiles, thermodynamic/transport perturbations, measured observations, numerical refinement and measured cost.
**Exit:** This connection supports predictive transfer development and assessment of parameter reuse in process calculations.

### I6 — State the objective and reading order once

**Lead:** The model is evaluated against seven NCCC one-bed cases and then examined locally around Case 3C.
**Develop:** Name the reference refinement, thermodynamic and transport sensitivity, and operating changes; mention computational cost as supporting information; state that the operating study is partial and no optimum is inferred.
**Exit:** Section 2 defines the coupling that these comparisons evaluate.
**Revision:** Replace the current contribution list and duplicated overview with one connected paragraph; correct the stale count of two missing operating conditions to one if counts are included here.

## 5. Section 2 — Absorber and Thermodynamic Model

Keep the order physical system → thermodynamic state → transport flux → column balances.
Define each quantity before using it, and explain each equation's role immediately after its introduction.

### 2.1 Physical system and model responsibilities

**M1 — Physical domain.**
**Lead:** The model describes steady countercurrent absorption in one packed bed.
**Develop:** Define phase directions, axial coordinate, bed geometry, inlet boundaries, conserved apparent components and the zero pressure-gradient assumption used for these cases.
Distinguish dimensional height from normalized plotted position.
**Exit:** Bulk states supply the inputs to the thermodynamic and transport calculations.

**M2 — Calculation sequence.**
**Lead:** At each axial position, reactive equilibrium and transport calculations supply the fluxes used in the column balances.
**Develop:** Introduce Figure F1 and Table T3; follow operating state → equilibrium/speciation → CO₂ fugacity → conventional enhancement and flux → material and energy derivatives.
Identify separately the empirical density used by column correlations and the EOS density used internally.
**Exit:** The reactive equilibrium calculation is the first part of this coupling.
**Correction:** Neither the diagram nor its caption may imply that ePC-SAFT supplies the adopted water-vapor closure, all caloric properties or a full flue-gas fugacity-coefficient calculation.
Identify the reduced neutral vapor composition before the EOS vapor calculation; M6 defines its mapping to the wet-gas CO₂ fugacity.

### 2.2 Reactive equilibrium and phase driving forces

**M3 — Components, species and reactions.**
**Lead:** Five reactions distribute the conserved MEA–CO₂–water composition among nine liquid species.
**Develop:** Introduce the complete species set and R1–R5; define apparent totals versus true species; state component balances and electroneutrality.
**Exit:** Equilibrium requires both reaction constants and species activities.

**M4 — Activities and standard states.**
**Lead:** Reaction equilibrium is evaluated on a declared activity basis using temperature-dependent equilibrium constants.
**Develop:** Present the reaction-product relation, molality convention and units/reference quantities; explain the conversion used for the selected effective constants; point to Appendix B for coefficients and source distinctions.
**Exit:** ePC-SAFT provides the nonideal contributions required by these activities.
**Correction:** Preserve the actual implemented convention; printed effective constants must not be treated as unconverted source constants or converted a second time.

**M5 — EOS physical contributions.**
**Lead:** The residual Helmholtz-energy model represents the physical interactions that enter activities and fugacity coefficients.
**Develop:** Identify the active hard-chain, dispersion, association and electrolyte terms; connect the existing compact equations to the calculated quantities; define symbols and avoid claiming inactive contributions.
**Exit:** The combined equilibrium and EOS calculation supplies the molecular CO₂ driving force.

**M6 — CO₂ and water driving forces.**
**Lead:** Interphase CO₂ transfer uses phase fugacities evaluated from the specified thermodynamic states.
**Develop:** Define the CO₂ fugacity difference and its sign; disclose that the vapor EOS uses normalized CO₂, a numerical trace of MEA and water, excluding N₂/O₂ from the composition supplied to its fugacity-coefficient calculation.
Write the mapping explicitly: the reduced composition gives the vapor CO₂ fugacity coefficient, which is multiplied by the original wet-gas CO₂ mole fraction and total pressure to obtain vapor CO₂ fugacity.
Distinguish the numerical MEA trace from a measured vapor MEA concentration and the reduced EOS composition from the actual wet-gas composition used in the column.
Separately state the ideal-vapor/vapor-pressure approximation for water; introduce the free molecular concentration used in M10.
**Exit:** These driving forces require transfer coefficients and an interfacial area to produce column fluxes.
**Revision:** Rename the current broad “Vapor–Liquid Equilibrium” heading if it implies a full reactive phase-equilibrium study that is not performed here.
**Limit:** Do not describe this mapping as a full-mixture flue-gas EOS treatment or assume that omitting inert-gas interactions has a quantified negligible effect.

**M7 — Parameter origin and independence.**
**Lead:** The calculation uses one specified parameterization assembled from inherited, fitted and derived inputs.
**Develop:** Briefly identify the input families and direct readers to Appendix B; state that the NCCC cases were not used to fit the adopted thermodynamic parameters; distinguish earlier selection evidence from untouched evaluation data.
**Exit:** The remaining constitutive choices describe transport and heat transfer.
**Revision:** Keep software hashes and detailed fitting-stage counts out of this paragraph; preserve them in reproducibility records.

### 2.3 Hydraulics, transfer coefficients and enhancement

**M8 — Hydraulic and property dependencies.**
**Lead:** Flow and mixture properties determine holdup, interfacial area and the transfer correlations.
**Develop:** Define velocities and packing variables; introduce the actual holdup and area expressions; identify the property correlations and their applicable ranges in Appendix A.
**Exit:** These quantities determine gas- and liquid-side transfer coefficients.

**M9 — Mass and heat transfer.**
**Lead:** Gas- and liquid-side transfer correlations provide the physical resistance to interphase exchange.
**Develop:** Present the mass-transfer and heat-transfer expressions with explicit phase/species subscripts; define hydraulic diameter and units; distinguish the heat-transfer coefficient from total heat flux.
**Exit:** The liquid CO₂ resistance is modified by the adopted reaction enhancement.
**Correction:** Resolve the current U versus U_T notation from the implementation before rewriting equations.

**M10 — Enhancement and its limits.**
**Lead:** A conventional enhancement factor represents the influence of reaction on liquid-side CO₂ transfer.
**Develop:** Define the Hatta number, concentration inputs and kinetic terms; state which free species from bulk equilibrium are used and explain the reduced concentration approximation.
Keep the explicit thermodynamics-to-film relation in this paragraph: \(H_{\mathrm{bulk}}=f_{\mathrm{CO₂}}^\ell/C_{\mathrm{CO₂}}^{\ell,\mathrm{free}}\), in Pa m³ mol⁻¹, uses fugacity and free molecular CO₂ concentration from the same bulk equilibrium state.
Explain that the bulk activity coefficient is held fixed across the film; this ratio is neither a derivative with respect to total absorbed carbon nor a separately fitted Henry constant.
Identify the empirical concentration divisor and state that its original calibration source is unavailable; it applies inside enhancement, not to EOS fugacity or component balances.
Introduce the documented source applicability ranges in Table T6; R10b compares those ranges with the calculated states.
**Exit:** Combining this enhancement with the phase resistances gives the interfacial flux.
**Limit:** Nine-species bulk equilibrium does not mean that nine coupled reaction–diffusion equations are solved through the liquid film.

**M11 — Flux construction.**
**Lead:** Interfacial mass and heat fluxes connect the constitutive calculations to the axial balances.
**Develop:** Write the combined CO₂ resistance, water transfer and heat-transfer relations; define positive directions and the effective interfacial area; state which dependencies are reevaluated when temperature or composition changes.
**Exit:** The column solution follows from conserving material and evaluating the adopted energy balance along the bed.

### 2.4 Column balances and boundary conditions

**M12 — Material balances.**
**Lead:** Phase component flows change through the interfacial fluxes with signs set by the countercurrent coordinate.
**Develop:** Present the governing flow derivatives and conservation relations; explain any retained numerical flow offset and its cancellation without describing it as physical generation.
**Exit:** Interphase transfer also couples the phase temperatures.

**M13 — Energy balances and caloric scope.**
**Lead:** Nonisothermal balances use the adopted empirical mixture-enthalpy and heat-transfer expressions.
**Develop:** Distinguish apparent-component enthalpy calculations from nine-species equilibrium; explain the full temperature chain rule and conservative interphase transfer terms.
State that the continuous implemented equations give a constant signed net enthalpy flow, \(d(\dot H^v-\dot H^\ell)/dz=0\), within the adopted additive component-enthalpy definition.
Then distinguish conservation of that defined quantity from whether the empirical enthalpy description accurately and consistently represents the reactive mixture's caloric properties.
**Exit:** Pressure and the specified inlet values complete the differential problem.
**Correction:** The exported profiles show nonzero axial net-enthalpy-flow variation; assign its measurement to E6 and its observed refinement response to R2.
Do not attribute this numerical conservation discrepancy to the empirical origin of the caloric properties, claim exact conservation by the exported solution, or assign a numerical cause without supporting diagnosis.

**M14 — State vector and pressure.**
**Lead:** The solution state comprises the phase flows, temperatures and pressure used by the column equations.
**Develop:** Introduce the full state vector outside a pressure-only heading; define the zero pressure derivative for the present calculations; show dimensional-to-normalized-coordinate scaling if both appear.
**Exit:** The countercurrent inlets prescribe different parts of this state at opposite ends.

**M15 — Boundary conditions.**
**Lead:** Gas-inlet and liquid-inlet specifications define the mixed boundary conditions.
**Develop:** Present the boundary equations, wet-feed reconstruction inputs and pressure condition; identify specified versus calculated outlet quantities.
**Exit:** Section 3 explains how this problem is solved and compared with observations.
**Revision:** Move solver tolerances and postprocessed acceptance checks to Methods.

## 6. Section 3 — Evaluation Methods

### 3.1 Experimental cases and reported quantities

**E1 — Experimental basis.**
**Lead:** Seven one-bed NCCC cases provide the experimental comparison.
**Develop:** State case-selection basis, source and geometry; introduce Table T4; distinguish dry reported gas flow/composition from reconstructed wet feed; identify the 318.15 K lean-inlet assumption for Cases 1C–3C.
**Exit:** The comparison uses outlet capture and axial packing temperatures.
**Move:** Place the operating-input table here, before Results, because it defines the evaluation rather than reporting a prediction.

**E2 — Observables and metrics.**
**Lead:** Capture errors and axial temperature comparisons describe different aspects of absorber performance.
**Develop:** Define capture, signed error and equally weighted MAE; explain the source-coordinate and Celsius-to-kelvin conversion; compare both phase curves to packing taps without assigning the taps to a phase.
Define the sampled peak liquid temperature using the 101 exported positions.
**Exit:** Numerical resolution must be assessed separately from these physical comparisons.
**Decision:** Remove the unused temperature-RMSE equation unless retained, source-traceable values are actually reported; qualitative profile comparison remains a valid result.

### 3.2 Numerical solution and verification

**E3 — Collocation solution.**
**Lead:** Adaptive residual-controlled collocation solves the mixed-boundary column equations.
**Develop:** State solver identity and relevant method description; explain simultaneous boundary enforcement, polynomial approximation and adaptive node insertion; identify state scaling and stopping limits.
**Exit:** The nonlinear calculation uses derivatives of the coupled equations.
**Correction:** Say that the polynomials satisfy collocation conditions and control differential residuals, not that they satisfy the ODE exactly everywhere in an interval.

**E4 — Derivatives and initialization.**
**Lead:** The assembled Jacobian includes the thermodynamic, transport and nonisothermal dependencies of the column residuals.
**Develop:** State the automatic-differentiation roles, identical-state derivative reuse and scaled centered-difference directional checks; give the documented 2e-5 relative and 2e-8 absolute derivative tolerances; explain Henry-profile and reactive-profile initialization in scientific terms.
Distinguish the initial axial column profile from the initial species amounts used by each reactive-equilibrium calculation.
Record Case 1C's reaction-extent starting fraction of 0.0001 rather than the usual 0.001, identify the retained case/run where it applies, and explain that it changes the chemical initial guess rather than reaction constants or inlet conditions.
**Exit:** A paired Case 3C calculation measures sensitivity to resolution settings.
**Revision:** Remove implementation jargon such as “native anchor”; do not reintroduce a finite-difference solver comparison.

**E5 — Reference refinement design.**
**Lead:** Case 3C is solved at two jointly varied mesh and tolerance settings.
**Develop:** Give 21/41 initial nodes, 0.5/0.05 collocation tolerances, 0.001 boundary tolerance and 1000-node limit; state what is held fixed; define recorded outer mesh iterations, Jacobian/equilibrium counts and normalized RMS residuals.
**Exit:** These diagnostics are interpreted alongside physical admissibility and balance checks.
**Limit:** This paired change is not an isolated mesh-order study or an error bound for all cases.

**E6 — Inclusion and diagnostics.**
**Lead:** Reported solutions must satisfy the documented numerical and physical checks.
**Develop:** Distinguish solver boundary residual, differential residual and the six-component postprocessed boundary-error norm; state the documented 1% norm threshold and the actual remaining checks; define component/charge discrepancies, species positivity and invalid-state handling.
Define axial net-enthalpy-flow variation as the maximum minus minimum of \(\dot H^v-\dot H^\ell\) over the exported positions, in W; the continuous equations predict zero variation.
Report it as a numerical conservation diagnostic without inventing an acceptance threshold or treating its absolute magnitude as a normalized error.
Explain that failed conditions remain missing rather than becoming zero responses.
**Exit:** The same physical model and stated settings define the controlled perturbations.
**Correction:** Do not invent numerical thresholds where only a diagnostic or observed range is recorded.

### 3.3 Thermodynamic, Transport and Operating Perturbations

**E7 — Thermodynamic sensitivity design.**
**Lead:** One-at-a-time thermodynamic perturbations test selected interactions and reaction constants around refined Case 3C.
**Develop:** Give ±5% multipliers for MEA–water and CO₂–water k_ij and R4/R5; explain the sign of scaling a negative coefficient; define the additive ln(s) change to ln K and unchanged temperature derivative; state unchanged empirical enthalpy and other inputs.
**Exit:** A complementary study varies transport inputs on the same refined reference.

**E8 — Transport sensitivity design.**
**Lead:** Separate ±10% perturbations test viscosity, molecular CO₂ diffusivity and liquid-side transfer coefficient.
**Develop:** State where multipliers enter, which dependencies are recalculated and which are held fixed; explain enhancement coupling and matched refined settings; describe the factor-one control and special initialization as procedures.
**Exit:** Changes in operating conditions are examined separately from uncertain constitutive inputs.
**Move:** Report the measured factor-one difference with Results, not as part of the procedure.

**E9 — Operating-response design.**
**Lead:** Discrete Case 3C perturbations vary solvent circulation, lean loading and inlet temperature separately.
**Develop:** Specify ±10% liquid flow at fixed reported dry-gas flow, loadings 0.225/0.275 around 0.25, and temperatures 313.15/323.15 K around 318.15 K; give coarse settings; describe the bounded retry procedure without reporting its outcome here.
**Exit:** These studies report responses at specified inputs, with numerical cost characterized independently.
**Limit:** No interpolation, simultaneous optimization or statistical input distribution is part of this design.

### 3.4 Computational-cost measurement

**E10 — Timing protocol.**
**Lead:** Three sequential fresh-initialization Case 3C runs characterize computational cost for the specified coarse calculation.
**Develop:** Give hardware, software versions and single-thread settings; distinguish BVP wall time, process CPU time, total workflow wall time and peak resident memory; state timer boundaries and the implementation identity through the reproducibility record.
**Exit:** Results first describe the reference solution and the observed changes under the tested refinement.
**Correction:** Describe the timer's actual inclusion of returned-solution conversion, initialization and export; do not equate these timings with refined or later operating-run cost.

This final subsection can be an unnumbered paragraph under numerical methods if a one-paragraph subsection looks visually excessive.
It supplies reproducibility detail, not a second numerical-method research objective.

## 7. Section 4 — Results and Discussion

### 4.1 Reference solution and seven-case comparison

**R1 — Read the reference profile.**
**Lead:** The refined Case 3C solution links CO₂ uptake to the predicted fugacity, speciation and temperature profiles.
**Develop:** Introduce Figure F2; give 91.54839% capture versus 89.50% observed and peak liquid temperature 347.86155 K; describe the positive CO₂ driving force and principal species using the actual curves.
**Exit:** The paired reference calculations show how much these reported quantities change with resolution settings.
**Limit:** Bulk species profiles illustrate the coupling; they are not independent speciation validation.

**R2 — Report the observed refinement response.**
**Lead:** The joint mesh/tolerance refinement changes capture and peak temperature only slightly for Case 3C.
**Develop:** Give 0.00510 percentage points and 0.04493 K; use Table T5 for mesh counts, residual histories and evaluation counts; summarize positive concentrations and component/charge checks.
Report that axial signed net-enthalpy-flow variation decreases from 293.219 to 273.006 W across the two reference calculations, although the continuous adopted equations conserve this quantity.
This persistent variation is a numerical conservation discrepancy; its cause is not established by the reported refinement pair and must not be attributed to empirical caloric properties.
**Exit:** Computational cost is characterized separately for an identified coarse configuration using the same physical equations.
**Limit:** Small observed output changes under one joint refinement do not establish a converged solution, a certified error bound or exact numerical energy conservation.

**R3 — Report computational cost compactly.**
**Lead:** The three coarse Case 3C repeats have a median BVP wall time of 34.66 s.
**Develop:** Give the 34.41–34.89 s range; identify the fastest run's 34.23 s CPU time, 51.45 s total wall time and 203.51 MiB peak RSS; state its 21→22 nodes and two mesh iterations; report agreement with the coarse reference at the retained numerical precision.
**Exit:** Numerical reproducibility establishes a basis for comparison with the seven physical cases, not their accuracy.
**Placement:** One paragraph under an unnumbered “Numerical verification and computational cost” heading; no separate solver-results subsection or timing figure.

**R4 — Evaluate capture across cases.**
**Lead:** Capture agreement varies substantially across the seven NCCC cases.
**Develop:** Introduce Figure F3; give MAE 5.85 points and the +11.45/−11.93-point errors for 6C/7C; discuss the sign and magnitude of disagreement rather than listing all seven predictions already plotted.
**Exit:** Outlet capture alone cannot show where the axial thermal response differs.

**R5 — Evaluate axial temperature behavior.**
**Lead:** The packing-temperature comparisons expose differences in predicted temperature level, bulge location and axial shape.
**Develop:** Use Figure F4; describe two or three specific, visible case examples after checking the final curves; explain the phase ambiguity and the assumed inlet temperature where relevant.
**Exit:** These physical discrepancies must be interpreted alongside campaign-wide numerical checks.
**Revision:** Do not fill this paragraph with generic “good agreement” wording or infer an unmeasured liquid-temperature error.

**R6 — Bound the campaign conclusion.**
**Lead:** All seven calculations meet the reported numerical checks at the campaign settings, while their physical agreement remains uneven.
**Develop:** Summarize 21/22 final nodes, residual and component/charge ranges, positive species and no invalid-state penalties; report the axial signed net-enthalpy-flow ranges of 106.49–325.13 W as separate nonzero numerical conservation diagnostics.
Avoid implying that all conservation diagnostics vanish or that an unstated energy-error threshold was passed; distinguish Case 3C refinement evidence from untested refinement of the other cases.
**Exit:** The studies examine whether selected inputs produce capture changes large enough to matter relative to the reference resolution change.

### 4.2 Thermodynamic and transport sensitivity

**R7 — Thermodynamic capture response.**
**Lead:** Among the tested equal relative thermodynamic changes, the MEA–water interaction produces the largest capture response.
**Develop:** Use Figure F5; report −0.82244/+0.75819 points, followed by the R5, R4 and CO₂–water ordering; state the refined baseline; compare response magnitudes with the reference refinement difference; note that the tested capture range still exceeds the observed 89.50%.
**Exit:** The temperature response provides a more limited basis for ordering these effects.

**R8 — Thermodynamic temperature response.**
**Lead:** The corresponding peak-temperature changes are small and only partly separated from the reference refinement indicator.
**Develop:** Give the MEA–water change near 0.10 K and the smaller reaction/CO₂–water changes; identify which are below or only modestly above 0.04493 K; summarize successful solution and balance diagnostics.
**Exit:** Transport inputs offer a distinct source of capture variation.
**Limit:** The reference indicator does not certify each perturbed solution's error.

**R9 — Transport capture response.**
**Lead:** The liquid-side transfer coefficient has the largest capture effect among the three tested transport inputs.
**Develop:** Use Figure F6; report −0.53137/+0.39813 points and the smaller viscosity/diffusivity effects; confirm the matched baseline and factor-one control; state that the ±10% transport study cannot be ranked directly against ±5% thermodynamic perturbations by absolute output change.
**Exit:** The diffusivity response shows why dependent transport and reaction quantities must be interpreted together.

**R10 — Explain the coupled response carefully.**
**Lead:** The negative capture response to increased molecular CO₂ diffusivity is consistent with the recalculated enhancement and transfer resistances.
**Develop:** Define every ratio as perturbed divided by reference at matched axial positions; give median ratios 1.047 for k_L, 0.931 for enhancement and 0.975 for their product; explain that separately solved profiles are not fixed-state derivatives.
State that peak-temperature shifts below 0.037 K are not resolved by the reference comparison; summarize admissibility without repeating every diagnostic.
**Exit:** These response mechanisms must also be interpreted against the source ranges of the adopted correlations.

**R10b — Compare calculated states with source ranges.**
**Lead:** The baseline calculation extends beyond the documented source ranges of two adopted transport descriptions.
**Develop:** Compare the retained baseline Hatta-number range, approximately 320–698, with the documented enhancement source comparison range of 35–165.
State that the baseline peak of 347.86 K also exceeds the MEA diffusivity source's 333 K upper temperature; this limit belongs to the named source correlation, not an assumed common limit for all diffusivity formulas.
These are concrete extrapolations; the ±10% perturbations measure response to the adopted formulas and do not establish their accuracy outside source ranges.
**Exit:** Operating inputs are then varied directly to examine a different, practical source of response.

### 4.3 Operating response

**R11 — State the observed operating changes.**
**Lead:** Six of the seven requested operating conditions have accepted predictions, including the baseline.
**Develop:** Use Figure F7; give the coarse baseline 91.54329%; report +0.31049 points for higher liquid flow, +3.75443/−6.86023 for lean loading, and +0.10730/−0.08482 for inlet temperature.
Identify the failed lower-flow condition explicitly.
**Exit:** The largest tested capture response is associated with solvent lean loading.

**R12 — Interpret capture and temperature together.**
**Lead:** Lower lean loading increases available solvent capacity and gives the largest positive capture response among these specified changes.
**Develop:** Relate the loading response to the model's capacity and driving force without claiming a uniquely proven mechanism; give peak-temperature changes +0.61395 K for higher liquid flow and +0.25063/−0.82713 K for the loading changes.
State that the ±inlet-temperature peak responses are below the 0.04493 K reference indicator.
**Exit:** The engineering interpretation is limited by incomplete coverage and the study design.

**R13 — State the operating study's boundary.**
**Lead:** These discrete calculations describe a response, not an operating optimum.
**Develop:** Summarize coarse numerical checks and report the 160.21–390.32 W axial signed net-enthalpy-flow ranges as observed numerical conservation discrepancies, without attributing them to the empirical caloric model.
State that lower liquid flow failed both tested initializations; explain the absence of symmetric flow coverage, simultaneous decision variables, hydraulic constraints and an energy/cost objective.
**Exit:** The combined evidence identifies both useful model behavior and the next scientific limitations to address.
**Limit:** Failure to obtain an accepted solution does not establish physical infeasibility.

### 4.4 Scope, limitations and implications

**R14 — Synthesize the dominant interpretation.**
**Lead:** The calculations connect reactive bulk thermodynamics to column response, while physical disagreement persists despite small capture changes under the tested refinement.
**Develop:** Relate the Case 3C refinement response to its remaining capture discrepancy and the larger 6C/7C errors; distinguish experimental comparison, sensitivity and limited numerical verification.
Synthesize the reduced vapor-EOS composition, extrapolated transport/enhancement correlations and assumed inlet inputs as model/input limitations without assigning them unmeasured shares of the error.
Keep the observed numerical enthalpy variation separate from the physical adequacy and thermodynamic consistency of empirical calorics.
**Exit:** Broader uncertainty and stronger operating conclusions require additional evidence.

**R15 — Develop predictive transfer and constrained optimization.**
Describe the author-confirmed liquid-film implementation already underway using full ePC-SAFT thermodynamic support without an empirical enhancement factor. Ground the precise implemented-versus-intended description in retained reactive-film work. Explain the intended bulk/interfacial connection without inventing unimplemented kinetics or equilibrium-manifold details. Then present optimization over operating variables with capture, hydraulic and thermal constraints and an energy/economic objective; no current optimum is claimed.

**R16 — Test other solvents and compare thermodynamic approaches in this column.**
Plan comparative DEA/MDEA/AMP/PZ/blend evaluations with appropriate solvent parameterization. Plan direct ePC-SAFT versus eNRTL column calculations with matched configuration and non-thermodynamic assumptions and internally consistent parameterizations. Compare capture/temperature, new data requirements and independently fitted quantities to assess the proposed reuse advantage.

## 8. Section 5 — Conclusions

The newer author instruction supersedes the earlier defensive ending. Use four connected paragraphs that represent the complete scientific work and end with its value and forward program.

**C1 — Coupled formulation and physical evaluation.**
Summarize the five-reaction/nine-species reactive EOS column, its bulk-to-column connection and seven-case capture/profile evaluation, including capture MAE.

**C2 — Numerical behavior and computational cost.**
Report refined capture, the tested capture/temperature changes and median coarse runtime. Keep one proportionate scope sentence referring to the detailed results, without repeating the full limitations list.

**C3 — Scientific insights.**
Summarize the largest thermodynamic and transport responses with their perturbation sizes and the larger loading effect. Explain their value for future process studies without calling the result an optimum.

**C4 — Scientific contribution and forward program.**
End with the value of connecting reusable molecular/ionic descriptions to column observables. Link predictive liquid-film development, direct eNRTL comparison, other-solvent testing and optimization as a coherent program for solvent selection and absorber design. Keep repository/process vocabulary out of concluding scientific prose.

The conclusion must contain no parameter, comparison, source claim or new numerical value that has not appeared in the body.

## 9. Appendix and back-matter paragraph assignments

Split the current long “Properties” appendix by scientific ownership of the content; retain its existing equation material and labels where possible.

### Appendix A — Property and transport correlations

| ID | Paragraph purpose and ordered content | Attached material |
|---|---|---|
| A1 | Define phase, composition and apparent-component bases used by the auxiliary correlations; distinguish these from true-species equilibrium. | Current property introduction and group-scope text |
| A2 | Define physical CO₂ solubility/Henry correlation, numerical unit convention inside logarithms, coefficients, source and role in initialization or transport. | Henry expression and coefficients |
| A3 | Define density and surface tension, including composition variables and applicability; replace misleading “pure species” wording for composition-dependent terms. | Density/surface-tension equations |
| A4 | Define heat capacity and enthalpy on the adopted apparent-component basis; give reference states and full derivatives needed in the continuous conservative energy balance. Distinguish physical caloric consistency from observed numerical conservation discrepancy. | Heat-capacity and enthalpy equations |
| A5 | Define thermal conductivity and heat-property mixing rules, with units and source ranges. | Conductivity equations |
| A6 | Define liquid and gas viscosity, mixture rules and coefficients; correct the Wilke self-term statement to the actual i=j condition. | Viscosity equations |
| A7 | Define molecular/ionic diffusivities, species included in each sum, diffusion volumes and dependence on viscosity. Identify the MEA source's 298–333 K range separately from CO₂ and ionic formulas, whose additional corrections lack recovered calibration ranges. | Diffusivity equations and coefficient values |
| A8 | Define the actual water-vapor-pressure closure and parameters; distinguish it from EOS CO₂ fugacity. | Add the missing explicit correlation from its authoritative implementation/source |
| A9 | Summarize correlation applicability and modifications, separating published fit errors from current column errors. Put documented source ranges beside calculated baseline ranges: Hatta number 35–165 versus approximately 320–698, and MEA diffusivity upper temperature 333 K versus peak 347.86 K. Mark unknown ranges explicitly. | Table T6; retain Table T7 only as a useful input locator |

### Appendix B — Reactive inputs and provenance

| ID | Paragraph purpose and ordered content | Attached material |
|---|---|---|
| B1 | Identify the nine components, units and temperature-dependent pure-component expressions. | Table T8 |
| B2 | Define ion diameters, active electrostatic/permittivity/solvation choices and mixture rules; identify all coefficients needed for reproduction. | Table T9 and current dielectric/solvation text |
| B3 | Define symmetric k_ij, any temperature dependence, zero defaults and fitted versus inherited pairs. | Table T10 with all 36 independent pairs |
| B4 | Define association topology and combining rules, including the actual geometric MEA–water volume rule. | Table T11 |
| B5 | Give effective R1–R5 equilibrium-constant functions, their units/conventions and relation to source values. | Existing reaction-coefficient material |
| B6 | Explain parameter origins by fitted, inherited, fixed and derived quantities; identify measurements used for fitting/selection and distinguish independent evaluation evidence. | Current source/fit provenance; selected-parameter documentation |
| B7 | Explain where full-precision values and exact calculation identities are available; keep source labels attached to parameter groups. | Data/code availability cross-reference |

Do not move indispensable definitions of reaction activities or the model's physical responsibilities out of Section 2.
The appendix supplies numerical completeness, not the only explanation of the scientific formulation.

### Back matter

| ID | Requirement |
|---|---|
| D1 | Preserve accurate author contribution, funding and competing-interest statements; do not invent declarations. |
| D2 | Data availability identifies the experimental sources, processed values and current numerical records, including the fresh operating rerun. |
| D3 | Code availability distinguishes accessible source from the exact executable dependencies required to reproduce results; verify package/version claims against retained run identities. |
| D4 | AI disclosure describes the actual author-approved use and responsibility in journal-required form. |
| D5 | References follow first-citation order and begin at [1]; refresh bibliography only through the established source workflow. |

## 10. Figure plan

Use semantic IDs below during revision; let LaTeX assign final figure numbers.
There are eight existing figures; the proposed main text uses seven by removing one redundant temperature close-up.
This is a placement decision, not authorization to delete retained scientific output.

| ID | Existing figure | Placement and claim | Required action and caption content |
|---|---|---|---|
| F1 | `figures/tikz/model-framework-flowchart.tex` | M2: how the coupled calculation works | Show reactive speciation, the reduced neutral vapor composition used for the EOS coefficient and its mapping to wet-gas CO₂ fugacity; distinguish water approximation, empirical calorics and conventional enhancement. Define arrows and inputs/outputs without implying full flue-gas EOS treatment. |
| F2 | `reactive-case3c-profiles.pdf` | R1–R2: bulk thermodynamics, capture and observed refinement response | Retain; distinguish 21/41 initial meshes and tolerances, identify refined-only panels, explain six displayed species from nine solved species, measured capture cross and axial direction. Do not call the profiles certified or fully resolved. |
| F3 | `reactive-seven-case-capture.pdf` | R4: uneven physical capture agreement | Retain; observed/predicted distinction, case labels, percentage-point error convention, observation source and MAE. |
| F4 | `reactive-seven-case-temperatures.pdf` | R5: axial thermal agreement and discrepancy | Retain; phase line styles, Morgan Table C2 taps, coordinate conversion, K units and assumed inlets for 1C–3C. Ensure all panels remain readable. |
| F5 | `reactive-parameter-sensitivity.pdf` | R7–R8: response to selected thermodynamic changes | Retain; define multipliers, baseline, R4/R5, capture points versus temperature K and refined settings. No statistical-uncertainty language. |
| F6 | `transport-sensitivity.pdf` | R9–R10b: response with dependent transport quantities recalculated | Retain; name viscosity, free molecular diffusivity and k_L multipliers; define baseline and shading as observed reference refinement magnitude, not confidence interval. State that responses use the adopted correlations and do not validate their extrapolation; put numerical range comparisons in R10b/Table T6. |
| F7 | `reactive-operating-response.pdf` | R11–R13: discrete operating response and missing coverage | Use fresh rerun; define mass L/reported dry G, changed liquid flow at fixed gas flow, loading basis and temperature units. Identify baseline, failed lower-flow condition and discrete predictions. |
| F8, remove from main text | `reactive-case3c-temperature.pdf` | Current standalone 3C close-up repeats F4 | Remove the float if F4's Case 3C panel is readable; carry any unique interpretation into R5. Retain the original output in the analysis. If close-up detail is essential, use it as an inset or appendix figure rather than a second unexplained main-text comparison. |

### Numerical source assignment

| Figure/result | Retained source location | What must match |
|---|---|---|
| F2 and T5 | `analyses/nccc_validation/figures/reactive_column/output/` | Selected parameters, paired meshes, capture/temperature values and refinement diagnostics |
| F3/F4/F8 | `analyses/nccc_validation/figures/reactive_parallel/` | Same physical formulation and experimental case definitions; exact run identities preserved |
| F5 | `analyses/nccc_validation/figures/reactive_column/output/sensitivity/` | Refined reference and eight ±5% cases |
| F6 | `analyses/transport_sensitivity/figures/response/output/` | Matched reference, six ±10% cases and propagated dependencies |
| F7 | `analyses/nccc_validation/figures/reactive_operating/output/` | Fresh `summary.csv` and `provenance.json`; six accepted conditions and one failed condition |
| Operating raw runs | `analyses/nccc_validation/results/runs/reactive_operating_rerun_20260904/` | Individual successful and failed attempts, settings and package identity |
| Cost paragraph | `analyses/nccc_validation/results/runs/runtime_diagnostics_20260904/` | Three identified repeats and same-run CPU/total/RSS figures |

Every figure must represent the selected current physical model.
Do not substitute legacy six-species output because a file is convenient or visually similar.
“Latest model” does not imply that all historical accepted campaigns used one identical wheel: preserve their actual package identities and establish equation/parameter equivalence before pooling claims.
If a new run changes a result, update its figure, associated prose, abstract, conclusion and reviewer response together.

## 11. Table plan

| ID | Existing table/location | Final role and placement | Required cell/caption work |
|---|---|---|---|
| T1 | `absorber_literature_comparison.tex` | I2: what previous absorber studies already establish | Separate fitting from evaluation coverage; name every reported error quantity; define abbreviations; trim cells to comparable facts. |
| T2 | Thermodynamic approach table in Introduction | I4: structural differences among approaches | Keep eNRTL, Kent–Eisenberg, CPA and ePC-SAFT; remove the Henry row if it distracts from this four-way positioning. Give source support and avoid accuracy ranking. |
| T3 | Responsibility table in Model | M2: which submodel supplies each quantity | Distinguish EOS, reaction equilibrium, water closure, enhancement and empirical calorics; disclose reduced neutral vapor EOS composition versus wet-gas fugacity and the fixed-bulk-activity film approximation. Replace “benchmark” in caption with the actual model. |
| T4 | `nccc_one_bed_case_scope.tex` | E1: experimental inputs and assumptions | Put dry basis in gas headings, loading units in its heading, pressure location if source establishes it, and “reported capture” where appropriate. Retain original units and explicit assumed values. |
| T5 | `reactive_numerical_verification.tex` | R2: observed paired refinement response | Preserve outer mesh iteration meaning, normalization, node/Jacobian/equilibrium counts and conservation diagnostics. Label 293.219/273.006 W as axial signed net-enthalpy-flow ranges, with zero expected for the continuous equations; do not imply a passed energy-error threshold or attribute the discrepancy to empirical calorics. Keep separate-run timing out of refinement columns. |
| T6 | `transport_applicability.tex` | A9 and R10b: source domains, calculated conditions and adopted modifications | Separate published correlation error from current prediction error; identify modified coefficients and clarify loading/flooding language. Compare source Hatta range 35–165 with baseline 320–698 and MEA diffusivity limit 333 K with peak 347.86 K. Unknown domains remain unknown, not implicitly covered. |
| T7 | `appendix_parameter_scope.tex` | A9: concise input-group locator | Retain only if it directs readers to actual equations/values; delete from manuscript if it merely repeats T3 and appendix headings. |
| T8 | Component table in `epcsaft_parameter_summary.tex` | B1: reproducible component inputs | Values, units, component indexing, temperature dependence, source/fitting labels. |
| T9 | Ionic diameter table in same file | B2: reproducible ionic inputs | Packing versus electrostatic diameter meaning, units, rounded-value/full-precision distinction and source labels. |
| T10 | Pair table in same file | B3: complete binary interactions | All 36 independent pairs, symmetry, temperature slopes, zero values and inherited/fitted distinctions. |
| T11 | Association table in same file | B4: complete association description | Site/topology definitions, energy and volume units, combining rules and source labels. |

Do not add a table solely to repeat values already readable in a figure.
Runtime is a few scalar measurements and fits in prose; reaction coefficients may remain in their current compact mathematical presentation if fully specified.

## 12. Evidence and analysis boundaries

| Question | Present answer | Permitted interpretation | Remaining limit |
|---|---|---|---|
| Can explicit reactive thermodynamics be coupled to the column? | Nine-species equilibrium, CO₂ fugacity and column profiles are computed with the selected model. | Demonstrated coupling for the reported MEA calculations. | Reduced vapor-EOS composition, fixed-bulk-activity enhancement and empirical calorics limit the formulation; no isolated incremental benefit is measured. |
| Does the model agree with experiments? | Seven-case capture MAE 5.85 points; largest errors approximately ±12 points; packing-temperature profiles available. | Case-dependent physical comparison with visible discrepancies. | One campaign/configuration, missing inlet information and no general accuracy guarantee. |
| How do reported outputs change under the tested refinement? | Joint refinement changes capture 0.00510 points and peak temperature 0.04493 K. | Small observed Case 3C changes to compare with physical discrepancies and perturbation responses. | No certified reference error bound, all-case convergence or all-perturbation resolution claim. |
| Does the exported solution conserve the adopted enthalpy? | Continuous equations conserve signed net enthalpy; exported reference ranges decrease from 293.219 to 273.006 W under refinement. | A persistent nonzero numerical conservation diagnostic. | The cause and acceptable relative magnitude are not established; empirical caloric accuracy is a separate question. |
| Which thermodynamic inputs matter locally? | Tested ±5% MEA–water change gives the largest capture effect of the four tested inputs. | Ordering within this chosen perturbation design. | No fitted uncertainty distribution, joint uncertainty or universal importance ranking. |
| Which transport inputs matter locally? | Tested ±10% k_L changes give the largest capture effect of the three tested inputs. | Dependent transport/enhancement responses matter. | Different design magnitude from thermodynamic study; baseline Hatta and temperature extend beyond named source ranges, so sensitivity does not validate correlation accuracy. |
| Which tested operating changes produce the largest response? | Loading perturbations produce +3.75/−6.86-point capture changes. | Strongest response among these particular tested changes. | One lower-flow condition unavailable; no optimum or hydraulic/economic assessment. |
| What does a calculation cost? | Median coarse BVP time 34.66 s for the identified three-run campaign. | Cost of that model implementation and configuration. | No comparative solver claim or extrapolation to other meshes/packages. |

Additional work needed to populate this outline is primarily editorial and evidential: inspect final plots, confirm notation against implementation, complete missing coefficient/source definitions and reconcile scalar claims.
This plan does not require launching a new solver comparison, thermodynamic-model comparison or optimization campaign.
If a desired sentence needs evidence beyond these records, narrow the sentence or identify that evidence as future work.

## 13. Reviewer coverage — secondary to the story

This mapping follows the argument above; reviewer order must not determine section order.
Coverage identifies where an issue is addressed, not automatic agreement with every requested expansion.
The original comments are in `docs/reviewer_comments.txt`; the current response and checklist remain the response documents.

| Comment | Natural home | How the story addresses it | Honest extent |
|---|---|---|---|
| R1.1 Theoretical justification | I3, M3–M6, F1, T3 | Separate physical nonideality from reaction equilibrium and show their coupling. | Formulation and scoped column evidence; no universal accuracy claim. |
| R1.2 Physical versus chemical thermodynamics | I3, M3–M5 | Identify EOS contributions, reactions, activities and equilibrium constants. | Explicit distinction throughout text, equations and diagram. |
| R1.3 Predicted quantities and schematic | M2, M6, M10, M13, F1, T3 | Assign speciation, reduced vapor-EOS coefficient, wet-gas fugacity, fixed-bulk-activity film relation, water transfer and calorics explicitly. | No full-mixture vapor EOS or complete reactive-film/caloric treatment is implied. |
| R1.4 Meaning of fugacity benchmark | I5–I6, title, M6 | Replace the ambiguous benchmark objective with reactive model evaluation and response. | Scope revised; no unsupported comparative benchmark remains. |
| R1.5 Complete parameters | M7, B1–B7, T8–T11 | Print parameter values, units, rules and full-precision access. | Completeness still requires verifying all dielectric/solvation and reaction definitions. |
| R1.6 Binary interactions | B3, T10 | Show all independent pairs, symmetry and temperature dependence. | No hidden zero/default pair assumptions. |
| R1.7 Fitted versus literature inputs | I4, M7, B6 | Identify inherited, fitted, fixed and derived inputs and fitting data. | Distinguish additional binary fitting from total fitted quantities; compare named calibration tasks fairly and preserve prior-selection disclosure. |
| R1.8 Thermodynamic sensitivity | E7, R7–R8, F5 | Quantify four one-at-a-time input responses. | Sensitivity, not robustness proof or statistical uncertainty. |
| R1.9 Transport uncertainty | E8, R9–R10b, F6, A7/A9, T6 | Quantify viscosity/diffusivity/transfer changes and compare actual Hatta/temperature ranges with source limits. | No joint uncertainty estimate or validation of extrapolated correlations. |
| R1.10 Numerical accuracy/convergence | E3–E6, R2, T5 | Report observed refinement changes, iterations, residual histories, evaluation counts, initialization exceptions and nonzero enthalpy variation. | Solver-comparison objective removed; no certified resolution, complete energy conservation or three-method comparison claimed. |
| R1.11 Computational expense | E10, R3 | Report timing protocol, median/range, CPU, memory and mesh details. | One identified coarse implementation; no comparative speed claim. |
| R1.12 Other amines | R16 | State chemistry, property, parameterization and validation requirements. | Extension procedure, not demonstrated second-solvent performance. |
| R1.13 Applicability and limits | M6/M10/M13, R10b, R13–R16, C2/C4 | Explain vapor approximation, correlation extrapolation, caloric adequacy, numerical enthalpy discrepancy, operating and industrial-mixture limits. | Separate observed numerical behavior from model-form limitations and unmeasured causes of physical error. |
| R1.14 eNRTL/Kent–Eisenberg/CPA/ePC-SAFT comparison | I4, T2, R16 | Compare physical structures, inputs and applicability. | Literature scope comparison; no numerical accuracy or data-efficiency ranking. |
| R2.1 Reference formatting | D5 and final PDF review | First-citation numbering and bibliography consistency. | Verify after final section order is established. |
| R2.2 Literature gap | I1–I6, T1–T2 | Build the specific coupling/evaluation question from existing capabilities. | No unsupported “first” claim. |
| R2.3 Reproducible modeling parameters | M7, Appendices A/B, D2–D3 | Supply equations, values, source roles and exact numerical inputs. | Dependency availability must not be described as public availability. |
| R2.4 Broader validation | E1–E2, R4–R6, F3–F4, T4 | Extend physical comparison across seven reported cases. | Broader within the selected NCCC campaign; not independent multi-campaign validation. |
| R2.5 Operating conditions and optimum | E9, R11–R13, F7 | Quantify discrete flow/loading/temperature changes and interpret them. | Partial response: one missing condition and no optimization objective or optimum. |
| R2.6 Fitting another amine | R16, B6 | Explain solvent-specific fitting and independent evaluation requirements. | No new solvent fit or transferability result claimed. |
| F.1 Current corrected results | Source assignment and final checks | Use current nine-species figures and reconcile all result statements. | Figure freshness and content presence alone do not establish scientific validity. |

## 14. Sentence, paragraph and cross-section editing rules

1. **One sentence, one primary job.** Introduce, define, report, interpret or qualify; avoid packing all five functions into one sentence.
2. **Carry information forward.** Begin with the quantity or question established immediately before; introduce the new consequence at the end.
3. **One paragraph, one scientific question.** The lead announces it, the middle supplies evidence, and the final sentence states the consequence or opens the next question.
4. **No result before its basis.** Define settings and response metrics in Methods before interpreting them in Results.
5. **No repeated scope paragraph.** Put a short necessary qualification beside the affected result and reserve combined implications for R14–R16.
6. **Use precise quantities.** Capture is percent; capture change/error is percentage points; temperature change is K; loading is mol CO₂ per mol MEA; mass ratio identifies the reported dry-gas basis.
7. **Distinguish reference calculations.** Campaign/operating Case 3C uses coarse settings; thermodynamic/transport sensitivity uses the refined reference; timing uses its identified coarse implementation.
   “Refined” identifies the changed settings; it is not a synonym for verified accuracy or complete conservation.
8. **Preserve chemistry.** Distinguish apparent components, true species, free molecular CO₂, activities, fugacity and equilibrium constants. Resolve overloaded symbols such as m before editing equations.
9. **Interpret with the right strength.** “Consistent with” is appropriate for profile-based mechanism interpretation; “demonstrates” requires evidence that excludes the relevant alternatives.
10. **Make captions independently interpretable.** State system, changed quantity, baseline, plotted quantities/units, marks and any qualification needed to avoid a false reading.
11. **Make table cells scientifically comparable.** Name the measured quantity, calibration/evaluation role, units and source; never use an unexplained error percentage.
12. **Apply CSE terminology by meaning.** Remove vague claims such as unqualified robustness or scalability and implementation jargon in scientific prose. Statistical regression and mathematical contraction remain valid technical terms.
13. **Remove process history.** The article describes the final calculation and evidence, not archival recovery, task handoffs, failed editorial approaches or reviewer-checklist progress.
14. **Keep source reviewable.** One prose sentence per LaTeX source line, stable paragraph comments if useful, and existing semantic equation/figure/table labels where possible.
15. **Separate equation properties from numerical observations.** Continuous conservation of the adopted enthalpy, variation in exported numerical profiles and physical caloric accuracy are three distinct statements; do not use one as an explanation for another without evidence.
16. **State actual applicability.** Pair named source ranges with calculated ranges, distinguish comparison ranges from proven validity boundaries, and never present a perturbation response as validation of an extrapolated correlation.
17. **Expose consequential approximations and initial guesses.** Distinguish reduced EOS vapor composition from wet gas, bulk-activity film approximation from resolved film chemistry, and reaction-extent seeds from initial column profiles.

## 15. Order of execution and completion checks

### Pass 1 — Settle definitions and evidence

Confirm the selected physical formulation and final run sources.
Resolve U/U_T, axial-coordinate scaling, water-vapor closure, reaction-standard-state wording and any missing coefficient/source definitions.
Carry the verified reduced vapor-composition mapping and H_bulk relation into the model text, diagram and responsibility table.
Separate continuous enthalpy conservation, exported numerical variation and empirical caloric adequacy in every assignment that discusses energy.
Check the retained baseline Hatta/temperature ranges against their named source limits and retain Case 1C's 0.0001 reaction-extent starting fraction separately from column-profile initialization.
Verify that the operating prose uses six accepted conditions and one unavailable lower-flow condition.
Record any unresolved scientific fact in the existing reviewer response rather than inventing a value.

**Complete when:** every planned claim has an identified source and no unresolved definition can change its meaning.

### Pass 2 — Build the body skeleton

Reorder existing prose and equations into M1–M15 and E1–E10; place the case-input table in Methods; remove solver-comparison material and duplicate introductions.
Split the appendix by property correlations and reactive parameter inputs.
Place floats by their first substantive discussion and remove the redundant main-text temperature close-up if the campaign panel is readable.

**Complete when:** section headings alone reproduce the central argument and every retained equation/float has an explanatory paragraph.

### Pass 3 — Populate results and interpretation

Write R1–R16, including the separate applicability paragraph R10b, using final retained values.
Inspect the final temperature panels before drafting case-specific thermal interpretation.
Use the actual correlation ranges when interpreting transport responses, and report persistent numerical enthalpy variation without guessing its cause.
Keep full scalar precision in data files and appropriate rounded values in prose.
Check the paired order of perturbations, sign conventions, baselines and unavailable conditions.

**Complete when:** each results paragraph states a finding, shows its evidence and explains its limited meaning.

### Pass 4 — Write the opening and ending

Write I1–I6 around the completed body.
Write C1–C4 from the established results, then AB1–AB8 and the final title.
Check that the abstract includes physical comparison, sensitivity, operating response and the main interpretation.
Attach loading levels to their abstract/conclusion responses and use observational refinement wording throughout.

**Complete when:** title, abstract and conclusion describe the same paper, and every abstract/conclusion claim points to a body result.

### Pass 5 — Read at every level

Read consecutive sentences for logical progression, then each paragraph as one argument, then subsection transitions, then the paper using only its topic sentences.
Audit captions, table cells and appendix prose with the same terminology and evidence standards.
Check all 20 reviewer comments against the secondary mapping without rearranging the scientific story around the checklist.

**Complete when:** no paragraph is a collection of unrelated facts, no section repeats another section's job, and partial responses remain honestly identified.

### Pass 6 — Verify the assembled manuscript

Use the existing LaTeX build and figure-freshness commands without rerunning the scientific model as part of rendering.
Verify references, labels, units, scalar consistency and the actual figure files included in the PDF.
Inspect final page layout, figure legibility, table overflow and float placement.
Apply the repository's final immutable-engine integration check when preparing final manuscript results; report any dependency failure rather than changing package state during editorial work.
Update the existing response/checklist only after the manuscript content is stable.

**Complete when:** the PDF tells the planned story, the numerical claims match their retained sources, and the remaining scientific limits are visible to a reader without access to the reviewer correspondence.

## Source documents used for this plan

- `docs/latex/main.tex`, the five current section sources, `appendices/appendix_properties.tex` and the current table sources.
- `docs/reviewer_comments.txt`, `docs/fallback_reviewer_response.md` and `docs/latex/scripts/reviewer_checklist.json`.
- `docs/code_to_paper_traceability.md`, `docs/selected-reactive-parameters.md` and `REPRODUCE.md` for numerical source assignments.
- The completed section-by-section editorial audit and its current-result follow-up.
- The main chat's consolidated three-Sol review of this plan, checked against `Thermodynamics/thermo_models.py`, `Thermodynamics/reactive_bundle.py`, current model energy/film equations and the transport-applicability table; source modules are under `src/mea_absorption_column/`.
- `analyses/transport_sensitivity/figures/response/output/profiles.csv`, baseline `Ha` values 319.8659895–698.2728584, and `analyses/nccc_validation/figures/reactive_parallel/output/summary.csv`, Case 1C `cold_start_seed_fraction` 0.0001, directly checked during revision.

The generated `docs/latex/builds/manuscript-story-map.md` describes an older manuscript structure and is not the authority for this plan.
The historical August coordination plan governs a different scientific/deadline sequence and is not being rewritten here.
