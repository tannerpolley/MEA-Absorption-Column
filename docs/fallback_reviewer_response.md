# Response to reviewers

**Manuscript:** Reactive ePC-SAFT Modeling of Carbon Dioxide Absorption in Aqueous MEA

**Authors:** Tanner W. Polley and John D. Hedengren

We thank the editor and both reviewers for their careful assessment and constructive comments. We have revised the manuscript to explain the coupling between ePC-SAFT and chemical equilibrium, document the parameters and numerical method, and present the absorber comparison and sensitivity studies more clearly. The revised article focuses on the coupled reactive absorber model rather than a comparison of numerical solution methods.

Each numbered comment is reproduced below, followed by our response and the corresponding manuscript locations. Locations refer to the clean revised manuscript supplied with this response; Table 11 is included in that version.

## Reviewer 1

### Comment 1

> 1. The theoretical justification for applying ePC-SAFT to chemical absorption is insufficient. This is the largest weakness of the manuscript. Throughout the manuscript, the authors present ePC-SAFT as the thermodynamic engine for predicting CO2 fugacity and absorber performance. However, they never explain why a molecular equation of state developed for physical interactions can accurately describe a strongly reactive absorption system.

**Response.** We agree that the equation of state alone does not describe chemical reaction equilibrium. We have substantially revised the formulation and its explanation. Five equilibrium reactions among nine liquid species are now solved with carbon and nitrogen conservation, charge neutrality and the stated activity convention. ePC-SAFT supplies the nonideal thermodynamic quantities entering these equilibrium relations and the CO2 fugacity driving force. The revised equations explain how molecular thermodynamics and chemical equilibrium are coupled, while conventional reaction enhancement and empirical caloric properties remain separate parts of the column model. The article evaluates this coupled formulation against absorber observations; it does not use that comparison as independent validation of every thermodynamic parameter.

**Manuscript changes.** Section 2 (pp. 3–11), especially reactive equilibrium and ePC-SAFT; Figure 1 (p. 4); Table 3 (p. 5); Appendix B (pp. 28–30).

### Comment 2

> 2. ePC-SAFT is essentially a physical equation of state rather than a reaction model. The manuscript currently blurs the distinction between molecular thermodynamics and chemical thermodynamics. The original PC-SAFT framework accounts for hard-chain repulsion, dispersion interactions, association (hydrogen bonding), dipolar interactions, ionic electrostatics (extended versions). These terms describe physical intermolecular interactions.

**Response.** We have made this distinction explicit. ePC-SAFT describes molecular and ionic nonideality; the reaction network and equilibrium constants determine the chemical-equilibrium constraints. Their coupled solution gives the liquid species distribution. Table 3 identifies the quantities supplied by each model component, including the separate enhancement, transport and caloric relations. This replaces the ambiguous presentation of the equation of state as a reaction model.

**Manuscript changes.** Section 2 (pp. 3–11); Table 3 (p. 5); Appendix B (pp. 28–30).

### Comment 3

> 3. The manuscript should explicitly identify which quantities are predicted by ePC-SAFT and which are obtained from reaction equilibrium. The workflow is currently ambiguous. A schematic workflow should be provided illustrating

**Response.** We have added a schematic that follows the column state through reactive speciation, thermodynamic and transport properties, reaction enhancement, interphase fluxes and the column balances. The accompanying table identifies which quantities come from ePC-SAFT, reaction equilibrium and the empirical property relations. The text also distinguishes the liquid equilibrium calculation from the reduced neutral-species vapor calculation.

**Manuscript changes.** Figure 1 (p. 4); Table 3 (p. 5); Section 2 (pp. 3–11).

### Comment 4

> 4. The manuscript repeatedly uses the term "ePC-SAFT fugacity benchmark", but its meaning is not sufficiently defined. The authors should clearly define this terminology in the Introduction to avoid ambiguity.

**Response.** We have removed the term “ePC-SAFT fugacity benchmark” and revised the title to “Reactive ePC-SAFT Modeling of Carbon Dioxide Absorption in Aqueous MEA.” The Introduction now defines the contribution as a coupled reactive absorber formulation evaluated using seven NCCC cases, Case 3C refinement and thermodynamic, transport and operating perturbations. The revised article no longer presents an isolated fugacity benchmark or a comparison of three numerical methods.

**Manuscript changes.** Title and Abstract (p. 1); Section 1 (pp. 1–3); Sections 3 and 4 (pp. 11–20).

### Comment 5

> 5. Provide more information on ePC-SAFT parameters, since these parameters determine prediction accuracy, a complete parameter table should be included either in the main text or Supporting Information.

**Response.** Appendix B now lists the nine-species component parameters, ionic diameters, binary interaction coefficients, association parameters and reaction-equilibrium correlations. The tables and accompanying text give units, model choices, literature sources and fitted quantities. Standard-state transformations and temperature ranges are stated so that the tabulated reaction inputs can be interpreted consistently with the equations.

**Manuscript changes.** Appendix B (pp. 28–30); Tables 7–9 (p. 28) and Table 10 (p. 29).

### Comment 6

> 6. The manuscript does not clearly state which binary interaction parameters (kij) were adopted,

**Response.** The complete symmetric matrix of all 36 off-diagonal binary interaction coefficients for the nine-species mixture is now printed, including zero entries and same-charge exclusions. The text also states the zero reciprocal-temperature slope and the association combining rule used in the calculation.

**Manuscript changes.** Appendix B, binary-interaction matrix in Table 9 (p. 28), and association description (pp. 29–30).

### Comment 7

> 7. whether they were fitted, or whether literature values were directly used. The predictive capability of the model strongly depends on these parameters, and additional clarification is necessary.

**Response.** We now distinguish literature values from coefficients fitted for this system. The MEA–water interaction was fitted to binary vapor–liquid equilibrium data; the amine-ion coefficients are attributed to the authors’ ion and speciation fit, to be reported separately; and the CO2 dispersion energy was chosen jointly with the R4 correlation. The appendix identifies the reaction-constant adjustments and explains which measurements had a role in parameter selection. It also states that the amine-ion fit has not been independently validated and that no thermodynamic parameter was fitted to the NCCC cases.

**Manuscript changes.** Section 2, parameter description (p. 7); Appendix B (pp. 28–30) and Tables 7–10.

### Comment 8

> 8. The authors should evaluate the sensitivity of the simulation results to key thermodynamic parameters. This analysis would improve confidence in the robustness of the proposed benchmark.

**Response.** We have added one-at-a-time ±5% perturbations of the MEA–water and CO2–water binary interactions and the R4 and R5 equilibrium constants. Section 3 defines the changes, including how equilibrium-constant multipliers preserve logarithmic temperature derivatives. Figure 5 reports the resulting capture and peak-temperature changes. The MEA–water interaction gives the largest capture response among the tested thermodynamic inputs, approximately −0.82 and +0.76 percentage points. We compare the responses with the Case 3C refinement changes to distinguish small numerical effects from the larger responses. These perturbations are a bounded sensitivity comparison, rather than statistical uncertainty intervals.

**Manuscript changes.** Section 3.3 (p. 13); Section 4.2 and Figure 5 (p. 18); Conclusions (p. 21).

### Comment 9

> 9. The manuscript focuses on thermodynamic uncertainty but provides little discussion regarding transport-property uncertainty. Mass-transfer coefficients, diffusivity, and liquid viscosity may significantly influence absorber performance and should at least be discussed qualitatively.

**Response.** We have added both a quantitative transport sensitivity study and a discussion of correlation applicability. Separate ±10% changes to liquid viscosity, free-CO2 diffusivity and the liquid-side mass-transfer coefficient are evaluated with the same Case 3C reference. The liquid-side transfer coefficient produces the largest capture response of the tested transport inputs, approximately 0.53 percentage points. Table 6 and Appendix A identify correlation ranges and distinguish published correlation errors from uncertainty in the present column calculation.

**Manuscript changes.** Section 3.3 (p. 13); Section 4.2 (pp. 18–19); Figure 6 (p. 19); Table 6 (p. 27); Appendix A (pp. 22–27).

### Comment 10

> 10. Three numerical methods are compared. However, the manuscript mainly reports convergence time. The comparison would be more convincing if the authors additionally reported iteration numbers, Jacobian evaluations, nonlinear residual histories, mesh refinement histories.

**Response.** The revised manuscript focuses on the reactive absorber formulation and no longer compares three numerical methods. For the collocation method used throughout the revised study, Table 5 now reports the original paired Case 3C refinement: initial and final node counts, two mesh iterations, iteration-level RMS residuals, final boundary residuals, right-hand-side evaluations and Jacobian evaluations. The two calculations use 21→22 and 41→42 nodes, with four Jacobian evaluation batches in each. The text distinguishes mesh-refinement iterations from inner Newton steps. Detailed diagnostics for the other reported calculations are consolidated in Table 11. Additional Case 3C refinement is described concisely, with its energy sampling grid identified.

**Manuscript changes.** Section 3.2 (pp. 12–13); Section 4.1 (pp. 14–17); Table 5 and additional refinement statement (p. 15); Table 11 (p. 31).

### Comment 11

> 11. Since one objective is benchmarking numerical methods, the computational expense should be quantified. For example, CPU time, memory usage, mesh size, nonlinear iterations, under identical hardware conditions.

**Response.** We retain computational cost as a measured property of the reported model, rather than a ranking of numerical methods. Three fresh-start coarse Case 3C repeats on an AMD Ryzen 5 5500 with single-thread libraries give a median BVP wall time of 34.66 s, with a range of 34.41–34.89 s. The fastest BVP run takes 34.41 s wall time and 34.23 s CPU time; the total including initialization and profile export is 51.45 s, with peak resident memory of 203.51 MiB. The methods identify hardware, timing boundaries, mesh settings and derivative evaluation. The abstract and conclusion now identify the approximately 35 s value specifically as BVP wall time.

**Manuscript changes.** Section 3.2 (pp. 12–13); Section 4.1 and Table 5 (p. 15); Abstract (p. 1); Conclusions (p. 21).

### Comment 12

> 12. The benchmark is demonstrated only for MEA. The manuscript should briefly discuss whether the proposed framework can be directly extended to DEA, MDEA, AMP, PZ, blended amines, or whether additional parameterization would be required.

**Response.** Section 4.4 now discusses extensions to DEA, MDEA, AMP, PZ and blends. Applicable molecular, association and ionic parameters may be reused, but each solvent requires a consistent reaction network, equilibrium constants and transport inputs. Additional calibration and comparison with thermodynamic and column measurements would be needed to establish performance for a new solvent. We present these extensions as proposed studies and do not claim that another amine has already been evaluated.

**Manuscript changes.** Section 4.4 (p. 20); Conclusions (p. 21).

### Comment 13

> 13. The Conclusion primarily emphasizes the advantages of the benchmark. It would be beneficial to briefly discuss the current limitations, such as dependence on reliable thermodynamic parameters, applicability to reactive electrolyte systems, extension to multicomponent industrial flue gases, future incorporation of reaction kinetics.

**Response.** We have revised the Conclusion to retain a concise statement of the model’s limits alongside its principal results. The discussion identifies uneven agreement across the seven cases and the use of conventional enhancement, empirical caloric properties and transport correlations with restricted applicability. Section 4.4 explains how these choices limit interpretation and motivates a liquid-film treatment and further solvent and operating studies. Extensions beyond the present MEA/flue-gas conditions and a more detailed reaction–transport treatment are future work, not demonstrated capabilities.

**Manuscript changes.** Section 4.4 (p. 20); Conclusions (p. 21); Table 6 (p. 27); Appendix A (pp. 22–27).

### Comment 14

> 14. The manuscript would benefit from adding a concise comparison between electrolyte-NRTL, Kent-Eisenberg, CPA, and ePC-SAFT. Although the focus is on ePC-SAFT, discussing the advantages and limitations relative to other widely used thermodynamic models would better demonstrate the novelty and necessity of the proposed benchmark. Such a comparison would also help readers identify the situations in which ePC-SAFT provides clear advantages over traditional electrolyte models.

**Response.** Table 2 now compares Kent–Eisenberg, electrolyte-NRTL, CPA and ePC-SAFT in terms of chemical-equilibrium treatment, reusable inputs, calibration requirements and applicability. The Introduction explains the motivation for the molecular description without claiming an established fitting-effort advantage. Appendix B identifies the fitted and literature inputs used here. Section 4.4 proposes a comparison within the same column model to assess accuracy and fitting requirements under consistent conditions.

**Manuscript changes.** Section 1 (pp. 1–3); Table 2 (p. 3); Section 4.4 (p. 20); Appendix B (pp. 28–30).

## Reviewer 2

### Comment 1

> 1. Formats of this paper need rearranged. For example, reference number should start from [1].

**Response.** We have revised the formatting and rebuilt the manuscript. The reference list starts at [1], citations follow numerical order, and equations, figures and tables use consistent numbering and cross-references. We have also simplified repeated numerical-reporting paragraphs and reduced excessive precision in the narrative while retaining detailed values in the tables.

**Manuscript changes.** Throughout the manuscript; References (pp. 32–33); Table 5 (p. 15); Table 11 (p. 31).

### Comment 2

> 2. Shortages of MEA absorber modeling should be clarified in literature review section.

**Response.** The Introduction and Table 1 now compare representative absorber studies in terms of thermodynamics, film treatment, experimental coverage and numerical and parameter reporting. This makes the motivation for the present coupling and evaluation clearer while distinguishing an omission in a published report from a limitation of the underlying model. The contribution is stated as the coupled reactive-EOS column formulation and its evaluation.

**Manuscript changes.** Section 1 (pp. 1–3); Table 1 (p. 2).

### Comment 3

> 3. Parameters for the modeling should be presented for peer repetition.

**Response.** We have expanded Appendix B to give the component, ionic, binary, association and reaction parameters, together with units, sources and standard-state conventions. Appendix A supplies the property correlations, and Section 3 gives the column inputs and solution settings. The availability statements identify the absorber repository and explain that the ePC-SAFT package is available from the authors on request, with public release planned. Detailed numerical diagnostics are collected in Table 11.

**Manuscript changes.** Section 3 (pp. 11–13); Table 4 (p. 12); Appendices A and B (pp. 22–30); Table 11 (p. 31); Data Availability and Code Availability (p. 32).

### Comment 4

> 4. There are more reported MEA absorber results. Modeling validations should also be conducted for the other results.

**Response.** We have expanded the comparison to seven NCCC one-bed cases, 1C–7C. Figure 3 compares capture for all seven cases, giving a mean absolute error of 5.85 percentage points; Figure 4 compares calculated phase-temperature profiles with the reported packing temperatures. The text retains the larger signed capture errors of +11.45 and −11.93 percentage points and identifies assumed inlet conditions. Because the source does not assign the temperature measurements to a specific phase, we use the temperature profiles as a qualitative comparison. The Case 3C paired refinement is identified separately from the seven-case calculations.

**Manuscript changes.** Section 3.1 (pp. 11–12); Table 4 (p. 12); Section 4.1 (pp. 14–17); Figures 3 and 4 (pp. 16 and 17); Table 5 (p. 15).

### Comment 5

> 5. Based on the model proposed in this paper, MEA absorber performance under various run conditions should be obtained and discussed for optimal operating.

**Response.** We have added operating perturbations in liquid-to-gas ratio, lean loading and inlet temperature. Six of the seven specified conditions converged, including the baseline; the lower liquid-to-gas condition did not converge and is explicitly omitted from Figure 7. Lean loadings of 0.225 and 0.275 mol CO2 per mol MEA change capture by +3.75 and −6.86 percentage points relative to 0.25, giving the largest responses among the tested operating changes. Higher liquid flow and the inlet-temperature changes have smaller effects. These results identify useful operating directions but do not establish an optimum. Section 4.4 explains that optimization requires an energy or cost objective together with capture, hydraulic and thermal constraints. Thus, the revised manuscript addresses the operating-response part of the request and retains constrained optimization as future work.

**Manuscript changes.** Section 3.3 (p. 13); Section 4.3 (pp. 19–20); Figure 7 and Section 4.4 (p. 20); Conclusions (p. 21).

### Comment 6

> 6. Besides MEA, other amines for CO2 capture process are more and more used. How to fit new amine to this model?

**Response.** We have clarified how a new amine would enter the formulation. Applicable component, association and ionic parameters can provide starting inputs, while solvent-specific reactions, equilibrium constants and transport properties must be supplied consistently. Missing or unsuitable inputs require calibration to thermodynamic measurements, followed by comparison with independent thermodynamic and column observations. Section 4.4 identifies DEA, MDEA, AMP, PZ and blends as proposed tests of which inputs transfer and which require further calibration. This is a description of the extension procedure; the revised article does not report a new-amine fit or validation.

**Manuscript changes.** Section 4.4 (p. 20); Appendix B for the MEA parameter example (pp. 28–30); Conclusions (p. 21).
