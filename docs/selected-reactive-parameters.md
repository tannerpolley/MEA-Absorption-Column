# Selected reactive parameters: supporting record

Updated 2026-09-04. Input documentation, not a new column result.

The exact selected inputs are [parameters.json](../src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle/parameters.json) (SHA-256 `a9186c93759f2e2c02a6c913350ad06a244fff3f82503820c9962b3df8dd40d9`) and [reaction-system.json](../src/mea_absorption_column/data/epcsaft_datasets/MEA_reactive_epcsaft_bundle/reaction-system.json) (SHA-256 `810dfec15760cf74451df91743d6e63684cee93ddaf3e1ff4e42bf4a686afe29`). These tables document the parameter file and effective reaction correlations, including typed fitted records that supersede the matching source export. The matching anchored thermochemistry and adoption receipt are retained with the validated bundle; no parameter fitting is performed here. Full precision and source locators remain in the JSON; this record is a supporting view, not an executable parameter copy. Software identity and replication commands are in [REPRODUCE.md](../REPRODUCE.md).

## Chemistry and model responsibility

The selected runtime couples nine-species EOS activities, five reaction equilibria, pressure, component balances and electroneutrality. Reaction constants depend on temperature, not the initial composition. The EOS describes nonideality; the reaction constants and conserved components determine chemical equilibrium jointly with it. EOS density supplies true concentrations and the same solved state supplies liquid CO2 fugacity. The conventional enhancement-factor approach is retained. Its coupled film-resistance conversion uses the bulk EOS fugacity/free-CO2 concentration ratio under a locally constant activity-coefficient approximation; this is not a derivative with respect to reactive loading or a resolved reactive-film replacement. The coupling derivation and numerical checks are retained in the reviewer-energy record.

The model selections in the parameter document are:

| Family | Choice |
| --- | --- |
| base | pc-saft |
| association | general-site |
| electrolyte | born |
| permittivity | solvent-only |

No polar parameter family is selected. The matching source reaction export describes Debye-Huckel plus original Born and Uyan solvent mixing; executable model selection and fitted coefficients come from the selected parameter document rather than free-text export descriptions.

## Component and fixed coefficients

| Species | Coefficient | Value | Unit | Source ID |
| --- | --- | --- | --- | --- |
| CO2 | dispersion_energy_over_k | 173.44025 | kelvin | mea-best-in-slot-campaign-2026-09-02 |
| CO2 | segment_count | 2.0729 | dimensionless | mea-retained-phase-two-artifact |
| CO2 | segment_diameter | 2.7852 | angstrom | mea-retained-phase-two-artifact |
| CO2 | solvation_factor | 1 | dimensionless | mea-retained-phase-two-artifact |
| CO2 | charge_number | 0 | elementary-charge | mea-retained-phase-two-artifact |
| CO2 | molar_mass | 0.04401 | kilogram / mole | mea-retained-phase-two-artifact |
| MEA | dispersion_energy_over_k | 277.174 | kelvin | mea-retained-phase-two-artifact |
| MEA | relative_permittivity | 32 | dimensionless | mea-retained-phase-two-artifact |
| MEA | segment_count | 3.0353 | dimensionless | mea-retained-phase-two-artifact |
| MEA | segment_diameter | 3.0435 | angstrom | mea-retained-phase-two-artifact |
| MEA | solvation_factor | 1 | dimensionless | mea-retained-phase-two-artifact |
| MEA | charge_number | 0 | elementary-charge | mea-retained-phase-two-artifact |
| MEA | molar_mass | 0.06108 | kilogram / mole | mea-retained-phase-two-artifact |
| H2O | dispersion_energy_over_k | 353.95 | kelvin | mea-retained-phase-two-artifact |
| H2O | segment_count | 1.2047 | dimensionless | mea-retained-phase-two-artifact |
| H2O | solvation_factor | 1.5 | dimensionless | mea-retained-phase-two-artifact |
| H2O | charge_number | 0 | elementary-charge | mea-retained-phase-two-artifact |
| H2O | molar_mass | 0.01801528 | kilogram / mole | mea-retained-phase-two-artifact |
| MEAH+ | born_diameter | 3.53322927146 | angstrom | mea-retained-phase-two-artifact |
| MEAH+ | debye_huckel_diameter | 3.0668752979568 | angstrom | mea-retained-phase-two-artifact |
| MEAH+ | dispersion_energy_over_k | 232.687201645 | kelvin | mea-retained-phase-two-artifact |
| MEAH+ | packing_diameter | 3.0668752979568 | angstrom | mea-retained-phase-two-artifact |
| MEAH+ | segment_count | 1 | dimensionless | mea-retained-phase-two-artifact |
| MEAH+ | segment_diameter | 3.48508556586 | angstrom | mea-retained-phase-two-artifact |
| MEAH+ | solvation_factor | 1 | dimensionless | mea-retained-phase-two-artifact |
| MEAH+ | charge_number | 1 | elementary-charge | mea-retained-phase-two-artifact |
| MEAH+ | molar_mass | 0.06209 | kilogram / mole | mea-retained-phase-two-artifact |
| MEACOO- | born_diameter | 3.54107030822 | angstrom | mea-retained-phase-two-artifact |
| MEACOO- | debye_huckel_diameter | 3.1111830263448 | angstrom | mea-retained-phase-two-artifact |
| MEACOO- | dispersion_energy_over_k | 453.265244384 | kelvin | mea-retained-phase-two-artifact |
| MEACOO- | packing_diameter | 3.1111830263448 | angstrom | mea-retained-phase-two-artifact |
| MEACOO- | segment_count | 1 | dimensionless | mea-retained-phase-two-artifact |
| MEACOO- | segment_diameter | 3.53543525721 | angstrom | mea-retained-phase-two-artifact |
| MEACOO- | solvation_factor | 1 | dimensionless | mea-retained-phase-two-artifact |
| MEACOO- | charge_number | -1 | elementary-charge | mea-retained-phase-two-artifact |
| MEACOO- | molar_mass | 0.10408 | kilogram / mole | mea-retained-phase-two-artifact |
| HCO3- | born_diameter | 3 | angstrom | mea-retained-phase-two-artifact |
| HCO3- | debye_huckel_diameter | 2.5780480000000003 | angstrom | mea-retained-phase-two-artifact |
| HCO3- | dispersion_energy_over_k | 70 | kelvin | mea-retained-phase-two-artifact |
| HCO3- | packing_diameter | 2.5780480000000003 | angstrom | mea-retained-phase-two-artifact |
| HCO3- | segment_count | 1 | dimensionless | mea-retained-phase-two-artifact |
| HCO3- | segment_diameter | 2.9296 | angstrom | mea-retained-phase-two-artifact |
| HCO3- | solvation_factor | 1 | dimensionless | mea-retained-phase-two-artifact |
| HCO3- | charge_number | -1 | elementary-charge | mea-retained-phase-two-artifact |
| HCO3- | molar_mass | 0.0610168 | kilogram / mole | mea-retained-phase-two-artifact |
| CO3^2- | born_diameter | 3 | angstrom | mea-retained-phase-two-artifact |
| CO3^2- | debye_huckel_diameter | 2.149136 | angstrom | mea-retained-phase-two-artifact |
| CO3^2- | dispersion_energy_over_k | 249.26 | kelvin | mea-retained-phase-two-artifact |
| CO3^2- | packing_diameter | 2.149136 | angstrom | mea-retained-phase-two-artifact |
| CO3^2- | segment_count | 1 | dimensionless | mea-retained-phase-two-artifact |
| CO3^2- | segment_diameter | 2.4422 | angstrom | mea-retained-phase-two-artifact |
| CO3^2- | solvation_factor | 1 | dimensionless | mea-retained-phase-two-artifact |
| CO3^2- | charge_number | -2 | elementary-charge | mea-retained-phase-two-artifact |
| CO3^2- | molar_mass | 0.06001 | kilogram / mole | mea-retained-phase-two-artifact |
| H3O+ | born_diameter | 1.218 | angstrom | mea-retained-phase-two-artifact |
| H3O+ | debye_huckel_diameter | 3.049552 | angstrom | mea-retained-phase-two-artifact |
| H3O+ | dispersion_energy_over_k | 500 | kelvin | mea-retained-phase-two-artifact |
| H3O+ | packing_diameter | 3.049552 | angstrom | mea-retained-phase-two-artifact |
| H3O+ | segment_count | 1 | dimensionless | mea-retained-phase-two-artifact |
| H3O+ | segment_diameter | 3.4654 | angstrom | mea-retained-phase-two-artifact |
| H3O+ | solvation_factor | 1 | dimensionless | mea-retained-phase-two-artifact |
| H3O+ | charge_number | 1 | elementary-charge | mea-retained-phase-two-artifact |
| H3O+ | molar_mass | 0.01902 | kilogram / mole | mea-retained-phase-two-artifact |
| OH- | born_diameter | 3.08107689400404 | angstrom | mea-retained-phase-two-artifact |
| OH- | debye_huckel_diameter | 1.775576 | angstrom | mea-retained-phase-two-artifact |
| OH- | dispersion_energy_over_k | 650 | kelvin | mea-retained-phase-two-artifact |
| OH- | packing_diameter | 1.775576 | angstrom | mea-retained-phase-two-artifact |
| OH- | segment_count | 1 | dimensionless | mea-retained-phase-two-artifact |
| OH- | segment_diameter | 2.0177 | angstrom | mea-retained-phase-two-artifact |
| OH- | solvation_factor | 1 | dimensionless | mea-retained-phase-two-artifact |
| OH- | charge_number | -1 | elementary-charge | mea-retained-phase-two-artifact |
| OH- | molar_mass | 0.01701 | kilogram / mole | mea-retained-phase-two-artifact |

## Pair coefficients

All retained pair coefficients, including explicit zeros, are listed; do not substitute the historical MEA/water coefficient.

| Pair | Family | Value | Source ID |
| --- | --- | --- | --- |
| HCO3- / CO3^2- | k_ij | 1 | figiel-2025 |
| HCO3- / H3O+ | k_ij | 0 | mea-retained-phase-two-artifact |
| HCO3- / OH- | k_ij | 1 | figiel-2025 |
| HCO3- / MEACOO- | k_ij | 1 | figiel-2025 |
| MEACOO- / CO3^2- | k_ij | 1 | figiel-2025 |
| MEACOO- / H3O+ | k_ij | 0 | mea-retained-phase-two-artifact |
| MEACOO- / OH- | k_ij | 1 | figiel-2025 |
| HCO3- / CO2 | k_ij | 0 | mea-retained-phase-two-artifact |
| MEACOO- / CO2 | k_ij | 0 | mea-retained-phase-two-artifact |
| CO2 / CO3^2- | k_ij | 0 | mea-retained-phase-two-artifact |
| CO2 / H3O+ | k_ij | 0 | mea-retained-phase-two-artifact |
| CO2 / OH- | k_ij | 0 | mea-retained-phase-two-artifact |
| CO2 / MEA | k_ij | 0 | mea-retained-phase-two-artifact |
| CO2 / MEA | k_ij_reciprocal_temperature_slope | 0 | mea-retained-phase-two-artifact |
| CO2 / MEAH+ | k_ij | 0 | mea-retained-phase-two-artifact |
| CO2 / H2O | k_ij | 0.013262879176919628 | mea-retained-phase-two-artifact |
| CO3^2- / H3O+ | k_ij | 0 | mea-retained-phase-two-artifact |
| CO3^2- / OH- | k_ij | 1 | figiel-2025 |
| H3O+ / OH- | k_ij | 0 | mea-retained-phase-two-artifact |
| HCO3- / MEA | k_ij | 0 | mea-retained-phase-two-artifact |
| MEACOO- / MEA | k_ij | 0 | mea-retained-phase-two-artifact |
| CO3^2- / MEA | k_ij | 0 | mea-retained-phase-two-artifact |
| H3O+ / MEA | k_ij | 0 | mea-retained-phase-two-artifact |
| OH- / MEA | k_ij | 0 | mea-retained-phase-two-artifact |
| MEA / MEAH+ | k_ij | 0 | mea-retained-phase-two-artifact |
| MEA / H2O | k_ij | -0.07352749874985018 | cai-1996-neutral-refit |
| HCO3- / MEAH+ | k_ij | 0 | mea-retained-phase-two-artifact |
| MEACOO- / MEAH+ | k_ij | -0.00201813457644 | mea-retained-phase-two-artifact |
| CO3^2- / MEAH+ | k_ij | 0 | mea-retained-phase-two-artifact |
| H3O+ / MEAH+ | k_ij | 1 | figiel-2025 |
| OH- / MEAH+ | k_ij | 0 | mea-retained-phase-two-artifact |
| HCO3- / H2O | k_ij | 0 | mea-retained-phase-two-artifact |
| MEACOO- / H2O | k_ij | 0 | mea-retained-phase-two-artifact |
| CO3^2- / H2O | k_ij | -0.25 | mea-retained-phase-two-artifact |
| H3O+ / H2O | k_ij | 0.25 | mea-retained-phase-two-artifact |
| OH- / H2O | k_ij | -0.25 | mea-retained-phase-two-artifact |
| MEAH+ / H2O | k_ij | 0 | mea-retained-phase-two-artifact |

## Association

| Species | Site | Role | Multiplicity |
| --- | --- | --- | --- |
| MEA | a | donor | 1 |
| MEA | b | acceptor | 1 |
| H2O | a | donor | 1 |
| H2O | b | acceptor | 1 |
| CO2 | a | donor | 1 |
| CO2 | b | acceptor | 1 |

| Site pair | Energy/k (K) | Volume | Rule |
| --- | --- | --- | --- |
| MEA:a / MEA:b | 2586.3 | 0.03747 | explicit |
| H2O:a / H2O:b | 2425.7 | 0.04509 | explicit |
| CO2:a / H2O:b | 1212.85 | 0.04509 | explicit |
| CO2:b / H2O:a | 1212.85 | 0.04509 | explicit |
| MEA:a / H2O:b | combining rule | combining rule | combining-rule |
| MEA:b / H2O:a | combining rule | combining rule | combining-rule |

Combining-rule inputs and source locators are explicit in `topology.edges[].source` in the parameter JSON. No additional association edges are introduced here.

## Temperature correlations and model coefficients

| Correlation | Form | Coefficients (exact JSON values) |
| --- | --- | --- |
| component/water/segment_diameter/constant-plus-sum-of-exponentials | constant-plus-sum-of-exponentials | {"constant":{"magnitude":2.7927,"unit":"angstrom"},"terms":[{"amplitude":{"magnitude":10.11,"unit":"angstrom"},"exponent_coefficient":{"magnitude":-0.01775,"unit":"1 / kelvin"}},{"amplitude":{"magnitude":-1.417,"unit":"angstrom"},"exponent_coefficient":{"magnitude":-0.01146,"unit":"1 / kelvin"}}]} |
| component/water/relative_permittivity/constant-plus-polynomial-in-temperature | constant-plus-polynomial-in-temperature | {"constant":{"magnitude":78.41043478578715,"unit":"dimensionless"},"reference_temperature":{"magnitude":298.15,"unit":"kelvin"},"terms":[{"coefficient":{"magnitude":-0.36133766233691506,"unit":"1 / kelvin"},"power":1},{"coefficient":{"magnitude":0.00076555618295,"unit":"1 / kelvin ** 2"},"power":2}]} |

| Coefficient | Value | Unit | Source ID |
| --- | --- | --- | --- |
| ionic_region_relative_permittivity | 8 | dimensionless | mea-retained-phase-two-artifact |

## Reactions and standard state

Species order: CO2, MEA, H2O, MEAH+, MEACOO-, HCO3-, CO3^2-, H3O+, OH-. Products have positive stoichiometric coefficients. Temperature is in kelvin and logarithms in the first correlation form are natural logarithms.

| Reaction | Stoichiometry in species order | ln K form | Effective coefficients | Offset | Calculation T domain (K) |
| --- | --- | --- | --- | --- | --- |
| R1 | [0,0,-2,0,0,0,0,1,1] | a + b_k / T + c * ln(T) + d_per_k * T + standard_state_offset | {"a":132.899,"b_k":-13445.9,"c":-22.4773,"d_per_k":0} | 8.0330699846 | [273.15,498.15] |
| R2 | [-1,0,-2,0,0,1,0,1,0] | a + b_k / T + c * ln(T) + d_per_k * T + standard_state_offset | {"a":232.33141533884407,"b_k":-11105.640030520277,"c":-36.7816,"d_per_k":0} | 0 | [293.15,393.15] |
| R3 | [0,0,-1,0,0,-1,1,1,0] | a + b_k / T + c * ln(T) + d_per_k * T + standard_state_offset | {"a":216.049,"b_k":-12431.7,"c":-35.4819,"d_per_k":0} | 4.0165349923 | [273.15,498.15] |
| R4 | [0,1,-1,0,-1,1,0,0,0] | a + b_k / T | {"a":1.505015374192114,"b_k":-1317.0489707842564} | 0 | [293.15,393.15] |
| R5 | [0,1,-1,-1,0,0,0,1,0] | -ln(10) * (a_k / T + b + c_per_k * T) | {"a_k":3037.6399534696106,"b":-1.0173150837285996,"c_per_k":0.0004277} | 0 | [293.15,393.15] |

These are effective runtime coefficients: R2/R4/R5 typed fitted records in the selected parameter document supersede the source correlations in the matching reaction export completely. R2's offset is already folded into its fitted `a`; it is not added again. R1/R3 retain source correlations and offsets. The common selected range is 293.15–393.15 K; wider R1/R2/R3 source ranges do not extend the coupled model's domain. Effective ln K values agree with the previous export at four checked temperatures. This is a selected calculation range, not proof of uniform predictive accuracy.

| Standard-state field | Value |
| --- | --- |
| id | aqueous-molality-infinite-dilution-water-v1 |
| log_activity_scale_factors | [-4.016534992299479,-4.016534992299479,0,-4.016534992299479,-4.016534992299479,-4.016534992299479,-4.016534992299479,-4.016534992299479,-4.016534992299479] |
| reference_convention | molality-infinite-dilution |
| reference_pressure_pa | 100000 |
| solvent_composition | [0,0,1,0,0,0,0,0,0] |
| standard_molality_mol_per_kg | 1 |

The pure-water solvent reference and 1 mol/kg standard molality are converted by the Engine to its EOS neutral reference. The offsets and logarithmic activity scale factors must not be applied a second time. The implementation is `Thermodynamics/reactive_bundle.py::compile_reaction_constants`.

## Sources, fitting and applicability

### Independent-coefficient accounting for the manuscript comparison

The selected JSON was checked byte-for-byte against MEA-Thermodynamics `analyses/mea_parameter_bundle/results/selected-current-best-parameters.json` on 2026-09-04. The SHA-256 remains `a9186c93759f2e2c02a6c913350ad06a244fff3f82503820c9962b3df8dd40d9`. Counts distinguish a stored coefficient, an independent fitting coordinate and an inherited fitted value; there is no claim that this is a globally smaller parameterization than eNRTL.

- **Inherited component inputs:** neutral segment and association parameters, six ions' dispersion/size/Born inputs, dielectric and solvation choices. Six inherited MEAH+/MEACOO− size, dispersion and Born-diameter values came from a historical seven-coordinate ion/speciation fit together with their one mutual interaction coefficient. Their provisional qualification is retained; a fit history does not establish independent predictive validation.
- **Inherited mixture interactions:** five nontrivial `k_ij` values besides the neutral MEA–water refit: CO2–water, MEAH+–MEACOO− and three water–ion values. Only the mutual amine-ion term is identified here as historically fitted; the others are retained inputs, not inferred fits.
- **New neutral interaction refit:** one scalar MEA–water `k_ij`, −0.07352749875, fitted to 25 Cai binary VLE rows. Two reciprocal matrix entries represent this one independent symmetric coefficient. The separate CO2 dispersion energy, 173.44025 K, was chosen by a joint CO2/R4 calibration grid, not an isolated optimizer estimate.
- **Fixed and derived inputs:** the 36 unique pair constants comprise the one neutral refit, five inherited nontrivial values, 23 zeros and seven unit same-charge dispersion exclusions. One additional reciprocal-temperature slope is fixed to zero. The six ion packing diameters and six Debye–Hückel diameters are derived as 0.88 times the relevant segment diameter; repeated storage does not add fitting freedom. Reciprocal association edges and cross quantities supplied by combining rules are not independent fits. Fixed species charges/masses and the standard molality are not regression coordinates.
- **Reaction/caloric inputs:** the six changed R2/R4/R5 intercept/slope coefficients are linked. The upstream reaction-temperature screen retained **two SVD fitting directions** from a five-reaction sensitivity matrix, not six independently fitted coefficients; the selected file adopts only its R2/R4/R5 shifts. Prior R4/R5 grid selections are separate stages. The anchored-reference polynomial coefficients are derived from reaction enthalpies, neutral anchors and a charge gauge, not separately fitted ion measurements. The column studies retain the empirical caloric formulas, not these polynomials as a replacement column enthalpy model.
- **Calibration and validation:** neutral binary VLE, pressure/speciation and calorimetry enter different fitting stages. Xu pressure and 120 °C calorimetry were excluded from the final reaction-temperature objective; Xu had earlier model-selection use, so it is not an untouched holdout for the entire model. The full five-shift replay is not the exact selected three-shift vector. Neither its aggregate fit statistics nor a count of stored coefficients establishes independent column validation.

Exact upstream accounting locators: `analyses/mea_parameter_bundle/notebook.qmd#fit-neutral`, `#fit-ions`, `#fit-co2`, `#fit-reactions`; `results/reaction-temperature-fit/screen-receipt.json` (`retained_direction_count=2`, five rows in each direction); `scripts/run_reaction_temperature_fit.py`, lines 615–625; and the selected JSON coefficient `provenance.locator` fields. The source notebook's numerical performance claims are not newly promoted here; only the count, source and dependency distinctions are used.

The following are the source and use distinctions recorded by the selected bundle, not newly audited primary-paper claims.

| Source ID | Citation | Recorded use |
| --- | --- | --- |
| mea-retained-phase-two-artifact | MEA-Thermodynamics retained Phase 2 parameter artifact | authority-neutral regression starts and fixed model inputs; row-level literature lineage, transfer limits, and provisional statuses are audited in docs/ePC-SAFT/full-component-parameter-source-audit.md |
| schick-2023-pabsch-2020-induced-association | Schick et al. (2023) using Pabsch et al. (2020) | CO2 2B sites and reciprocal CO2-water cross-association with the source combining rule |
| zuber-et-al-2014 | Zuber et al. (2014), Fluid Phase Equilibria 376, 116-123, doi:10.1016/j.fluid.2014.05.037 | Ion-specific aqueous dielectric-suppression coefficients and charge-class fallback values |
| uyan-2015-permittivity-transfer | Uyan Eq. 5; Wangler campaign Archer-Wang approximation. | Explicit transfer approximation, not an exact Floriano-Nascimento replay. |
| tong-2012-aroua-1999 | Tong et al. (2012), Table 5 K5 MEA, via Aroua et al. (1999). | R4 carbamate hydrolysis source-centred correlation. |
| bates-pinching-1951 | Bates and Pinching (1951), equation 7. | R5 monoethanolammonium dissociation source-centred correlation. |
| cai-1996-neutral-refit | Project-authored ePC-SAFT parameter fit result. | Accepted fitted coefficient values and exact numerical evidence. |
| mea-best-in-slot-campaign-2026-09-02 | MEA-Thermodynamics bounded direct Engine campaign, 2026-09-02. | Current-best joint pressure/speciation calibration of the R4 correlation and carbon-dioxide dispersion energy; initialized from the retained literature bundle and not an independent validation. |
| mea-born-permittivity-study-2026-09-02 | MEA-Thermodynamics Born-permittivity and reaction comparison, 2026-09-02. | Selected R5 temperature coefficient for the extended-Born bundle from the common-target pressure and speciation comparison; initialized from Bates and Pinching (1951). |
| figiel-2025 | Figiel et al. (2025), Industrial & Engineering Chemistry Research 64, 9406-9420, doi:10.1021/acs.iecr.5c00475 | Same-sign ion dispersion exclusion and the coupled SSM+DS diameter, solvation-factor, and dielectric-suppression formulation. |
| mea-reaction-temperature-fit-2026-09-03 | MEA-Thermodynamics reaction-temperature fit, 2026-09-03 (analyses/mea_parameter_bundle/results/reaction-temperature-fit) | constant R2/R4/R5 reaction-enthalpy shifts fitted to pressure, speciation, and 40/80 C calorimetry with Xu 2011 pressure and 120 C calorimetry held out; ln K preserved at 313.15 K |

Retained Phase 2 inputs include literature-lineage values, provisional historical ion values and transferred diagnostic values; they must not all be labeled directly measured or independently fitted. Per-coefficient `provenance.locator` records distinguish these statuses. The neutral refit, joint pressure/speciation fit and reaction-temperature fit are project-fitted inputs. Source-centred R4/R5 correlations are starting relationships, not the final selected coefficients.

The reaction-temperature fit records pressure/speciation and 40/80 C calorimetry as calibration, with Xu 2011 pressure and 120 C calorimetry held out. That split is not absorber validation. No new fit, held-out evaluation or column calculation was performed for this transcription.

The upstream full-replay summaries include nonzero R1/R3 shifts absent from this selected JSON. Their aggregate metrics must not be attributed to this exact selected vector. The retained `calorimetry/current-selected-direct-enthalpy-summary.json` is a separate selected-vector result; no numerical claims from it are promoted into this manuscript here.

| Domain ID | Kind | Temperature (K) | Pressure (Pa) |
| --- | --- | --- | --- |
| mea-diagnostic-293-15-to-393-15-k | fit-range | 293.15 to 393.15 | 10 to 300000 |
| mea-provider-source-qualified-313-15-k | reported-conditions | 313.15 to 313.15 | 100000 to 100000 |
| mea-candidate-293-15-to-393-15-k | fit-range | 293.15 to 393.15 | 1 to 10000000 |
| cai-1996-neutral-refit | unknown | not specified | not specified |

The selected rows carry `candidate_extrapolation` qualifications. Candidate calculation bounds are broader than some source-qualified conditions; they are not a replacement for the source domains. The historical manuscript parameter tables and figures remain unchanged and must be replaced together after accepted matching column runs.
