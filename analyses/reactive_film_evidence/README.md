# Reactive-film physical evidence

This analysis identifies what can presently support a physical, chemical-potential-driven MEA film model and preserves one exact aggregate comparison figure. It reuses the retained Issue 34/35/41/42 records; source coefficients are not copied into a second executable model.

## Evidence now available

- Putta2016 provides reversible F1/F2 concentration- and activity-form rate laws and Arrhenius fits over its reported temperature, MEA-label, loading, and pressure-driving-force domain. The retained record preserves the printed unit discrepancy rather than silently correcting source metadata. **[verified]**
- Putta2016 Table 4 provides 20 exact aggregate AARD values across SDC, two WWC data groups, and a laminar-jet group; the Luo WWC group was used to estimate the Present models, so these compare reported groups but do not expose row-level measured fluxes or uncertainty. **[verified]**
- Luo2015 reports 227 WWC/SDC observations and states on printed p. 65 that all raw data are supplementary, but that supplement is not local. **[verified]**
- Luo2015 Eq. 21 and Eq. 22 and Snijder1993 Table III/Eq. 8 are already reconstructed by Issues 35/42 as scalar transport correlations with their original quantity limitations. **[verified]**
- Polat2023 Table 2 supplies Arrhenius fits for the self-diffusivity of infinitely dilute CO2 in 10-50 wt% aqueous MEA over 293-353 K; these are tracer/self-diffusion results, not Fick or Maxwell-Stefan coefficients. **[verified]**
- Melnikov2019 gives a 313 K, 30 wt% MEA/water MD trend in which CO2 self-diffusion at loading 0.5 is approximately one-half of the pre-reaction value; the exact ESI Table S4 values are not present locally, so the retained point is an assumption/UQ band, not a digitized table. **[verified]**
- Jerng2022 reports DOSY anchors at 0.1 M MEA/D2O: MEA 8.8e-10 m2/s before CO2, unresolved MEA/MEAH+ 8.4e-10 m2/s after CO2, and MEACOO- 6.8e-10 m2/s after CO2. Lv2026 reports ferrocenemethanol-probe solution diffusion in CO2-loaded 2 M amines: MEA 3.94e-10, DEA 3.06e-10, and TEA 1.91e-10 m2/s. **[verified]**
- Ramezani2021 supplied SI Tables S1-S5 now retain exact density, viscosity, and kinetics rows for 30 wt% MEA with 0-20 wt% sugar, 0-0.4 CO2 loading, and 298.15-343.15 K. The supplied DOCX XML resolves the S5 schema as K_G, D_CO2, H_CO2, k_l, k_g, k_ov, Ha, and E_inf with the reported scale factors; those semantic columns are retained without fitting. **[verified]**
- Ganesan2026 supplied SI retains exact Aspen model choices (S3.1), beta corrections (S4.2), VLE rows (S6.1), and specific-absorption-flux rows (S6.2). The SI uses E-NRTL for the liquid thermodynamics, PC-SAFT for the gas, Wilke-Chang/Nernst-Hartley liquid diffusion, and Bravo correlations for transfer, area, and holdup. **[verified]**
- No retained source supplies species-resolved transport for the six ions or the off-diagonal Maxwell-Stefan/Onsager friction terms needed for a complete nine-species mobility matrix. **[verified]**

## Exact source locators

- Putta et al. (2016), DOI `10.1016/j.ijggc.2016.08.009`: printed pp. 341-342, Eqs. 1-5 and 9-22; printed pp. 345-346, Eqs. I-II; printed p. 349, Table 4. **[verified]**
- Luo et al. (2015), DOI `10.1016/j.ces.2014.10.013`: printed pp. 59-60 for apparatus/domain; printed p. 61, Eqs. 21-22; printed p. 65 for supplementary raw data. **[verified]**
- Snijder et al. (1993), DOI `10.1021/je00011a037`: printed p. 477, Table III and Eq. 8; printed p. 479 for concentration extension. **[verified]**
- Polat et al. (2023), DOI `10.1016/j.fluid.2022.113587`: PDF page 4, Table 2, 293-353 K. **[verified]**
- Melnikov and Stein (2019), DOI `10.1039/C9CP03976G`: PDF Fig. 5, 313 K, 30 wt% MEA/water; ESI Table S4 is referenced but absent locally. **[verified]**
- Jerng and Gallant (2022), DOI `10.1016/j.isci.2022.104558`: PDF p. 10, transport paragraph and Fig. 7A. **[verified]**
- Lv et al. (2026), DOI `10.1039/D6SC00859C`: PDF p. 11979, mass-transport paragraph, Fig. 6a, and Fig. S8. **[verified]**
- Ramezani et al. (2021), DOI `10.1016/j.molliq.2021.115569`: supplied SI pp. 2-8, Tables S1-S5; S5 headers checked in the supplied DOCX XML. Zotero parent `VMZKH34U`, SI child `NCLG54A3`. **[verified]**
- Ganesan et al. (2026), DOI `10.1016/j.ces.2025.122487`: PDF Eq. 22 and fitting paragraph; supplied SI p. 3 Table S3.1, p. 5 Table S4.2, pp. 10-11 Tables S6.1-S6.2. **[verified]**
- Dugas and Rochelle (2011), DOI `10.1021/je101234t`: PDF printed p. 2189, Table 1 MEA rows; uncertainty note in text. **[verified]**

## What can be used now

- The Putta F1/F2 forms can drive a finite-rate prototype only after their standard-state mapping is made consistent with the thermodynamic provider; they cannot yet be validated from the retained aggregate AARD table. **[inference]**
- The Luo/Snijder reduced scalar correlations and Polat self-diffusion fits can define labeled comparator ranges, but converting them into a multicomponent mobility matrix would add unsupported physics. **[inference]**
- The central figure shows that Table 4 errors depend on both model basis and reported apparatus/data group; it is aggregate model-comparison evidence, not row-level film validation. **[verified]**
- The three-panel synthesis separates (a) thermodynamic-force changes under common unit-diagonal mobility, (b) a fixed-chemistry viscosity perturbation proxy from Ramezani S4 at 313.15 K, and (c) dimensional Dugas Table 1 film observations. **[verified]**
- Panel (b) is normalized as μ(no sugar)/μ(sugar) at identical MEA/CO2 loading. It is a falsification/sensitivity check for mobility dependence, not a dimensional flux prediction: a viscosity scalar does not identify the nine-species mobility matrix or its cross-friction terms. **[inference]**
- Panel (a) now uses relative deltas. Its integrated boundary-response comparison is -2.35e-4% because the path-integrated generalized coordinate gradient is solved separately for each closure. The local same-gradient check at 313.15 K, 101325 Pa, normalized `[1,20,70,3,2,0.5,0.25,0.5,0.5]` composition, CO2-only `q=1e-4`, and the same total-flux/zero-current projection gives -4.781% (the earlier rounded -4.75% report). The full state, species order, matrices, projection, and normalizations are retained in `chemical_potential_definition_trace.csv`; these are distinct diagnostics, not interchangeable flux claims. **[verified]**
- Dugas Table 1 retains 50 MEA rows across 7/9/11/13 molal and 40/60/80/100 °C; one 9 molal, 40 °C, loading 0.231 kg' entry is not reported. The reported kg' span is approximately 27x, conventionally summarized as a roughly 30x envelope. **[verified]**

`evaluate_onsager_closure.py` now makes the missing transport assumption explicit instead of using equal ion diffusivities. It forms harmonic-mean pair estimates from the retained CO2/MEA/MEAH+/MEACOO- anchors and broad labeled bounds for H2O and the inorganic ions, builds a symmetric pair Onsager matrix, and removes total-molar-flow and electrical-current modes by a Schur complement. The resulting matrix is positive semidefinite to roundoff, has nonnegative entropy production, and recovers binary ideal Fick diffusion. This is a bounded formulation/sensitivity calculation; the estimated pairs are not measurements or a validation dataset. **[inference]**

The same calculation evaluates the exact ePC-SAFT tangent at the fresh homogeneous equilibrium. The constrained Hessian is symmetric and positive, while its condition number is about `5.3e10` because H3O+ is a trace species. Putta F1/F2 forward scales combined with provider affinities give a local reaction time orders of magnitude below the 100 micrometre diffusion time. A fully differential nine-species BVP is therefore expected to be stiff; fast-reaction manifold reduction is required before column coupling. **[inference]**

## Eight-case column-film campaign

The adopted calculation replaces the explicit enhancement-factor CO2 flux with a fast-equilibrium nine-species film resistance. It integrates the ePC-SAFT chemical-potential tangent along the local equilibrium manifold, applies a symmetric positive-semidefinite diagonal pair-mobility approximation, and projects out total-molar-flow, electrical-current, nonvolatile-MEA, and water-component modes before closing the CO2 flux against the gas-film resistance. Water transfer remains the column model's separate two-film calculation, so the CO2 film is explicitly a decoupled-component approximation. The packed-column balances remain a mixed-boundary BVP because vapor enters at the bottom and lean liquid enters at the top; the local film calculation is a one-dimensional quadrature/root problem, not a second spatial BVP. **[verified]**

Across K18, K19, and 1C--6C, the film calculation changes predicted capture by -0.031 to +8.526 percentage points relative to the former enhancement-factor model. The observed-capture mean absolute error is 6.677 percentage points for the all-case film aggregate and 4.038 percentage points for the enhancement-factor comparison. This campaign therefore does not show improved aggregate agreement with the pilot data; it shows the consequence of replacing an empirical enhancement closure with the documented thermodynamic-film formulation without fitting to capture. **[verified]**

All eight packed-column solves satisfy their boundary conditions at numerical precision, and their maximum collocation RMS residual is 0.066 under the declared 0.1 tolerance. Five cases (K19, 2C, 3C, 4C, and 6C) also meet the declared outer-film stopping rule: less than 0.05 percentage-point capture change and less than 2% maximum change in both the interpolated conductance and reactive bulk-fugacity fields. K18, 1C, and 5C reach the common 15-update cap first and remain explicitly marked provisional/non-field-converged in the aggregate and figure. **[verified]**

The retained film axial profiles reproduce the six 2017 C-case temperature comparisons without reusing the former enhancement-factor profiles. Their liquid-temperature tap RMSE values range from 3.73 to 8.05 K, with a mean of 5.65 K; Case 3C has a 4.18 K tap RMSE. These are descriptive comparisons rather than acceptance criteria. **[verified]**

The result is predictive only in the conditional engineering sense: case inputs, the selected nine-species ePC-SAFT bundle, gas/liquid transfer correlations, and labeled diffusivity estimates determine the outputs without capture fitting. The selected bundle was evaluated against pressure/speciation data through 393.15 K, but its R5 source correlation is used beyond its 323.15 K source-qualified range and remains an explicit temperature extrapolation. It is not a source-complete transport prediction because the nine-species Maxwell--Stefan/Onsager matrix has not been measured or independently identified, and the campaign does not validate the estimated mobility closure. **[inference]**

## Best next strategy

1. Acquire the exact Luo2015 supplementary file and retain all 227 rows with apparatus membership, units, state variables, and reported measurement accuracy. **[inference]**
2. Seek loaded-MEA ionic self-diffusion/conductivity/NMR or validated molecular-dynamics evidence for MEAH+ and MEACOO- first, then the inorganic ions; retain uncertainty and composition basis. **[inference]**
3. Replace the present estimated diagonal pair mobility only when species-resolved evidence supports a better identified closure, then repeat the same eight-case comparison. **[inference]**
4. Validate the local film flux against held-out WWC/SDC/laminar-jet rows before describing the transport closure as independently predictive. **[inference]**

Regenerate the retained tables and both figure formats with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python analyses/reactive_film_evidence/scripts/build_reactive_film_evidence.py
uv run python analyses/reactive_film_evidence/scripts/render_reactive_film_evidence.py
uv run python analyses/reactive_film_evidence/scripts/render_putta2016_aard.py
for case in K18 K19 1C 2C 3C 4C 5C 6C; do uv run python analyses/reactive_film_evidence/scripts/run_column_film_comparison.py --case-ids "$case" --film-nodes 5 --outer-iterations 15 --workers 2 --relaxation 0.5 --mesh-points 21 --tol 0.1 --max-nodes 1000 --output-dir "analyses/reactive_film_evidence/results/runs/current_bundle_campaign_$case"; done
uv run python analyses/reactive_film_evidence/scripts/promote_column_film_results.py
uv run python analyses/reactive_film_evidence/scripts/render_column_film_comparison.py
```

The main column outputs are `results/final/tables/column_film_capture_comparison.csv`, `results/final/tables/column_film_nodes.csv`, `results/final/tables/column_film_axial_profiles.csv`, `results/final/tables/column_film_temperature_metrics.csv`, and the `column_film_capture_comparison`, `column_film_temperature_overlay`, and `column_film_3c_temperature` figures in both PNG and PDF formats. The earlier source-evidence panels remain explicitly labeled non-predictive or non-validation where a physical scale or complete transport closure is missing.
