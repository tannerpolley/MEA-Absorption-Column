# Reactive-film physical evidence

This analysis identifies what can presently support a physical, chemical-potential-driven MEA film model and preserves one exact aggregate comparison figure. It reuses the retained Issue 34/35/41/42 records; source coefficients are not copied into a second executable model.

## Evidence now available

- Putta2016 provides reversible F1/F2 concentration- and activity-form rate laws and Arrhenius fits over its reported temperature, MEA-label, loading, and pressure-driving-force domain. The retained record preserves the printed unit discrepancy rather than silently correcting source metadata. **[verified]**
- Putta2016 Table 4 provides 20 exact aggregate AARD values across SDC, two WWC data groups, and a laminar-jet group; the Luo WWC group was used to estimate the Present models, so these compare reported groups but do not expose row-level measured fluxes or uncertainty. **[verified]**
- Luo2015 reports 227 WWC/SDC observations and states on printed p. 65 that all raw data are supplementary, but that supplement is not local. **[verified]**
- Luo2015 Eq. 21 and Eq. 22 and Snijder1993 Table III/Eq. 8 are already reconstructed by Issues 35/42 as scalar transport correlations with their original quantity limitations. **[verified]**
- Polat2023 Table 2 supplies Arrhenius fits for the self-diffusivity of infinitely dilute CO2 in 10-50 wt% aqueous MEA over 293-353 K; these are tracer/self-diffusion results, not Fick or Maxwell-Stefan coefficients. **[verified]**
- No retained source supplies species-resolved transport for the six ions or the off-diagonal Maxwell-Stefan/Onsager friction terms needed for a complete nine-species mobility matrix. **[verified]**

## Exact source locators

- Putta et al. (2016), DOI `10.1016/j.ijggc.2016.08.009`: printed pp. 341-342, Eqs. 1-5 and 9-22; printed pp. 345-346, Eqs. I-II; printed p. 349, Table 4. **[verified]**
- Luo et al. (2015), DOI `10.1016/j.ces.2014.10.013`: printed pp. 59-60 for apparatus/domain; printed p. 61, Eqs. 21-22; printed p. 65 for supplementary raw data. **[verified]**
- Snijder et al. (1993), DOI `10.1021/je00011a037`: printed p. 477, Table III and Eq. 8; printed p. 479 for concentration extension. **[verified]**
- Polat et al. (2023), DOI `10.1016/j.fluid.2022.113587`: PDF page 4, Table 2, 293-353 K. **[verified]**
- Ramezani et al. (2021), DOI `10.1016/j.molliq.2021.115569`, remains a candidate whose quantity and domain have not been inspected locally. **[unknown]**

## What can be used now

- The Putta F1/F2 forms can drive a finite-rate prototype only after their standard-state mapping is made consistent with the thermodynamic provider; they cannot yet be validated from the retained aggregate AARD table. **[inference]**
- The Luo/Snijder reduced scalar correlations and Polat self-diffusion fits can define labeled comparator ranges, but converting them into a multicomponent mobility matrix would add unsupported physics. **[inference]**
- The central figure shows that Table 4 errors depend on both model basis and reported apparatus/data group; it is aggregate model-comparison evidence, not row-level film validation. **[verified]**

## Best next strategy

1. Acquire the exact Luo2015 supplementary file and retain all 227 rows with apparatus membership, units, state variables, and reported measurement accuracy. **[inference]**
2. Acquire and classify Ramezani2021 and the Polat2023 supplement; use self-diffusion only as tracer evidence unless the source provides a defensible conversion or cross-correlation information. **[inference]**
3. Seek loaded-MEA ionic self-diffusion/conductivity/NMR or validated molecular-dynamics evidence for MEAH+ and MEACOO- first, then the inorganic ions; retain uncertainty and composition basis. **[inference]**
4. Build the smallest symmetric positive-semidefinite mobility closure consistent with those data and electroneutral zero-current transport, and compare it against the existing effective-Fick reduction at identical thermodynamics and kinetics. **[inference]**
5. Validate against held-out WWC/SDC/laminar-jet flux rows before coupling to the packed column; only then promote parity/residual and sensitivity figures toward the manuscript. **[inference]**

Regenerate the exact plotted table and both figure formats with:

```bash
uv run python analyses/reactive_film_evidence/scripts/render_putta2016_aard.py
```
