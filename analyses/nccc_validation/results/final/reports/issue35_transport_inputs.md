# Issue 35: source-faithful transport inputs and unequal-ion closure

Status: **supported-negative source-faithful transport record complete**. The retained result reconstructs positive source correlations and preserves the missing-input decision. It does not adopt a physical transport closure or compare film fluxes.

## Decision

Candidate A, the equal-ion effective-Fick reduction, is retained as a reduced-form requirement only. A physical admission would require a source-complete effective diffusivity for every retained species or an explicit source-defined ion lump, plus an accepted local concentration basis.

Candidate B, the unequal-ion zero-current form, is not executable. Its minimum requirements are the true-species state, ePC-SAFT Gamma, a complete cited mobility law, source-complete unequal ion diffusivities, molar density, electroneutrality, zero current, and the potential gauge. The retained sensitivity table therefore has no flux values, charge residual, current residual, or transfer direction.

## Source authority and reconstruction

`inputs/issue35_transport.json` is the authoritative coefficient record. Every evaluated correlation row carries its numeric source parameters in `source_parameters_json`; no coefficient is duplicated independently in the resolver. Luo2015 Eq. 21 retains `D_CO2_water = 2.35e-6 exp(-2119/T)` in `m2/s`. Luo2015 Eq. 22 retains `D_CO2_amine = D_CO2_water (mu_water/mu_amine)^0.8`; the evaluated viscosity input uses the Weiland parameters reproduced by Amundsen2009, not a claim that Amundsen originated that correlation.

Snijder1993 Eq. 8 retains `ln(D_MEA) = -13.275 - 2198.3/T - 7.8142e-5 C`, with `C` in `mol/m3` and `D` in `m2/s`. The 16 Table III observations retain density and viscosity. Snijder source viscosity is reported in `mPa s`; its emitted CSV value is converted to `Pa s`. The maximum absolute residual is `6.401227067e-11 m2/s` and the maximum relative residual is `0.054757` against the displayed, rounded diffusivities. Snijder's source statement that the fit is within 5% is preserved as source metadata; the rounded table alone does not reproduce that statement at every displayed row.

Amundsen2009 Weiland density parameters and loaded 30 mass% density observations are retained with the source `g/cm3` units. The Weiland density equation is numerically evaluated at those retained loaded states; because Amundsen Table 10 does not print V2, V2 is reconstructed at each temperature from the retained Amundsen Table 1 unloaded 30 mass% density anchor. The reconstructed values are source-labeled residual checks, not an independent fit or physical admission. Hartono2014 density observations are Table 3 30 mass% observations, retained in `kg/m3`; Hartono2014 viscosity observations are Table 7 30 mass% observations, reported in `mPa s` and emitted in `Pa s`. Hartono Eq. 5 density and the loaded viscosity correction are numerically evaluated at the retained states using the same-temperature alpha=0 source observation as the unloaded anchor. The viscosity check is therefore an anchored loaded-correction reconstruction, not a full independent Eq. 6--9 prediction. All model values, references, absolute residuals with units, relative residuals, domains, and source parameters are retained in the correlation table as source-labeled and non-admitted. Density is not converted to a total molar-density film state while Issue 33 remains basis_unresolved.

Putta2017 Eq. 12 is retained as a blocked N2O analogy because the cited Ko, Jamal, and Ying/Eimer primary N2O-water and N2O-amine inputs were not recovered with source-complete coefficients and uncertainty. The legacy scalar ion expression in `src/mea_absorption_column/Properties/Transport_Properties.py` is rejected as unattributed and is not retained as a default.

## Dependency and claim boundary

Issue 33 remains `basis_unresolved`: Position 1 analytical MEA is `4.889309897097635 mol/L` and free MEA is `2.491683471902737 mol/L`; neither is rounded to exact 5 M. Issue 34 remains the merged supported-negative kinetics record. No rate comparison, Case 3C tuning, physical film result, production transport-adoption change, or packed-column capture claim is made here.

The generated tables are `issue35_transport_correlations.csv`, `issue35_transport_sensitivity.csv`, and `issue35_transport_summary.json`.
