# Issue 35: source-faithful transport inputs and unequal-ion closure

Status: **supported-negative source-faithful transport record complete**. The retained result reconstructs positive source correlations and preserves the missing-input decision. It does not adopt a physical transport closure or compare film fluxes.

## Decision

Candidate A, the equal-ion effective-Fick reduction, is retained as a reduced-form requirement only. A physical admission would require a source-complete effective diffusivity for every retained species or an explicit source-defined ion lump, plus an accepted local concentration basis.

Candidate B, the unequal-ion zero-current form, is not executable. Its minimum requirements are the true-species state, ePC-SAFT Gamma, a complete cited mobility law, source-complete unequal ion diffusivities, molar density, electroneutrality, zero current, and the potential gauge. The retained sensitivity table therefore has no flux values, charge residual, current residual, or transfer direction.

## Source reconstruction

Luo2015 Eq. 21 retains `D_CO2_water = 2.35e-6 exp(-2119/T)` in `m2/s`. Luo2015 Eq. 22 retains the modified Stokes-Einstein form `D_CO2_amine = D_CO2_water (mu_water/mu_amine)^0.8`; Amundsen2009 Weiland Eqs. 9--10 and Table 12 supply the source-labeled viscosity relationship for the evaluated 30 mass% and 0--0.5 loading rows. These values are source reconstructions, not absorber inputs.

Snijder1993 Eq. 8 retains `ln(D_MEA) = -13.275 - 2198.3/T - 7.8142e-5 C`, with `C` in `mol/m3` and `D` in `m2/s`. The 16 Table III observations retain density and viscosity in `kg/m3` and `mPa s`; the density is not converted to molar density while Issue 33 remains basis_unresolved. The maximum relative residual against the displayed, rounded Table III diffusivities is `0.054757`. Snijder's source statement that the fit is within 5% is preserved as source metadata; the rounded table alone does not reproduce that statement at every displayed row.

Putta2017 Eq. 12 is retained as a blocked N2O analogy because the cited Ko, Jamal, and Ying/Eimer primary N2O-water and N2O-amine inputs were not recovered with source-complete coefficients and uncertainty. The legacy scalar ion expression in `src/mea_absorption_column/Properties/Transport_Properties.py` is rejected as unattributed and is not retained as a default.

## Dependency and claim boundary

Issue 33 remains `basis_unresolved`: Position 1 analytical MEA is `4.889309897097635 mol/L` and free MEA is `2.491683471902737 mol/L`; neither is rounded to exact 5 M. Issue 34 remains the merged supported-negative kinetics record. No rate comparison, Case 3C tuning, physical film result, or production transport-adoption change is made here.

The input record is `inputs/issue35_transport.json`. Generated tables are `issue35_transport_correlations.csv`, `issue35_transport_sensitivity.csv`, and `issue35_transport_summary.json`.
