# Issue 34: reversible MEA film kinetics and reaction partition

Status: **supported-negative source-faithful record complete**. This record preserves the source reaction space, reversible rate forms, unit reconstruction, observations, and explicit non-admission boundaries. It does not implement or fit a reactive film.

## Decision

- F1, `CO2 + 2 MEA <=> MEACOO- + MEAH+`, and F2, `CO2 + MEA + H2O <=> MEACOO- + H3O+`, remain finite-rate candidates because Putta2016 supplies reversible concentration/activity rate equations and Arrhenius correlations.
- F3, `CO2 + OH- <=> HCO3-`, remains a source relationship but is rejected as a physical finite-rate input because Putta2016 attributes its coefficient to Gondal2015 and the primary coefficient was not recovered in local Zotero.
- R1/R3/R4/R5 remain equilibrium-closure candidates only. Putta2016 omits H3O+/OH- transport kinetics in favor of water equilibrium and electroneutrality, but no quantitative film timescale evidence is available for admission.
- No reaction may be applied as both an exact local-equilibrium constraint and an independently applied finite rate in a future partition.

## Source and basis boundary

Putta2016 (DOI `10.1016/j.ijggc.2016.08.009`, attachment SHA-256 `fac3789d1ff6baa53e226638a2505ee3f3ff10433e4af89ef0a8e27785771e99`) is the primary finite-rate source. Luo2015 (DOI `10.1016/j.ces.2014.10.013`) supplies secondary mechanism context. The cited Gondal2015 source (DOI `10.1016/j.ces.2014.10.038`) is not present in local Zotero, so no F3 coefficient is invented.

Putta's 1 M and 5 M values are source labels only. The immutable issue 33 dependency remains `basis_unresolved`: Position 1 analytical MEA is `4.889309897097635 mol L^-1` and free MEA is `2.491683471902737 mol L^-1`; it is not rounded or admitted as exact 5 M. No capture or kinetic tuning was performed.

## Reaction-space and units

The retained species order is `CO2, MEA, H2O, MEAH+, MEACOO-, HCO3-, CO3^2-, H3O+, OH-`. The checked projections are `F1 = R2 - R4 - R5`, `F2 = R2 - R4`, and `F3 = R2 - R1`. All three have Δν = -1, so a raw concentration quotient has units m^3 kmol^-1 when concentrations are in kmol m^-3. A future dimensionless conversion is `K° = K_raw c°`; the sources do not specify a provider-compatible standard state.

Putta prints `m^6 kmol^-2 s^-2` for the third-order F1/F2 coefficient, but a rate in kmol m^-3 s^-1 requires `m^6 kmol^-2 s^-1`. The printed unit is retained as rejected source metadata and the dimensionally required unit is recorded separately. F3 would require `m^3 kmol^-1 s^-1`, but its coefficient is unavailable.

The source-state closure rows use a strictly positive synthetic state at 313.15 K only to verify `k_reverse = k_forward/K_raw`; the maximum absolute ln(Q/K) is `0.000e+00`. This is not a retained NCCC state, a fitted result, or an activity closure.

## Timescale evidence and observations

The Putta source domain is 198 points over 293.15--343.15 K, source-labeled 1/5 M solutions, loading 0.0--0.4 mol CO2 per mol MEA, and LMPD 0.58--14.7 kPa. These are source-domain records, not an absorber admission basis.

A quantitative reaction time or Damköhler comparison is **not evaluable**: the retained evidence does not jointly provide an accepted physical basis, film thickness, diffusivity, and state-specific rate evaluation. Putta's reversible F1/F2 forms support finite-rate candidacy; its H3O+/OH- closure prescription supports only a qualitative equilibrium-closure candidate.

Putta Table 4 aggregate AARD values are retained as summary-only observations. Raw paired observations, row uncertainty, and non-overlapping fit/validation membership are unavailable, so no rate-data admission or coefficient uncertainty fit is claimed.

## Outputs

The input record is `inputs/issue34_kinetics.json`. The generated tables are `issue34_finite_reactions.csv`, `issue34_partition_decisions.csv`, `issue34_rate_observation_comparisons.csv`, and `issue34_kinetic_sensitivity.csv`; gate and identity data are in `issue34_kinetics_summary.json`.
