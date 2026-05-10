# ePC-SAFT Appendix Parameter Table Handoff

Purpose: give the main manuscript-editing thread a narrow, evidence-based path for fixing Table A3 / `epcsaft_parameter_summary.tex` references and placeholder ion values without accidentally changing the scientific record.

## Current Files To Inspect

- Manuscript table: `docs/latex/tables/epcsaft_parameter_summary.tex`
- Current run dataset: `src/mea_absorption_column/data/epcsaft_datasets/MEA_CO2_H2O_ionic_fit/pure/any_solvent.csv`
- Final artifact copy: `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_pure_parameters.csv`

Do not edit `docs/latex/references.bib` manually. It is Zotero-owned.

## Main Finding

The neutral MEA parameter row is supported by Baygi 2015. Some current ion rows appear to be literature-backed analog values, while several carbonate/water-ion rows are placeholder-like values used by the current committed runs. The `MEAH+` and `MEACOO-` rows should also be treated as estimates in this manuscript, not independently regressed MEA-system parameters.

Do not silently replace current parameter values in Table A3 unless the corresponding model dataset is changed and the affected results are rerun. If no rerun is planned, Table A3 should report the actual parameters used and mark assumed auxiliary values clearly.

## Parameter Provenance Notes

| Species | Current table status | Suggested handling |
|---|---|---|
| CO2 | Literature/inherited PC-SAFT value. Current `m = 2.0790` differs slightly from the Baygi table value `2.0729`; verify exact upstream source before adding a precise citation claim. | Cite the current dataset source carefully, or mark source as inherited from the ePC-SAFT dataset if exact literature source is not confirmed. |
| MEA | Supported by Baygi 2015: `m = 3.0353`, `sigma = 3.0435`, `epsilon/k = 277.174`, association terms. | Cite Baygi 2015. |
| H2O | Current table uses temperature-dependent water `sigma(T)` and association values from the active ePC-SAFT dataset, not the Baygi table's simple water entries. | Cite the dataset/source actually used; verify against the upstream ePC-SAFT parameter source before naming Baygi as the source. |
| MEAH+ | Current `sigma = 3.5630`, `epsilon/k = 228.71` matches protonated amine analog values such as MDEAH+/MeHQ+ in ePC-SAFT ion tables, not a clean MEA-specific fitted value found in this search. | Treat as an estimated analog parameter. For this work, it can be described as assumed close enough to MDEAH+ because both represent protonated alkanolamine cations; independent MEAH+ regression remains ongoing/future work. |
| MEACOO- | Current `sigma = 3.5605`, `epsilon/k = 533.11` resembles acetate-like anion values in ePC-SAFT ion tables, not a clean MEA carbamate-specific fitted value found in this search. | Treat as an estimated analog parameter. For this work, it can be described as assumed close enough to a carboxylate/acetate-like anion for the present validation scope; independent MEACOO- regression remains ongoing/future work. |
| HCO3- | Current `sigma = 2.9296`, `epsilon/k = 70.00` matches published ePC-SAFT ion table values. | Cite the carbonate-ion source. |
| CO3^2- | Current `sigma = 3.0000`, `epsilon/k = 300.00` is placeholder-like. Literature values found: `sigma = 2.4422`, `epsilon/k = 249.26`. | If not rerunning, keep current values but mark with `\approx` and an assumption note. If rerunning, replace with literature values and regenerate validation evidence. |
| H3O+ | Current `sigma = 3.0000`, `epsilon/k = 300.00` is placeholder-like. Literature values found for H+/H3O+ convention: `sigma = 3.4654`, `epsilon/k = 500`. | Same as CO3^2-. Do not silently swap without rerun. |
| OH- | Current `sigma = 3.0000`, `epsilon/k = 300.00` is placeholder-like. Literature values found: `sigma = 2.0177`, `epsilon/k = 650`; Held 2008 gives another OH- set tied to its own water model. | Same as CO3^2-. If replacing, choose the value consistent with the water/ePC-SAFT model source and rerun. |

## Recommended Table Edits

If the main thread is only editing the manuscript:

1. Keep the current numerical values that match the committed run artifacts.
2. Add `\approx` or a superscript note to `CO3^{2-}`, `H3O+`, and `OH-` placeholder-like `sigma`, `epsilon/k`, and Born-diameter entries.
3. Add a short table note such as:

   `Auxiliary carbonate and water-ion parameters marked with approximately equal signs were retained as assumed hydrated-ion-scale values in the committed validation runs; they were not independently regressed in this work. Literature ePC-SAFT values are available for these ions and should be used in any future rerun that claims them as source-backed parameters.`

4. For `MEAH+` and `MEACOO-`, use wording like `estimated amine-cation analog` and `estimated carboxylate-anion analog` unless a direct MEA-specific source is found. A suitable note is:

   `The MEAH+ and MEACOO- segment-energy and segment-size values were treated as estimated analog parameters in this work. MEAH+ was assumed close to protonated MDEA-type alkanolamine cations because of similar charged amine chemistry, while MEACOO- was assumed close to carboxylate-anion analog values. Dedicated regression of these MEA-specific ionic parameters is ongoing work and was outside the scope of the present validation runs.`

## Segment Number `m` For Ions

Do not replace ion `m` values with a dash if the table is meant to report model parameters used by the run. Published ePC-SAFT ion tables commonly set ion segment numbers to `m = 1` as a fixed spherical-ion convention. A dash could incorrectly imply the parameter is absent from the model.

Better options:

- show `1` with a footnote: `fixed to unity for charged species`;
- show `1^{\dagger}` and explain it in the table note;
- only use a dash if the column is renamed to mean `fitted m`, not `m used`.

## Sources Found

- Baygi 2015 supports the MEA pure-component row.
- Bülow et al. 2021 ePC-SAFT sour-gas / aqueous solvent tables compile several charged-species parameters relevant to HCO3-, CO3^2-, OH-, H+/H3O+, and amine ions.
- RSC ePC-SAFT pH article with ion parameter table: <https://pubs.rsc.org/en/content/articlehtml/2015/cp/c5cp06166k>
- Held, Cameretti, and Sadowski 2008 contains an alternative OH- parameter set, but it is tied to that paper's water/electrolyte parameterization and should not be mixed without checking consistency.

## Main Thread Decision Point

Choose one of these before editing:

1. Conservative manuscript-only fix: annotate current run-used values as assumed where needed, add citations/notes, and do not change model data.
2. Source-backed parameter update: replace placeholder-like ion values with literature-backed values, update the dataset, rerun affected validation/speciation evidence, regenerate Table A3, and then revise the manuscript.

The conservative path is safest for a quick submission cleanup because it preserves consistency with the current committed figures and tables.

## Requested Born / SSM / DS Cleanup Direction

The author now wants the main thread to test a simpler electrolyte presentation:

1. Remove the `dBorn` column from the appendix parameter table.
2. Remove paper-facing SSM and DS terminology/equations/claims if the selected manuscript model no longer needs those terms.
3. Keep the Born contribution if it remains part of the active electrolyte ePC-SAFT thermodynamic calculation.
4. Replace the `dBorn` column with dielectric-constant information for the relevant components, or with the dielectric-rule parameters actually used by the code.
5. Add a brief appendix expression for the dielectric constant or dielectric mixing rule used in the model.

Important validation gate: before removing SSM/DS from the manuscript, the main thread should rerun or otherwise validate that disabling SSM/DS while retaining the Born contribution gives the same relevant validation outputs within the accepted tolerance. If the values change, the manuscript should report the model actually used for the committed figures and tables, not the simplified description.

Suggested checks for the main thread:

- Identify the exact ePC-SAFT option set used for the current committed NCCC figures/tables.
- Run a bounded comparison with SSM/DS disabled and Born retained for at least the accepted validation rows or a representative subset if the full sweep is too slow.
- Compare capture, temperature-profile, runtime, and any full-speciation feasibility metrics against the committed artifacts.
- Only remove SSM/DS paper-facing language if the comparison proves those terms are inactive or numerically immaterial for the reported results.

Table caution: if the code uses a temperature- or composition-dependent dielectric rule, do not present a single dielectric constant as though it were an independent fitted pure-component parameter. In that case, Table A3 should list the dielectric-rule value/coefficients and the appendix should define the expression used to evaluate it.
