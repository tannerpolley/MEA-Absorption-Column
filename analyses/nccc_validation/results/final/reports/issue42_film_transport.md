# Issue 42: source-only species-resolved film transport

Status: **supported-negative source-only transport evidence complete**. The retained record extends the Issue 35 source reconstruction with the nine-species transport inventory, explicit quantity types, source chains, conversions, uncertainty states, and a structurally blocked Candidate A/B comparison. No physical film flux is calculated or adopted.

## Source recovery and reconstruction

The seven retained local Zotero PDFs were rehashed before generation. The Issue 35 correlation table was replayed from its committed input and resolver with an exact CSV hash match. Its retained counts remain 23 Weiland loaded-density states, 48 Hartono density states, 48 Hartono viscosity states, and 16 Snijder diffusivity source states. The maximum Snijder reconstruction differences remain 0.054757 relative and 6.401227067e-11 m2/s absolute against the displayed rounded values.

The molecular records preserve the Luo CO2-water and modified Stokes-Einstein relationships and the Snijder free-MEA relationship with their source quantity labels, units, domains, conversions, and uncertainty statements. Snijder's dispersion-derived coefficient remains `not_reported` for tracer/Fick/Maxwell--Stefan classification at the retained locator. The retained source chain does not supply a source-complete H2O self-diffusivity record or the primary Ko, Jamal, and Ying--Eimer N2O inputs. It also supplies no species-resolved ionic diffusivity record for MEAH+, MEACOO-, HCO3-, CO3^2-, H3O+, OH-, no source-defined equal-ion lump, and no complete primary unequal-ion mobility/friction law. The legacy scalar ion expression remains rejected and is not used.

## Candidate decision

Candidate A, `A_equal_ion_effective_fick`, is **blocked** because all nine source-complete effective diffusivities or a cited source-defined lump are unavailable, and the concentration basis is unresolved. Candidate B, `B_unequal_ion_zero_current`, is **blocked** because its complete generic unequal-ion electrochemical-potential mobility/friction law, unequal-ion inputs, ePC-SAFT Gamma, and admitted true-species state are unavailable. Its quantity type remains `not_reported`; scalar diffusivities are not converted into a mobility matrix.

The comparison table retains 5 declared states: two source-label rows, two out-of-common-temperature rows, and the packet-evaluated Position 1 row. All 5 rows are `not_attempted`; evaluated states, CO2 fluxes, species fluxes, paired Delta J intervals, uncertainty widths, numerical-error bounds, charge/current residuals, transfer directions, and positivity results remain blank. The packet and kinetic dependencies admit zero physical rows.

## Provenance and claim boundary

Source revision: `2ef7706a2c328701c1deffed0f4df460428011b8`; generator SHA-256: `14f8b3f17970fcf96a6a5772d457cfaa431fd00dff85bbaf278777213c511354`; input SHA-256: `7add22f65906ae9548827896c6f19dd3062e5c272d6999e75b43e76b562bd181`; machine: `Linux-7.0.0-30-generic-x86_64-with-glibc2.39`; workers: `1`; run identity: `issue42_source_only_2ef7706a2c32`.

No ePC-SAFT package, parameter document, parameter bundle, or mutable sibling checkout was used. The result does not establish Candidate A or Candidate B adequacy, universal unequal-ion mobility/friction adequacy, thermodynamic or kinetic validation, packed-column capture, or a manuscript result. Physical transport selection remains unresolved until source-complete inputs, an admitted common true-species/kinetic state, and the stated physical checks exist.

Regenerate with:

```text
uv run python analyses/nccc_validation/scripts/resolve_issue42_film_transport.py
```
