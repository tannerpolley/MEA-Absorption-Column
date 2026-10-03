# Amine #160 absorber candidate inputs

This directory holds the unpromoted F1″ candidate for the Stage 5 absorber
check. The column reads only these local files. The parameter file is an exact
copy of Amine-Thermodynamics commit
`083083c55c3c5490100063c251507fe3879128d9`,
`analyses/mea_parameter_bundle/results/density-cp-160/D1-evidence/F1-double-prime-candidate-parameters.json`
(SHA-256 `756fec502d3a1433d538abb073c4caeabf91902013ab7e27d5459ce70396543b`).
Removing its density-correction metadata leaves the F1-A physical parameter
values; the remaining record differences are `document_id` and `purpose`.
F1-A `parameters.json` has SHA-256
`cd1ad71b551848e04a6841d4a4ccc017371910ae77f6bc0984b34322a3785a94`.

`engine-reactions.json` was generated with that commit's
`shared_evaluation.py` helpers `_engine_reaction_records` and
`_neutral_reference_data`, using `reaction_values(candidate)` and the corrected
source packet `results/source-corrections-152/state-packet.json.gz` (SHA-256
`3fca278be8790d58ab30e5474b2b3892c47225d8230ac40fd4f87c066e8eaf2d`). Its
`parameter_sha256` names the exact local `parameters.json`; its provenance
records the source commit, files, helper, candidate reaction coefficients,
standard-state basis, and runtime Born defaults. Candidate reaction records
retain the corrected packet's nine-species order and water neutral reference.

`ideal-gas-thermochemistry.json` is an unchanged copy of the accepted
`MEA_greenfield_exploratory` record (SHA-256
`b0af93ab2650b7aec9c8214ca2a8c187ad7462e7681af466d17bb2d1a499e607`). Its CO2,
water, and MEA coefficients match the physical ideal-gas calorics used by
Amine-Thermodynamics' `verify_reference_calorics.py`, the producer used by the
three Hilliard checks. The absorber record also retains the N2 and O2 vapor
records; the Hilliard temperatures lie within the common neutral-record ranges.

The candidate's `empirical_density_correction` field is the only owner of the
translation constant. `reactive_bundle.engine_liquid` reads that field and
passes the converted volume-per-amount value to `EnginePhase`. For liquid
amounts `n_i`, native raw molar density `rho_EOS`, total amount `n`, and the
two named ion amounts, the CasADi output density is
`n / (n / rho_EOS + c_SI * (n_MEAH+ + n_MEACOO-))`. Multiplying this molar
density by the mixture molar mass gives the accepted translated mass-density
equation. The translation changes only the density returned to the column;
native equilibrium states, activities, enthalpy, and heat capacity remain
unchanged.
