# Nine-species liquid on the greenfield Engine (exploratory)

Inputs for the twelve-state callbacks (`reactive_bundle.engine_liquid`,
`engine_vapor`); numerical qualification is in
`analyses/greenfield_node_qualification/`.

- `parameters.json`: byte copy of MEA-Thermodynamics
  `analyses/mea_parameter_bundle/results/selected-current-best-parameters.json`
  at `04a9328`, SHA-256 `868a5018…`. Exploratory, not the adopted record (#61).
- `engine-reactions.json`: R1–R5 source-basis `ReactionLogPolynomial` and
  `ReactionReference` records, the neutral reference and the Born runtime
  defaults, as produced by that commit's `shared_evaluation.py`
  (`_engine_reaction_records`, `_neutral_reference_data`). It names the
  parameter hash it belongs to; loading checks it.
- `ideal-gas-thermochemistry.json`: physical ideal-gas records shared by the
  liquid and the four-gas vapor: NIST Shomate CO2, water (extrapolated below
  500 K), N2, O2; Zhang–Que–Chen 2011 MEA. The Engine completes the ion
  records from the reactions. Interim copy until MEA-Thermodynamics#112
  retains the same records. It replaces the liquid-fitted effective Cp
  inputs of `MEA_reactive_epcsaft_bundle/`, which the seven-state route keeps.

Liquid mass balances use element-formula molar masses, checked against the
declared ones to 1e-5 kg/mol, so reactions conserve mass exactly.
