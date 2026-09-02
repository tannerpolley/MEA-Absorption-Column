# Issue 41: source-rate evidence and packet-consistent reversible MEA kinetics

Status: **supported-negative evidence record complete**. The source laws, exact stoichiometric projections, raw-observation inventory, declared estimation/validation split, and immutable provider equilibrium relationships are retained. No physical reactive film is adopted.

## Retained result

- Fixed species order: `CO2, MEA, H2O, MEAH+, MEACOO-, HCO3-, CO3^2-, H3O+, OH-`.
- Fixed projections: `F1 = R2 - R4 - R5`, `F2 = R2 - R4`, and unresolved `F3 = R2 - R1`.
- Source-rate evidence contains 5 rows: Putta2016 F1/F2 concentration and activity forms are retained, including their source units and domains; F3 remains unavailable because the cited Gondal2015 coefficient was not recovered.
- The source-printed `m^6 kmol^-2 s^-2` F1/F2 unit is retained as rejected metadata; the dimensionally required rate coefficient unit is `m^6 kmol^-2 s^-1`.
- The raw inventory contains 23 records, including the 20 Putta2016 Table 4 aggregate AARD cells. No row-level rate observations or row-level uncertainty weights are available, so no estimation or validation fit is performed.

## Provider relationship and packet boundary

The immutable handoff bundle `mea-reactive-epcsaft-parameter-bundle` is internally hash-consistent. Its source standard state is `aqueous-molality-infinite-dilution-water-v1`, with products-positive reaction orientation. Provider `K(T)` is compiled at the three retained anchors (293.15, 313.15, 323.15 K) for all three projections, but `ln Q`, residuals, and detailed-balance pass/fail are intentionally blank: Issue 40 retains all five candidate rows as `basis_unresolved` and admits zero scientific rows.

The detailed-balance criterion is `abs(ln Q - ln K) <= 1e-07` only for a scientifically admitted true-species state on the bundle standard state. That prerequisite is absent here. Reaction-rate evaluation, reaction timescales, and film partition therefore remain not attempted.

## Evidence gaps and next gate

The source apparatus split is declared but remains `predeclared_only_no_row_ids`; the source does not provide retained row IDs, raw rates, or a usable uncertainty covariance. F3 has no admitted primary coefficient. Issue 40's packet mapping is numerically retained but not scientifically admitted because the prepared/loaded concentration basis is unresolved. A future update may evaluate detailed balance only after a source-basis resolution admits a packet-bound activity vector; transport admission remains a separate downstream gate.

The exact bundle identities are retained in `issue41_reversible_kinetics_summary.json`; the outer archive SHA-256 is `4139fecd9b5192e7cadd12883d2ff1bff71c20d74950af5256e4f0447995f27b` and the parameter, wheel, state-packet, and chemistry member hashes are recorded there. No bundle provenance mismatch was found.

Regenerate with:

```text
uv run python analyses/nccc_validation/scripts/resolve_issue41_reversible_kinetics.py --bundle /home/tnnrpolley21/Workspaces/Engineering/MEA-Thermodynamics/analyses/mea_parameter_bundle/results/handoff/mea-reactive-epcsaft-parameter-bundle.zip
```
