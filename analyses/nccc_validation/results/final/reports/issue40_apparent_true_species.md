# Issue 40 apparent-to-true species mapping

This retained analysis reconstructs source apparent component flows and maps the in-domain retained Case 3C Position 1 state through the authoritative nine-species reactive ePC-SAFT packet. It does not replace the Issue 33 source basis: the exact source values remain **4.889309897097635 mol/L analytical MEA** and **2.491683471902737 mol/L free MEA**, with `basis_unresolved` admission.

## Reproduction and identities

Command: `python3.13 analyses/nccc_validation/scripts/resolve_issue40_apparent_true_species.py --bundle /home/tnnrpolley21/Workspaces/Engineering/MEA-Thermodynamics/analyses/mea_parameter_bundle/results/handoff/mea-reactive-epcsaft-parameter-bundle.zip`<br>
Repository commit: `616ff267253cfecde4a395984a94c1b455db8c85`<br>
Generator SHA-256: `d436d86c960b8f9a410709446e796b4eac2b724bd5956f3cb1231aa6a4c62d5f`<br>
Machine: `Linux-7.0.0-30-generic-x86_64-with-glibc2.39`; workers: `1`

The immutable bundle identities are outer zip `4139fecd9b5192e7cadd12883d2ff1bff71c20d74950af5256e4f0447995f27b`, parameter document `2666914f0f9cfebdf230e96565de843f9aadc9424035c940883147ff66af035c`, ePC-SAFT wheel `d7b4fc5ba5cbf0e979b65af83442d565496d11b771bb559233ad9dc3a4f8414a`, state packet `41017bcf727a486a8f3feb280e19c111a15c5dda5a3cca4e8c7dc5b051168fef`, and parameter fingerprint `sha256:c1fc2665e94d136eb85f27c793b7defbd16d1d82cb3173cb50a9aaf6513c8940`. The bundle verifier was run before this analysis.

## Definition and method

The true species order is `CO2, MEA, H2O, MEAH+, MEACOO-, HCO3-, CO3^2-, H3O+, OH-`. Apparent inputs are normalized component flows `(CO2, MEA, H2O)` in `mol per mol analytical MEA`, with CO2 representing apparent total inorganic carbon. The retained transform rows are `analytical_MEA, total_inorganic_carbon, water_equivalent, elemental_C, elemental_N, charge`. Prepared concentration, analytical concentration, free MEA, apparent totals, true species, and diagnostic density are separate definitions in the Issue 40 input record. Every apparent input records units and reporting-interval status in the result table; absent experimental intervals remain explicitly `null`/unreported.

The packet request is one finite liquid phase at `T=318.15 K`, `P=109500 Pa`, with no vapor phase and no VLE fugacity equality. Reaction constants are compiled at the requested temperature from the bundle reaction correlations, including the explicit R1--R3 standard-state offsets. The packet's electroneutral positive interior seed is used only as a solver start; exact apparent C/N totals are conserved independently. The returned true state is forward-transformed and inverse-replayed by conserved totals, and the same request is solved twice for deterministic identity.

## Retained row accounting

| Row class | Count |
|---|---:|
| Source rows | 5 |
| Literature label rows | 2 |
| Case 3C retained profile rows | 3 |
| Packet candidates | 1 |
| Packet evaluated | 1 |
| Packet not attempted | 4 |
| Scientific admissions | 0 |
| Basis-unresolved rows | 5 |

Position 1 is packet-evaluated with a single `strict_stable_local_minimum` liquid branch. It remains scientifically unadmitted because the source prepared/loaded volume basis is unresolved. Positions 0 and 0.5 are retained as out-of-common-domain, not attempted; Putta labels are retained as source-label-only rows.

## Claim boundary

The mapping is numerical evidence for a specified single-liquid state, not a packed-column result. It infers no thermodynamic, kinetic, transport, area, hydraulic, or capture quantity, and it performs no parameter fit. Historical packet failures remain represented by their immutable state-packet/non-evaluable-state identities; no historical continuation state is reused as a process initial condition.
