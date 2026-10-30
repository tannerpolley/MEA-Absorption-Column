# MDEA thermodynamic construction and native actions, checkpoint 1

The existing `engine_liquid` constructor consumes the identified B1 input and
passes two-state value/action/reference/permutation checks. **Numerical
verification at two thermodynamic states on a candidate wheel: development
evidence, no absorber prediction.** Checkpoint 2 remains blocked research/design
work. Delivered independent review and investigator promotion remain open.

## Identities and input basis

`design.md` is the corrected accepted design, #85 comment 6531482275. The owner
authorized Build after its digest correction. Base is
`b2e2d2a087fb36ca40da80a330dad9638fd06eff` on the issue-85 branch.
`identity.json` records the exact source/input hashes and upstream objective
citation; objective assembly/reproduction remains upstream.

The source bytes are owned by
`src/mea_absorption_column/data/epcsaft_datasets/MDEA_B1_217026ab/`;
`inputs/` contains byte-identical run snapshots of them. B1 parameter SHA-256 is
`1d0d80832f282cde85acab294501ca9ec8dbc34cddf8edf2eebf6e1e1d03949d`.
The source CSV SHA-256 is
`488940ecf3fc984edeee3fa145fe6f76936396fce802b5af881bcbee70f677ef`.
`declared/` and `permuted/` retain the derived declaration/reaction records.
The reference pressure is absent/None for R1–R3 and fixed 100000 Pa for MDEA.
The candidate is Engine commit `217026ab60b8e244cd7d9dac126b8b3d16bccb67`,
wheel SHA-256 `b4c0f6d8eb8971abd3c0bd801cb9167b8ca602cb2a63b16a22e924d5b6727496`.
It is installed only in `.tmp_local/issue85_stage4/candidate`. The repository
pin remains `f4566b3d` / `57bdf6c4277722fdf378107fa940fd321a4e9caeed8875165a27c318239dbd4a`;
`uv.lock`, `pyproject.toml` and the integration identity are unchanged.
`candidate-requirements.txt` and `candidate-install.log` identify the separate
installation; no Engine source or HSL library is imported or bundled.

Both fixtures use 313.15 K, 500000 Pa, 1 mol analytical MDEA,
15.433850227880626 mol water; B has 0.128 mol CO2 and I has 0.258 mol.
They select Böttinger Table 2 rows 26/27, `comparison:229` / `comparison:233`,
with 30% unloaded MDEA mass fraction. Row 27 retains the impurity indicator.
The loading-path point uses lambda=ln(0.258/0.128), not an interphase solve.

## Result locations and acceptance

- `values_{B,I}_{declared,permuted}.json`: fixture, ordered labels/controls,
  state outputs, all first actions, conservation, direct-native comparisons,
  pressure directions, solver/linear solver, actual wheel and runtime.
- `actions_{B,I}_{declared,permuted}.json`: consumed CO2-by-all-control second
  actions, symmetry, loading-path values/Jacobian, explicit chain-rule checks
  and both centered-difference step errors.
- `reference_B_{declared,permuted}.json`: one frozen native state per order,
  complete coordinates/composition/density and the mixed/all-fixed/all-system
  evaluations at that unchanged point. No mutated construction is equilibrated.
- `permutation.json`: ratios after mapping state/action axes and reaction IDs
  back to the declared order.

The maximum first-action ratios at h=1e-3 / 5e-4 are
**0.0249076382 / 0.0308368502**; second-action ratios
**0.00225980393 / 0.000618463691**; loading-path ratios
**0.691469400 / 0.172894240**. Each ratio must be <=1, using
`abs(native-difference)/(1e-8+1e-5*max(abs(native),abs(difference)))` on the
accepted scaled quantities. Pressure displacements are 10/5 Pa and T
displacements 0.001/0.0005 K. Second-action symmetry and loading-tangent
value differences are zero; maximum explicit chain-rule ratio is 4.43257e-10.
The maximum permutation ratio is **0.000382739015 <=1**, using rtol=1e-7,
atol=1e-9 after the same scaling.

Direct-native value ratio is at most **1.53214e-8 <=1** with rtol=1e-8,
atol=1e-10 on scaled values. Maximum element/component/charge errors are
**1.42109e-14 / 4.16334e-15 / 2.49801e-16 mol**, all <=1e-10 mol.
Minimum supported amount is **1.78376e-9 mol >0**; all outputs are finite,
and native solves meet requested row tolerance 1e-10.

Mixed-reference frozen affinity agreement is **0 <=1e-9** in A/(RT).
The all-fixed control differs by at least **0.005153896778**, and all-system
by at least **0.0004202765353**, both **>1e-8** as required. R2 and MDEA
respectively carry the largest discrepancies. Affinities in J/mol of reaction
as written are also retained. These are numerical discrimination checks,
not experimental deviations.

## Replay and limits

Use the accepted design for complete scales and units. Create a new output
root, retaining/linking the same four `inputs/` files; preserve previous runs.
For each group, fixture B/I and declared/permuted order run (add `--permuted`
for the reordered case):

```bash
nice -n 10 timeout 45s timeout 30s env UV_PROJECT_ENVIRONMENT=.tmp_local/issue85_stage4/candidate PYTHONPATH=src uv run --frozen --no-sync python analyses/bvp_solution_methods/scripts/verify_mdea_thermodynamics.py NEW_ROOT values --fixture B
nice -n 10 timeout 45s timeout 30s env UV_PROJECT_ENVIRONMENT=.tmp_local/issue85_stage4/candidate PYTHONPATH=src uv run --frozen --no-sync python analyses/bvp_solution_methods/scripts/verify_mdea_thermodynamics.py NEW_ROOT actions --fixture B
nice -n 10 timeout 45s timeout 30s env UV_PROJECT_ENVIRONMENT=.tmp_local/issue85_stage4/candidate PYTHONPATH=src uv run --frozen --no-sync python analyses/bvp_solution_methods/scripts/verify_mdea_thermodynamics.py NEW_ROOT reference
nice -n 10 timeout 45s timeout 30s env UV_PROJECT_ENVIRONMENT=.tmp_local/issue85_stage4/candidate PYTHONPATH=src uv run --frozen --no-sync python analyses/bvp_solution_methods/scripts/verify_mdea_thermodynamics.py NEW_ROOT permutation
```

Each group records `execution.wall_limit_s=30`; the external 30-second worker
bound includes all evaluations, not 30 seconds per difference side. Final
value/action groups take 0.45–1.35 seconds inside the verifier under shared
machine load; interpreter startup is additional and included in the external
bound. Each command has a separate log. There is no new executor or numerical
fallback. Finite differences are verification only; the production callbacks
still consume native exact actions.

The constructor changes are confined to `reactive_bundle.py` and
`casadi_reactive.py`. They retain the existing reaction/parameter/caloric owners
and native solver. The pinned tests exercise the same construction path and
frozen-reference check in both orders. The existing MEA test's wall bound is
reduced from 600 to the design's 300 seconds; equations/tolerances are unchanged.

## Failures and corrections

`values_B_declared_attempt1.log` retains the first verifier failure: a native
request duplicated CO2/water log observables already present in the state
observable list. The corrected verifier uses separate nonduplicated requests;
`attempt2` and final value records pass with unchanged physics/tolerances.
`pinned_r2.log` retains a postprocessing KeyError for an incorrectly printed
`errors` key, after the declared numerical comparisons were saved. The ordinary
existing verifier supplies the final declared/permuted records under
`pinned_r2_final/`, with 51 comparisons each. No numerical model failure was
hidden by that reporting correction.

The reference verifier's final2 logs use the direct native upstream compiled
problem as the affinity target. Earlier reference evaluations had reused the
downstream compiled target; they are not the final independent comparison.
No production equations or native method changed for either verifier correction.

`candidate_final_rejection.log` has expected exit 1: candidate hash rejected
against the final frozen identity. `candidate_development.log` has exit 0.
`pinned_integration.log` has exit 0 on the unchanged pinned wheel. The candidate
final-check output's `Engine commit` line is the repository frozen identity;
use its actual wheel hash and `identity.json` for the candidate commit.

## Scientific limits

The assessed source ranges are prepared-row extrema, not a continuous validity
volume. The action neighborhood, especially ±P around the 5-bar observation
plane, is numerical evidence only. Calorics at 313.15 K are not validated by
298.15 K Cp/density data; water Shomate calorics are extrapolated below 500 K.
The absolute enthalpies depend on the declared anchors/reaction completion.
This checkpoint selects no MDEA transport coefficients, kinetic/film model,
packing or absorber feed, and makes no physical-validation or capture claim.
The notebook uses tables because these are two fixed thermodynamic fixtures
and finite acceptance checks, not a continuous physical trend.

Native fixture solves report MA57. **HSL, a collection of Fortran codes for
large-scale scientific computation. https://www.hsl.rl.ac.uk/**

## Pinned persistent behavior checks and delivery budget

The pinned full suite passes **332 tests in 521.43 s**, under the 900 s suite
worker / 960 s outer bound. Its one warning is the existing legacy transport
probe's deliberately recorded invalid-log evaluation. The two new parametrized
frozen-reference cases also pass independently in 0.87 s.

The unchanged MEA 3C gate accepts both seed/cosine solves in both species orders.
Cosine captures are **89.41488041587351% / 89.41488041586858%**, versus retained
**89.41488041587351%**, within 1e-6 percentage point. The 306 native state/action
comparisons have maximum scaled difference **2.69069e-11 <=1e-7**; film
comparison maximum is **5.43046e-13 <=1e-7**. Every existing mixed-unit profile
and physical-certificate criterion passes. The four solve runtimes are
108.5134, 98.9339, 109.3827 and 95.9997 s, each under its 300 s worker bound.

The pinned stage-2 film at 129 points gives **0.0008333330846335338 mol/(m² s)**,
relative flux/integral error **2.98440e-7 <=1e-5**, interface error
**0.0024870 Pa <=0.1 Pa**, and closure **5.42101e-19 <=1e-12 mol/(m² s)**.
Both orders pass. The mathematical R2 gate passes **51 comparisons per order**.

Both pinned stage-3 finest cases (129 nodes / 129 film points) give
**31.454926062128543%**, with zero capture permutation/preservation difference,
and **0.07205477434 <=0.1 percentage-point** continuum error. Maximum profile
error is **1.3210041963e-4 <=2e-4 mol/s**, maximum flux error
**8.1516465662e-7 <=1e-5 mol/(m² s)**, and maximum scaled equation residual
**5.91194e-14 <=1e-9**. Minimum species amount is 0.05 mol; maximum component
and affinity errors are **1.11022e-16 mol/s** and **4.44089e-16 in A/(RT)**.
Runtimes are **252.1625 / 250.5742 s**, each below 600 s. Existing coarser
refinement records are reused by identity from stage 3; the current finest
cases pass the existing refinement/permutation/preservation comparison against
them. No old refinement numbers are presented as new runs.

Gross executable additions against b2e2d2a, including replacement lines and
comments/blanks: reactive_bundle.py **12/80**, casadi_reactive.py **5/20**,
one verifier **200/200**, total **217/300**. Test additions are **21/75**:
20 lines in one result-level test parametrized over two orders and one changed
existing wall-limit line. No new reaction owner or abstraction is introduced.
Data, prose and generated evidence are accounted separately.

### Correct candidate refusals and scope decision

Candidate MEA and both stage-3 gate attempts stop before numerical solving with
**capability_refusal: Loaded epcsaft distribution does not identify the selected
wheel**. Their unchanged requests select the pinned wheel. The first stage-3
launches also omitted PYTHONPATH=src and stopped at import; these separate logs
are retained. No identity checks or runners were changed to bypass either
refusal. The owner confirmed that candidate-wheel MEA, stage-2 and stage-3
repeats are **unavailable in checkpoint 1 and belong to a future repin**, as in
the stage-4 repin requirements. Stage 2 was not run on the candidate. Only the
MDEA thermodynamic verifier uses the isolated candidate wheel for accepted
numerical evidence. The pinned gates and final integration remain mandatory
and pass.

The first notebook refresh refused run-input symlinks; input snapshots are now
byte-identical regular files linked by provenance/hash to the canonical retained
source data. This is a presentation failure, not a numerical failure. Its log is
retained. This documentation also corrects the initially transcribed caloric
README digest to the hand-off/file value 278e7b9832dc051f2896615f870f0f91482a8642b7196d2a2a89b669b905ce6c;
the verifier always checked that value against the hand-off before solving.
