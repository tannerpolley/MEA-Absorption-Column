# August 27 predictive reactive ePC-SAFT revision plan

Status: canonical
Schedule status on 2026-08-24: `FALLBACK_ACTIVE`; the August 22 predictive-model freeze was missed, so August 23--26 are reserved for the fixed-chemistry submission package and reviewer response
Submission deadline: 2026-08-27
Internal model freeze: 2026-08-22
Internal submission-ready deadline: 2026-08-26

## Authority and objective

This is the canonical cross-repository execution plan for the revised
MEA-absorption manuscript. It supersedes informal date estimates for this
revision, but it does not supersede repository-local scientific contracts,
governance decisions, immutable data receipts, or accepted engine promotion
evidence.

The primary objective is to deliver a submission-ready revised manuscript by
August 27 with a scientifically accepted predictive reactive ePC-SAFT model if
all dated acceptance gates pass. The already revised fixed-chemistry fugacity
benchmark remains the fail-closed submission path. A missed scientific gate
does not permit weaker evidence to be relabeled as predictive.

The three repositories have distinct ownership:

| Repository | Owner responsibility |
| --- | --- |
| `ePC-SAFT-project` | EOS, homogeneous and phase-equilibrium values, exact CppAD derivatives, solver certificates, regression machinery, installed wheel, and capability receipts |
| `MEA-Thermodynamics` | MEA reaction/source contract, canonical observations and splits, application-owned residual definitions, parameter-selection evidence, and held-out thermodynamic validation |
| `MEA-Absorption-Column` | Installed-engine adapter, column integration, NCCC validation, sensitivity and numerical evidence, manuscript, and reviewer response |

No production runtime may import a sibling source checkout. Cross-repository
work moves through an immutable wheel, copied source-traceable data packets,
fingerprints, and retained receipts.

## Scientific endpoint and claim ladder

The final scientific endpoint is true reactive VLE: from temperature and total
MEA/H2O/CO2 composition or loading, the model predicts equilibrium pressure,
vapor composition, and liquid speciation without using the observed pressure
or observed CO2 partial pressure as a model input.

Work proceeds through a strict claim ladder:

1. **Certified reacting liquid:** fixed-`T,P` homogeneous chemical equilibrium
   predicts liquid speciation and fugacity using independently sourced
   activity-basis reaction constants. This is a tracer and a valid column-local
   thermodynamic state calculation, but it is not by itself predictive VLE.
2. **Reduced coupled tracer:** one independently observed pressure/fugacity
   quantity and one eligible speciation quantity establish the exact
   value/Jacobian/regression path for no more than two preregistered
   coordinates.
3. **Predictive reactive VLE:** a certified reactive bubble/VLE solve predicts
   pressure, vapor composition, and liquid speciation with exact total
   parameter sensitivities.
4. **Predictive absorber integration:** the frozen reactive model runs in the
   column without case-specific fitting, hidden fallback, or parameter drift.

The following do not establish predictive reactive thermodynamics:

- reaction constants rebased from legacy compositions;
- a feasible stationary point that fails the local-minimum certificate;
- a fit evaluated only on its training observations;
- observed-pressure fixed-`T,P` calculations presented as pressure prediction;
- model selection and parameter fitting performed on the reserved set; or
- a converged column run containing rejected local chemistry states.

## Current evidence baseline

As of August 12:

- The new monorepo has one `epcsaft` runtime, a public homogeneous chemical
  equilibrium API, exact active-parameter sensitivities, reduced-Hessian and
  local-minimum diagnostics, CppAD derivative authority, a general nonreactive
  phase-equilibrium owner, and generic positive-observation regression
  transport.
- The retained source-complete MEA reacting-liquid sentinel is
  `FEASIBLE_ONLY`: its solver, balance, charge, pressure, reaction, and EOS
  checks pass, but negative reduced-Hessian curvature rejects the local-minimum
  claim.
- The current unified-Engine governance defers Equilibrium-owned observation
  families from Regression. General upstream issue 30 now requests certified
  fixed-topology homogeneous reactive observation blocks, and dependent issue
  31 requests certified reactive bubble/VLE blocks. Publication closes the
  issue-shaping part of E0; implementation and admission remain open.
- MEA-Thermodynamics owns source-adjudicated five-reaction chemistry,
  pressure/speciation observations, volumetric observations, and the
  application-side data/model/stage manifests. Its historical 147/220 split
  remains frozen evidence. If Jou rows enter training, a new campaign-blocked
  split identity and hashes must be frozen before optimization; the historical
  split must not be relabeled.
- The active predictive-v2 draft contains 319 admissible observations in 26
  indivisible campaign blocks across five folds (121 pressure and 198
  speciation targets). The live MEA evidence-library validator passes this
  draft, but its independent reviews and repository freeze are still in
  progress, so it is not yet G0 acceptance evidence.
- The absorber repository has a frozen before-revision PDF, a visually checked
  31-page quick-revision PDF, and a working fixed-chemistry/ePC-SAFT fallback.
  The 7C/K19 validation contradiction and first-citation numbering are
  reconciled. Its historical parameter family must not be promoted as the
  predictive reactive model.
- The frozen downstream v1 application contract passes its structure and exact
  tracked-artifact hashes. Its clean-checkout source mode is temporarily
  expected to fail while the active MEA-Thermodynamics task owns modified and
  untracked predictive-v2 context, roadmap, cross-validation, model, parameter-
  stage, inventory, and validator files. The bound v1 source artifacts have not
  changed. Final mode fails closed because the required upstream capabilities
  and installed receipt are pending.
- The current upstream wheel was rebuilt under an unchanged filename. Its
  SHA-256 is `81a6b876c4ddee639e0a3da92444f97293b427fd288c7ebc6171467aa3567de9`,
  while the downstream lock expects
  `8e282a13b405f44a7178d53abd7b9637244fd84bce3e89eb784f48e178a3a3a2`.
  This is an E7 packaging blocker; the downstream lock must not be refreshed
  against another mutable in-place wheel.

## Model-form discipline

Induced association, polar contributions, SSM+DS Born corrections, and improved
relative-permittivity formulations are model structures, not a license to fit
every available coefficient simultaneously.

Before regression, one preregistered model hierarchy must be frozen:

1. source-consistent association topology and ordinary cross-association;
2. one source-defined induced/cross-association correction, only when its
   parameter identity and exact chain-rule derivative are advertised;
3. one polar contribution supported by independent observables, with its
   discrete formulation fixed outside the optimizer;
4. the Born formulation fixed to an explicit coefficient pair, including
   SSM+DS only through the engine's admitted `(1,1)` identity; and
5. one relative-permittivity formulation with a source-backed domain and exact
   derivatives for every fitted coefficient.

Discrete model selection occurs before parameter estimation. The reserved set
may reject a chosen model but may not be used to redesign it. If the available
observations cannot identify a contribution, that contribution is either fixed
from an independent source or removed from the fitted model.

The first reaction-correlation experiment releases only the physical
`ln K_4(T) = A + B/T` coordinates. The coordination task reports a diagnostic
three-temperature run in which this pair strongly changed pressure but did not
reproduce the observed temperature curvature. Its earlier receipt is not
retained in the current MEA checkout, so the run contributes a design warning,
not gate evidence or promotable coefficients.
Release `C` only after campaign-blocked held-out improvement, and release `D`
only after profile-likelihood or cross-validation evidence supports it. Keep
R1--R3 source-fixed; defer R5 until EOS, mixture, ionic, and R4 stages are
qualified. Compare the transferred baseline and the source-reference-
transformed polar candidate on the same folds, but exclude the latter from
promotion until its public source-reference transformation is installed.

The first promoted joint pressure/speciation claim is limited to
313.15--353.15 K. Lower-temperature speciation is support evidence; pressure
rows above 353.15 K are a domain-extension challenge, not part of the initial
qualified fit. Corrected SSM+DS/permittivity formulations may be active as fixed
model choices, but their MEA-specific electrostatic coordinates remain fixed
until direct dielectric or ion-activity evidence supports fitting them.

## ePC-SAFT-project critical deliverables

| ID | Required upstream result | Acceptance evidence | Due |
| --- | --- | --- | --- |
| E0 | Reactive-VLE scientific slice and governance admission | Published issues 30/31; typed inputs/outputs; units and bases; fixed topology and row identities; branch, nonuniqueness, conditioning, and Non-Evaluable Trial policy; accepted implementation plans | Aug 13 |
| E1 | Certified stable homogeneous MEA state | Distinct continuation/multistart design; balance, charge, pressure, reaction, KKT, positivity, packing, EOS-domain, reduced-Hessian, and local-minimum certificates all pass | Aug 15 |
| E2 | Installed reacting-liquid observable block | Ordered speciation and neutral-CO2 fugacity values; exact selected-parameter columns; source-reference transform; complete fingerprints and typed failures | Aug 16 |
| E3 | Rank-sufficient reduced tracer | Preregistered `N <= 2` coordinates; full-rank sensitivity matrix at every declared start; immutable two-row receipt; no rebased constants | Aug 17 |
| E4 | Certified reactive bubble/VLE owner | Predicts pressure, vapor composition, and liquid speciation; fixed phase/reaction topology; stable branch certificate; no observed-pressure input | Aug 18 |
| E5 | Exact reactive-VLE sensitivities and regression block | Centered re-solve agreement; complete CppAD-derived total columns; fail-closed boundary, coalescence, singular, ill-conditioned, and branch-change behavior | Aug 19 |
| E6 | Selected-parameter capability for the frozen model | Exact active support for each selected segment, interaction, association, Born, polar, or permittivity coordinate; discrete formulations excluded from the fit vector | Aug 19 |
| E7 | Installed immutable release candidate | Non-editable wheel; commit/tree and packet hashes; capability fingerprint; no legacy/split distribution; affected and full checks pass | Aug 21 |

### Upstream capability decisions

- The first regression must remain small. Start with the existing staged
  MEAH+ and MEACOO- coordinate proposal and add a coordinate only after a rank
  and source-evidence amendment.
- `k_hb_ij` is fit-eligible only if the resolved association-energy transform
  and its exact chain rule are part of the advertised parameter identity.
  Otherwise fit the resolved association energy under its actual meaning.
- Polar model inputs currently represented in the EOS but excluded from the
  Regression roadmap require an explicit governance amendment before they can
  become fit coordinates. They may be evaluated as fixed, source-defined model
  alternatives without claiming they were identified by the MEA fit.
- The SSM+DS Born value path is already represented. Parameter fitting still
  requires exact derivative advertisement and observation support for each
  active Born, solvation, or permittivity coordinate.
- True reactive bubble/VLE is a new requirement. The older decision that it was
  not a tracer gate remains valid for the fixed-`T,P` tracer, but it cannot
  support the final predictive-VLE claim.
- Regression needs one general application-declared scalar-coordinate
  extension so MEA may supply physical R4 correlation coordinates and their
  exact chain rule without adding an MEA-specific observation family to the
  Engine.
- Final certification uses the native Regression/Ceres owner. Application-side
  screening code may diagnose rows or starts but cannot produce the promoted
  parameter receipt.

## Dated execution gates

### August 12-13 — G0: freeze the contract

- Resolve the active MEA-Thermodynamics and monorepo changes before binding
  hashes; do not bind uncommitted working-tree state.
- Freeze species order, charges, molecular weights, elemental balances,
  reactions, standard states, reaction-constant transforms, oxazolidone
  inclusion/exclusion domain, observation roles, model hierarchy, fitted
  coordinates, starts, bounds, scales, and failure semantics.
- Preserve the historical 147/220 split. If Jou is admitted to training, freeze
  a new campaign-blocked cross-validation/all-data-refit partition with a new
  identity and hashes before any outcome-guided fitting.
- Correct MEACOO- molecular weight and select one internally consistent
  parameter family.
- Reconcile the absorber's 7C and K19 artifact contradictions.

Pass condition: one immutable predictive-v2 application contract and accepted
implementation plans for upstream issues 30 and 31 exist. If G0 is not
accepted by the end of August 13, the predictive lane is behind schedule.

### August 14-16 — G1: certify reacting-liquid chemistry

- Complete E1 and E2.
- In parallel, complete the source-closure inputs that can use already admitted
  Engine families: pure-MEA property evidence, source-valued molecular moments,
  primary MEA--H2O binary evidence, physical CO2-solubility data, and any
  proposed induced-association topology and edge values.
- Traverse the intended temperature, loading, concentration, and pressure
  domain using continuation and independent starts.
- Verify conservation, equilibrium, positivity, EOS domain, reduced-Hessian
  stability, and exact active-parameter sensitivities independently.

Pass condition: an installed artifact certifies stable states across the
preregistered sentinel domain. `FEASIBLE_ONLY` does not pass.

### August 17-19 — G2: establish reactive VLE and the reduced fit

- Complete E3 through E6.
- Run the two-row tracer before any broad fit.
- Implement and verify the reactive bubble/VLE owner.
- Freeze the smallest rank-sufficient coordinate set.
- Reject, rather than patch after inspection, any underidentified or
  branch-unstable design.

Pass condition: true reactive VLE values and exact Jacobians pass independent
checks, and the reduced tracer converges from every declared start without
active-bound or rank failure.

### August 20-21 — G3: select, refit, challenge, and package

- Run staged regression under the frozen campaign-blocked cross-validation
  design.
- Select and freeze one model/parameter block from cross-validated evidence,
  then refit that fixed choice once using all scientifically admissible rows.
- Preserve all failed, skipped, inapplicable, and evaluated rows.
- Reject any candidate that fails an admitted pressure row or admitted
  speciation state; failed admitted rows may remain visible as diagnostic
  evidence but may not be removed, reclassified, or hidden to pass the gate.
- Keep cross-validated predictions, all-data calibration residuals, and the
  high-temperature challenge set as distinct evidence products.
- Report objective decomposition, rank, conditioning, parameter correlation,
  active bounds, basin dependence, cross-validated errors by family, profile
  likelihoods for material coordinates, and campaign bootstrap results where
  runtime permits.
- Build the immutable E7 wheel and receipts.

Pass condition: the selected model is scientifically accepted on preregistered
criteria and materially stable to starts and bounds. Optimizer success alone
does not pass.

### August 22 — G4: model freeze

- Integrate the immutable wheel into the absorber.
- Run a local state, one external axial profile, Case 3C, and representative
  low/middle/high-loading cases.
- Confirm that no local chemistry call uses legacy constants, an unaccepted
  state, a mutable parameter set, or a hidden fallback.
- Record runtime and call-count budgets.

Pass condition: freeze the predictive model, parameter packet, engine wheel,
column configuration, and evidence commands. After this gate, model changes are
prohibited unless a discovered defect invalidates the evidence. If G4 fails,
activate the fixed-chemistry manuscript fallback immediately.

### August 23-24 — G5: regenerate manuscript evidence

- Run the baseline Henry, fixed-chemistry ePC-SAFT, and predictive reactive
  ePC-SAFT lanes.
- Complete thermodynamic sensitivity, representative transport sensitivity,
  solver diagnostics, same-hardware performance measurements, and independent
  validation.
- Regenerate all claim-bearing tables and figures from frozen artifacts.

Pass condition: every manuscript number has a source artifact and every failed
case remains visible.

### August 25 — G6: manuscript and response

- Rewrite theory, methods, parameter provenance, validation, sensitivity,
  numerical-performance, limitations, conclusion, and the point-by-point
  reviewer response.
- Call the model predictive reactive VLE only if G2-G5 passed.

### August 26 — G7: submission audit

- Run tests, installed integration, final evidence validation, manuscript
  freshness, LaTeX build, citation/cross-reference checks, and visual PDF
  inspection.
- Compare the final PDF with the frozen before-revision PDF.
- Verify hashes, repository states, and the submission bundle.

Pass condition: submission-ready PDF and response package by end of day.
August 27 is reserved for submission, not model development.

## Tracking protocol

Every working session must begin by reading this plan and must report:

1. current date and active gate;
2. schedule status: `ON_TRACK`, `AT_RISK`, `MISSED`, or `FALLBACK_ACTIVE`;
3. completed acceptance evidence with file or command anchors;
4. the single next critical-path action;
5. blockers, owner, and latest acceptable resolution date; and
6. whether the August 22 model freeze remains achievable.

The Codex goal records the submission objective for this task. The app heartbeat
`mea-august-27-revision-tracker` performs a status-only check every day at 09:00
local time through August 27. It may inspect evidence and report schedule
state, but it may not edit, commit, push, or start long campaigns.

Gate status changes only from retained evidence, never from percent-complete
estimates. Update the table below when evidence changes.

| Gate | Deadline | Status on Aug 24 | Evidence or blocker |
| --- | --- | --- | --- |
| G0 contract freeze | Aug 13 | `MISSED_FOR_PREDICTIVE_V2` | The retained application contract remains v1 with predictive-v2 admission pending; its source binding no longer matches the current MEA-Thermodynamics checkout. |
| G1 reacting-liquid certification | Aug 16 | `MISSED` | No immutable application artifact certifies every preregistered sentinel state with the required stability and derivative evidence. |
| G2 reactive VLE and tracer | Aug 19 | `MISSED` | Upstream now evaluates true reactive pressure/speciation states, but the downstream frozen contract and admitted observation set have not been reconciled and promoted. |
| G3 fit and reserved evaluation | Aug 21 | `MISSED` | A retained 119-state/257-target candidate replays with zero failures, but it is not the contract-admitted all-row fit, the optimizer reports `NO_CONVERGENCE`, and conditioning remains weak; it is diagnostic, not promotable evidence. |
| G4 model freeze | Aug 22 | `MISSED` | No accepted parameter packet, immutable installed artifact receipt, or downstream column freeze exists. The documented fallback is active. |
| G5 regenerated evidence | Aug 24 | `FALLBACK_ACTIVE` | Preserve and validate the fixed-chemistry Henry/ePC-SAFT evidence; do not generate a late predictive lane. |
| G6 manuscript and response | Aug 25 | `IN_PROGRESS` | Quick revisions and the frozen before-version exist; finish the bounded response package with predictive work stated as future work. |
| G7 submission audit | Aug 26 | `NOT_STARTED` | Before-revision PDF is frozen |

Execution evidence retained on August 24:

- The fixed-chemistry fallback manuscript builds to a fresh 31-page PDF from
  the canonical Zotero bibliography; all 31 rendered pages were visually
  inspected.
- The point-by-point response in `docs/reviewer_response.md` addresses every
  reviewer comment and preserves the missing-sensitivity, controlled-cost,
  cross-facility-validation, and optimization boundaries.
- The installed fixed-state ePC-SAFT check passes from the content-addressed
  wheel with SHA-256
  `e14288867d4fb5bc1367dd0de490aeb1551f1613074aced0a8d28432ca762f23`;
  this wheel identity does not promote the upstream predictive parameter set.
- The NCCC evidence validator passes, all 137 repository tests pass, and the
  flat Elsevier source archive passes its ZIP integrity check.

Single next critical-path action: author/coauthor scientific review of the
revised manuscript and point-by-point response before the August 25 text
freeze. Predictive admission, upstream Engine provenance, quantitative
parameter sensitivity, same-hardware solver profiling, independent
cross-facility validation, and operating optimization remain outside this
submission lane.

Escalation rules:

- Any gate with less than one day of float and an unresolved scientific blocker
  is `AT_RISK`.
- A missed G1, G2, or G3 deadline does not automatically end the attempt, but
  the August 22 freeze does not move.
- Failure of G4 activates the fixed-chemistry fallback and redirects August
  23-26 exclusively to submission evidence and reviewer response.
- No result from a mutable sibling checkout, uncertified state, or changed
  reserved set may enter the manuscript.

## Coordination anchors

- Predictive-regression sanity-check task:
  `019f6c45-232d-7e40-92ab-25b1ef808402`
- That task has selected staged true reactive VLE, a factorized neutral/
  reactive model ladder, campaign-blocked cross-validation followed by one
  all-admissible-data refit, and one promoted parameter set. It has published
  upstream issues 30 and 31 and is building the MEA-owned context, partition,
  model, and stage manifests. Its current three-temperature R4 result is
  diagnostic only and demonstrates the need for curvature/identifiability
  testing rather than promotion of `A,B` by optimizer success alone.
- Monorepo working branch observed on August 12:
  `codex/cap-11-final-cutover-audit` at commit
  `ce3de1aaa8aa9088b83a2cad41466f8acb194509`, with concurrent cutover edits.
  Those changes belong to their current task and must not be overwritten from
  this repository.
- MEA-Thermodynamics working branch observed on August 12:
  `codex/unified-engine-cutover` at bound commit
  `ac5ff017870ecf2c7987cba39f243b0399b8f106`; the active task currently owns the
  predictive-v2 context, roadmap, 319-row/26-block/five-fold campaign manifest,
  factorized model configurations, S0--S7 parameter stages, inventory, and
  validator changes. The validator passes; independent review and freeze are
  in progress, so the draft does not yet count as G0 evidence.
- Frozen downstream application contract:
  `integration/reactive_mea_application_contract.json`.
- Source verification command:
  `python3.13 scripts/check_reactive_mea_application_contract.py --mode source --source-root /home/tnnrpolley21/Workspaces/Engineering/MEA-Thermodynamics`.
- Upstream acceptance handoff:
  `docs/coordination/epcsaft_reactive_vle_upstream_handoff.md`.

## Submission fallback

The fallback is not a failure artifact. It is the bounded, reviewer-revised
paper comparing Henry-law and ePC-SAFT fugacity closure under the documented
fixed chemistry. It must retain transparent parameter limitations, sensitivity
discussion, numerical diagnostics, and validation reconciliation. The archived
activity-rebased nine-species sweep remains numerical feasibility evidence and
must not be presented as predictive chemistry.
