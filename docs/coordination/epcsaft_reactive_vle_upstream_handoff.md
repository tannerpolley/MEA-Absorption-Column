# ePC-SAFT reactive-VLE upstream acceptance handoff

Status: downstream v1 source contract frozen; upstream issues 30 and 31 published, implementation pending
Required for G0: accepted source-neutral implementation plans and frozen predictive-v2 MEA manifests by 2026-08-13
Required installed release candidate: 2026-08-21

## Requested scientific capability

The final endpoint is true reactive VLE. Given temperature and overall
MEA/H2O/CO2 composition or loading, the installed engine must predict
equilibrium pressure, vapor composition, and liquid speciation without using
the observed pressure or observed CO2 partial pressure as a prediction input.

The implementation must be general and source-neutral. MEA supplies reactions,
standard states, observations, and residual policy; Equilibrium supplies
certified values and exact total derivatives; Regression supplies residuals,
weights, bounds, multistart, optimization, and result policy.

## Required admission order

### 1. Certified homogeneous reactive observations

This family exposes a fixed-`T,P` reacting-liquid state as an ordered observable
block. It is the reduced-tracer capability and a necessary column-local state
calculation. It does not predict equilibrium pressure and must not be labeled
reactive VLE.

Required inputs:

- immutable `Parameters` and ordered `ActiveParameterSet`;
- temperature in kelvin and pressure in pascals;
- ordered species and reaction topology;
- total composition or source-defined balance specifications;
- source-to-provider standard-state transform;
- fixed phase identity and liquid model;
- continuation/multistart policy and explicit budgets.

Required outputs:

- ordered species amounts, mole fractions, and activities;
- neutral-CO2 liquid fugacity;
- requested scalar or linear-aggregate observables;
- exact total derivatives in `ActiveParameterSet` order;
- complete parameter, reaction, reference-state, phase, and artifact
  fingerprints.

Required certificates:

- material and elemental balances;
- electroneutrality;
- reaction affinity;
- pressure and EOS closure;
- positivity, packing, and EOS-domain validity;
- KKT stationarity and complementarity;
- reduced-Hessian local-minimum status;
- derivative availability, rank, and conditioning.

The existing `FEASIBLE_ONLY` MEA state is negative evidence because its reduced
Hessian has negative curvature. The new slice must use a distinct
continuation/multistart design and must return a certified stable state before
Regression evaluates it.

### 2. Certified reactive bubble/VLE observations

This family is the final predictive capability. It couples homogeneous reaction
equilibrium to an ordered liquid-vapor equilibrium boundary without taking
observed pressure as an input.

Required inputs:

- temperature and overall composition/loading;
- immutable liquid and vapor phase topology;
- the same reaction, standard-state, parameter, and active-parameter contracts
  used by the homogeneous family;
- fixed support and phase-role ordering;
- bounded search/continuation and branch-selection policy.

Required outputs:

- equilibrium pressure in pascals;
- ordered vapor composition;
- ordered liquid speciation;
- phase fugacity/potential residuals;
- exact total derivatives of pressure, vapor composition, and requested liquid
  observables in active-parameter order;
- stable row identities suitable for Regression.

Required certificates and failure policy:

- all homogeneous-reacting-state certificates;
- common pressure and component-potential closure;
- phase material balance and normalization;
- mechanical stability and ordered phase identity;
- distinctness from a coalesced one-phase state;
- full-rank and acceptably conditioned defining Jacobian;
- explicit nonuniqueness and branch-identity status;
- fail-closed boundary, support-change, coalescence, singular,
  ill-conditioned, search-exhaustion, and uncertified-state results.

No penalty value may convert a Non-Evaluable Trial into a physical prediction.
Regression consumes the ordered value/Jacobian block; it does not reimplement
reaction or phase equilibrium.

The promoted vapor phase contains neutral CO2, H2O, and MEA and excludes ions.
Final promotion uses EOS vapor fugacities; an ideal-gas vapor may appear only as
a separately quantified low-pressure approximation.

## First admitted MEA evidence

The downstream frozen application contract is
`integration/reactive_mea_application_contract.json`. It binds the clean
MEA-Thermodynamics commit
`ac5ff017870ecf2c7987cba39f243b0399b8f106` and its exact artifact hashes.

The first reduced tracer contains no more than two fitted coordinates and uses:

- Hilliard `vle_obs_0137`: `T = 313.15 K`, unloaded MEA mass fraction
  `0.30`, loading `0.466 mol CO2/mol MEA`, measured
  `p_CO2 = 574 Pa`, measured state pressure `7326.7 Pa`; and
- Böttinger `cheq_canon_00194`: the same temperature, MEA fraction, and
  loading, with measured `x_MEACOO- = 0.0502`. Its `7326.7 Pa` evaluation
  pressure is an application anchor, not a Böttinger measurement.

The two-row scaled sensitivity matrix must have rank `N` for every declared
start before optimization. Missing rank rejects the coordinate selection; it
does not authorize changing rows, weights, or parameters after inspection.

The historical 147/220 training/reserved split remains immutable evidence. If
Jou rows enter regression, MEA must publish a new campaign-blocked split and
hashes before the optimizer sees outcomes. Model selection uses blocked
cross-validation; after the model and parameter block are frozen, one final
refit may use all scientifically admissible rows. Cross-validated predictions
and all-data calibration residuals remain distinct evidence products.

## Model and parameter support

- Molecular dipole/quadrupole moments remain source-fixed discrete model input.
  Polar DD/QQ/DQ physics may be compared as a nested fixed model, but polar
  values are not silently promoted to fit coordinates.
- Born correction coefficients are discrete model definitions. The required
  SSM+DS identity is `(c_shell, c_dielectric) = (1, 1)`; the coefficients are
  not continuous fit parameters.
- A relative-permittivity formulation is selected outside the optimizer. Every
  fitted coefficient requires an exact advertised derivative and independent
  observable support.
- Induced association is explicit fixed topology, not a magic model switch.
  A generic `k_hb_ij` coordinate is admitted only with a source-defined
  combining rule, stable parameter identity/fingerprint, and exact chain-rule
  derivative. Otherwise the actual resolved association energy/volume identity
  is used.
- Simultaneous dispersive `k_ij` and resolved cross-association fitting is
  rejected unless preregistered rank and profile-identifiability evidence
  separates them.
- Reaction correlations remain fixed through the first EOS/mixture/ionic
  stages. They cannot absorb EOS model error during the reduced tracer.
- The first reaction-correlation release is physical R4 `A,B` in
  `ln K_4(T) = A + B/T`. Release `C` only after held-out improvement and `D`
  only after profile-likelihood or cross-validation support. Keep R1--R3 fixed
  and defer R5 until the preceding stages are qualified.
- The first promoted joint pressure/speciation domain is 313.15--353.15 K.
  Treat lower-temperature speciation as support and pressure rows above
  353.15 K as an explicit domain-extension challenge.
- MEA-specific Born, solvation, and permittivity coordinates remain fixed until
  direct dielectric or ion-activity evidence identifies them.

## Additional Engine work required

1. Implement and admit general issue 30: the fixed-topology homogeneous
   reactive observation block above.
2. Implement and admit dependent issue 31: the certified reactive bubble/VLE
   block above.
3. Add a general application-declared scalar-coordinate descriptor so a source
   application can supply physical reaction-correlation coordinates and exact
   chain rules without an MEA-specific Engine branch.
4. Make both observation families consumable by the native Regression/Ceres
   problem and result path, including typed Non-Evaluable Trials, row ledgers,
   multistart, rank, conditioning, and profile evidence.
5. Install the public source-reference transform required by the polar
   candidate before that model is eligible for promotion.
6. Preserve CppAD as the sole production derivative authority and verify exact
   total columns by centered re-solves.
7. Produce a clean, immutable wheel and installed capability receipt rather
   than rebuilding the current development wheel in place.

## Installed artifact and packaging

The current wheel at
`build/environment-wheel/epcsaft-0.2.0.dev0-cp313-cp313-linux_x86_64.whl`
was rebuilt in place. Its current SHA-256 differs from the absorber's locked
hash. Do not ask downstream projects to refresh locks against this mutable
filename.

The release candidate must have a new immutable identity and retain:

- monorepo commit and clean-tree evidence;
- wheel SHA-256 and installed module origin;
- Data packet and materialized-file hashes;
- equation, parameter, reaction, reference-state, and topology fingerprints;
- admitted capability fingerprint;
- full affected and slow numerical check results; and
- installed-wheel validation receipts for the homogeneous and reactive-VLE
  families.

## Downstream checks

The frozen contract structure and tracked source-artifact hashes pass. During
the active MEA-Thermodynamics task, source mode reports the required clean-tree
failure because that task owns modified and untracked predictive-v2 contract
work; the bound v1 artifacts remain unchanged. Re-run the following command
after the owning task finishes and freezes its work:

```text
python3.13 scripts/check_reactive_mea_application_contract.py \
  --mode source \
  --source-root /home/tnnrpolley21/Workspaces/Engineering/MEA-Thermodynamics
```

Final mode is intentionally fail closed until every upstream capability in the
contract is accepted and an installed artifact receipt is bound:

```text
python3.13 scripts/check_reactive_mea_application_contract.py \
  --mode final \
  --source-root /home/tnnrpolley21/Workspaces/Engineering/MEA-Thermodynamics
```
