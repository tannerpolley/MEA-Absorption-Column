# Boundary-value solution methods (Section 4.3)

The four adapters are implemented and verified on an independent analytic DAE.
The corrected 3C full-film interface has also passed recovery and the original
algebraic equations. The corrected public two-node trapezoidal attempt in
`results/conserved_public_physical_20260915T204019Z/attempt.json` returned a
numerically and physically accepted full-column candidate in 472.343 s (14
IPOPT iterations, scaled residual infinity norm 6.371460913581593e-08, zero
bound violation). All five native-grid and film-quadrature checks pass the
fixed 1e-7 physical criterion. This is a first coarse result; refinement and a
matched-accuracy timing comparison remain required, and no manuscript claim or
physical method ranking follows from this attempt.

The first public three-node axial attempt is retained in
`results/conserved_public_axial_n3_20260915/attempt.json`. It reached the fixed
20-iteration limit in 889.577 s with a finite, bounded candidate and no native
callback failure, but its scaled residual remained `8.449045858900252e-03`.
Numerical and physical acceptance are rejected, so it supplies no refinement
estimate and is not evidence of formulation infeasibility. The preceding
`results/trapezoidal_axial_n3_20260915/run.json` setup failure records that the
older comparison runner still targets its frozen candidate snapshot; no model
was assembled in that attempt.

The matched retained-profile rerun is preserved in
`results/conserved_public_axial_n3_seeded_20260915/attempt.json`. It admitted
the accepted two-node profile with matching model, inputs, Engine identity,
scales, and non-node solver settings, then interpolated its twelve physical states onto
the three-node grid. Under the unchanged 20-iteration budget, its scaled
solver residual decreased from `1.6129296196679564e+02` to
`1.64310785649288e+01`, so numerical and physical acceptance are rejected.
The interpolated seed was worse than the unseeded initialization under this
fixed budget; neither rejected attempt supplies refinement evidence or proves
formulation infeasibility. No further retry is justified without a new design
for algebraically consistent interior initialization or continuation.

This analysis set owns the numerical-method evidence. Imported physics retains the source hashes
in `results/candidate_snapshot.json`; the centered discretization and
`Conserved_Reduction.py` are this task's extensions. Historical seven-state
`Run_Model` solver dispatch uses different physics and is not a comparison path.

## Runtime and retained evidence

The current public attempt declares and expects Engine commit
`7b1e62f0483571f8a194b32441467bdb58b45ce7` and verifies installed wheel SHA-256
`91632d2812429cbd293aae70fe8d4efb00000efe2377a91546dd7374dca67ee4`.
Earlier records keep their actual source and wheel identities.
The repaired wheel changes native mixed-action precision, not model equations,
physical tolerances or root selection. Final integration and the repaired-wheel local recovery replay passed.
`results/consistent_3c_reduction_balanced_actions/run.json` retains the latter:
588.96 s fresh-process wall, 476.50 s CPU, one local value and one Jacobian
evaluation, unchanged state, scaled residual 6.8092e-14 and condition 4786.86.
These durations are diagnostic observations, not a matched-cost comparison.

Parameter SHA-256: `a9186c93759f2e2c02a6c913350ad06a244fff3f82503820c9962b3df8dd40d9`.
Reference SHA-256: `a24a6b3c8b506fc659fc1bbd8a470b55919ba93da23eea27ffdf882645706185`.
`results/evaluator_adoption.json` records the supplied stateless liquid evaluator:
loading anchor .25, maximum log-loading step .1, at most 32 steps.

| Retained record | Result and limit |
|---|---|
| `results/analytic_cse_sampling/` | Eight successful analytic adapter checks; no absorber evidence |
| `results/central_a1_charge_seed/run.json` | Rich-state A1 finite; no interface or A2 claim |
| `results/consistent_3c_interface_check/run.json` | Original interface residuals [0,0,1.5994e-14,0,0]; 86.24 s fresh-worker wall |
| `results/consistent_3c_matrix_verified.json` | Owner's actual-node directional check passed at 5.3033e-8 against 1e-5; scaled matrix condition 4786.86, linear residual 1.4211e-14 |
| `results/consistent_3c_reduction_check/run.json` | Direct local recovery passed with one value and one Jacobian evaluation; 349.21 s including setup and rechecks |
| `results/native_adapter_point*.json` | Historical, algebraically inconsistent verification points; not the corrected 3C local root |

The physical records in this table use the earlier charge-seed wheel. The
repaired-wheel replay is identified separately above. Matrix
conditioning depends on the recorded physical scales; it cannot be compared to
an earlier differently scaled condition number as a precision improvement.
The owner's two-node full-column attempt is retained unchanged in
`results/owner_3c_baseline_charge_seed.json`, with source provenance beside it.
It reached its 900 s limit before a reported IPOPT iteration or returned candidate.
The last checkpoint records 107 completed liquid A1 queries (825.74 s), seven
A2 queries (49.14 s), and 46 vapor queries (6.00 s), with one A1 call unfinished.
No native equilibrium failure was recorded. This is a computational-budget
failure, not evidence of infeasibility. No physical profile can be extracted.
A fixed-bulk state is not a full column.

The shared-loading evaluation delivery is recorded in
`results/shared_evaluation_adoption.json`; its copied owner verification is
`results/owner_shared_evaluation_verification.json`. Two actual node evaluations
used 8 liquid A1 queries instead of 14, with bitwise-identical outputs; the full
Jacobian also matched exactly and passed the original directional criterion.
The optional enhancement-reference branch is outside this primary comparison.
The coupled owner records an unresolved pure-water vapor Cp discrepancy, so
thermal predictions under the common reference remain provisional.

## Equations and adapters

The common `node(z,u)->(B,R,a)` has seven conserved balances and five algebraic
relations. Its twelve states are four apparent component flows, liquid and gas
temperatures, pressure, holdup, CO2 and water fluxes, total energy flux and
interface log-loading. Height increases upward from the gas inlet. Both phase
flow magnitudes use negative interphase-transfer sources. Total enthalpy already
includes transported enthalpy; adding reaction heat would change the model.

Inference from these verified functions: one shared, consistently initialized
local elimination can provide both shooting and adaptive collocation, without
reimplementing thermodynamics or film equations. Use the seven conserved
quantities `q = B(z,u)` as the differential state. At each requested `(z,q)`,
solve `B(z,u)-q = 0, a(z,u) = 0`, preserving bounds, branch selection and all
native physical certificates. Then evaluate `q' = R(z,u)`.

This requires a nonsingular, acceptably conditioned
`M = [B_u; a_u]` on the accepted trajectory. Implicit derivatives satisfy
`M u_q = [I; 0]` and `f_q = R_u u_q`; the inlet Jacobians use the same chain
rule. These derivatives use the existing native node Jacobian, including its
equilibrium A2 and vapor enthalpy second actions. This formulation avoids
differentiating a temperature-state mass-matrix inverse, which could require
additional derivative orders. It does require local enthalpy-to-temperature
inversion and consistent interface roots; neither is established by a single
unconstrained node Jacobian check. Evaluate regularity only at certified
algebraically consistent states. A local refusal remains a failed evaluation.

Conservative trapezoidal differences and centered differences solve all nodes
simultaneously with IPOPT. The centered scheme uses nonuniform three-point
interior and one-sided endpoint derivatives of B; boundary slots
`[(0,-1),(1,-1),(2,0),(3,0),(4,-1),(5,0),(6,0)]` replace one endpoint differential
row per balance. Every algebraic row and every raw endpoint defect is retained. An explicit
constraint Jacobian with common-expression sharing reduces repeated value work.
`results/solver_graph_callbacks.json` records the static two-node graph counts
before shared-loading callback adoption:
liquid value/tangent callback leaves fall from 42 to 28 per Jacobian evaluation;
derivative callback counts stay unchanged. This is not a measured speedup.
On the current interface (`results/k1_k2_148/solver_graph_callbacks.json`) the
value callback leaves in the constraint Jacobian fall from 90 to 60 with shared
expressions; the 30 derivative callback leaves are unchanged.
The solver also disables IPOPT's additional gradient-based NLP scaling because
the variables and equations already use the declared physical scales. The
installed-IPOPT zero-iteration check in `results/solver_startup_shared_evaluation.json`
records two initial Jacobian calls under the default and one with `none`.
Original bounds and residual admission remain unchanged; this is a startup-call
count result, not a measured column speedup. See the [IPOPT option](https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_nlp_scaling_method).

Local recovery uses bounded least squares with the exact native Jacobian.
Optimizer termination and original-equation admission have separate tolerances.
It uses a fixed supplied seed and caches only identical coordinates. Height is
ignored in the last-result cache only when CasADi proves the node structurally
independent of height. No thermodynamic seed is warmed across queries.

Shooting integrates conserved states and sensitivities with DOP853 and supplies
the exact endpoint chain rule to its boundary root. Adaptive SciPy collocation
receives both RHS and boundary Jacobians. Collocation differential residuals are
checked at nodes and midpoints. Shooting reconstructs the documented degree-seven
DOP853 dense polynomial on eight Chebyshev points, verifies it at four separate
points, then differentiates it analytically. This checks original conserved
residuals without finite-differencing the EOS. Polynomial and residual failures
remain failed attempts. See the official [IVP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
and [BVP interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html).

The first executable method checks use an independent analytic two-balance,
one-algebraic system: `u=(c,T,j)`, `B=(c,T²)`, `R=(-j,-2j)`, `a=j-c`, on
`0 <= z <= 1` with `c(0)=2`, `T(1)=3` and positive T. Its solution is
`c=2 exp(-z)`, `j=c`, `T²=9+4(exp(-z)-exp(-1))`. Thus `u_q` is diagonal
in its first two rows `(1,1/(2T))` and its last row is `(1,0)`, while the
reduced RHS Jacobian is `[[-1,0],[-2,0]]`. These closed expressions check
the conserved-coordinate chain rule and mixed-end boundary mapping without
using the implementation as its own expected answer. Refinement must recover
the analytic capture and temperature profile for each distinct discretization.
The singular case T=0, a deliberately inconsistent algebraic state, a rejected
local evaluation and an exhausted iteration budget must remain failures.

## Corrected 3C inputs

The user selected corrected published 2017 run **3C**: L = 7517 kg/h,
G = 2013 kg/h, dry CO2 = 0.093, dry O2 = 0.090 and dry N2 balance;
loading = 0.25 mol CO2/mol MEA; unloaded-solvent MEA mass fraction = 0.30;
gas inlet T = 316.75 K. Liquid inlet T = 318.15 K is **assumed**, not measured.
Water follows the documented inlet saturation assumption. The paper leaves
gas mass-flow moisture basis unspecified; retain the delivered convention and
resolved wet molar feeds. Top pressure = 109500 Pa and gas inlet pressure =
110900 Pa are distinct measurements. The inspected builder uses the latter
as its boundary; do not impose both or silently substitute top pressure.
The retained `input/case_3c.json` supplies height 6 m, diameter .64 m, packing,
resolved wet molar feeds and the explicit species-diffusivity closure.

The durable source reconciliation is `input/case_3c_source.md`: Morgan et al.
(2020), DOI `10.1016/j.apenergy.2020.114533`, Tables A1, A4, C2 and Table 6.
It was recovered from the repository's archived source review; this task did
not independently reacquire or re-read the paper. Legacy `C_cases_data.csv`
and the intermediate campaign CSV do not contain the selected gas composition.

## K1–K2 on the current interface (Engine #148, wheel `48a639e7…`)

Five predeclared attempts on case 3C through the public twelve-state path, nine-point
film quadrature, each ≤ 20 IPOPT iterations, tolerance 1e-7 and ≤ 1200 s, seeded by
the accepted two-node profile (`results/k1_k2_148/summary.json`, one `attempt.json`
per directory).

| Attempt | Iterations | Wall s | Scaled residual | K1 (≤ 1e-7) | Capture % |
|---|---|---|---|---|---|
| trapezoidal 2 (replay) | 18 | 299 | 2.6e-11 | accepted; largest 2.6e-11 (material) | 97.46 |
| trapezoidal 3 | 20 (limit) | 487 | 9.6e-3 | rejected | 97.30 |
| trapezoidal 5 | 20 (limit) | 827 | 1.4e-1 | rejected | 93.86 |
| central 3 | 10 | 261 | 2.2e-10 | rejected: material 1.4, energy 24 | 76.01 |
| central 5 | 20 (limit) | 806 | 6.1e-1 | rejected | 97.81 |

K1 passes on the two-node replay (material 2.6e-11, energy 1.0e-12, interface 5.1e-12,
boundary 5.0e-13, charge 2.4e-15). The 17-point quadrature control at that solution
changes the film integral by at most 1.0e-4 relative (criterion 1e-3). K2 capture
refinement has no pair of accepted refinements, so it is not established.

Diagnosis (`run_case.py diagnose`; `results/k1_k2_148/diagnosis_trapezoidal_n2.json`,
`diagnosis_trapezoidal_n3.json`). Every resolved node-Jacobian entry agrees with
Richardson-extrapolated centred differences to 4.6e-8 or better at all nodes of the
accepted two-node solution and of the stalled three-node iterate, so the Engine actions
and the CasADi chain are exact there. The three-node collocation Jacobian at the stall
is nearly singular (condition 4.9e9; smallest singular value 2.3e-7 against 1.2e-2
for the next), with the null vector in the interior vapor water flow and the
alternating interface water and heat fluxes, and full IPOPT steps no longer reduce
the residual: a fold of the discrete equations, consistent with the 2026-09-16 source
homotopy that turned back between multipliers 0.28 and 0.31. The five-node iterates
alternate node to node (liquid temperature 310, 353, 332, 357, 318 K). On the manifold
of the interface equations, the stiffest mode of dB/dz = R has eigenvalue −10.4 m⁻¹
(gas inlet) and −12.6 m⁻¹ (top) on the accepted two-node states and +14.6 m⁻¹ at the
hot interior node of the stalled iterate: a gas-side relaxation length of 7–10 cm.
The trapezoidal amplification (1 + λh/2)/(1 − λh/2) of that mode is −0.88 to −1.10
at h = 3 m and −0.77 to −1.20 at h = 1.5 m, an undamped sawtooth; a non-oscillating
trapezoidal mesh needs h ≤ 2/|λ| ≈ 0.14 m. The central scheme converges at three nodes
but does not conserve the native-grid invariants. The predeclared 2 → 3 → 5 ladder
therefore lies in the oscillatory range of this column's stiff gas-side relaxation;
completing K2 needs a new mesh or discretization design, which the frozen budget
does not include.

## Running and interpreting the study

Run one native attempt at a time. The runner retains a new directory per attempt,
its source identities, exact inputs, original residuals, native call counts,
solver candidate/iteration snapshots, measured wall/CPU/RSS and failures. The
600 s baseline and 3600 s initial campaign limits bound work; failure diagnosis
precedes multiplying an unsuccessful baseline. Refinements and comparisons are
authorized work, not a new review or permission gate.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=analyses/bvp_solution_methods/results/k1_k2_148
uv run --frozen python analyses/bvp_solution_methods/scripts/run_case.py attempt --method trapezoidal --nodes 2 --output $R/trapezoidal_n2
uv run --frozen python analyses/bvp_solution_methods/scripts/run_case.py attempt --method trapezoidal --nodes 3 --initial-profile $R/trapezoidal_n2/attempt.json --output $R/trapezoidal_n3
uv run --frozen python analyses/bvp_solution_methods/scripts/run_case.py film-control $R/trapezoidal_n2/attempt.json --output $R/film_control_17.json
uv run --frozen python analyses/bvp_solution_methods/scripts/run_case.py summarize $R/*/attempt.json --film-control $R/film_control_17.json --output $R/summary.json
```

`run_case.py` drives the public twelve-state path (`mea_absorption_column.column.run_column`
on the current Engine interface): one verified worker per attempt with its wall limit,
the case-owned scaling and interface initialization, and the K1 native-grid physical
certification. `--initial-profile` admits one accepted attempt with matching model,
inputs, Engine identity and solver settings and interpolates its twelve states; it
remains an uncertified numerical guess. The reduced shooting/collocation methods and
the earlier `--stage` probes used the retired Engine callbacks and are not migrated;
the public path refuses them with a typed capability refusal.
`input/initial_3c_newton_predictor.json` is a half-step Newton predictor derived
from the retained two-node Jacobian, not a column result. The undamped step
predicts negative outlet CO2 and is retained as rejected initialization evidence.
`make_initial_predictor.py` reproduces the bounded step and linearization check.

Global numerical success remains separate from physical certification. Original
native-grid material and energy invariants, bulk/film-quadrature charge,
interface and inlet residuals, bounds and capture all pass the fixed 1e-7
physical criterion. Between-node and refinement evidence is also required.
Temperature profile differences use common physical heights. Adaptive methods
export the native mesh plus dense residual-check states, retaining `solver_grid`
separately from profile samples. Reported thermal maxima and locations remain
sampled extrema; continuous-peak accuracy requires further sampling/refinement. The finest numerical reference needs its own
refinement evidence and does not supply an absolute error automatically.

Refine axial intervals for the two difference schemes with film resolution
fixed, then film quadrature with axial settings fixed. For shooting refine
IVP tolerance with a fixed absolute/relative tolerance ratio; sampling the same
IVP at more heights is not refinement. For adaptive collocation refine its BVP
tolerance. Do not average timings across different settings, starts or contexts.
At matched achieved capture/profile differences and physical residuals, repeat
fresh-process runs sequentially and retain failures alongside successful costs.

```bash
uv run python analyses/bvp_solution_methods/scripts/compare.py RUN_JSON MORE_RUN_JSON --reference REFERENCE_RUN_ID --output NEW_COMPARISON_DIRECTORY
uv run python analyses/bvp_solution_methods/figures/method_comparison/scripts/render_comparison.py NEW_COMPARISON_DIRECTORY/comparison.json --refinement axial --output NEW_FIGURE_DIRECTORY
```

The figure-owned renderer supports `axial`, `film`, `ode` and `bvp` refinement,
checks that other settings are fixed and emits exact plotted CSV plus SVG/PNG/PDF.
The cost view requires explicit `--capture-limit-pp` and `--temperature-limit-k`.
Filled/open points distinguish inside/outside achieved-difference limits;
failed or uncertified attempts remain crosses at measured durations. Timing
environments must match, and timeout budgets never substitute for measurements.
Manufactured method-check figures are temporary validation, not manuscript
results. The working non-executing notebook is `notebook.qmd`, with its
rendered HTML at `builds/notebook.html`. It reports the two-node result as
endpoint tables only: two nodes cannot resolve an interior profile or
continuous peak without overstating evidence.

## Retained attempt requirements

Each JSON file contains one attempt. `result` is the native conservative
solver result dictionary, including its unchanged `grid`, `profile`, defects,
algebraic/boundary residuals, iterations, acceptance and failure fields, or
null when no candidate returned. Method adapters must export the same physical
twelve-state order and original diagnostics. Preserve unsuccessful candidates;
do not replace them with the initial state.

The surrounding fields are:

- `run_id`, `method`, `settings`: unique attempt identity, distinct numerical
  scheme, and all numerical settings, initialization identity, mesh, film
  quadrature, interface-root accuracy, global tolerances and scales. Numerical
  method source identity is required as `method_source_sha256`, separate from
  shared physical equations.
- `problem`: complete resolved `physical_inputs` (including `height_m`),
  `engine_commit`, `wheel_sha256`, `parameters_sha256`, `reference_sha256`,
  `model_source_sha256` (physical/transport source-path to hash mapping),
  `coordinate = "height_m_from_gas_inlet"`, and `state_layout` exactly as
  declared in `compare.py::STATE_LAYOUT`. All available problem fields must match exactly
  across records. Setup failures with no problem identity remain visible with
  `problem_identity_available=false`; they are ineligible for physical metrics. The required `physical_inputs` are `height_m`, three-component
  `liquid_feed_mol_s`, four-component `vapor_feed_mol_s`,
  `liquid_temperature_k`, `vapor_temperature_k`, `bottom_pressure_pa`, `area_m2`,
  seven packing values in the builder order, `gas_mass_flow_basis`,
  `liquid_mass_flow_basis`, `humidity_assumption` and `case_id`. Additional
  closure inputs and provenance remain the exporter's responsibility;
  equality of incomplete metadata does not establish identical physics.
- `physical_certification`: separate `accepted` boolean or null, original
  dimensional/scaled residual evidence in `original_residuals`, and `criteria`.
  Both evidence fields must contain `material`, `energy`, `charge`, `interface`
  and `boundary`. These are supplied by the physical evaluator; the reducer preserves them and
  does not independently certify them. Native `result.accepted` alone is
  insufficient for reference selection or comparison eligibility.
- `measurements`: measured `wall_s`, `cpu_s`, `peak_rss_bytes`, optional
  `setup_wall_s`, `thermo_wall_s`, `film_wall_s`, `global_wall_s`, and native
  counts/other diagnostics. Missing measurements stay null or absent.
  `context` carries hardware, software, thread settings and exact timing scope
  under `hardware`, `software`, `threads`, `scope`. An unknown execution
  environment suppresses timing statistics. Identical declared method, settings,
  and execution environment are grouped;
  no averaging across meshes, film settings or starts. Actual wall time and
  timeout budget (`limit_seconds`) are different fields.
- `execution_status` is required: `completed`, `failed`, `timeout`,
  `interrupted` or `not_run`. Only completed attempts can supply successful
  timings. Retain `termination`, `failure` and `limit_seconds` where applicable. Every record,
  including all residual arrays, rejected profiles, native counters and
  diagnostic fields, survives unchanged in `comparison.json`.

The command writes `comparison.csv`, `comparison.json` and `profiles.csv` to a
new directory and refuses overwrite. It records input-file hashes. The
temperature CSV uses the union of all eligible exported physical grids.
Piecewise-linear differences attain their sup norm on this union, but are
not the native adaptive-collocation spline error or a continuous ODE residual.
Peak first/last heights delimit all maximizing exported nodes; equal maxima
can be separated by lower temperatures and do not imply an entire plateau.
Native dense profile extrema and original between-node residuals must be
retained separately for final figures. Rejected/uncertified profiles remain in
JSON but do not contribute derived solution metrics or successful timings.

The output reports **differences from the explicitly selected reference**,
not automatic refinement error estimates or a matched-accuracy ranking. For
axial refinement select the next finer grid with film settings fixed; for film
refinement select the next finer quadrature with the axial grid fixed. Validate
those pairs from the complete settings before manuscript use. Repeated-run
statistics are min/median/max of measured eligible samples, alongside total
attempt and ineligible counts. Null data never become zero. Timing components
are summarized separately and never added or subtracted into an invented
exclusive breakdown.
