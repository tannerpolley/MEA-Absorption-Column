# Greenfield migration consumer specification

Status: **twelve-state consumer contract drafted 2026-09-21 for Engine #91,
correction round 2 applied after the Sol Scientific Basis and Execution
lenses; application qualification pending; no production code, run, wheel or
real-repository change.** This draft records what the current absorber
actually consumes from the Engine, the physical cases and inputs it consumes
them at, the frozen falsifiers, budgets and proposed physical criteria for
#62, and the finite Engine gaps. The seven-state route keeps its retained
requirements in its own section; nothing is retired here.

Publication boundary: the checkpoint is based on GitHub `main`
`b93a3741ac947f5f045f18e6b211d2f1b20062e2`. That tree does not contain the
inspected twelve-state `column.py`, `BVP/Coupled_Column.py`,
`Thermodynamics/casadi_reactive.py` or `MEA_neutral_vapor` records. The consumer
contract below describes the selected `b7172296` source snapshot, not a runnable
twelve-state implementation on that `main`. Reconcile that source branch before
the dependent #62 implementation; this documentation checkpoint does not merge
or qualify its runtime. The shared reactive-bundle identities remain unchanged.

Inspected owners, read-only:

| Owner | Revision inspected |
|---|---|
| Real absorber checkout | `b7172296071d093c62a604809f888a3f3c744597`, branch `manage-casadi-film-workflow`, clean tree: `src/mea_absorption_column/column.py`, `BVP/Coupled_Column.py`, `BVP/ABS_Column.py`, `Thermodynamics/casadi_reactive.py`, `Thermodynamics/reactive_bundle.py`, `Transport/Reactive_Film.py`, `Transport/Enhancement_Factor.py`, `config/column.py`, `tests/test_conserved_assembly.py`, `analyses/bvp_solution_methods/` inputs and retained results, `analyses/reactive_film_evidence/README.md` |
| MEA-Thermodynamics | `main` `3b2071197e801401162f24ff7d0aa99518369c26`: `data/reference/MEA/film_chemistry_inputs/1/input.json`; `analyses/mea_parameter_bundle/results/selected-current-best-parameters.json`, `results/parameter-record-history.csv`, `results/calorimetry/thermal-reference-validation.json` and its script |
| Engine | `main` `f909bb21a93171c38465adba8777cd3b0bc6c53d`; qualified wheel SHA-256 `9f73aeb466c54eff80c89a8776c90d49cb9c9c2d130f0eb15bf0311c1f76e606`; `engine/src/epcsaft/equilibrium/`, `_core.pyi`, `engine/docs/equations.md` (`cp_ideal_polynomial`, `total_enthalpy`), `engine/docs/science/algorithms.md`, Engine #53 D7/D8 record |
| Sources checked by the parent in Zotero | Hilliard 2008 (parent `DNQUPRMT`, PDF `LB7LUETY`) Appendix G.2, printed p. 937 / PDF p. 994; Weiland 1997 (parent `66XLM3LX`, PDF `27V9SYLA`) Table 3, p. 1004 |
| This draft clone | `9b75bbe24beb666f11ba82334b6d28269c20eff6`; older than the real checkout |

## Selected consumer

The production route is `column.py` → preset `twelve_state_conserved` →
`BVP/Coupled_Column.py::build_coupled_column_functions`. Two film modes share
the same balances, callbacks and interface equations:

- `model.film_model = "equilibrium_manifold"` (default, `co2_model="reactive_film"`):
  liquid CO2 resistance is the trapezoidal quadrature of the nine-species
  equilibrium-manifold conductance from bulk to the signed interface log-loading.
  This mode consumes the nine-species log-activity loading tangent and its
  second solved-state actions (`capabilities.a2_equilibrium_actions =
  "required_on_outer_derivative"`, `column.py:571`).
- `model.film_model = "enhancement_reference"`: replaces only the liquid CO2
  relation with `E k_L (c_int − c_bulk)` using native bulk and interface
  concentrations and the retained Luo-type reference kinetics hard-coded in
  `Transport/Enhancement_Factor.py::enhancement_reference_expression`. It does not
  consume the loading tangent or equilibrium A2 and cannot qualify the manifold
  mode; it still consumes the caloric actions below.

Twelve states per node, `z` upward in m: apparent `Fl_CO2, Fl_H2O, Fv_CO2,
Fv_H2O` [mol/s]; `Tl, Tv` [K]; `P` [Pa]; raw holdup `h_L` [bed-volume
fraction]; `j_CO2, j_H2O` [mol/(m² s)]; total interphase energy flux `e`
[W/m²], positive gas to liquid; signed interface log-loading `λ`. Fixed by
feeds: `Fl_MEA, Fv_N2, Fv_O2`. Seven conserved rows `B = (four apparent
flows, H_L, H_V [W], P)`, sources `R = −a_e A (j_CO2, j_H2O, j_CO2, j_H2O,
e, e, ΔP/Δz)`; five algebraic rows: holdup residual and the four interface
equations of `Reactive_Film.py:21` (CO2 film resistance, CO2 gas film, water
gas film, energy flux `e = h_T (Tv − Tl) + j_CO2 h̄_CO2^V + j_H2O h̄_H2O^V`).
Seven inlet conditions (`Coupled_Column.py:102-105`): liquid CO2/H2O flows
and `Tl` at the top, vapor CO2/H2O flows, `Tv` and `P` at the bottom. Total
energy flux already contains transported enthalpy; no separate reaction heat
is added. The outer solver (`BVP/Methods/Casadi_Collocation.py`) uses the
exact node Jacobian (`ca.jacobian`) with a limited-memory Ipopt Hessian, so
the consumed derivative order is exactly one outer derivative of every node
output; no third-order action is consumed.

## Consumed parameter and reference records

All paths are under `src/mea_absorption_column/data/epcsaft_datasets/` in the
real checkout; `config/column.py:24-25, 626-632` select them by default.

| Record | Path | SHA-256 | Identity and status |
|---|---|---|---|
| Nine-species EOS parameters | `MEA_reactive_epcsaft_bundle/parameters.json` | `a9186c93759f2e2c02a6c913350ad06a244fff3f82503820c9962b3df8dd40d9` | `mea-co2-h2o-nine-species-estimation-candidate-v1`, document version 2; exploratory incumbent `mea-reactive-exploratory-incumbent-2026-09-03` |
| Reaction system R1–R5 | `MEA_reactive_epcsaft_bundle/reaction-system.json` | `810dfec15760cf74451df91743d6e63684cee93ddaf3e1ff4e42bf4a686afe29` | molality/infinite-dilution source basis with additive offsets |
| Liquid reference thermochemistry | `MEA_reactive_epcsaft_bundle/anchored-reference-thermochemistry.json` | `a24a6b3c8b506fc659fc1bbd8a470b55919ba93da23eea27ffdf882645706185` (fingerprint `sha256:6919acbc…0648`) | `mea-anchored-reaction-consistent-reference-thermochemistry-v2`, 353.15 K datum, 293.15–393.15 K |
| Bundle and adoption receipt | `MEA_reactive_epcsaft_bundle/bundle.json`, `adoption-receipt.json` | `19fe7554…fb05`, `bf178ed7…37e6` | `adopted_as_exploratory_incumbent; not final column validation` |
| Neutral four-gas parameters | `MEA_neutral_vapor/parameters.json` | `42b0e9e6363f9cab3440d138eb6c0b6b939a73bc1475d3d37ffa2f101cc8e89e` | `mea-neutral-vapor-candidate-2026-09-03`; unaccepted candidate, five unfitted zero binaries |
| Vapor reference thermochemistry | `MEA_neutral_vapor/reference-thermochemistry.json` | `72b944e0221233152c4cdc85652ad792c7158e5a319366742939273024b77347` (fingerprint `sha256:097ade6e…8658`) | `absorber-neutral-vapor-reference-2026-09-03`; CO2/water rows copied from the liquid reference, N2/O2 NIST Shomate |

Relation to the MEA #82 current candidate, checked 2026-09-21: MEA's
`analyses/mea_parameter_bundle/results/selected-current-best-parameters.json`
has SHA-256 `568f7a5f6379acebacea584d707d5a3222db1022a85a4092b52553248e48524d`.
Its content is identical to the absorber's `a9186c93…` document (same
`document_id` and version; 1583 flattened leaves, none differing); the byte
hashes differ only by the 2026-09-11 storage-only JSON compaction recorded in
`results/parameter-record-history.csv` (`00049473…` → `a9186c93…` on
2026-09-03 with the R2/R4/R5 shifts, then `a9186c93…` → `568f7a5f…` on
2026-09-11). MEA's `results/calorimetry/current-selected-reference-thermochemistry.json`
is byte-identical to the absorber's liquid reference (`a24a6b3c…`, fingerprint
`6919acbc…`). Neither repository holds an adopted packet: MEA's context
records "no active (accepted) MEA parameter set", and the absorber bundle is
labelled exploratory. No common adopted packet is implied by this identity.
#62 must consume the eventual #61-qualified candidate by its own identity, and
every retained absorber result above is tied to the hashes in this table.

## Representative physical cases, units, conventions and domain

### Case 3C anchor (`analyses/bvp_solution_methods/input/case_3c.json`)

| Quantity | Value | Basis and source |
|---|---|---|
| Packed height, area | 6.0 m; 0.3216990877 m² (0.64 m diameter), one bed, no intercooler | Morgan et al. 2020, Table A1 run 3C |
| Packing vector | `[a_p 250 m²/m³, ε 0.97, C_lp 0.203, C_vp 0.35, C_s 0.017, C_p0 0.292, C_h 0.119]` | application constants |
| Liquid feed (top) | `[CO2 2.4323205276, MEA 9.7292821105, H2O 76.9751925073]` mol/s; loading 0.25; 30 wt% MEA on the CO2-free basis; 7517 kg/h loaded liquid | reported loaded mass and unloaded `w_MEA`; molar masses 0.044009/0.061084/0.018015 kg/mol |
| Vapor feed (bottom) | `[CO2 1.6540435157, H2O 1.5624169375, N2 14.5306833586, O2 1.6006872733]` mol/s; wet `y = [0.08549, 0.08075, 0.75102, 0.08273]`; dry `y_CO2 = 0.093`, dry `y_O2 = 0.090`; 2013 kg/h | reported dry compositions; 2013 kg/h interpreted as **total wet** mass (`gas_mass_flow_basis: reported_total_wet; paper moisture basis unspecified`); water saturated at inlet `Tv/P` by the repository `vapor_pressure` correlation; molar masses 0.04401/0.01801528/0.02801/0.031999 kg/mol |
| Temperatures | `Tl` 318.15 K (**imputed**, not measured); `Tv` 316.75 K (43.6 °C measured) | Table C2 |
| Pressure | 110900 Pa at the gas inlet (boundary); 109500 Pa reported at the top (comparison only) | |
| Liquid branch policy | loading anchor 0.25 (CO2/MEA), max log-loading step 0.1, ≤ 32 steps | application continuation policy for the native liquid start |
| Species diffusivities [m²/s] | `D_CO2 = 3.14e-7 exp(−15230/(R T))`; MEA 8.8e-10; H2O 8.8e-10; MEAH⁺ 8.4e-10; MEACOO⁻ 6.8e-10; HCO3⁻ 6.8e-10; CO3²⁻ 6.8e-10; H3O⁺ 1.8439e-9; OH⁻ 1.8439e-9; pairs by harmonic mean | **assumptions**: half of the Polat 2023 30 wt% CO2 self-diffusion correlation (Melnikov 2019 loading trend); Jerng 2022 DOSY anchors for MEA/MEAH⁺/MEACOO⁻; the rest are labelled estimates with no species-resolved source (`analyses/reactive_film_evidence/scripts/evaluate_onsager_closure.py:33-60`) |
| Retained interface initialization | twelve-state point `[…, h_L 0.0527151, j_CO2 0.0183159, j_H2O 0.0149621, e −10940.41, λ 0.7745669]`; original algebraic residuals `[0, 0, 1.6e-14, 0, 0]` | `input/consistent_3c_interface.json`; interface loading 0.25 e^0.7746 = 0.54 is a **numerical start**, not a consumed final state |
| Retained accepted two-node profile | bottom node: `Fl_CO2 4.04747, Fl_H2O 76.0902, Tl 329.39 K, Tv 316.75 K, P 110900 Pa, λ 0.1682`; top node: `Fv_CO2 0.038893, Fv_H2O 2.44739, Tl 318.15, Tv 329.571 K, P 110155 Pa, λ 0.1830`; bulk loading 0.416 → 0.25; interface loading 0.492 → 0.300; 97.6 % capture on two nodes | `results/conserved_public_physical_20260915T204019Z/attempt.json`, wheel `91632d28…` (Engine `7b1e62f0`), all 1e-7 physical criteria met; **coarse, not a physical result** |
| Capture observation | 89.5 ± 1.2 % | observation only |

### Operating envelope and target domain

| Quantity | Case 3C | Ten one-bed NCCC cases (K18–K20, 1C–7C) | All 52 catalogued cases | Consumed domain |
|---|---|---|---|---|
| Inlet liquid / gas temperature | 318.15 / 316.75 K | 314.05–318.45 / 316.25–319.33 K | 313.6–321.7 / 314.2–320.8 K | Retained reactive profiles reach 348.5 K (peak bulge); **owner target 315–360 K** |
| Pressure | 110.9 kPa | 107.3–109.7 kPa (reported) | 106.4–110.2 kPa | ≈ 1.06–1.11 bar; column drop ≈ 0.7 kPa |
| Lean loading | 0.25 | 0.075–0.34 | 0.062–0.399 | Retained 3C final states: bulk ≤ 0.416, interface ≤ 0.492 |
| MEA mass fraction (CO2-free) | 0.30 | 0.27–0.33 | 0.271–0.329 | 30 wt% class only |
| Dry inlet `y_CO2` | 0.093 | 0.077–0.110 | 0.076–0.117 | |
| Liquid / gas mass flow | 7517 / 2013 kg/h | 3175–11790 / 1000–2700 kg/h | 3016–11790 / 1000–3000 kg/h | L/G (mass) 1.2–11.8 |

Source coverage of the consumed models against this domain:

- EOS parameter candidate (table above): declared fit range 293.15–393.15 K,
  exploratory, not adopted (#82/#61). R1–R3 source domains 273–498 K. R4 was
  fitted jointly to pressure/speciation rows over 293.15–393.15 K; the R5
  source correlation is qualified only to 323.15 K, so 323–360 K is an
  extrapolation that the owner accepted pragmatically (#82). Broad total-fit
  statistics over 293–393 K do not assess it. Before qualification, #82/#61
  must bound the extrapolation on the observables this column consumes at
  323–360 K: bulk CO2 fugacity, carbamate/bicarbonate speciation (which sets
  the loading tangent), and the liquid enthalpy slope with loading (heat of
  absorption). The retained 120 °C heat holdout under-predicts by 18.8 kJ/mol
  CO2 on average (MEA reaction-temperature-fit README), which is the nearest
  retained evidence on the high-temperature trend.
- Reference thermochemistry: 293.15–393.15 K polynomial domain; the water and
  MEA `Cp°` rows are liquid-fitted effective inputs, not ideal-gas data (own
  section below).
- Film-chemistry record (`film_chemistry_inputs/1/input.json`): common
  admission domain 293.15–323.15 K, MEA 1 and 5 M only, loading < 0.5 with no
  oxazolidone treatment. Its temperature and molarity limits govern the
  finite-rate kinetic inputs, which the equilibrium-manifold mode does not
  consume. The loading limit is a model-domain limit of the nine-species
  equilibrium chemistry and applies to consumed final states. On the retained
  3C result every final state is below it (bulk 0.416, interface 0.492); the
  0.54 initialization start is a transient numerical evaluation. The Engine
  does not reject a fit-interval exit; a **final** consumed state above 0.5,
  which higher-loading cases such as 6C/7C (lean 0.31–0.34) may produce at
  the rich end, needs an owner decision when it occurs. No extension of the
  domain is proposed here.
- Transport: estimated mobilities above; `D_CO2` bulk correlation and the
  liquid/gas transfer, holdup, area, pressure-drop, viscosity, surface-tension
  and conductivity correlations are application-owned and unchanged.

## Consumed quantity contract

Species orders are frozen. Reactive liquid (index: id, charge): 0
carbon-dioxide 0; 1 monoethanolamine 0; 2 water 0; 3
protonated-monoethanolamine +1; 4 carbamate-anion −1; 5 bicarbonate-anion −1; 6
carbonate-anion −2; 7 hydronium-cation +1; 8 hydroxide-anion −1. Reactions
R1–R5 in the MEA packet order with products positive (water autoionization,
CO2/bicarbonate, bicarbonate/carbonate, carbamate hydrolysis, MEAH⁺
dissociation). Vapor: 0 carbon-dioxide, 1 water, 2 nitrogen, 3 oxygen, all
neutral, no reactions. The liquid callback is evaluated at `(T, P, F_CO2,
F_MEA, F_H2O)` and the vapor callback at `(T, P, F_CO2, F_H2O, F_N2, F_O2)`; all
`F` are positive absolute apparent amounts (mol, numerically mol/s).

Current callback rows (`casadi_reactive.py:80-177`): 0–8 true amounts per
mole of apparent feed, 9–17 mu/RT, 18–26 fugacities [Pa], 27 molar density
[mol/m³], 28 total enthalpy [J] per mole of apparent feed. Vapor: 0–3
fugacities, 4 molar density, 5 total enthalpy per mole of feed. Of these the
column reads:

| Consumer (real line) | Quantity, species, units | Chart and orders actually consumed | Engine producer at `f909bb21` |
|---|---|---|---|
| `Coupled_Column.py:76` volume, mass density, `true_x` | nine true amounts `n_i` [mol]; liquid molar density `ρ_L` [mol/m³] | value + first action in `(T, P, n_CO2, n_MEA, n_H2O)` | `PhaseComponentAmount` × 9, `PhaseMolarDensity` |
| `:92` conserved liquid enthalpy flow | extensive `H_L = N_L h_L` [W] on the apparent feed | value + first action in `(T, P, n)` | `PhaseProperty(TotalEnthalpy)` × `PhaseAmount`; product rule in the application |
| `:194-197` film conductance, `quadrature_points` states per node | at loading `n_CO2 = F_CO2 e^{fλ}`: nine amounts (composition for the mobility weights), `ρ_L`, and the **nine-component loading tangent** `t_i = D ln a_i[v]`, `v = n_CO2 e_CO2` | tangent value = order-1 `PhaseLogActivity` action; its outer derivative = order-2 actions `D² ln a_i[v, e]`, `e ∈ {e_T, e_P, e_CO2, e_MEA, e_H2O}` plus the chain terms below | `PhaseLogActivity` × 9, orders 1 and 2 |
| `:209` CO2 driving force | `f_CO2` at the interface loading; `f_H2O` at the bulk [Pa] | value + first action | `PhaseLogActivity` components 0 and 2; caller forms `f_i = a_i R T ρ_0`, `ρ_0 = 1 mol/m³` |
| `:153, :179, :211` vapor caloric closure | `Cp_V = (∂H_V/∂T)_{P,n}/N_V` [J/(mol K)]; partial enthalpies `h̄_CO2^V, h̄_H2O^V = (∂H_V/∂n_i)_{T,P,n_j}` [J/mol] | value = first `TotalEnthalpy` actions; outer derivative = order-2 `TotalEnthalpy` actions `D²H[e_T, e]`, `D²H[e_i, e]` for `e ∈ {e_T, e_P, e_CO2, e_H2O, e_N2, e_O2}` | `PhaseProperty(TotalEnthalpy)` × `PhaseAmount`, orders 1 and 2 |
| `:76, :92, :209` vapor | `ρ_V` [mol/m³]; `f_CO2^V, f_H2O^V` [Pa]; extensive `H_V` [W] | value + first action in `(T, P, n_V)` | `PhaseMolarDensity`, `PhaseLogActivity` 0–1, `TotalEnthalpy` |
| Enhancement mode only (`:201-205`) | bulk and interface `c_i = ρ_L x_i` for CO2, MEA, H2O, MEAH⁺, MEACOO⁻ | value + first action | amounts and density above |

Explicit nonrequirements, unchanged from #53 D8: the nine mu/RT value rows,
the seven MEA/ionic fugacity rows, any absolute chemical potential or ionic
gauge, `d(mu/RT)/dT` alone, N2/O2 fugacities, entropy, active-parameter
actions, and any third-order action. `reactive_bundle.py:816-819` currently
refuses a state unless all nine mu and nine fugacity rows are certified; under
this contract that refusal narrows to the consumed rows.

Kinetics, precise finding: the film-chemistry record admits three finite
reaction directions (F1, F2, F3) and rejects all four coefficient records
(Putta 2016 `k_MEA,c`, `k_H2O,c`, `k_MEA,a`, `k_H2O,a`) for source unit
inconsistency or model-specific activity basis; F3 has no locally retained
coefficient source. The equilibrium-manifold mode consumes no finite-rate
coefficient. The enhancement mode consumes only the hard-coded Luo-type
`k2 = 2.003e4 exp(−4742/T) c_MEA + 4.147 exp(−3110/T) c_H2O` with the
concentration divisor 1.0454. Finite-rate chemistry is therefore not a
requirement of the selected consumer and is not declared missing.

Interface mismatch to be closed in #62 Build, demonstrated by inspection: the
real callbacks import `EquilibriumStateInputDerivatives`,
`EquilibriumStateInputAction`, `general_reactive_equilibrium_problem_from_mapping`,
`GeneralReactiveEquilibriumProblem`, `ChemicalEquilibriumProblem`,
`ReactivePhase`, `EosModel`, `FinitePhaseStart`, `AllComponents`,
`EquilibriumOutput`, `EquilibriumEnthalpy`, `ReferenceThermochemistry`,
`ComponentReferenceThermochemistry`, `IdealHeatCapacityPolynomial` and
`NonEvaluableTrial` (`reactive_bundle.py:593`, `casadi_reactive.py:249, 381,
463`). None of these names exists in `epcsaft` at `f909bb21`. The current
surface is `Mixture(parameters, thermochemistry=ThermochemistryRecord)`,
`equilibrium.Problem(phases=[Phase(kind="liquid"|"vapor")], T, P,
feed=Amounts({...}), reactions=[Reaction(stoichiometry, correlation=
ReactionLogPolynomial(...))])`, `compile_problem`, `solve_equilibrium`, and
`solved_state_actions(compiled, central, SolvedStateActionRequest(observables,
directions))` with `SolvedStateActionDirection(temperature_k, pressure_pa,
feed_amounts_mol, active_parameters)`. Values are the order-0 field of the same
results. The migration is a re-targeting of the two callback modules only; the
balances, mobility projection, interface equations, collocation and physical
certification are unchanged.

## Coordinate chain rules (application-owned, exact)

Directions are absolute feed increments in the nine-species vector (ions
zero). With `FeedKind.Amounts` there is no unit-feed normalization, so the
curvature blocks now carried in `casadi_reactive.py::_LoadingTangent.input_jacobian`
and `_VaporCaloric.input_jacobian` disappear; the outputs become extensive
directly (`H_L`, `H_V` in J for the mol feed, i.e. W for mol/s).

- Loading coordinate at callback level, inputs `u = (T, P, u_CO2, u_MEA,
  u_H2O)`, `v = u_CO2 e_CO2`: `t = D ln a[v]`; `∂t/∂u_CO2 = D² ln a[v, e_CO2] +
  D ln a[e_CO2]`; `∂t/∂u_j = D² ln a[v, e_j]` for MEA, H2O; `∂t/∂T = D² ln a[v,
  e_T]`; `∂t/∂P = D² ln a[v, e_P]`. CasADi supplies `u_CO2 = F_CO2 e^{fλ}`, so
  `∂u_CO2/∂F_CO2 = e^{fλ}` and `∂u_CO2/∂λ = f u_CO2`; along the path `dt/dλ =
  f (D² ln a[v, v] + D ln a[v])`.
- Fugacity: `f_i = a_i R T ρ_0`; `∂f_i/∂T = f_i (∂ ln a_i/∂T + 1/T)`;
  `∂f_i/∂u_j = f_i ∂ ln a_i/∂u_j`. Every composition or loading derivative of
  `ln a_i` at fixed `T, P` equals that of `mu_i/RT` because `g_i°(T)/RT` has
  no composition dependence (#53 D7/D8); the mixed loading–temperature action
  is likewise datum-free.
- Extensive and partial calorics: `H = N h`; `DH[e] = h DN[e] + N Dh[e]`;
  `h̄_i = ∂H/∂n_i`; `Cp = (∂H/∂T)_{P,n}/N`; second actions by the same product
  rule. Vapor `N_V = Σ F_V` is fixed by the feed; liquid `N_L = Σ n_i` changes
  with reaction and is a `PhaseAmount` action. At fixed `T, P` the solved
  liquid state is homogeneous of degree one in the **three apparent feed
  coordinates**, so `Σ_{j∈{CO2,MEA,H2O}} F_j ∂H_L/∂F_j = H_L`; the vapor
  identity is on its four feed coordinates. Partial derivatives with respect
  to the nine true-species amounts are not independent coordinates of this
  consumer.
- Conductance: `g(λ) = c^T M(x(λ), ρ(λ)) t(λ)`, `c = (1,0,0,0,1,1,1,0,0)`
  (conserved CO2), `M` the symmetric Onsager mobility with pair weights `ρ x_i
  x_j D_ij` projected to zero charge, MEA, water and total-flux modes
  (`Reactive_Film.py:147-191`); its outer derivative uses `∂x/∂u`, `∂ρ/∂u`
  (first actions) and `∂t/∂u` (second actions). Film integral `∫_0^λ g dλ'` by
  trapezoid on `quadrature_points` states; thickness `δ = D_CO2^{bulk}/k_L`.
- Energy flux: `e = h_T (Tv − Tl) + j_CO2 h̄_CO2^V + j_H2O h̄_H2O^V`, with
  `h_T` from `Transfer_Coefficients.py::heat_transfer_expression(P, k_g, k_t,
  Cp_V, ρ_V, D_v)` ∝ `Cp_V^{1/3}`.

## Input inventory

| Input | Status | Owner and source |
|---|---|---|
| Nine-species EOS parameters, pairs, association, Born/permittivity fixed inputs | present, **exploratory, unqualified**; content-identical to the MEA current candidate `568f7a5f…` (records table) | MEA-Thermodynamics; adoption #82/#61 |
| R1–R5 K(T): R1–R3 `a + b/T + c ln T` on the molality/infinite-dilution water basis with additive `standard_state_offset` (4.0165 for R2/R3, 8.0331 for R1 = `ln 55.508` × net change in molality-basis solute count); R4 `a + b/T` (fitted); R5 `−ln10 (a/T + b + cT)` | present; **conversion to the EOS standard state unqualified** (`source_standard_conversion.status = algebraic_identity_only`, provider activity correction never ran) | #82 defines, #84 supplies the native chain; the application's `compile_reaction_constants` will map to `ReactionLogPolynomial(a, b, c, d, T_ref)` on `EOS_STANDARD_STATE_ID` |
| Liquid reference thermochemistry: 353.15 K datum, degree-8 `(T − 353.15)` polynomials, CO2 NIST Shomate `Cp°`, water and MEA `Cp°` = Hilliard liquid Cp minus the EOS liquid residual, hydronium = water gauge, five reaction-enthalpy constraints | present; **water and MEA `Cp°` are not ideal-gas data** (liquid-fitted effective inputs); provisional | MEA-Thermodynamics calorimetry; ideal-gas records #82 |
| Vapor reference: CO2/water copied from the liquid reference; N2/O2 NIST Shomate | present; water row carries the same non-ideal-gas `Cp°` | absorber `MEA_neutral_vapor/reference-thermochemistry.json`; the application requires shared components to be identical (`reactive_bundle.py:71-74`) |
| Neutral four-gas parameters | present, **unaccepted candidate** (five unfitted zero binaries; N2 source 63–126 K, O2 fit range unreported) | absorber `MEA_neutral_vapor/parameters.json`; no additional active family (#85 nonrequirement) |
| Species mobilities and harmonic-mean pair closure | **assumed** (model assumption; not a validated nine-species mobility matrix) | application; retained sensitivity only |
| Finite-rate kinetics F1–F3 (Putta 2016) | **not consumed** by the equilibrium-manifold mode; admitted reaction directions, all four coefficient records rejected | MEA film-chemistry record; no requirement here |
| Reference enhancement kinetics | present, hard-coded, historical | application `Enhancement_Factor.py`; enhancement mode only |
| Bulk `D_CO2`, `D_MEA`, `D_ion` correlations; `k_L`, `k_g`, `h_T`, holdup, area, pressure drop, viscosity, surface tension, conductivity | present, application-owned empirical | application |
| Hydraulic (empirical) versus EOS true-species density | present: the twelve-state route uses the **EOS** `ρ_L` for volume, velocity and mass density; the empirical `density_expression` is not called by `Coupled_Column.py` | application; note for the seven-state route only |
| Feed conventions: dry compositions, saturation humidity, gas mass basis, imputed `Tl` | present, assumed as listed in the anchor table; owner selected `reported_total_wet` for 2013 kg/h on 2026-09-21, explicitly as an assumption | case record |
| Boundary conditions | present (seven inlet conditions) | `Coupled_Column.py:102-105` |
| Loaded-solution heat capacity | **source located, record not yet retained**: Hilliard 2008 Appendix G.2 experimental rows (used by C5 below); Weiland 1997 Table 3 is 25 °C only; Hilliard Table 13.6-1 (p. 494) is eNRTL prediction, not observation | #82 is the single future data owner |
| Heat-of-absorption comparison rows (Kim–Svendsen calorimetry, finite loading-interval semidifferential heats) | present in MEA calibration blocks (not held out); the exact loading pairs are defined by #82's heat observation basis | #82 |
| Loaded density/viscosity rows (Amundsen 2009) | present, admitted observations | MEA record |
| Ideal-gas `Cp°` records for water and MEA | **missing** as retained data records (MEA ideal-gas Cp not retained per the MEA validation record; NIST/JANAF water not retained as a record) | #82 |

## Water `Cp°`: input inadequacy, not a reference choice

Engine definition (`cp_ideal_polynomial`, `total_enthalpy`): `Cp°_i(T)` is
the physical ideal-gas heat capacity and is model content; the reference
enthalpy and entropy offsets are gauges. The absorber's records supply for
water (and MEA) a liquid-fitted effective `Cp°` = Hilliard 2008 liquid Cp minus
the ePC-SAFT liquid residual Cp at 101325 Pa (`anchors.water.cp_kind =
liquid_correlation_minus_eos_residual`), copied unchanged into the vapor
reference. That input is not ideal-gas data. CO2's row is NIST Shomate and
matches to 0.00 %.

| T [K] | anchored water `Cp°` [J/(mol K)] | NIST/JANAF ideal-gas water Cp | anchored CO2 `Cp°` vs NIST Shomate |
|---|---|---|---|
| 298.15 | 44.92 | 33.58 (+33.7 %) | 37.13 vs 37.13 (0.00 %) |
| 315.00 | 44.33 | 33.70 (+31.6 %) | 37.91 vs 37.91 |
| 345.00 | 43.50 | 33.90 (+28.3 %) | 39.21 vs 39.21 |
| 360.00 | 43.14 | 34.00 (+26.9 %) | 39.82 vs 39.82 |

Consumed consequence of the current input, from source data (no wheel run):
vapor mixture Cp high by **+2.8 % to +4.6 %** over the 3C column (`y_H2O`
0.081 to 0.131 × (44.3 − 33.7) J/(mol K) on ≈ 30 J/(mol K)); `h_T` high by
+0.9 % to +1.5 % (`Cp_V^{1/3}`); water vapor sensible enthalpy relative to the
353.15 K datum offset by ≈ −380 J/mol at 316.75 K and −50 J/mol at 348.5 K
(latent part ≈ 43 kJ/mol). The pure-water saturation latent heat is unaffected
by the ideal polynomial in the current same-polynomial construction; the MEA
record checks the EOS residual difference against steam tables at ≤ 1.34 %.

Consequence of the physically correct input: with the ideal-gas water `Cp°`
the EOS pure liquid water Cp is 11–15 % below Hilliard
(`water_ideal_gas_anchored_cp_error_percent` −15.0 % at 25 °C to −10.7 % at
120 °C in the MEA validation record). That is a **liquid residual-Cp
deficiency of the candidate model**, and in this column it lands on the
dominant term (liquid capacity rate 7410 W/K versus 608 W/K vapor at 3C).

The MEA ideal-gas Cp candidate is now located in the retained Zhang, Que and
Chen 2011 PDF (DOI 10.1016/j.fluid.2011.08.025; Zotero
`MMAGNPIH/M3QDCZM7`, Table 3, printed p. 68 / PDF p. 2): molar-SI
polynomial coefficients `[13.207, 0.28158, −0.0001513, 3.1287e-8]`,
283 < T < 1000 K. The paper attributes it to the Aspen databank; uncertainty
is unreported. It is a located physical ideal-gas model candidate, not an
adopted application payload or experimental Cp series. #82 owns its source
record and comparison with the effective liquid-fitted input.

Dispositions, separated by owner: #82 retains source ideal-gas `Cp°` records
for water and MEA; #84 owns the exact reference and reaction chains (gauges,
standard-state shifts); #61 owns the liquid residual deficiency and what an
accepted candidate may do about it. The liquid-fitted effective variant may
remain as a labelled empirical historical input for exploratory runs; it
cannot qualify generic or vapor calorics, and the vapor-Cp falsifier C3
below is expected to fail against it by the stated amount. That failure is
recorded, not waived. No absorber-level decision exists here.

## Frozen falsifiers and budgets for #62 (set before any run)

States: S1 = 3C lean bulk `(318.15 K, 110900 Pa, F_L = [2.4323205, 9.7292821,
76.9751925])`; S2 = accepted-profile bottom `(329.39 K, 110900 Pa, F_L =
[4.04747, 9.72928, 76.0902])`; S3 = target-edge probe `(360 K, 106400 Pa,
loading 0.45, same MEA/water)`, value-only, an evaluation failure is a
recorded domain limit. Loadings: at S1 `λ ∈ {−0.2, 0, 0.05, 0.4, 0.7746}`
(0.05 is the retained node-test value, 0.7746 the retained start); at S2
`λ ∈ {0, 0.168}`. Vapor V1 = 3C inlet `(316.75 K, 110900 Pa, F_V = [1.6540435,
1.5624169, 14.5306834, 1.6006873])`; V2 = accepted-profile top `(329.571 K,
110155 Pa, F_V = [0.038893, 2.44739, 14.5306834, 1.6006873])`.

Criteria reuse the retained Engine E5 evidence: `CRITERION = {5e-6, 5e-5,
5e-4}` relative for orders 1–3, structural zero `1e-12 × max(1, largest
term)` **only for proved identities**, negative controls that must fail their
own criterion. No stricter threshold is promised. Finite-difference references
use three-step ladders; a component is **resolved** when the two finest
Richardson estimates agree within one tenth of their magnitude; unresolved
components are reported, never passed by an absolute allowance, and a
resolved nonzero action set to zero must fail.

| ID | Check | Criterion |
|---|---|---|
| N1 state | At S1, S2: charge `Σ z_i n_i`, C/N moieties and mass equal the feed; every classified row residual; liquid branch identity; `f_i = a_i R T ρ_0` for CO2 and water against the state-level `x_i φ_i P` on the same root | structural zero for the balances (proved); row residual ≤ 1e-7 (case tolerance); fugacity identity ≤ 1e-10 relative (algebraic identity on one state) |
| N2 Gibbs–Duhem | At S1 (five loadings) and S2 (two loadings): `Σ_i x_i D ln a_i[e] = 0` at fixed `T, P` for `e ∈ {v, e_MEA, e_H2O}` | proved identity: target `1e-12 × max(1, Σ_i |x_i D ln a_i[e]|)`; the attainable defect is bounded by the central row residual propagated through the linearization, so report the achieved ratio with the batch `central_residual` and `central_floor`; a miss goes to #87 without loosening the rule. Negative control: dropping the six ionic terms must fail the same rule; record the ratio |
| N3 tangent ladder | At the same states: `D ln a_i[v]` versus centred differences of `ln a_i` along `λ`, `h ∈ {2e-2, 1e-2, 5e-3}` | resolved components: relative defect ≤ 5e-6; unresolved components reported; negative control: any resolved component zeroed must fail |
| N4 outer tangent actions | `D² ln a_i[v, e]` for the five `e` versus centred differences of the first action; `e_T` step 0.4/0.2/0.1 K, `e_P` 400/200/100 Pa, feed steps 1e-2/5e-3/2.5e-3 of the coordinate | resolved components: ≤ 5e-5 relative; chain control: omitting `D ln a[e_CO2]` from `∂t/∂u_CO2` must fail for `e = e_CO2`; record the ratio |
| N5 conductance | At each quadrature state: `M z = 0`, `M c_MEA = 0`, `M c_H2O = 0`, `M 1 = 0`; `t^T M t ≥ 0`; `g(λ) > 0` | projections are proved zeros: `1e-12 × max(1, ‖M‖_∞)`; signs exact |
| N6 node Jacobian | Retain `tests/test_conserved_assembly.py::test_full_native_node_jacobian_is_19_by_12_and_finite` at the 3C point and add S2's node: 19×12 exact versus centred difference along the retained direction, step 1e-3 | relative error ≤ 1e-5 on rows whose two-sided difference is resolved (retained criterion; last passed at 5.3e-8 on wheel `ee6c34b5…`) |
| C1 apparent-feed Euler | Liquid at S1, S2: `Σ_{j∈{CO2,MEA,H2O}} F_j ∂H_L/∂F_j = H_L`; vapor at V1, V2: `Σ_{i=1..4} F_i ∂H_V/∂F_i = H_V` | exact degree-one homogeneity in the apparent feed (E5 H3 evidence): relative defect ≤ 5e-6 against `max(|H|, Σ_j |F_j ∂H/∂F_j|)`; negative control: omitting one term must fail |
| C2 caloric ladders | Vapor: `∂H_V/∂T`, `∂H_V/∂F_i` versus centred differences of `H_V` (T steps 0.4/0.2/0.1 K; feed steps as N4); the second actions behind `Cp_V` and `h̄_i^V` versus centred differences of the first actions. Liquid: `∂H_L/∂T`, `∂H_L/∂F_j` versus centred differences of `H_L` | order 1 ≤ 5e-6, order 2 ≤ 5e-5, resolved components only; no tighter promise |
| C3 vapor Cp physical | `Cp_V` at V1, V2 versus `Σ y_i Cp°_i` (NIST Shomate/JANAF) plus the EOS residual | ≤ 2 %; **expected to fail by +2.8 % to +4.6 % against the current water record** (input gap #82) |
| C4 latent heat | Pure water, vapor and liquid at the same saturation state `(T, P_sat(T))`, `T ∈ {313.15, 329.4, 348.5} K`, versus IAPWS/NIST `Δh_vap` (the MEA record's `vaporization_rows` method, reused, not duplicated) | ≤ 2 % (MEA record: 1.34 % maximum). Separate diagnostic, no acceptance: mixture `h̄_H2O^V(T,P,y) − h̄_H2O^L(T,P,F)` at S1/V1 and S2/V2 reported against the pure value (no reference exists for the partial molar enthalpy of water in loaded MEA) |
| C5 liquid Cp, source-matched | Hilliard 2008 Appendix G.2 experimental rows at 7 mol MEA/kg water (`w_MEA` 0.2995 CO2-free): loading 0 at 45 °C = 3.7195; loading 0.358 at 45 °C = 3.3675 and at 80 °C = 3.4707 kJ/(kg solution K). Composition uses kg water; specific heat uses kg loaded solution. Engine quantity `(∂H_L/∂T)_{P,F} / Σ_j F_j M_j` uses that total loaded-solution mass | ≤ 3 % (the unloaded→loaded change is 9.5 %, so 3 % discriminates the ionic `Cp` content). The unloaded point is near-tautological (pure-component anchors) and tests excess Cp only; the two loaded points are the informative ones. Locator: Zotero parent `DNQUPRMT`, PDF `LB7LUETY`, printed p. 937 / PDF p. 994. Not Table 13.6-1 (eNRTL predictions); Weiland 1997 Table 3 (25 °C) is outside the column range |
| C6 heat of absorption | Kim–Svendsen positive heat release is a finite loading-interval semidifferential quantity: `q_release = [Δn_CO2 h_CO2,feed^ref(T) − (H_L,b − H_L,a)] / Δn_CO2`, converting J/mol to kJ/mol. Use #82's exact 353.15 K pair, α 0.047 → 0.090, target 90.904 kJ/mol CO2. The incoming CO2 ideal/reference enthalpy is the declared calorimeter convention used by `evaluate_direct_absorption_heat.py:589–593`; it is not an arbitrary equilibrium-vapor partial enthalpy | the #82/#61 paired-heat criterion; until its review is complete, the comparison remains diagnostic. Retain calibration RMSE 11.8 and accessed 120 °C comparison RMSE 31.2 kJ/mol CO2 as historical model-error context |
| K1 column | Retain `column.py::_equilibrium_physical_certification`: native-grid material/energy drift, film-quadrature charge, interface, boundary, bounds, capture direction | scaled residuals ≤ 1e-7 (retained) |
| K2 refinement | Trapezoidal 2 → 3 → 5 nodes and central 3 → 5 on 3C from the accepted two-node profile; one 17-point quadrature control at the two-node solution | capture change between successive accepted refinements ≤ 0.5 percentage points before any physical comparison; film-integral change 9 → 17 points ≤ 1e-3 relative (the film solver's own tolerance); the two-node 97.6 % is not compared |

Frozen budget (no campaign):

| Item | Frozen value |
|---|---|
| Node/loading/vapor matrix | S1 × 5 loadings, S2 × 2 loadings, S3 value-only, V1, V2 |
| Ladder evaluations | N3: 7 loadings × 6 = 42 liquid solves; N4: 7 × 5 directions × 6 = 210 liquid solves + 35 order-2 batches; C2: 2 vapor states × 6 directions × 6 = 72 vapor solves + liquid 2 × 4 × 6 = 48. At the retained costs (A1 ≈ 0.6 s, A2 ≈ 0.9 s, vapor ≈ 0.03 s), these ladders take about 214 s before auxiliary checks; retain the 600 s total node ceiling, including all auxiliary state/action work. Exhaustion is a reported numerical limit, not permission to omit a state or exceed the budget |
| Column attempts | five: trapezoidal 2 (replay), 3, 5; central 3, 5; each ≤ 20 Ipopt iterations (retained setting), tolerance 1e-7, ≤ 1200 s wall (investigator decision 2026-09-11); total ≤ 6000 s; no sixth attempt without a new initialization design (retained three-node attempts stalled at 8.4e-3 scaled residual) |
| Quadrature | 9 points (retained default); one 17-point control |
| Wheel | the one shared non-editable wheel identified by the parent; no build or install by the worker |

## Proposed physical acceptance criteria (for owner/reviewer freeze)

Justified from engineering use, retained data and baseline models; numerical
convergence (K1, K2) and thermodynamic consistency (N1–N5, C1–C2) are judged
separately.

| Observable | Proposed limit | Basis |
|---|---|---|
| 3C CO2 capture | within 5 percentage points of 89.5 % after K2 | observation ± 1.2 %; retained enhancement-factor campaign MAE 4.0 pp, film campaign 6.7 pp; conditional prediction without capture fitting |
| Liquid temperature taps (2017 3C) | RMSE ≤ 6 K; peak within 5 K | retained film-campaign tap RMSE 3.7–8.1 K (3C 4.18 K); liquid inlet temperature imputed |
| Bulk liquid density | Reuse #82/#61's 1.6 % column-use screen only at matched Amundsen (T, w, α) source states; report unmatched column-state densities as diagnostics | Amundsen's retained satisfactory column-use baseline supplies the engineering rationale; ±0.002 g/cm³ measurement uncertainty remains metadata, and the source pressure is unreported |
| Bulk CO2 fugacity at S1, S2 | **diagnostic only**: reported against the #82 cohort at matched (T, w, α) in #82's pressure metric; acceptance is #82/#61's | the exploratory candidate supports no absorber-level band |
| Water fugacity | **diagnostic only**: reported against `x_w P_sat(T)` from the repository correlation | Raoult reference has no electrolyte validation |
| Vapor Cp, latent heat, liquid Cp, heat of absorption | C3 2 % (expected fail against the current record), C4 2 %, C5 3 %, C6 per #82 | as tabulated |
| Film flux | no physical acceptance: mobilities are assumed; report sensitivity to the thickness multiplier and mobility scale only | reactive-film evidence README |

## Retained #62 requirements and their disposition

#62 explicitly retains enthalpy-state inversion chains, absorption-only
smooth-cap derivatives, nonzero gas-velocity area effects, promoted
vapor/dry-wet conventions and caloric-reference consistency. Selecting the
twelve-state consumer does not retire them; each is mapped here or left open.

| Retained requirement | Twelve-state consumer evidence | Disposition |
|---|---|---|
| Enthalpy-state inversion | The conserved rows carry `H_L, H_V`; the retained conserved-reduction design (`BVP/Methods/Conserved_Reduction.py`, `analyses/bvp_solution_methods/README.md`) solves `B(z,u) − q = 0, a(z,u) = 0`, which needs `∂(H_L,H_V)/∂(Tl,Tv)` nonsingular; `column.py` requires the capacity rates (7410 and 608 W/K at 3C) to be finite positive. The seven-state `thermal_state_mode='enthalpy'` (`ABS_Column.py:35, 64, 115, 354`) is not exercised by the twelve-state route | conserved-chart invertibility and C2 do not complete the separately required seven-state enthalpy-state Jacobian; that check remains required and open under #62 |
| Absorption-only smooth cap | `ABS_Column.py:298-299, 503`; the twelve-state interface equations are bidirectional and contain no cap | not consumed by the selected twelve-state route; remains a required open check under #62 |
| Nonzero gas-velocity area | `ABS_Column.py:229, 511`; twelve-state `interfacial_area_expression` uses the liquid velocity only | not consumed by the selected twelve-state route; remains a required open check under #62 |
| Vapor dry/wet conventions | Twelve-state consumes SI feeds fixed upstream, so no derivative through the convention exists. Owner selected `dry_saturated` with `reported_total_wet` for the first accepted Case 3C. The older seven-state `reported_dry_mass` run is a different historical input | wet mass is a settled explicit assumption, not a pending decision; retain the old dry-basis discrepancy without treating its results as the same case |
| Caloric-reference consistency | Water `Cp°` section | open: #82 ideal-gas records, #84 chains, #61 liquid residual deficiency; C3–C6 frozen |

## Engine and program gap map

| Gap demonstrated here | Owner |
|---|---|
| Callback modules target retired Engine names; re-target to `Problem/Phase/Amounts/Reaction(correlation)`, `compile_problem`, `solve_equilibrium`, `solved_state_actions` with `PhaseComponentAmount`, `PhaseMolarDensity`, `PhaseLogActivity`, `PhaseProperty(TotalEnthalpy)`, `PhaseAmount`; drop the unit-feed curvature blocks; narrow the certified-row refusal; consume the eventual #61 candidate by identity | #62 Build (application) |
| Ideal-gas `Cp°` records for water and MEA; K(T) source-basis definitions; Hilliard G.2 and Kim–Svendsen pair records; observable-specific bounding of the R4/R5 extrapolation at 323–360 K | #82 (inputs) |
| Source-basis to EOS-basis reaction conversion and thermal shift for R1–R5; reference gauges and standard-state chains | #84 |
| Liquid residual-Cp deficiency of the candidate (pure water 11–15 % low with the physical `Cp°`) and its treatment in an accepted candidate | #61 |
| Active parameter families | none consumed: the absorber requests no parameter action; explicit nonrequirement | #85 |
| Trace species in the tangent (H3O⁺, OH⁻, CO3²⁻ at 1e-6 to 1e-4 mole fraction) enter `g(λ)` through mobility weights `ρ x_i x_j D_ij`; the retained film evidence reports a constrained-Hessian condition ≈ 5e10; absolute feeds of ≈ 89 mol are nonunit; N2 misses are routed here | #87: hand S1, S2 and the λ path as trace/nonunit cases |
| Liquid pressure-root identity at 293–360 K, 1.06–1.11 bar, loaded 27–33 wt% MEA (`reactive_bundle.py:738` currently rechecks the native root) | #89: hand S1–S3 as regular one-root liquid cases |
| The application's loading-anchor continuation (anchor 0.25, ≤ 32 log steps of 0.1) exists because cold starts fail at rich loading; `continue_equilibrium` exists natively | #90: hand S1 → S2 and the λ path as initialization/continuation cases |
| Callback lifetime, typed unavailable-action failures, no finite-difference fallback | retained in the application (#62) |

## Settled decisions and conditional limit

The owner selected the current twelve-state wet-mass interpretation of the
2013 kg/h feed on 2026-09-21. The source does not settle that basis, so it
remains an assumption in every accepted Case 3C input record. The older dry
interpretation changes molar flow by about 8 % and is a separate lineage.
The seven-state enthalpy-state, absorption-only and gas-velocity-area checks
remain required under #62; no retirement or scope change was authorized.

- **Conditional: a final consumed film state above loading 0.5.** None occurs in
  the retained 3C result; if a higher-loading case produces one, the model
  domain limit needs an owner decision at that time.

## Seven-state route (retained requirements, not acceptance)

The earlier draft selected `analyses/nccc_validation/scripts/run_reactive_column.py`
Case 3C with `thermo_model=epcsaft_reactive_nine`, the seven-state
`[Fl_CO2, Fl_H2O, Fv_CO2, Fv_H2O, Hlf, Hvf, P]` collocation and the 7×7
`ReactiveColumnJacobian` (temperature state, bidirectional CO2 flux, zero
gas-velocity exponent, `dry_saturated` plus `reported_dry_mass`). Its retained
refined report (max RMS residual 0.043138, 91.5484 % capture) was never rerun
on a qualified wheel and is not #62 acceptance. Its open enthalpy-state,
absorption-only, gas-velocity-area and vapor-convention items are carried in
the disposition table above; the neutral CO2/MEA/H2O vapor state it used is
replaced in the twelve-state route by the four-gas
`FixedCompositionVaporCallback` (MEA not in the vapor support).
