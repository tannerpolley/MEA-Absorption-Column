# Research options

`notebook.qmd` is the research-options page in the root site at
[`../index.qmd`](../index.qmd). Render the registered site from the repository
root with `bash analyses/render.sh --to html`; execution is disabled.

From the repository root:

```bash
python3 src/mea_absorption_column/research.py options.toml --list-options
python3 src/mea_absorption_column/research.py analyses/research_options/options.toml
uv run python -m mea_absorption_column.research analyses/research_options/options.toml --run
```

The example selects one nine-species Case 3C calculation. Change case IDs,
thermodynamics, film or method in the file rather than reconstructing a long shell
command. It is not a reproduction claim or a required default. Each actual run
requires a new `output_dir`. `[cases]` selects existing case groups; `[settings]`
sets benchmark options; `[solver_settings]` carries the model and numerical options.
The run records the resolved choices and benchmark defaults.

## Seven-state options

- Thermodynamics: `ideal_henry`, `epcsaft_neutral`, `epcsaft_ionic`, `epcsaft_reactive_nine`.
- Film: `enhancement_factor` or `reactive_film_linearization`.
- Method: `single`, `scipy-bvp`, `finite`.
- Enhancement: `solver_settings.enhancement_type = "explicit"` or `"implicit"`.
- Reactive bundle: `solver_settings.reactive_dataset` is an explicitly selected
  directory with internally consistent bundle hashes, species and reaction inputs.
- Initialization: `reactive_loading_anchor`, solver initial profiles and existing
  seed options remain explicit numerical choices, not validation status.

A film linearization requires `[positions, conductances, bulk_fugacities]` in
`solver_settings.reactive_film_linearization`; positions span normalized `[0,1]`.
It is a frozen outer-iteration input, not an automatic coupled-film solution.
It cannot be used for staged beds until per-bed coordinates are supplied.
The native seven-state Jacobian currently supports explicit enhancement only.
Select `jacobian_mode = "numerical"` for other implemented closures instead of
reusing an incorrect analytic derivative. This choice is recorded, not hidden.
Implicit enhancement failure remains a failure; it never substitutes explicit enhancement.

## Conservative methods with supplied equations

Set `formulation = "conserved"`, `problem_factory = "my_study:build_problem"`
and a method from `trapezoidal`, `central`, `shooting`, `collocation`.
`thermo_model` and `film_model` are explicit labels interpreted by your builder;
they are not substitutes for implementation. The importable builder receives the
resolved configuration and returns the existing solver's keyword arguments:

- Trapezoidal/central: `solve_conservative_collocation` inputs, excluding `scheme`.
- Shooting/collocation: `solve_reduced_bvp` inputs, including a constructed
  `ConservedReduction`, excluding `method`.

`solver_settings` supplies additional numerical keyword arguments. Duplicate
arguments and signature mismatches are refused. Python callers can directly use
`research.run_conserved(problem, method, settings)` with the same mapping.
The result retains arrays and diagnostics; exceptions remain in `failure.json`.

The copied independent method implementations are tested with an analytic DAE.
The integrated `twelve_state_conserved` preset selects `reactive_epcsaft`,
`equilibrium_manifold` or `enhancement_reference`, native total enthalpy,
hydraulic pressure drop and trapezoidal, central or upwind CasADi/IPOPT.
`numerics.solver_settings.liquid_density_source` selects `solved_state` (default)
or `correlation`. Reduced shooting/collocation preset execution remains unavailable.
The earlier experiment is preserved at `archive/coupled-solver-2026-09-08`.
Built-in eNRTL/MDEA column selections remain unavailable.

The seven-state preset fixes empirical enthalpy, constant pressure and correlation
density. Its legacy mapping additionally accepts a supplied frozen film profile;
it refuses unconsumed energy, pressure and density selectors. Staged pumparound
intercoolers require explicit `thermal_state_mode = "temperature"` in the legacy
solver settings. The seven-state preset itself requires enthalpy states and cannot
represent that combination. Selected Engine fugacity errors refuse explicitly;
they cannot substitute ideal-Henry fugacities.

The notebook's October capability section retains the complete resolver product,
original requests, bounded attempts and failure causes. Reproduce from the root
with `nice -n 10 agent-heavy uv run --frozen python
analyses/research_options/scripts/check_capabilities.py`, using a fresh output
label in the script for an independent repeat.

Observed-capture agreement is diagnostic unless `success_capture_error_max_pct` is explicitly supplied. Failed solver termination is not accepted by default; the older `accept_low_residual_final_iterate` option requires an explicit request and is not a physical validation claim.
