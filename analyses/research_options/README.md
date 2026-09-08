# Research options

`notebook.qmd` is the research index. Render HTML with `bash render.sh notebook.qmd --to html`.
Execution is disabled; no model calculation occurs during preview or rendering.

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
The twelve-state MEA builder remains a separate experiment at
`archive/coupled-solver-2026-09-08`, together with its exact source and retained
checks. It has a different derivative/caloric interface and is not silently
substituted into the seven-state model. Built-in eNRTL/MDEA columns remain future
implementation, with MDEA groundwork preserved on `codex/mdea-support`.
