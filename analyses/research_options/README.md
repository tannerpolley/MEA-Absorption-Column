# Research options

The ordinary absorber command resolves one immutable case, model and numerical
selection, then executes it through `run_column`. Preview the editable example
from the repository root without importing thermodynamics or a solver:

```bash
python3 src/mea_absorption_column/research.py --list-options
python3 src/mea_absorption_column/research.py analyses/research_options/options.toml
uv run python -m mea_absorption_column.research analyses/research_options/options.toml --run
```

The example selects a Henry Case 3C calculation with one bed and empirical
enthalpy states. Change the TOML or submit the same request as JSON. Each
new run requires a new `execution.output_dir`; existing directories are refused
unless the existing explicit resume policy is requested. The attempt retains
resolved inputs, configuration and wheel identities, native states, raw solver
stages, available observables and failures. Numerical completion does not
establish physical accuracy.

For a Python caller, use the same request mapping with the existing column call:

```python
import json
from mea_absorption_column import run_column

request = json.load(open("case.json"))
attempt = run_column(request)
encoded = json.dumps(attempt, allow_nan=False)
```

The returned attempt equals retained `attempt.json`. A Python request may omit
`execution.output_dir`; the command requires one to preserve its calculation.
Nonfinite diagnostic values retain their explicit `value_class` representation.
No arrays or data frames require a second serializer in the caller.

```bash
python3 src/mea_absorption_column/research.py case.json
uv run --no-sync python src/mea_absorption_column/research.py case.json --run > attempt.json
```

Preview returns the resolved selection and limitations without importing the
scientific runtime. Execution prints one complete attempt JSON to stdout and
redirects Python solver messages to stderr. Exit 0 means completed execution
without a retained failure. Exit 1 accompanies a failed, timed-out or interrupted
attempt. Exit 2 emits a separate `error` object with the exception `kind` and
`cause` for file loading, invalid requests and overwrite refusal; it invents no
attempt. Completion is distinct from numerical, physical and scientific acceptance.
Read the returned quantities and their units, native layout and failure before
using them. The current prepared Henry command has parseable stdout; arbitrary
native-library writes are not redirected by the Python stdout context.

For a seven-state multistart, process isolation and the configured wall limit are
applied by the existing worker launcher to each candidate. The parent coordinates
and selects candidates; its total elapsed time can exceed one candidate's limit.

- `[case]` declares a source and case identity, mass/molar basis, wet/dry gas
  basis, or an identified `physical_input_file`.
- `[model]` selects thermodynamics and the film closure. `[model.parameters]`
  carries physical overrides such as enhancement, transfer factors, gas-velocity
  area parameters, reactive dataset, intercooler settings and `staged_beds`.
- `[numerics]` selects the existing method. `[numerics.solver_settings]` carries
  mesh, tolerances, numerical derivatives, state representation and seeds.
- `[initialization]`, `[acceptance]`, `[execution]`, `[engine]` and
  `[dependencies]` retain their existing meanings. Duplicate physical declarations
  in solver settings and conflicting chemistry selections are refused.

## Seven-state column

Use `preset = "seven_state_legacy"`. Thermodynamic selections are `ideal_henry`,
`epcsaft_neutral`, `epcsaft_ionic` and `epcsaft_reactive_nine`. Coupled nine-species
chemistry follows the thermodynamic choice. Its old adapters require interfaces
absent from the current Engine wheel; selecting them retains an unavailable
calculation with the exact missing-interface cause before the outer solve.
A selection label does not establish runtime availability.

Methods are `single`, `scipy-bvp` and `finite`. Select `thermal_state_mode` as
`enthalpy` (the default) or `temperature` in numerical settings; the resolved
layout identifies that representation. The existing numerical owner handles
`raw`, `bounded_guarded_raw_state`, `positive_flow_pressure` and
`case_bounded_flow_pressure` transformations. The last uses the existing
one-bed substitution to `positive_flow_pressure`; the returned solver record
identifies the actual transformation.

Film choices are `enhancement_factor` and `reactive_film_linearization`.
Enhancement is `model.parameters.enhancement_type = "explicit"` or `"implicit"`.
A frozen film requires `model.parameters.reactive_film_linearization =
[positions, conductances, bulk_fugacities]` on normalized `[0,1]`. It is a supplied
outer-iteration profile; per-bed frozen-film coordinates remain unavailable.
`staged_beds` accepts `true`, `false` or `"auto"`; the existing case and numerical
owners decide actual staging. Explicit thermal representation is retained.
Native seven-state derivatives require the existing unstaged nine-species,
raw temperature-state, bidirectional, explicit-enhancement equations without
gas-velocity area tuning. Unsupported combinations retain the owner's refusal.

## Twelve-state column

Use `preset = "twelve_state_conserved"` with an identified physical-input JSON,
for example `analyses/bvp_solution_methods/input/case_3c.json`. This preserves the
existing fixed case policy; arbitrary cases and physical overrides are not
implemented by this selection. The film is `equilibrium_manifold` or
`enhancement_reference`. Available numerical methods are `trapezoidal`, `central`
and `upwind`; reduced `shooting` and `collocation` are disclosed as unavailable
and refused before outer assembly. Supply an explicit worker wall limit.
The coarse retained evidence, mobility assumptions and caloric limitations keep
this an exploratory calculation rather than a physical-validation claim.

`research.run_conserved(problem, method, settings)` remains a separate low-level
utility for already constructed numerical equations. Ordinary TOML runs use the
column selections above; they do not import an external `problem_factory`.
Benchmark campaigns keep their existing enumeration interface. Built-in eNRTL
and MDEA columns remain unavailable.

The working notebook is `notebook.qmd`, registered at [`../index.qmd`](../index.qmd).
Render without model execution using `bash analyses/render.sh --to html`.
