# MEA absorption-column research

Explore how thermodynamic, film, transport and numerical choices affect absorber
capture, axial temperature, conservation and computational cost. The manuscript
revision has been submitted; new work belongs in `analyses/` and Quarto notebooks.

## Submitted revision

The clean remote branch `codex/fallback-manuscript` at `eef9dce` preserves the
submitted document, source, supporting results, reviewer records and reproduction
instructions (`REPRODUCE.md`). The exact PDF, letters, bibliography and source are
in that branch's `docs/submitted_revision/revision_submission_2026-09-04.zip`:

```bash
git show eef9dce:docs/submitted_revision/revision_submission_2026-09-04.zip > revision_submission_2026-09-04.zip
```

Its checksum list covers that archive commit only. Do not edit manuscript prose here.

## Select a research calculation

Preview options without importing the scientific runtime or running a calculation:

```bash
python3 src/mea_absorption_column/research.py --list-options
python3 src/mea_absorption_column/research.py analyses/research_options/options.toml
```

Set up a fresh checkout or worktree with one command. It downloads the private
Engine wheel named in `integration/epcsaft_contract.json` into the ignored
`build/engine-wheels/<tag>/` (the relative path pinned in `pyproject.toml`),
checks the release digest, tag target and SHA-256 against the values recorded in
that file, and runs
`uv sync --frozen --group test`. Rerunning it only re-verifies the hash:

```bash
bash scripts/setup_environment.sh
```

Keep the wheel in that ignored directory: Engine provenance reads it. The wheel
requires glibc 2.39 or newer and is for the owner's own use; do not add it to Git
or redistribute it.

Edit the TOML selections, case IDs, output directory and numerical settings.
Then explicitly run a new calculation in the uv environment:

```bash
uv sync --frozen --group test
uv run python -m mea_absorption_column.research analyses/research_options/options.toml --run
```

Existing output directories are refused. The resolved selection and benchmark
settings are saved beside each new result. Presets are starting points, not
scientific admission rules. An unknown or incompatible implemented option fails
explicitly; numerical completion and physical agreement remain separate questions.

| Formulation | Thermodynamics / film | Methods |
|---|---|---|
| `seven_state` | Henry, neutral/ionic ePC-SAFT, or nine-species reactive ePC-SAFT; explicit/implicit enhancement or a supplied frozen film linearization | `single`, `scipy-bvp`, `finite` |
| `conserved` | A user-supplied `problem_factory` builds compatible balances and boundary equations, interpreting the selected thermo/film labels | `trapezoidal`, `central`, `shooting`, `collocation` |

The immutable `twelve_state_conserved` configuration connects trapezoidal and
central methods to a bounded worker. Always supply `execution.wall_limit_s` for
an actual run. `prepare_conserved_column` provides graph preparation separately.
Set `model.film_model` to `equilibrium_manifold` (the default) or
`enhancement_reference`; unsupported finite-rate choices fail before assembly.
The first retained native attempt is a numerically accepted two-node
trapezoidal result. An independent final-tree capability review found no
remaining blocker in the bounded execution path; physical certification,
refinement, and notebook review remain open. The solver connection is
bounded to its retained coarse result and establishes no resolved profile or
physical-accuracy claim.
Reduced shooting and collocation remain explicit unavailable choices.
Their existing generic research interface remains available. The earlier experiment remains under
`archive/coupled-solver-2026-09-08`. Built-in eNRTL and MDEA column paths are not implemented by a menu
label. Their future implementation can use the same explicit problem selection.
See `analyses/research_options/README.md` and its rendered notebook for evidence,
compatibility and candidate studies.

For a reproducible single case, the same preview/run command also accepts an
immutable preset configuration:

```toml
preset = "seven_state_legacy"
[case]
source = "C_cases_data"
id = "3C"
[numerics]
method = "scipy-bvp"
[execution]
output_dir = "analyses/research_options/results/runs/my-3c"
process_isolation = true
wall_limit_s = 600
```

The attempt retains resolved SI inputs, configuration and source identities,
the selected and verified Engine wheel, native profiles, raw solver stages,
and every multistart candidate. Solver termination, numerical checks,
observation agreement and physical certification are distinct. A changed wheel
requires an explicitly prepared interpreter and its exact hash; no environment
is installed automatically. The conserved preset additionally requires an
identified physical-input JSON, such as
`analyses/bvp_solution_methods/input/case_3c.json`. Estimated mobilities and
provisional caloric references permit exploratory calculations, not a thermal
validation claim.

## Organization and ownership

- `src/mea_absorption_column/`: reusable column equations, adapters, methods and
  packaged input data.
- `analyses/`: study inputs, scripts, retained attempts and Quarto notebooks.
- `tests/`: focused behavior and independent analytic checks.
- `docs/scientific/latex/`: preserved submitted manuscript; no ongoing editorial work.
- Engine (`../ePC-SAFT`, GitHub `tannerpolley/ePC-SAFT`): generic equations, equilibrium and derivatives.
- `Amine-Thermodynamics`: parameter fitting and thermodynamic parameter adoption.

Use identified, non-editable Engine wheels. The pinned dependency is a convenient
starting environment, not an assertion that all experiments must use its inputs.
An intentional candidate wheel requires its exact identity and explicit dependency
selection. `scripts/check_epcsaft_integration.py --mode dev --self-only` inspects
research dependencies without enforcing the submitted wheel hash; stable/final
modes retain archive identity checks. Parameter bundles validate their own input
hashes, species, charges and units. Do not import a mutable sibling source tree.

## Notebook and validation workflow

```bash
python3 analyses/manuscript.py validate analyses
bash analyses/render.sh --to html
```

The registered analysis pages appear in the root Quarto website. Rendering is
explicitly non-executing. Run studies separately into new output folders, then
describe their observations, numerical checks, uncertainty and next questions in
the notebook. Promotion is a later investigator decision; a failed or preliminary
calculation may still be useful research evidence.

For an optional native PDF of one retained page, use the same root renderer
with that page and `--to pdf`; its layout settings remain with the source page:

```bash
bash analyses/render.sh bvp_solution_methods/notebook.qmd --to pdf
```

Engine 1.0.0 `main-f4566b3d` passes the final integration check and all 289 tests,
including both ideal-Henry 3C positive-flow pressure transform checks, at unchanged
test tolerances. The shared transform uses an affine interior with reciprocal
tails and rejects floating-point values or derivatives that cannot be represented
inside its bounds. Numerical verification is retained under
`analyses/bvp_derivative_trials/results/runs/issue81_transform_domain/`.
See `analyses/engine_wheel_adoption/notebook.qmd` and
`integration/epcsaft_adoption.json` for the wheel adoption evidence and approved
infinite-dilution reference updates.

Focused configuration and method checks:

```bash
uv run --frozen pytest -q -p no:cacheprovider tests/test_research_options.py tests/test_research_model_dispatch.py tests/test_epcsaft_contract.py
uv run --frozen pytest -q -p no:cacheprovider tests/test_column_config.py tests/test_conserved_assembly.py
uv run python scripts/check_epcsaft_integration.py --mode final
```

These checks do not reproduce the manuscript results.

### Declared MEA inputs for the twelve-state calculation

The equilibrium-manifold route accepts the identified MEA records in
`analyses/bvp_solution_methods/input/issue85/thermodynamics.json` and
`transport.json`. Select them with `dependencies.thermodynamics_input` and
`dependencies.transport_input` in the column request. The transport record
identifies the physical case and the retained 3C initialization policy.
For example, the existing case runner accepts:

```bash
nice -n 10 uv run --frozen python analyses/bvp_solution_methods/scripts/run_case.py attempt \
  --method upwind --nodes 17 --wall-limit 600 \
  --thermodynamics-input analyses/bvp_solution_methods/input/issue85/thermodynamics.json \
  --transport-input analyses/bvp_solution_methods/input/issue85/transport.json \
  --output .tmp_column_runs/mea-declared-seed
```

Without the two explicit declarations, the route retains its built-in MEA
inputs and does not read the analysis example files.

A cosine-grid replay supplies `--end-clustering 1` and `--initial-profile`
pointing to that accepted seed's `attempt.json`, with a new output directory.
The two records retain parameter/reaction/caloric hashes, explicit phase
supports, species/feed/component mappings, species diffusivities and selected
unchanged MEA property equations. Liquid formula masses and vapor hydraulic
masses keep their distinct retained conventions. Missing laws, changed hashes,
unsupported declarations and dependent stationary-film constraints refuse
before native evaluation. Invalid evaluated states, native evaluation/action
unavailability and outer-solve failures retain distinct causes.

This is MEA numerical reproduction and declaration-order verification only.
The selected 3C temperature/packing policy and stationary analytical amine/water
film approximation remain unchanged. Generalized enhancement, expression
parsing, seven-state input injection, variable-count blend verification and
qualified second-solvent prediction remain unavailable.
