# MEA absorption-column research

Explore how thermodynamic, film, transport and numerical choices affect absorber
capture, axial temperature, conservation and computational cost. The manuscript
revision has been submitted; new work belongs in `analyses/` and Quarto notebooks.

## Submitted revision

The clean remote branch `codex/fallback-manuscript` at `eef9dce` preserves the
submitted document, source, supporting results and reproduction instructions.
Retrieve its `docs/submitted_revision/revision_submission_2026-09-04.zip` for the
exact PDF, letters, bibliography and source. Its checksum list is for that archive
commit, not for this evolving research checkout. Do not edit manuscript prose here.

## Select a research calculation

Preview options without importing the scientific runtime or running a calculation:

```bash
python3 src/mea_absorption_column/research.py --list-options
python3 src/mea_absorption_column/research.py analyses/research_options/options.toml
```

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
- `docs/latex/`: preserved submitted manuscript; no ongoing editorial work.
- `ePC-SAFT-project`: generic Engine equations, equilibrium and derivatives.
- `MEA-Thermodynamics`: parameter fitting and thermodynamic parameter adoption.

Use identified, non-editable Engine wheels. The pinned dependency is a convenient
starting environment, not an assertion that all experiments must use its inputs.
An intentional candidate wheel requires its exact identity and explicit dependency
selection. `scripts/check_epcsaft_integration.py --mode dev --self-only` inspects
research dependencies without enforcing the submitted wheel hash; stable/final
modes retain archive identity checks. Parameter bundles validate their own input
hashes, species, charges and units. Do not import a mutable sibling source tree.

## Notebook and validation workflow

```bash
cd analyses/research_options
bash render.sh notebook.qmd --to html
```

Rendering is explicitly non-executing. Run studies separately into new output
folders, then describe their observations, numerical checks, uncertainty and next
questions in the notebook. Promotion is a later investigator decision; a failed
or preliminary calculation may still be useful research evidence.

Focused configuration and method checks:

```bash
uv run --frozen pytest -q -p no:cacheprovider tests/test_research_options.py tests/test_research_model_dispatch.py tests/test_epcsaft_contract.py
uv run --frozen pytest -q -p no:cacheprovider tests/test_column_config.py tests/test_conserved_assembly.py
uv run python scripts/check_epcsaft_integration.py --mode final
```

These checks do not reproduce the manuscript results.
