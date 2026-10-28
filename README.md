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

Edit the TOML or JSON case, physical choices, output directory and numerical settings.
Then explicitly run it in the prepared uv environment:

```bash
uv sync --frozen --group test
uv run python -m mea_absorption_column.research analyses/research_options/options.toml --run
```

The direct script above provides runtime-free preview; module invocation imports
the package runtime. One immutable configuration owner resolves ordinary runs,
and `run_column` executes them. Existing output directories are refused. The
attempt retains resolved physical inputs, model and numerical selections,
wheel identity, native states, raw solver stages and failures. Model parameters
belong in `[model.parameters]`; numerical controls belong in
`[numerics.solver_settings]`. The example selects a Henry Case 3C calculation.

Python callers pass the same request mapping to `run_column`. Its returned attempt
is safe for `json.dumps(attempt, allow_nan=False)` and equals retained `attempt.json`:

```python
import json
from mea_absorption_column import run_column

request = json.load(open("case.json"))
attempt = run_column(request)
capture = attempt.get("observables", {}).get("capture")
failure = attempt["failure"]
```

The command accepts the same JSON file:

```bash
python3 src/mea_absorption_column/research.py case.json
uv run --no-sync python src/mea_absorption_column/research.py case.json --run > attempt.json
```

Execution prints one attempt JSON object to stdout; Python solver messages go
to stderr. Exit 0 means execution completed without a retained failure, exit 1
returns a failed, timed-out or interrupted attempt, and exit 2 returns a separate
error JSON for loading, invalid requests or overwrite refusal. Numerical, physical
and scientific acceptance remain separate fields; inspect them before using values.
This supplies a configured calculation for a caller without a graphical app or server.

| Preset | Thermodynamics / film | Methods |
|---|---|---|
| `seven_state_legacy` | Henry, neutral/ionic ePC-SAFT or nine-species reactive ePC-SAFT; enhancement or supplied frozen film | `single`, `scipy-bvp`, `finite` |
| `twelve_state_conserved` | Existing reactive ePC-SAFT with `equilibrium_manifold` or `enhancement_reference` film and identified physical inputs | `trapezoidal`, `central`, `upwind` |

Seven-state temperature and enthalpy representations retain distinct resolved
layouts. Coupled chemistry follows the selected thermodynamics. The old
seven-state ePC-SAFT adapters need interfaces absent from the current Engine pin;
the selected prepared runtime is checked before the outer solve, and exact
missing-interface failures are retained without substituting a model.

The twelve-state selection retains its existing fixed case policy, explicit
worker wall limit, exploratory mobility assumptions and caloric limitations.
Reduced shooting and collocation are disclosed as unavailable and refused before
outer assembly. Existing coarse numerical evidence does not establish a resolved
profile or physical accuracy. The separate `research.run_conserved` utility
accepts already constructed numerical equations; ordinary TOML runs use the two
column presets. Built-in eNRTL and MDEA columns remain unavailable.
See `analyses/research_options/README.md` and its notebook for selections,
retained evidence and limitations.

For a reproducible single case, the preview/run command accepts this
immutable configuration:

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
For a seven-state multistart, the configured wall limit applies separately to
each candidate worker; it is not an aggregate deadline for the parent call.

## Organization and ownership

- `src/mea_absorption_column/`: reusable column equations, adapters, methods and
  packaged input data.
- `analyses/`: study inputs, scripts, retained attempts and Quarto notebooks.
- `tests/`: focused behavior and independent analytic checks.
- `docs/scientific/latex/`: preserved submitted manuscript; no ongoing editorial work.
- `ePC-SAFT-project`: generic Engine equations, equilibrium and derivatives.
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

Focused configuration and method checks:

```bash
uv run --frozen pytest -q -p no:cacheprovider tests/test_research_options.py tests/test_research_model_dispatch.py tests/test_epcsaft_contract.py
uv run --frozen pytest -q -p no:cacheprovider tests/test_column_config.py tests/test_conserved_assembly.py
uv run python scripts/check_epcsaft_integration.py --mode final
```

These checks do not reproduce the manuscript results.
