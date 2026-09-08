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

The coupled twelve-state column experiment is preserved under
`archive/coupled-solver-2026-09-08`; its derivative/caloric interface is not yet
integrated with the seven-state runtime. The independent conservative solvers are
available now. Built-in eNRTL and MDEA column paths are not implemented by a menu
label. Their future implementation can use the same explicit problem selection.
See `analyses/research_options/README.md` and its rendered notebook for evidence,
compatibility and candidate studies.

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
```

These checks do not reproduce the manuscript results.
