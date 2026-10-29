# Two-amine film numerical verification

Question and frozen method: https://github.com/tannerpolley/MEA-Absorption-Column/issues/85#issuecomment-6509541282.
The owner returned independent Ready with no required revisions before Build.
The allowed clarification separates species-flux quadrature error (relative 1e-5)
from stationary-component and total-flux invariant residuals (absolute 1e-12 mol/(m2 s)).

Base: `04a9d399bc1c292a10df1fbcddce63ccb53e71cc`.
The mathematical phase has six neutral species and four components, no Engine mixture,
and no physical solvent parameter adoption. Inputs, units, exact target and limits
are in `../../../input/issue85/manufactured_film.json` and the notebook section
`variable-count-two-amine-film`. No MEA test, reference or tolerance was edited.

From the repository root, run:

```bash
bash scripts/setup_environment.sh
nice -n 10 uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_film.py analyses/bvp_solution_methods/results/runs/issue85_stage2_20261029
uv run --frozen python analyses/bvp_solution_methods/scripts/render_manufactured_film.py analyses/bvp_solution_methods/results/runs/issue85_stage2_20261029
ISSUE85_RUN_ROOT="$PWD/analyses/bvp_solution_methods/results/runs/issue85_stage2_20261029/mea_final" nice -n 10 agent-heavy uv run --frozen pytest -q
nice -n 10 uv run --frozen python scripts/check_epcsaft_integration.py --mode final
python3 analyses/manuscript.py refresh analyses analyses/bvp_solution_methods/notebook.qmd
```

Use a fresh output directory for replay: the unchanged MEA test creates its own
permuted-input directory and does not overwrite a completed replay. `agent-heavy`
on this machine is a direct command adapter, with no admission or resource limits.
Each MEA column solve has the unchanged 600 s wall limit in its request.

`manufactured_film.json` contains both actual declaration orders, 17 state vectors
per order, full projected mobility matrices, loading tangents and second loading
derivatives, conductances and their derivatives, all five fixed-quadrature solves,
species fluxes, maximum absolute reference errors and duplicate-row refusals.
It also records input/source SHA-256 and the pinned immutable Engine identity.
`quadrature_errors.csv` owns plotted values; rendering reads retained numbers only.
`mea/` contains fresh declared/permuted uniform seeds and cosine solves, mapped
native first/second actions, raw film differences and original-equation certificates.
The original MEA baseline is read without modification from the stage-1 retained run.

`pytest.txt`, `integration.txt` and `render.txt` retain the final command output.
The first integration probe found a disallowed direct Engine import in the new test;
the test-only import/guard was removed without changing any calculation. The first
suite reported that earlier import (322 passed, 1 failed); its output is retained
as `pytest_initial.txt`, with numerical records under `mea/`. The complete clean
rerun reports **323 passed, 1 warning in 553.46 s**, with fresh numerical records
under `mea_final/`. The existing warning is the deliberately invalid logarithm probe.
Final integration and notebook rendering both exit zero. `verification.json`
records command outcomes; `manufactured_film.json` source hashes match the final
tested files.

This is numerical verification of the manufactured neutral film and MEA reproduction,
not physical validation, uncertainty quantification or a full-column blend admission.
Full-column MEA count/property restrictions remain explicit. Independent delivered
review and promotion for manuscript use remain open.

MA57 evidence acknowledgement: HSL, a collection of Fortran codes for large-scale
scientific computation. https://www.hsl.rl.ac.uk/

Executable accounting against the fixed base (comments and blank lines included):

| File | Added | Removed |
|---|---:|---:|
| `Transport/Reactive_Film.py` | 31 | 0 |
| `BVP/Coupled_Column.py` | 4 | 16 |
| `scripts/verify_manufactured_film.py` (analysis) | 206 | 0 |
| `scripts/render_manufactured_film.py` (analysis) | 29 | 0 |
| **Executable total** | **270** | **16** |
| New `tests/test_manufactured_film.py` (separate) | 10 | 0 |

270 gross executable additions are below the 350-line ceiling. The separate
29-line renderer earns retention by producing the required notebook comparison
from retained numbers without running the model. Records, mathematical inputs,
notebook/navigation and figure are separate from executable accounting.
