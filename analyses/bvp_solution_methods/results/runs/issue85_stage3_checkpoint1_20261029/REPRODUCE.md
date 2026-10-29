# Two-amine column admission and assembled equations

Design: https://github.com/tannerpolley/MEA-Absorption-Column/issues/85#issuecomment-6510746468.
The owner returned independent Ready and authorized two bounded checkpoints.
Base: `3221748d53dbdc92297327428dac078c60ebdbb5`. This is checkpoint 1.

The ordinary configuration, declared feed preparation and twelve-state assembly
accept the mathematical six-species, four-component, two-amine liquid and
three-species vapor. Mathematical phases have no native solve or Engine result.
They report mathematical values/actions and failures; the immutable Engine
identity describes the executable environment. The Engine-backed and undeclared
MEA equations, masses, properties and numerical policies are unchanged.

`r2_final/declared/prepared_check.json` and `r2_final/permuted/prepared_check.json`
retain 51 independent comparisons per order. These include the individual A/B
feeds and reporting, total-amine loading 13/30 mol/mol, extensive capacities
45/30 W/K, actual initialization scales, nonzero water/energy source rows,
heat residual -25 W/m2 at the off-solution state, and vapor molar-enthalpy first
and second actions. Values, expected numbers, SI comparison units as specified
in the design, errors, ordering and source identity are retained. The off-solution
state is an equation probe, not an accepted column solution.

`gates_final.txt`: **18 passed in 480.45 s**. These unchanged MEA/film checks and
new admission/equation/failure checks include fresh uniform seeds and cosine
solves. `mea_final/comparison.json` retains captures 89.41488041587351% and
89.41488041586858%, 306 mapped native comparisons (maximum scaled difference
2.690685453529136e-11), and maximum film difference 5.430458370518347e-13.
The four MEA solve times were 99.90/98.87 s declared seed/cosine and
105.89/96.77 s permuted seed/cosine. All original certificates pass.

`film/manufactured_film.json` retains the unchanged stage-2 independent
reference checks: 129-point flux 0.0008333330846335338 mol/(m2 s), relative
error 2.9843975946075574e-7 against 1/1200. `config_assembly.txt` retains
71 passing existing configuration/assembly tests in 14.53 s.
`seven_state_comparison.json` records the separate unchanged seven-state
capture 92.08186315973761%, zero reference difference, maximum RMS residual
0.22481082656001908 and runtime 3.53 s. Its formal numerical/physical acceptance
fields remain not evaluated, as in the reference; no new acceptance is claimed.
`integration.txt` records final-mode integration with Engine f4566b3d and its
pinned immutable wheel.

From the repository root (choose new output paths on replay):

```bash
bash scripts/setup_environment.sh
nice -n 10 uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_column.py <new-r2-directory>
nice -n 10 uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_film.py <new-film-directory>
ISSUE85_RUN_ROOT=<new-mea-directory> nice -n 10 uv run --frozen pytest -q tests/test_solvent_declarations.py tests/test_manufactured_film.py tests/test_manufactured_column.py
nice -n 10 uv run --frozen pytest -q tests/test_column_config.py tests/test_conserved_assembly.py
nice -n 10 uv run --frozen python scripts/check_epcsaft_integration.py --mode final
```

Failed development checks are not solver failures or physical evidence. The first
R2 probe stopped on a missing `settings` variable after extraction of the common
initializer. The first MEA gate (`gates.txt`, `mea/declared_seed/attempt.json`)
stopped after successful algebraic initialization, before the nonlinear solve,
on missing `homotopy_step` in its caller. Both missing assignments were corrected
in that extraction. No iteration budget, tolerance or reference changed. The
corrected gates use `mea_final/`; the failed attempt is preserved.

The full variable-count column study has not run in checkpoint 1. Numerical
verification of these mathematical equations is not physical validation or
real second-solvent prediction. The existing MEA density/holdup and empirical
area limitations remain. Delivered independent review remains required.

HSL acknowledgement for MA57 evidence: HSL, a collection of Fortran codes for
large-scale scientific computation. https://www.hsl.rl.ac.uk/

Final checkpoint-1 accounting against the fixed base: **550 gross executable
additions, 175 removals**, below 850. Per file (added/removed):
`verify_manufactured_column.py` 128/0; `verify_manufactured_film.py` 2/38;
`Coupled_Column.py` 45/28; `manufactured_neutral.py` 129/0;
`solvent_inputs.py` 73/0; `column.py` 132/82; `config/column.py` 41/27.
Tests add 55 lines separately, below 90. Every per-file ceiling passes.
`accounting.json` retains the measurement; source files pass whitespace checks.
The raw failed pytest output preserves its original trailing whitespace.
