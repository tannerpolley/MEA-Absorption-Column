# Stage 1: declared MEA input reproduction and permutation

This is numerical verification on the fixed twelve-state MEA model, not
solvent qualification, blend verification or experimental validation.
The engineering specification is issue 85, comment 6507122557; the owner
reported independent Ready for this stage only. The implementation base after
`git rebase main` is `16d5862cdfa52023b381e81da7ad96ce8b80a1be` (main at
`9a4f29c23f1cddd4893812a48d00696e9964a10a`). Earlier baseline preparation is
separate from this stage's executable-line accounting.

## Inputs and equations

The ordinary input pair is `analyses/bvp_solution_methods/input/issue85/`.
The permutation input pair and its referenced parameter, reaction, caloric,
gas and physical-case documents are retained in `permuted_inputs/` here.
Each attempt retains resolved declarations' content hashes, physical feeds,
Engine and source identity, initialized state, scales, full native profile,
solver result and original-equation certificate. The numerical reference is
`../issue85_current_pin_20261029/cosine_repeat_1/attempt.json`.

Both calculations use the same 6 m corrected published 2017 3C bed, direct
SI molar feeds and phase-specific masses, 17 cosine-upwind nodes, 9 film
quadrature points and 1e-7 nonlinear/certificate tolerance. Each begins with
a fresh uniform 17-node seed through the same declarations, then uses the
existing accepted-profile interpolation/admission for the cosine run. The
state layout remains the twelve original physical quantities. Input-order
permutations change the actual Engine species/feed orders; the component
rows, charges, calorics, reactions, supports and diffusivities follow them.
Stationary analytical amine/water film constraints and zero charge flux remain.

Engine 1.0.0 commit `f4566b3d0c64d05822b41fe363de47ab5cfa6a0e`, wheel SHA-256
`57bdf6c4277722fdf378107fa940fd321a4e9caeed8875165a27c318239dbd4a`, is unchanged.
Liquid hydraulics use formula masses, with the existing 1e-5 kg/mol allowance
against rounded Engine masses; vapor hydraulics and property-law mixing
masses retain their distinct conventions. No mass reconciliation, new
coefficient, solver, enhancement law or seven-state input injection is added.

## Replay

The full suite's result-level test owns the four bounded column solves and
calls `scripts/compare_solvent_declarations.py` for the numerical comparison.
Set a **new** output directory; attempts refuse to overwrite existing runs:

```bash
ISSUE85_RUN_ROOT="$PWD/analyses/bvp_solution_methods/results/runs/issue85_stage1_new_label" nice -n 10 uv run --frozen pytest -q
nice -n 10 uv run --frozen python scripts/check_epcsaft_integration.py --mode final
```

The tests give every column case a 600 s wall limit. All commands use
`nice -n 10` directly. No admission wrapper is required.

To repeat the native comparison alone without solving columns, use the
interpreter recorded in the attempts (`.venv/bin/python` for these records).
This preserves the existing strict runtime-identity admission; a different
Python executable alias is not silently substituted:

```bash
nice -n 10 timeout 600s uv run --frozen .venv/bin/python analyses/bvp_solution_methods/scripts/compare_solvent_declarations.py analyses/bvp_solution_methods/results/runs/issue85_stage1_delivery_20261029
```

The comparison refuses a different package source-tree hash. `comparison.json`
contains capture, profile differences, film/native maxima and raw native
permuted-minus-declared vectors at 306 phase/film-state comparisons. Its
layout names the raw quantities, observables, derivative directions and units;
derivative units divide the observable unit by each direction unit. Native
Engine amounts use the same one-second extensive basis as the column flows.
Native values and first/second input-direction actions of log activities and
molar enthalpy are checked; amount and density derivatives were absent from
this earlier comparison. The corrected run supersedes that coverage claim.
No finite-difference or unavailable-action substitution is used.

Capture tolerance is 1e-6 percentage point. Profile checks preserve the
accepted flow/temperature/pressure/holdup/flux tolerances. Film differences
use relative 1e-7 with absolute SI floor 1e-10. Native differences use the
explicit SI scale max(1, abs(reference value)) and criterion 1e-7, with raw
differences retained for small or cancelled quantities. Every candidate must
also pass the full original-equation certificate. Missing conductivity law
refusal is checked before any native mixture construction; an evaluated
out-of-domain temperature raises a distinct state failure.

## Seven-state comparison

`seven_state/attempt.json` replays the earlier
`analyses/research_options/results/runs/engine_ionic_3c_verified_20261029/`
configuration with current runtime/output paths only. Its 3C feed basis,
21 starting nodes, scipy-bvp tol=0.5 and bc_tol=0.001 are unchanged.
`seven_state_comparison.json` checks capture, solver termination and the
original residual criteria. The formal seven-state numerical acceptance
field remains `not_evaluated`, as in its reference; this check does not
promote that field to accepted. Its input basis is distinct from the
conserved case, so the two captures are not a controlled model comparison.

## Limits

Repeated/order-invariant numbers do not bound between-node error or establish
physical accuracy. The thermal reference, liquid-inlet temperature imputation
and wet-gas convention retain their recorded limitations. Solved-state liquid
density remains the default; its approximately +11% loaded-density bias lowers
3C holdup about 12–16%. Variable-count multi-amine numerical verification and
qualified second-solvent prediction are still unavailable. Delivered review
and any scientific promotion remain separate.

For MA57 use: HSL, a collection of Fortran codes for large-scale scientific
computation. https://www.hsl.rl.ac.uk/
