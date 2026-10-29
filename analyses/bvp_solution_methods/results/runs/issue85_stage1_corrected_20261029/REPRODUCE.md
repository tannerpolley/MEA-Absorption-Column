# Corrected stage-1 MEA numerical verification

These records supersede the action-coverage and input/failure-admission claims
in `../issue85_stage1_delivery_20261029/`. The earlier capture reproduction
remains valid. The scope is input-driven reproduction and ordering invariance
of the fixed twelve-state MEA equilibrium-manifold calculation, not blend
verification or physical validation.

The correction branch was rebased onto main `de8c36e`. Executable accounting
uses the rebased baseline-only commit
`cb253a0fb686ffb977bd0050db3beb771593197a`, so all stage-1 implementation
additions remain counted and the earlier 51-line baseline reduction script is
separate. Each attempt records the exact tested package source-tree hash and
the dirty patch against its launch head; the correction commit retains those
tested bytes.

## Physical and numerical basis

The inputs are `analyses/bvp_solution_methods/input/issue85/thermodynamics.json`
and `transport.json`, plus their identified native assets and physical case.
The case remains the corrected published 2017 3C, 6 m bed, liquid inlet
318.15 K (imputed), vapor inlet 316.75 K, bottom pressure 110900 Pa and wet-gas
basis. Direct mol/s feeds and distinct liquid formula, vapor native and legacy
property-law masses remain unchanged. Solved-state density is retained.

Both declared and permuted calculations use fresh uniform 17-node upwind
seeds, then 17-node cosine-upwind solves admitted from their respective seeds,
nine film quadrature points, nonlinear/certificate tolerance 1e-7 and a 600 s
wall limit per column solve. The reference remains
`../issue85_current_pin_20261029/cosine_repeat_1/attempt.json`.

The immutable Engine is 1.0.0, commit
`f4566b3d0c64d05822b41fe363de47ab5cfa6a0e`, wheel SHA-256
`57bdf6c4277722fdf378107fa940fd321a4e9caeed8875165a27c318239dbd4a`.

## Replay and retained quantities

Use a new output label because existing attempt directories refuse overwrite:

```bash
ISSUE85_RUN_ROOT="$PWD/analyses/bvp_solution_methods/results/runs/issue85_new_label" nice -n 10 uv run --frozen pytest -q
nice -n 10 uv run --frozen python scripts/check_epcsaft_integration.py --mode final
```

The full test run owns the four bounded solves and their expanded native
comparison. `pytest.log` and `integration.log` retain the command output.
The run reports 322 passed, 1 existing invalid-log warning in 601.86 s;
final Engine integration exits successfully on the pinned immutable wheel.
The permuted input directory consistently reorders native species, feeds,
reactions, calorics, component rows, charges, supports and diffusivities.
Each attempt retains numerical acceptance, original-equation certification,
native profiles, initialization, input hashes and native construction orders.

`comparison.json` contains capture, profile and film differences and raw native
permuted-minus-declared vectors at 306 phase/film-state comparisons. The native
action observables are every phase species' log activity and amount, molar
density and molar enthalpy. Each has its value and every first and second
T/P/feed-direction action compared after mapping by identity. This is a
specified observable set, not a claim about every Engine observable. The raw
column output comparison also includes CO2/water fugacities and total enthalpy.
Native amounts use the column's one-second extensive basis. Vector layouts
retain observable and direction units; action units divide the observable unit
by each direction unit.

Capture allowance is 1e-6 percentage point. Profile tolerances and the 1e-7
original-equation certificate are unchanged. Native differences use the SI
scale max(1, abs(reference entry)) with criterion 1e-7 and retain raw differences
for small/cancelled quantities. Film diagnostics also pass relative 1e-7 and
absolute SI allowance 1e-10. To recompute comparison vectors without solving,
use the recorded interpreter identity:

```bash
nice -n 10 timeout 600s uv run --frozen .venv/bin/python analyses/bvp_solution_methods/scripts/compare_solvent_declarations.py analyses/bvp_solution_methods/results/runs/issue85_stage1_corrected_20261029
```

## Refusal and failure verification

Eight declaration cases check missing conductivity, bad hash, unsupported
execution policy, an unknown nested domain field, dependent film rows,
duplicate analytical-component IDs, inconsistent quadrature and inconsistent
density selection. They refuse before assembly or native mixture construction.
Both seed and cosine solves must be numerically and physically accepted.

`state_domain_failure/` and `native_evaluation_failure/` retain deliberately
forced interface callback failures from the same worker entry point. They
check that CasADi wrapping preserves the originating class and diagnostic;
these are synthetic failure-class checks, not physical nonconvergence cases.
Real initialization/outer-solver failures retain their separate meaning.
Undeclared calculations use the prior built-in MEA basis and do not read these
analysis declarations implicitly. Seven-state equations and mass handling are
unchanged.

## Limits

Stationary analytical amine/water film constraints and zero charge flux remain.
No variable-count multi-amine verification, generalized enhancement or qualified
second-solvent prediction is supplied. Repeat/order invariance does not bound
discretization error or physical uncertainty. The provisional calorics, wet-gas
basis and liquid-inlet temperature imputation retain their recorded limits.
Solved-state liquid density reads approximately 11% high on loaded comparison
rows and lowers 3C holdup about 12–16%; these checks do not resolve that bias.
Delivered independent review remains separate from this correction evidence.

For MA57 use: HSL, a collection of Fortran codes for large-scale scientific
computation. https://www.hsl.rl.ac.uk/
