# Mathematical full-column comparison, incomplete delivery

Accepted design: https://github.com/tannerpolley/MEA-Absorption-Column/issues/85#issuecomment-6510746468.
Checkpoint 1: 52eb1680a13f93c18cf091361e3cf83b6788f133.
Checkpoint 2 remains uncommitted because its finest full-column permutation
gate timed out. `study.json` marks `delivery_complete: false` and gives no
permutation capture difference. The six declared-order comparisons are useful
numerical results, not a substitute for the missing gate.

Each `n<N>_q<Q>/` directory has actual inputs, ordinary `run_column` attempt,
source/Engine identities and the independent scalar-reference comparison.
The last accepted declared case has capture 31.454926062128543%, capture error
0.0720547743439397 percentage point, and endpoint-flow error
0.0001321004196305764 mol/s; runtime 303.3053 s. `n129_q129_permuted/` has the
failed 600 s worker attempt, with accepted initialization but no completed
profile. Its complete error output is retained in `attempt/attempt.json`.
The `.log` files retain stdout of successful commands; worker stderr and causes
are inside attempt records. Main stdout/stderr from the failed assertion is
also retained through its attempt record and the session transcript.

For fresh replay, choose a new root directory. Cases were run in this order:
17/33, 33/33, 65/33, 129/33, 129/65, 129/129, then 129/129 permuted.

```bash
nice -n 10 timeout 150s uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_column.py NEW_ROOT --case 17 33
nice -n 10 timeout 660s uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_column.py NEW_ROOT --case 129 129
nice -n 10 timeout 660s uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_column.py NEW_ROOT --case 129 129 --permuted
```

The requests themselves set execution.wall_limit_s=120 for 17/33/65 nodes and
600 for 129 nodes; method upwind, tolerance 1e-9, maximum 40 iterations, uniform
grids, declared feed-seeded algebraic initialization and no homotopy/seed reuse.
The main study's `--compare` command cannot pass until permutation is available.
The retained incomplete summary was assembled from the six completed comparison
records and two scalar-reference evaluations at integration tolerances 1e-10 and
1e-12. Their capture difference passes 1e-7 percentage point. Rendering reads
that summary without executing a column or reference calculation:

```bash
nice -n 10 uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_column.py RESULTS_ROOT --render
nice -n 10 python3 analyses/manuscript.py refresh analyses analyses/bvp_solution_methods/notebook.qmd
```

`pytest.txt`: 329 passed, 1 existing invalid-logarithm warning in 576.97 s.
`full_suite_mea/` retains the unchanged MEA ordering/action/film comparisons.
`integration.txt`: final-mode integration passes, Engine f4566b3d immutable wheel.
`provenance.json` records the analysis-driver/canonical-input hashes and Engine
identity. `render.txt` records the notebook/root HTML refresh.

The 95 s disposable diagnostic trace locates the bottleneck at construction of
`column_constraint_jacobian` with common-subexpression elimination, before
nonlinear solver construction (`permutation_diagnosis_trace.txt`). A separate
150 s disposable candidate changed only the two `node.call` sites in a temporary
copy of `solve_conservative_collocation` from always_inline=True to False.
It reached nonlinear iterations after 5.052 s, but did not complete before its
preselected diagnostic cutoff (`permutation_candidate_trace.txt`). No candidate
profile or corrected production result exists. Retained production source was
never changed by either diagnostic. A second production integration pass needs
the owner's Design decision under the binding hard stop; unused line allowance
is not authority for it.

Earlier missing-variable failures during checkpoint-1 extraction are described
and retained with that checkpoint. No failed or incomplete run is physical
validation or evidence about a real second solvent. No runtime cap was raised.

HSL acknowledgement for MA57 evidence: HSL, a collection of Fortran codes for
large-scale scientific computation. https://www.hsl.rl.ac.uk/

The diagnosis and proposed bounded return to Design are posted and read back at
https://github.com/tannerpolley/MEA-Absorption-Column/issues/85#issuecomment-6511556341.
Current accounting against main 3221748 is **662 executable additions and 175
removals**, including **112 additional analysis-verifier lines after checkpoint
1**. Tests remain 55 added lines. All currently applicable per-file ceilings
pass. `accounting.json` retains the measurement. No production or test source
changed after checkpoint-1 commit 52eb168; checkpoint-2 data/source/notebook are
staged but uncommitted pending the new Design decision. The notebook and root
HTML render from retained numbers, and the prose/source-whitespace checks pass.

This file is the pre-correction evidence snapshot. The subsequent independently
authorized correction and final-source completed checks are retained separately
in `../issue85_stage3_final_20261029/`; the original failed attempt remains unchanged.
