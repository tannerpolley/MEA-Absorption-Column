# Mathematical two-amine full-column verification on the final source

Both 129-node, 129-point ordinary `run_column` cases pass with accepted profiles
and original-equation certificates. This is mathematical numerical verification
only, with no physical validation or real second-solvent prediction. Canonical
equations and limits are in issue #85 comments 6510746468 and 6511556341.

`study.json` combines the unchanged retained spatial/film refinement from
`../issue85_stage3_20261029/` with final-source original/permuted finest runs.
Its preservation comparison uses that retained original 31.454926062128543%
record, not a replacement. `profile_comparison.json` records quantity-specific
differences and certificate summaries. Each attempt retains its source identity,
configuration, original equations, profile, initialization and failure status.

## Numbers and commands

Scalar reference: bottom liquid C 0.2744661315335329 mol/s, top vapor C
0.1255338684664671 mol/s, capture 31.526980836472482%. Both finest captures are
31.454926062128543%; maximum endpoint/profile C error 0.0001321004196305764 mol/s,
capture error 0.0720547743439397 percentage point, maximum flux error
8.151646566222515e-7 mol/(m2 s). Limits remain 2e-4 mol/s, 0.1 percentage point
and 1e-5 mol/(m2 s). Certificates and domain checks pass in both orders.

For replay use a new root; never overwrite these retained attempts:

```bash
nice -n 10 timeout 660s uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_column.py NEW_ROOT --case 129 129
nice -n 10 timeout 660s uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_column.py NEW_ROOT --case 129 129 --permuted
nice -n 10 uv run --frozen python analyses/bvp_solution_methods/scripts/verify_manufactured_column.py NEW_ROOT --compare --previous analyses/bvp_solution_methods/results/runs/issue85_stage3_20261029
```

Worker limit 600 s, outer limit 660 s, tolerance 1e-9, max 40 iterations,
initialization, scales and common-subexpression elimination are unchanged.
Only the two authorized collocation node calls preserve function calls instead
of forcing inlining. Timings (original/permuted): construction 1.491052/1.455754 s,
solve 244.418200/243.724929 s, whole column 257.983597/256.463397 s.
`*_timing.jsonl` holds pass-through timing measurements: construction is entry
to collocation through return from nlpsol construction, solve is native CasADi
`t_wall_total`. The temporary sitecustomize observer changed no solver arguments,
solver objects or numerical outputs. Final finest cases and the suite ran
alongside one another; these are observed wall times, not a speed benchmark.
The earlier graph-only completed runs remain in `../issue85_stage3_graph_corrected_20261029/`.

`pytest.txt`: 330 passed, 1 existing invalid-logarithm warning, 537.19 s.
The full suite ran with `nice -n 10 uv run --frozen pytest -q`; environment
ISSUE85_RUN_ROOT selected `full_suite_mea/`, ISSUE85_DOMAIN_RUN_ROOT selected
`domain_failure/`, and the pass-through observer wrote `mea_timing.jsonl`.
`r2.txt`: 51 independent comparisons in each order; expected inlet capacities
45/30 W/K and nonzero off-solution thermal/water rows pass.
`film.txt` and `film/` retain the unchanged stage-2 verification.
`integration.txt` is the subsequent final Engine integration check, exit 0:
Engine 1.0.0 f4566b3d0c64d05822b41fe363de47ab5cfa6a0e, immutable wheel
SHA256 57bdf6c4277722fdf378107fa940fd321a4e9caeed8875165a27c318239dbd4a.
All above commands exit 0. Source identities in attempts apply before the
checkpoint-2 commit; no executable change followed these runs.

## Failed attempts and classification fix

The preceding study retains the 600 s finest permutation timeout, without a
profile or capture, and the 95 s graph-construction diagnostic plus 150 s
candidate probe. The candidate reached iterations but did not finish its
preselected probe limit. Those failures were not promoted into results.
Independent review localized construction, using cse=False only in a diagnostic
stopping before nlpsol: about 18/21 s permuted/original, identical reported graph
sizes 6,657 and 11.53 million nodes. Why permutation alone was slow remains
unestablished; cse stays enabled in every accepted final run.

The required checkpoint-1 fix is committed separately at c8dfe6f. Its new
previous-failure check drives an actual invalid zero-C feed through the assembled worker.
One shared check now preserves state_domain_failure with inputs and cause for
both value/action evaluations. Before/after failures and the 8-test focused
check are retained in `../issue85_stage3_domain_fix_20261029/`.
No further production changes, limits, existing tests, references or tolerances
were changed. Initial checkpoint-1 missing-variable failures are retained with
that checkpoint. The full suite now includes 83 added test lines, below 90.

The refreshed notebook records conditions, reference, refinement, limits, MEA
timing comparison and the distinction between extensive and molar enthalpy.
Delivered independent review and investigator promotion remain open.

HSL acknowledgement for MA57 evidence: HSL, a collection of Fortran codes for
large-scale scientific computation. https://www.hsl.rl.ac.uk/

## Addition accounting

Against fixed main 3221748, final executable additions are 680 and removals 177;
tests are 83 additions. The sum of separately measured commit additions is 687:
checkpoint 1 550, domain fix 15, checkpoint 2 122. Seven lines added at checkpoint
1 are replaced by the domain fix, explaining the difference; neither count
uses removals to increase an allowance. Both totals are below 1050. Checkpoint 2
is 122/200; tests are 83/90. The graph-only correction is 2/10 solver additions.

| File | Checkpoint 1 additions | Domain fix additions | Checkpoint 2 additions | Final additions vs main | Ceiling |
|:--|--:|--:|--:|--:|--:|
| verify_manufactured_column.py | 128 | 0 | 120 | 248 | 290 |
| verify_manufactured_film.py | 2 | 0 | 0 | 2 | 10 |
| Coupled_Column.py | 45 | 0 | 0 | 45 | 125 |
| Casadi_Collocation.py | 0 | 0 | 2 | 2 | 10 |
| manufactured_neutral.py | 129 | 10 | 0 | 133 | 220 |
| solvent_inputs.py | 73 | 0 | 0 | 73 | 180 |
| column.py | 132 | 5 | 0 | 136 | 175 |
| config/column.py | 41 | 0 | 0 | 41 | 75 |
| Executable total | 550 | 15 | 122 | 680 | 1050 |
| test_manufactured_column.py | 55 | 28 | 0 | 83 | 90 |

The domain fix separately removes 7 lines; checkpoint 2 removes 2 solver lines.
Per-file ceilings also hold using sums of each commit's additions (139 for the
mathematical phase and 137 for column.py). `accounting.json` retains the fixed-base
measurement; executable/test counts exclude declarations and retained data.
