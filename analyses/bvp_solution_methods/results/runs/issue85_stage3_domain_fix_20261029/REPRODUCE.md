# Mathematical value-evaluation domain failure

The owner relayed an independent Not supported finding for checkpoint 1 at
52eb168, and explicitly required this correction within the phase/column/test
ceilings and the 1050-line total. Other checkpoint-1 findings were Supported.

The new test passes an actual zero liquid CO2 flow into a production node from
the initialized state, through `benchmark_worker.main`. It does not inject a
typed action exception. Before correction, `before/worker.json` records
`outer_solve_failure` and no `evaluation_failure`, despite the CasADi domain
assertion. `initial.txt` retains the failing test.

After correction, `after/worker.json` records `state_domain_failure`, phase
`mathematical_liquid`, T=320 K, P=100000 Pa and an ID-keyed feed with C=0 mol/s,
plus the actual domain message. The new worker test verifies these values.
Mathematical value evaluation now uses the existing exact-action callback;
values, derivatives and capacity evaluation share one phase-owned domain check.
The generic graph assertion is removed. No message parsing, fake native result,
fallback value, changed bound or additional physics is introduced.

`focused.txt`: **8 passed in 3.70 s** (all mathematical-column checks and the
unchanged stage-2 film verification). This includes all 51 R2 comparisons in
both declaration orders. Full finest and shared-caller checks will be rerun
on the combined final source before checkpoint 2 is committed.

Correction accounting relative to checkpoint 1, excluding the separately
accepted graph/analysis work: `manufactured_neutral.py` 10 additions/6 removals;
`column.py` 5 additions/1 removal; one new test adds 28 lines, for 83 test lines
in total. Thus the fix is **15 gross executable additions, 7 removals**, with
28 test additions separately. Both executable files remain below their
220/175 ceilings. Different-base cumulative totals account for replacements
within already-added code; these stage counts must not simply be summed.

Run the single failure test or the focused group with fresh output paths:

```bash
ISSUE85_DOMAIN_RUN_ROOT=NEW_DIRECTORY nice -n 10 uv run --frozen pytest -q tests/test_manufactured_column.py::test_invalid_evaluated_feed_keeps_domain_failure_in_worker
nice -n 10 uv run --frozen pytest -q tests/test_manufactured_column.py tests/test_manufactured_film.py
```

These checks concern truthful numerical failure reporting, not physical
validation or real second-solvent prediction. Delivered review of the correction
remains open. The two-call graph-construction correction retains its separate
authority and unchanged numerical criteria.
