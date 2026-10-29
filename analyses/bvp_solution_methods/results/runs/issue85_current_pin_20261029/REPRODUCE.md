# Current-pin 3C repeats for issue 85

Production source HEAD: `91ced24149ce7bbed71b1cfba5c78e26bcd9c57d`.
Engine 1.0.0 commit: `f4566b3d0c64d05822b41fe363de47ab5cfa6a0e`.
Wheel SHA-256: `57bdf6c4277722fdf378107fa940fd321a4e9caeed8875165a27c318239dbd4a`.

Run from the task checkout with `bash scripts/setup_environment.sh` completed.
The commands below use the existing runner, with 600 s per-case wall limit
and an outer 660 s limit. Each was launched through `nice -n 10 agent-heavy`.
The selected one-thread BLAS setting reproduces the #21 numerical conditions;
it is not a machine-wide resource cap. No source homotopy or old-pin profile
was used. Each cosine run independently interpolates the same accepted fresh
uniform seed, using the existing row-wise interpolation and identity checks.

```bash
set -euo pipefail
run_root=analyses/bvp_solution_methods/results/runs/issue85_current_pin_20261029
nice -n 10 agent-heavy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 660s uv run --frozen python -u analyses/bvp_solution_methods/scripts/run_case.py attempt --method upwind --nodes 17 --wall-limit 600 --output "$run_root/seed_uniform_n17" 2>&1 | tee "$run_root/seed_uniform_n17.log"
nice -n 10 agent-heavy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 660s uv run --frozen python -u analyses/bvp_solution_methods/scripts/run_case.py attempt --method upwind --nodes 17 --end-clustering 1 --initial-profile "$run_root/seed_uniform_n17/attempt.json" --wall-limit 600 --output "$run_root/cosine_repeat_1" 2>&1 | tee "$run_root/cosine_repeat_1.log"
nice -n 10 agent-heavy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 660s uv run --frozen python -u analyses/bvp_solution_methods/scripts/run_case.py attempt --method upwind --nodes 17 --end-clustering 1 --initial-profile "$run_root/seed_uniform_n17/attempt.json" --wall-limit 600 --output "$run_root/cosine_repeat_2" 2>&1 | tee "$run_root/cosine_repeat_2.log"
uv run --frozen python analyses/bvp_solution_methods/scripts/compare_issue85_repeats.py
```

The runner refuses an existing output directory. For a new replay, choose a
new run label/root rather than overwriting these records, and point the
reduction at those records. Relocate recorded absolute input/seed paths when
replaying in another checkout; never alter scientific identity fields to
admit a seed. Each attempt retains resolved inputs, Engine and source
identity, initialization, native profiles and original-equation certificate.
`repeat_comparison.json` records their file hashes and scalar differences.

The current-pin capture is 89.41488041587351% in both cosine runs. All twelve
profile rows are bit-identical. The original-equation infinity norm is
1.7860243639219501e-12; the certificate passes its existing 1e-7 criteria.
The proposed new-input reproduction tolerance is 1e-6 percentage point,
max(10 times measured repeat difference, 1e-6). This is an explicit small
reordering allowance, not an estimate of numerical discretization error or
physical uncertainty. The old 89.41487679044005% result used Engine 464a989
and is not this baseline.

The model has stationary analytical amine/water film constraints and zero
charge flux, separate column water transport, solved-state hydraulic density
and the retained phase-specific mass conventions. The inlet liquid T=318.15 K
is an imputation; the wet-gas basis and thermal reference retain their stated
limits. A repeated accepted certificate is numerical reproduction, not
experimental validation or mesh convergence. MEA loaded-density bias is about
+11%, lowering 3C holdup by about 12–16% (issue 83).

HSL, a collection of Fortran codes for large-scale scientific computation.
https://www.hsl.rl.ac.uk/
