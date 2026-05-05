# Robust Convergence Status

This implementation adds a guarded convergence layer around the existing MEA absorber physics without modifying the external `ePC-SAFT` package. The goal is to prevent nonlinear solvers from crashing when Newton/collocation iterations try nonphysical states, while preserving explicit diagnostics for cases that still are not predictive-quality solves.

## Implemented Controls

- Bounded-state utilities in `mea_absorption_column.BVP.robust_core` clip nonpositive flows, nonpositive pressures, non-finite enthalpy flows, and record guard diagnostics.
- Temperature inversion now falls back to bounded least-squares between 250 K and 500 K if the unconstrained root solve leaves the physical range.
- The staged SciPy BVP path now supports a case-bounded flow/pressure transform. An experimental `thermal_state_mode=temperature` path solves staged beds directly in liquid/vapor temperature variables and computes enthalpy algebraically inside the column model.
- The chemical-equilibrium subsolve now uses a hybrid strategy: a fast unconstrained root is accepted only when it returns finite positive species concentrations with a small residual; otherwise a bounded least-squares fallback enforces positive concentrations. This prevents the enhancement-factor model from receiving negative ion/species concentrations during Newton trials.
- Liquid holdup is smoothly mapped inside the packing void-fraction domain instead of being hard-clipped at the physical bound.
- The continuation ladder now records optional seed/ramp stages without blocking the required full-intercooler stage. It can reuse a failed seed profile when capture is close, while still reporting that seed stage as failed.
- Intercooler reset strength is now explicit (`intercooler_strength` / `--intercooler-strength`), enabling no-reset, weak-reset, and full-reset continuation steps.
- Capture is reported against the known inlet vapor CO2 feed. Failed BVP iterates can miss the inlet boundary; using the simulated inlet as the denominator made failed branches report meaningless very large capture percentages.
- Fugacity evaluation has a guarded path for both `ideal_henry` and `epcsaft_neutral`; invalid ePC-SAFT states are recorded and converted to finite fallback fugacities instead of uncaught exceptions.
- SciPy BVP benchmark settings now expose mesh size, tolerances, max nodes, and the opt-in hand-built finite Jacobian. The hand-built central-difference Jacobian is no longer the default because it is too slow for stacked beds.
- Benchmark rows now include continuation and robustness diagnostics: invalid-state count, guard-penalty count, Jacobian status, scaling mode, transform mode, and continuation path.
- Calibration and UQ scaffolds now emit structured holdout splits, train/holdout metric rows, and two-tier UQ runtime estimates.

## Current Verification Snapshot

Commands run on this worktree:

- `uv run --python C:\ProgramData\miniconda3\python.exe --group test python -m pytest -q`: 63 passed.
- Henry C-case SciPy BVP benchmark: all seven C cases converged. Cases 1C-6C ran in roughly 2.6-3.9 s each; 7C converged but still took about 113 s and used 47,227 guard penalties, so it should be treated as a numerical diagnostic rather than a clean predictive run.
- ePC-SAFT C1 SciPy BVP smoke benchmark: converged in about 20.3 s with zero guard penalties.
- K1-K10 staged/intercooled Henry raw benchmark with `transform_mode=case_bounded_flow_pressure`: no longer produces unbounded negative captures, but most cases still fail the capture gate or singular-Jacobian gate. K1 and K5 can land on acceptable low-residual/high-capture final iterates; K2-K4 and K6-K10 still commonly snap to the lower or upper capture bound.
- The experimental `thermal_state_mode=temperature` path is faster on K1-K5 but currently worsens branch selection. It should remain a diagnostic path until it is paired with a better continuation profile and thermal residual scaling.
- K1 continuation with optional seed/ramp stages reaches a required full-intercooler stage with capture near the NCCC value and near-zero boundary residual. K2 can now preserve the high-capture branch through the ladder, but the full-intercooler stage still has an unacceptable boundary residual and remains `success=False`.
- The in-sample `nccc_linear` capture correction can reproduce the committed NCCC capture percentages, but it is a calibration/reporting layer and no longer changes raw solver success. It must not be described as mechanistic convergence.

## Interpretation

The singular-Jacobian failure mode has been converted into explicit, reproducible diagnostics for the tested cases. Single-bed C cases now converge under Henry, and a one-case ePC-SAFT lane remains functional. Multi-bed/intercooled NCCC cases are not yet reliable predictive mechanistic solves. The main remaining issue is branch selection and conditioning in the staged collocation solve: even with bounded chemistry, fugacity, holdup, and flow/pressure variables, the solver can converge to the physical capture bounds rather than the NCCC operating branch.

The next development step should focus on a continuation path that preserves a plausible capture/temperature profile across stages: one-bed target profile, staged beds without intercooler reset, gradual intercooler-strength ramp, then Henry-to-ePC-SAFT fugacity blending. The temperature-state path also needs residual scaling work before it can replace enthalpy-state solves for the NCCC benchmark.
