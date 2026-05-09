# Handoff: Diagnose the `eta_Psi = 0.3` Driving-Force Scale

## Purpose

This handoff is for a fresh Codex agent working in an isolated worktree. The goal is to diagnose the impact of the hard-coded `0.3` multiplier in the CO2 driving-force correction and determine what happens when the multiplier is set to `1.0`.

Do not start by rewriting the manuscript. First produce run evidence. Only update paper language after the model behavior and evidence are clear.

## Current Source Location

The active code path is:

```text
src/mea_absorption_column/Transport/Enhancement_Factor.py
```

Current expression:

```python
Psi = E * kl_CO2 / kv_CO2
Psi_H = Psi / (Psi + H_CO2_mix)*.3
```

This value enters the CO2 molar flux through:

```text
src/mea_absorption_column/Transport/Flux.py
```

Current flux form:

```python
Nv_CO2 = -kv_CO2 * a_eA * (fv_CO2 - fl_CO2) * Psi_H
```

So changing `.3` directly scales the effective CO2 driving-force factor, not the thermodynamic fugacity calculation itself.

## Git History Already Traced

Use this history as the starting point:

```powershell
git log -L 116,120:src/mea_absorption_column/Transport/Enhancement_Factor.py --patch
```

Observed timeline:

- `e18d188` introduced the unscaled expression:
  `Psi_H = Psi / (Psi + H_CO2_mix)`
- `440a5b4` changed it to `*.5`.
- `e5d2af8` changed `*.5` to `*.3`.
- `9823e28` removed `.3` and restored the unscaled expression.
- `3a0d167` re-added `.3`, and that version survives today.
- Later commits moved the file into `src/` and added diagnostics, but did not explain the numeric source.

The commits do not document a literature citation for `.3`. Treat it as an empirical/lumped correction until proven otherwise.

## Literature Context Already Found

The standard resistance expression is literature-backed:

```tex
\frac{1}{K_G} = \frac{H}{E k_L} + \frac{1}{k_G}
```

Rearranged:

```tex
K_G = k_G \frac{E k_L/k_G}{E k_L/k_G + H}
```

This matches the code shorthand:

```tex
\Psi = E k_L/k_G
\Psi_H = \frac{\Psi}{\Psi + H}
```

Useful source already found:

```text
C:\Users\Tanner\Zotero\storage\A22GYL6U\Harun - 2012 - Dynamic Simulation of MEA Absorption Process for CO 2 Capture from Power Plants.pdf
```

Harun 2012, Appendix page 204 / PDF page 226, gives:

```tex
\frac{1}{K_G} = \frac{H}{k_L} + \frac{1}{k_G}
```

For chemical enhancement, the liquid-side term becomes `H/(E k_L)`.

Separate literature trail:

```text
C:\Users\Tanner\Zotero\storage\HS9JKJIH\Banerjee - 2020 - Carbon Dioxide Capture Using Aqueous MEA Solutions in a Countercurrent Adiabatic Packed-bed Absorber.pdf
```

Banerjee 2020 supports an interfacial-area correction factor of `3` for Cho vs Billet-Schultes wetted-area behavior. That is not direct evidence for a `0.3` multiplier on `Psi_H`.

## Main Diagnostic Question

Answer this with committed or clearly saved evidence:

> Does setting `eta_Psi = 1.0` produce clean, physically reasonable, converged NCCC validation runs, or does the historical `.3` multiplier hide a model/parameter/solver issue?

If `eta_Psi = 1.0` works cleanly, the model should not keep `.3` without a documented calibration reason.

If `eta_Psi = 1.0` breaks clean runs, diagnose the first real failure rather than restoring `.3` reflexively.

## Recommended Implementation Path

Make the multiplier configurable. Do not keep editing the hard-coded number manually.

Preferred minimal change:

- Add an optional parameter or environment-controlled setting for `eta_Psi`.
- Default should initially preserve current behavior (`0.3`) so existing runs remain reproducible.
- Add a clear way to run `eta_Psi=1.0` and optionally `eta_Psi=0.5`.
- Record the selected value in diagnostics/run metadata.

Possible implementation choices:

- Function argument on `enhancement_factor(...)`, threaded from benchmark/model settings.
- Environment variable such as `MEA_ETA_PSI=1.0` for fast diagnostic work.
- Config field in the benchmark run settings if that pattern already exists nearby.

Use the repo's existing pattern; avoid broad refactors.

## Experiment Matrix

Run the same cases under at least:

| Case Set | Thermo Lane | Solver | eta_Psi |
|---|---|---|---|
| Accepted NCCC one-bed set | ideal/Henry if available | Collocation | 0.3 |
| Accepted NCCC one-bed set | ideal/Henry if available | Collocation | 1.0 |
| Accepted NCCC one-bed set | ePC-SAFT ionic | Collocation | 0.3 |
| Accepted NCCC one-bed set | ePC-SAFT ionic | Collocation | 1.0 |
| Case 3C only | ePC-SAFT ionic | Collocation | 0.3, 0.5, 1.0 |

Accepted current manuscript-facing scope is K18, K19, and 1C--6C. K20, 7C, and multi-bed/D rows are known boundary/failure cases unless separately repaired.

## What to Capture

For every row, capture:

- case id
- thermo model label
- `eta_Psi`
- solver method display label
- convergence status
- capture percent
- outlet gas CO2
- max/min temperature
- final mesh nodes
- runtime seconds
- guard counts
- chemistry residual summary if available
- any first exception or timeout reason

Also snapshot enough profile data to compare:

- liquid temperature profile
- vapor temperature profile
- CO2 gas profile
- CO2 liquid/fugacity/driving-force quantities where available
- `Psi`, `Psi_H`, `E`, `H_CO2_mix`, `kl_CO2`, `kv_CO2`

The key comparison is whether `eta_Psi=1.0` shifts capture/temperature in a physically interpretable way or creates solver/domain failures.

## Suggested Commands

Use the repo-local venv when available:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_run_model.py
```

For ePC-SAFT lanes, ensure the sibling package is installed into the venv if needed:

```powershell
uv pip install 'C:\Users\Tanner\Documents\git\ePC-SAFT'
```

Before broad runs, run a bounded smoke on Case 3C with `eta_Psi=0.3` and `eta_Psi=1.0`. Use subprocess timeouts. Do not run unbounded all-case loops.

Look first for existing NCCC scripts under:

```text
analyses/nccc_validation/scripts/
```

Curated outputs belong under:

```text
analyses/nccc_validation/results/final/
```

Disposable exploratory outputs belong under:

```text
analyses/nccc_validation/results/runs/
```

Do not scatter CSVs or plots into the repo root.

## Debugging If eta_Psi = 1.0 Fails

If clean cases fail with `eta_Psi=1.0`, diagnose in this order:

1. Confirm the failure is not just a too-strict timeout or mesh cap.
2. Check whether CO2 flux becomes too large and drives concentrations negative.
3. Check whether `Psi_H` exceeds expected bounds or becomes non-finite.
4. Check fugacity driving force sign and magnitude: `fv_CO2 - fl_CO2`.
5. Check whether liquid-side chemistry receives impossible total concentrations after the stronger flux.
6. Check hydraulic/pressure-drop guards and temperature-domain guards.
7. Compare profiles for the last successful iterate if diagnostics exist.
8. Only after the failing mechanism is known, decide whether the model needs a physical correction, better scaling, or solver continuation.

Do not label `.3` as literature-derived unless a source is found that directly supports this exact placement and value.

## Expected Analysis Outputs

Create a short report with:

- Exact branch/worktree name and commit used.
- Whether the code was parameterized or manually patched.
- Table comparing `eta_Psi=0.3` vs `1.0`.
- Case-by-case pass/fail summary.
- Capture/runtime/temperature deltas.
- First failure mode for any failed case.
- Recommendation:
  - keep `0.3` only as documented empirical calibration,
  - remove it and use `1.0`,
  - or replace it with a different physically justified correction.

## Manuscript Guidance After Evidence Exists

Only update manuscript language after the diagnostic report exists.

If `eta_Psi=1.0` is accepted:

- Present the standard overall gas-side resistance expression.
- Remove any implication that `.3` is a literature driving-force correction.

If `.3` is retained:

- Describe it as a lumped empirical correction or calibration factor.
- State what it corrects: effective interfacial area, mass-transfer closure, or driving-force scale.
- Cite Banerjee only if the text is about interfacial-area correction, not as proof of `eta_Psi=0.3`.

If failures expose a model issue:

- Keep the manuscript on validated runs only.
- Put the unresolved correction in limitations/future work.
- Do not overstate the ePC-SAFT thermodynamic novelty as solving transport calibration.

## Completion Criteria

The handoff is complete when the new agent can report:

- `eta_Psi=0.3` baseline reproduced.
- `eta_Psi=1.0` results produced or first failure diagnosed.
- All output artifacts are organized under `analyses/nccc_validation`.
- No root-level scratch files were created.
- Any code changes are narrow, validated, and committed on the current worktree branch if requested by the user.
