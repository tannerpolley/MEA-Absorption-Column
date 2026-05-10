# PLAN.md — Codex Implementation Plan for MEA Absorber Manuscript Revision

**Purpose:** Implement a controlled, reviewer-facing revision of the MEA absorber benchmark manuscript and repository. The plan synthesizes the two `answers.md` fact-finding passes, the manuscript PDF, and the prior senior-review audit. It is designed for a Codex agent working in a Git worktree.

**Primary objective:** Make the manuscript technically clearer, harder to misread, more reproducible, and more aligned with journal-level chemical-engineering expectations without changing unsupported numerical results.

---

## 0. Operating Mode for Codex

Use **Agent / implementation mode**, not fact-finding-only mode, but keep the edits constrained.

Codex may:

- Edit manuscript `.tex` files.
- Edit README / workflow documentation.
- Add `REPRODUCE.md`.
- Add or update LaTeX tables if values are supported by committed artifacts or the two `answers.md` files.
- Regenerate figures/tables only through existing scripts or clearly documented minimal scripts.
- Run tests, validation scripts, greps, and manuscript builds if available.

Codex must not:

- Invent numerical values.
- Invent citations.
- Invent solver settings.
- Rerun long 200–350 s full-species sweeps unless explicitly instructed by the author.
- Silently resolve conflicts between result artifacts.
- Describe the full activity-coupled path as routine validation.
- Imply that routine ePC-SAFT uses ePC-SAFT activity coefficients inside the chemical-equilibrium equations.
- Claim a large accuracy improvement from ePC-SAFT.

If a value is unsupported or conflicting, insert:

```text
[AUTHOR VERIFY: concise description]
```

At the end, produce a revision report.

---

## 1. Required Inputs to Read First

Before editing, Codex must read the following, if present:

1. `answers.md` or equivalent answer file from the primary Codex fact-finding pass.
2. The second `answers.md` file from the clean-context/new-agent pass.
3. The current manuscript source, likely under:
   - `docs/latex/main.tex`
   - `docs/latex/sections/*.tex`
   - `docs/latex/tables/*.tex`
4. The final paper-facing result artifacts, likely under:
   - `analyses/nccc_validation/results/final/tables/`
   - `analyses/nccc_validation/results/final/reports/`
   - `analyses/nccc_validation/results/final/figures/`
5. The relevant model code, especially:
   - `src/mea_absorption_column/Thermodynamics/thermo_models.py`
   - `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`
   - `src/mea_absorption_column/BVP/ABS_Column.py`
   - `src/mea_absorption_column/Run_Model.py`
   - `src/mea_absorption_column/benchmark.py`
   - `src/mea_absorption_column/BVP/Methods/Single_Shoot_Solve.py`
   - `src/mea_absorption_column/BVP/Methods/Finite_Difference_Solve.py`
   - `src/mea_absorption_column/BVP/Methods/Scipy_BVP_Solve.py`

If files are named differently, locate them by content and report the actual paths.

---

## 2. Non-Negotiable Technical Interpretation

All manuscript and README edits must preserve this interpretation:

> The routine validation campaign uses a liquid-side ionic ePC-SAFT CO2 fugacity closure while retaining concentration-based chemical equilibrium with unit activity coefficients. The concentration-based solve converts apparent species to true species concentrations/mole fractions, and those true species are passed to the `epcsaft` package for fugacity-coefficient calculation. A separate nine-species activity-coupled ePC-SAFT chemistry-and-fugacity path also converges, but it uses relaxed feasibility settings and has 200–350 s runtimes, so it is a feasibility and acceleration target rather than the routine benchmark model.

Use this as the controlling distinction throughout the paper.

---

## 3. Confirmed Facts to Preserve

Use these values and labels unless the current repository artifacts directly contradict them.

### 3.1 Thermodynamic model labels

- Figure 4 / accepted one-bed ePC-SAFT comparison uses `thermo_model = epcsaft_ionic`.
- The routine ePC-SAFT model should be called:

```text
liquid-side ionic ePC-SAFT fugacity closure with concentration-based chemistry
```

- The slow full path should be called:

```text
nine-species activity-coupled ePC-SAFT chemistry-and-fugacity path
```

or, when referring to the code label:

```text
epcsaft_reactive_nine_activity_rebased
```

- The paper-facing ePC-SAFT dataset is:

```text
MEA_CO2_H2O_ionic_fit
```

- The code default may fall back to `MEA_CO2_H2O_draft` unless the dataset environment/configuration is applied. This is a reproducibility risk. Fix documentation so paper reproduction explicitly selects `MEA_CO2_H2O_ionic_fit`.

### 3.2 Routine ePC-SAFT chemistry and fugacity

- Routine ePC-SAFT keeps `chemical_equilibrium_model = legacy` / concentration-based chemistry.
- Routine liquid ePC-SAFT receives normalized true-species mole fractions from the six-species concentration-based chemistry solve:
  - `CO2`
  - `MEA`
  - `H2O`
  - `MEAH+`
  - `MEACOO-`
  - `HCO3-`
- Liquid CO2 fugacity is based on true molecular CO2 only:

```text
fl_CO2 = x_true_CO2 * phi_l_CO2 * P
```

- Vapor CO2 fugacity is:

```text
fv_CO2 = y_CO2 * phi_v_CO2 * P
```

- Routine H2O flux/fugacity remains vapor-pressure based, not ePC-SAFT based:

```text
fv_H2O = P * y_H2O
fl_H2O = P_sat,H2O * x_H2O,true
```

Therefore, do not state that the routine campaign replaces all water/CO2 thermodynamics with ePC-SAFT. State that the routine comparison changes the **CO2 fugacity driving force**.

### 3.3 Full activity-coupled path

- The full path computes species activity coefficients and uses `x_i gamma_i` activity-like terms in the reaction products.
- The rebased standard state is `mole_fraction_activity`.
- The path uses `calibrate_activity_to_legacy=True` or equivalent rebasing to align with the legacy concentration-based state.
- The full nine-species state includes:
  - `CO2`
  - `MEA`
  - `H2O`
  - `MEAH+`
  - `MEACOO-`
  - `HCO3-`
  - `CO3^2-`
  - `H3O+`
  - `OH-`
- The expanded reaction basis includes five reactions:
  - water autoionization,
  - CO2/HCO3,
  - HCO3/CO3,
  - MEACOO hydrolysis,
  - MEAH dissociation.
- Auxiliary parameters for `CO3^2-`, `H3O+`, and `OH-` are placeholder/diagnostic in the committed parameter summary. Do not overstate the full nine-species path as a fully predictive thermodynamic model.

### 3.4 Accepted and attempted validation set

- The accepted one-bed validation set for the routine comparison is:
  - `K18`
  - `K19`
  - `1C`
  - `2C`
  - `3C`
  - `4C`
  - `5C`
  - `6C`
- This is **eight** accepted rows, not seven.
- `K20` and `7C` are attempted but excluded from conditional accepted-row accuracy statistics.
- Do not hide `K20` and `7C`. Add attempted-case accounting.

### 3.5 Validation gate

The repository-supported gate is approximately:

- method success required, unless a low-residual collocation override applies;
- default maximum boundary residual is `1.0`;
- `success_capture_error_max_pct` is optional and only applies when set;
- no temperature-RMSE threshold is implemented in the acceptance script;
- no guard/fallback-count gate is applied to the broad accepted table because accepted `K19` contains guard counts;
- primary C-case validation uses common settings, no case-specific tuning, mass-transfer factor `1.0`, and heat-transfer factor `1.0`.

Write the manuscript so the gate is explicit but not stronger than the artifacts support.

### 3.6 K20 and 7C failure status

- `K20` failed in the attempted accepted-row artifact because SciPy BVP exceeded maximum mesh nodes and profile export fell back after invalid hydraulics/pressure-drop states. The ePC-SAFT K20 row had physically bad capture around `115.134%`, error `+19.584` percentage points, `invalid_state_count=39092`, and `guard_penalty_count=32720`.
- `7C` failed in `nccc_one_bed_all_attempted_results.csv` by subprocess timeout at `subprocess_timeout_s=90`.
- A separate temperature-overlay artifact has a successful 7C row with runtime about `19.253 s`, capture about `72.981%`, capture error `-3.419` percentage points, and temperature RMSE about `16.128 K`.
- This is a scope conflict. For the manuscript, keep Figure 4 accepted-only and report 7C as failed under the accepted-row validation gate unless the author explicitly decides otherwise.

### 3.7 Routine accepted-row summary values

Preserve, if confirmed by artifacts:

- ePC-SAFT accepted-row capture MAE: `3.73` percentage points, or `3.725` if using more precision.
- Henry-law accepted-row capture MAE: `3.78` percentage points, or `3.775` if using more precision.
- ePC-SAFT median runtime: `9.86 s`.
- Henry-law median runtime: `8.62 s`.
- ePC-SAFT fugacity blend for published accepted rows: `1.0`.

### 3.8 Guard/fallback counts

- Routine accepted `epcsaft_ionic` rows are zero guard/invalid state counts except `K19`, which has `invalid_state_count=108` and `guard_penalty_count=108`.
- The primary accepted C-case rows `1C–6C` have zero guard penalties.
- Full Case 3C handoff reports zero invalid states and no domain guards.
- Full seven-row handoff reports all rows with zero invalid states and zero domain guards.
- Because `K19` has guard counts, do **not** say “no guards/fallbacks were used in accepted rows” unless the scope is explicitly “accepted C-case rows” or “full Case 3C handoff.”

### 3.9 Full-path timing and accuracy values

Use cautiously because the raw full-run CSV was not present in the checked-out repo.

Case 3C full nine-species path:

- predicted capture: `89.855%`
- measured capture: `89.5%`
- error: `+0.355` percentage points
- runtime: `212.809 s`
- ePC-SAFT chemistry solve time: `160.191 s`
- no domain guards reported
- chemistry residuals near `1e-8`

Seven C-row sweep, `1C–7C`, relaxed feasibility settings:

| Case | Predicted capture (%) | Measured capture (%) | Error (p.p.) | Runtime (s) |
|---|---:|---:|---:|---:|
| 1C | 79.258 | 97.1 | -17.842 | 336.655 |
| 2C | 86.651 | 92.3 | -5.649 | 289.707 |
| 3C | 89.855 | 89.5 | +0.355 | 281.664 |
| 4C | 94.678 | 88.9 | +5.778 | 308.746 |
| 5C | 90.525 | 86.4 | +4.125 | 444.874 |
| 6C | 70.549 | 60.2 | +10.349 | 420.357 |
| 7C | 98.515 | 76.4 | +22.115 | 380.387 |

Summary:

- seven-row MAE: `9.459` percentage points
- mean runtime: `351.770 s`
- median runtime from listed rows: `336.655 s`
- relaxed settings: `mesh-points 7`, `tol 10`, `bc-tol 0.5`, `max-nodes 80`, timeout `900 s`

Important: These values do **not** support saying that the full seven-row activity-coupled path is “just as accurate” overall. They support saying that Case 3C agrees closely and that the seven-row sweep converged, but accuracy was uneven under relaxed feasibility settings. If the author has newer full-path results showing similar accuracy to the routine model, request/commit those raw artifacts before strengthening the claim.

### 3.10 Solver settings

Accepted NCCC rows use:

- method string: `scipy-bvp`
- paper-facing method name: `Collocation BVP`
- implementation: SciPy `solve_bvp`
- `mesh_points = 21`
- `max_nodes = 1000`
- `tol = 0.5`
- `bc_tol = 0.001`
- `scaling_mode = legacy_flow_enthalpy`
- `transform_mode = bounded_guarded_raw_state`
- `continuation_stage = direct`
- `continuation_success = True`
- `co2_capture_guess_pct = 95.0`
- `h2o_capture_guess_pct = -100.0`
- `epcsaft_fugacity_blend = 1.0`

Explain that `tol = 0.5` is applied in scaled/transformed solver space, not as a raw physical-unit error.

Shooting defaults:

- custom Euler integration by default;
- optional `solve_ivp` modes exist;
- SciPy `root(..., method='Krylov')` for the mismatch solve.

Finite difference:

- implemented as a nonlinear algebraic solve;
- paper-facing rows show convergence can differ from physical acceptance;
- smooth finite-difference row has `107.65%` capture and should not be labeled as physically accepted.

### 3.11 Other key facts

- `eta_Psi = 0.3`, dimensionless, from code expression `Psi_H = Psi / (Psi + H_CO2_mix) * .3`.
- Provenance of `eta_Psi` is not documented. Treat it as a fixed heuristic/global factor unless the author supplies provenance.
- The 45.0 °C lean-inlet values for `1C–3C` are reconstruction assumptions from blank source rows. Do not call them independently validated unless sensitivity evidence is added.
- Normalized column position `0` is vapor inlet/bottom; normalized position `1` is liquid inlet/top.
- Temperature taps shown in Figures 2 and 3 are liquid-temperature taps.
- The sharp liquid-temperature drop near normalized position `1.0` is the top liquid-inlet boundary feature; explain this in captions unless a plot regeneration removes a misleading segment.

---

## 4. Manuscript Revision Plan

### 4.1 Abstract — replace with a stand-alone version under 250 words

Replace the abstract with the text below unless source artifacts require small corrections. Keep it under 250 words.

```text
Post-combustion carbon dioxide capture with aqueous monoethanolamine (MEA) is a mature rate-based separation, but absorber benchmarks often change thermodynamics, solver formulation, and validation scope simultaneously. This work uses one steady-state packed-column model to compare boundary-value solvers and CO2 fugacity closures while holding chemical equilibrium, reaction enhancement, transport correlations, hydraulics, and material and energy balances fixed. The routine thermodynamic campaign replaces the Henry-law CO2 closure with a liquid-side ionic electrolyte perturbed-chain statistical associating fluid theory (ePC-SAFT) fugacity calculation while retaining concentration-based chemical equilibrium with unit activity coefficients. Eight of ten evaluated one-bed National Carbon Capture Center (NCCC) rows satisfied the common solver-acceptance gate; K20 and 7C are retained as attempted rows and excluded from conditional accuracy statistics. For accepted rows, ePC-SAFT gave a capture mean absolute error of 3.73 percentage points, compared with 3.78 percentage points for Henry-law closure, while median runtime increased from 8.62 to 9.86 s. Collocation produced the most reliable accepted NCCC solutions; shooting remained faster on smoother high-liquid-to-gas-ratio probes. A separate nine-species activity-coupled ePC-SAFT chemistry-and-fugacity path converged for Case 3C but required 212.809 s, so it is reported as a feasibility and acceleration target. The benchmark separates solver conditioning from fugacity-closure effects and reports attempted cases, residuals, runtime, and validation error for reproducible absorber-model comparison.
```

After inserting, count words and report the count.

Do not include citations or undefined abbreviations in the abstract.

### 4.2 Title

Keep the current broad title unless the author requests a change:

```text
A Reproducible MEA Absorber Benchmark for ePC-SAFT CO2 Fugacity Driving Forces
```

Do not force “ionic” into the title. Instead, define the ionic liquid-side model in the abstract and methods.

### 4.3 Introduction — narrow and sharpen the novelty claim

Search `docs/latex/sections/introduction.tex` and related included files for over-broad novelty language.

Replace strong first claims such as:

```text
first solved packed-column absorber benchmark to embed a full ionic liquid-side ePC-SAFT fugacity calculation
```

with:

```text
To the authors' knowledge, this is the first reproducible MEA packed-column benchmark to isolate a liquid-side ionic ePC-SAFT CO2 fugacity closure under fixed chemistry, enhancement, transport, hydraulics, balances, solver settings, and validation gates.
```

Use “to the authors’ knowledge” consistently.

Do not write:

- `ePC-SAFT replaces the MEA thermodynamic model`
- `full activity-coupled ePC-SAFT was used for the routine campaign`
- `ePC-SAFT provides improved performance` without immediate quantification.

### 4.4 Scope and Contributions — fix seven/eight and routine/full distinction

In Section 1.2, replace:

```text
all seven accepted one-bed NCCC validation cases
```

with:

```text
all eight solver-accepted one-bed NCCC validation cases, K18, K19, and 1C–6C
```

or:

```text
the eight solver-accepted one-bed NCCC validation rows, with K20 and 7C retained as attempted rows but excluded from conditional accuracy statistics
```

Revise the contribution bullets to something like:

```text
• It evaluates shooting, finite-difference, and collocation BVP solution paths against the same absorber equations and case data.
• It expands validation beyond the original Case 3C illustration to the eight solver-accepted one-bed NCCC rows, while retaining K20 and 7C in attempted-case accounting.
• It isolates a liquid-side ionic ePC-SAFT CO2 fugacity closure while keeping concentration-based chemistry, enhancement factors, transport, hydraulics, and balances fixed.
• It reports solver settings, convergence messages, residual norms, guard diagnostics, runtime, and validation gates so the numerical comparison can be checked and reproduced.
```

### 4.5 Model Framework — clarify routine chemistry and H2O treatment

In Sections 2.1.1 and 2.1.2, make the following distinctions explicit:

1. Routine chemistry solves concentration-based equilibrium with unit activity coefficients.
2. The solved true species concentrations/mole fractions feed transport and ePC-SAFT fugacity calculations.
3. Routine ePC-SAFT changes the CO2 fugacity driving force only.
4. H2O fugacity expressions remain the baseline vapor-pressure/ideal expressions in the routine flux model.
5. The full activity-coupled path is separate and reported only as feasibility/timing evidence.

Recommended insertion near the end of Section 2.1.1:

```text
The routine validation campaign uses this concentration-based equilibrium calculation. It assumes unit activity coefficients in the equilibrium equations but still produces true ionic species concentrations for CO2, MEA, H2O, MEAH+, MEACOO−, and HCO3−. Those true species states are passed to the transport-property and ePC-SAFT fugacity calculations. Thus, the routine ePC-SAFT campaign changes the CO2 fugacity closure, not the chemical-equilibrium model.
```

Recommended sentence in Section 2.1.2 after Equations 26–29:

```text
Equations (26)–(27) define the thermodynamic change tested in the routine ePC-SAFT campaign. The H2O flux uses the same vapor-pressure-based expressions in Equations (28)–(29), so the routine comparison remains localized to the CO2 driving-force closure.
```

If the current code contradicts this, update wording to match code and report the contradiction.

### 4.6 Remove textbook-style prose in the model/transport sections

Where the manuscript says things like:

- “Quantifying the molar flux is paramount...”
- “complete the full picture”
- “This phenomena”

replace with direct process language.

Examples:

```text
Interfacial molar flux determines the axial transfer rate of CO2 and H2O between phases.
```

```text
Transport coefficients and fugacity differences together determine the local absorption rate.
```

```text
Because absorbed CO2 reacts in the liquid phase, the CO2 flux includes the enhancement correction ΨH.
```

---

## 5. Methods Revision Plan

### 5.1 Add a validation-gate subsection before Results

In `docs/latex/sections/methods.tex`, add a concise subsection near the end of Section 3, likely after Section 3.4:

```text
\subsection{Validation gate and reported metrics}
```

Use this content, adjusting only where artifacts dictate:

```text
A result was treated as solver accepted when the selected method satisfied the recorded success gate or, for collocation, the low-residual acceptance rule implemented in the benchmark runner. The default boundary-residual gate is \(\|r_b\| \le 1.0\) in the reported scaled boundary residual. Capture-error gates are applied only in workflows that explicitly set `success_capture_error_max_pct`; no temperature-RMSE threshold is used in the accepted-row gate. Rows that fail the common gate remain in the attempted-case record but are excluded from conditional accuracy statistics. For the primary one-bed comparison, the accepted set is K18, K19, and 1C--6C. K20 and 7C are retained as attempted rows and reported with failure reasons.
```

Then add a sentence on guard diagnostics:

```text
Guard and invalid-state counts are reported as diagnostics rather than as a blanket acceptance criterion. This distinction matters because the accepted K19 ePC-SAFT row contains guard counts, whereas the accepted C-case ePC-SAFT rows have zero guard penalties.
```

If this is too detailed for Methods, move the K19-specific sentence to Results.

### 5.2 Lightly but substantially tighten Section 3.1 Shooting

Do not leave Section 3.1 as a generic tutorial. Replace with implementation-specific prose, but do not over-expand.

Target length: 2–4 concise paragraphs plus equations already needed.

Use these details:

- Shooting converts the BVP into repeated IVP integrations.
- Unknown inlet-side values are adjusted to satisfy the opposite boundary.
- Code default uses custom Euler integration and SciPy `root(method='Krylov')` unless CLI options select `solve_ivp` modes.
- Reported final method table does not record integrator/root settings for historical rows if that remains true; avoid claiming exact historical settings beyond artifacts.
- Main conclusion: fast on smoother rows, fragile on NCCC 3C.

Recommended replacement skeleton:

```text
The shooting implementation treats the unknown boundary values as a nonlinear parameter vector and integrates the column equations from one end of the bed to the other. The mismatch between the computed terminal state and the specified boundary conditions defines the root problem. In the code path used for the benchmark, the default shooting solver uses a custom Euler integrator and SciPy's Krylov root method; optional `solve_ivp` modes are available for alternative IVP integration. Because the final method-contrast artifacts do not store every integrator/root option, the manuscript reports shooting primarily as a benchmarked solver configuration rather than as a new algorithm.

Shooting is retained because it exposes a useful speed/conditioning contrast. It is fast for smoother high-liquid-to-gas-ratio probes but is sensitive to initial guesses and physical-domain excursions for the NCCC 3C thermal-pinch case. Its reported failure in the NCCC 3C comparison is therefore interpreted as solver-conditioning evidence, not as a change in the absorber equations.
```

Retain only the mismatch equation if useful. Remove “The steps are,” “Summary,” and generic lists of Newton/bisection/secant unless tied to the actual implementation.

### 5.3 Lightly but substantially tighten Section 3.2 Finite Difference

Replace the tutorial-style finite-difference explanation with implementation and interpretation.

Use these details:

- finite-difference method discretizes the same seven-state BVP;
- nonlinear algebraic solve;
- useful as a diagnostic discretization check;
- convergence is not physical acceptance;
- smooth finite-difference row converges numerically but predicts `107.65%` capture;
- NCCC 3C finite-difference row is rejected by strict capture/physical gate and has a zero-capture branch in the paper-facing table, with provenance conflict between `method_slice_3c.csv` and `method_case_contrast.csv`.

Recommended replacement skeleton:

```text
The finite-difference implementation discretizes the seven-state BVP on a fixed axial mesh and assembles the governing-equation and boundary-condition residuals into one nonlinear algebraic system. The same thermodynamic, transport, enhancement, hydraulic, and balance functions are evaluated at the mesh nodes, so differences from collocation reflect discretization and nonlinear-solve behavior rather than a different absorber model.

Finite difference is retained as a diagnostic solver rather than as the reference validation method. The method can satisfy the algebraic residual while reaching a physically unacceptable branch. In the smooth one-bed contrast, for example, the finite-difference solve reports numerical convergence but predicts 107.65% capture, which fails physical acceptance. In the NCCC 3C contrast, the paper-facing finite-difference row is reported as a rejected zero-capture branch. These outcomes are used to separate numerical convergence from validation-grade absorber predictions.
```

If exact mesh or nonlinear solver details can be verified, add them. If not, insert `[AUTHOR VERIFY: finite-difference mesh/nonlinear solver settings for reported rows]` only if necessary.

### 5.4 Tighten Section 3.3 Collocation

Keep most of the collocation section, but add exact settings and scaled tolerance explanation.

Add content like:

```text
For the accepted one-bed NCCC rows, the paper-facing collocation configuration uses `mesh_points=21`, `max_nodes=1000`, `tol=0.5`, and `bc_tol=0.001` with `legacy_flow_enthalpy` scaling and `bounded_guarded_raw_state` transformation. The tolerance is applied in scaled/transformed solver space, not directly to raw flow, enthalpy, or pressure units. Accepted rows report the final boundary residual norm separately.
```

Avoid over-explaining orthogonal collocation if SciPy `solve_bvp` is the actual implementation.

### 5.5 Reproducibility settings

In Section 3.4, add or verify:

- Python version and package versions are recorded in result rows.
- The paper-facing dataset is `MEA_CO2_H2O_ionic_fit`.
- ePC-SAFT fugacity blend is `1.0` for published accepted rows.
- The full activity-coupled path uses relaxed feasibility settings, not the same accepted-row gate.

Recommended sentence:

```text
The paper-facing ePC-SAFT rows use the `MEA_CO2_H2O_ionic_fit` parameter dataset and `epcsaft_fugacity_blend=1.0`; no Henry/ePC-SAFT continuation blend is used in the reported accepted rows.
```

If the repo does not force this dataset in reproduction commands, fix README/REPRODUCE as described below.

---

## 6. Results Revision Plan

### 6.1 Section 4.1 NCCC validation — add attempted-case accounting

Keep Figure 2 and Figure 3, but add an attempted/accepted table or appendix table.

Options:

1. Main-text compact table after Table 1.
2. Appendix table, with a concise main-text pointer.

Minimum table columns:

- Year
- Case
- Accepted under primary gate?
- Reason / note
- Method
- Thermo model
- Boundary residual if available
- Capture prediction if available
- Capture error if available
- Guard/invalid count if available

Recommended compact table for main text:

```text
Table X. Attempted one-bed NCCC validation status under the common accepted-row gate.
```

At minimum, include rows for all ten cases:

- `K18`: accepted
- `K19`: accepted, note ePC-SAFT guard diagnostics if included
- `K20`: failed, max mesh nodes / invalid hydraulics / nonphysical capture in ePC-SAFT attempt
- `1C`: accepted
- `2C`: accepted
- `3C`: accepted
- `4C`: accepted
- `5C`: accepted
- `6C`: accepted
- `7C`: failed under accepted-row artifact by `subprocess_timeout_s=90`; successful only in separate overlay scope, not accepted-row aggregate

If a table is too large, add it to appendix and add one paragraph in main results.

Main text should say:

```text
Accuracy statistics below are conditional on the solver-accepted rows. K20 and 7C remain part of the attempted validation scope but are excluded from the conditional aggregate because they did not satisfy the common accepted-row gate.
```

### 6.2 Figure 2 and Figure 3 captions — explain coordinate and terminal drop

Update captions to state:

- normalized position `0` = vapor inlet/bottom;
- normalized position `1` = liquid inlet/top;
- terminal liquid-temperature drop corresponds to the specified lean-liquid inlet boundary condition.

For Figure 2, add:

```text
The axial coordinate is normalized from the vapor inlet at 0 to the liquid inlet at 1; the terminal drop corresponds to the specified lean-liquid inlet boundary condition.
```

For Figure 3, add a shorter version:

```text
The coordinate is normalized bottom-to-top, so position 1 is the liquid inlet.
```

If Codex can regenerate a cleaner plot that avoids a visually misleading vertical boundary segment without changing data interpretation, it may do so through existing scripts. Otherwise, caption clarification is sufficient.

### 6.3 Figure 4 — accepted-only clarity

Keep Figure 4 accepted-only.

Update caption:

```text
Thermodynamic driving-force comparison for the eight solver-accepted one-bed, no-intercooler NCCC rows. K20 and 7C failed the common accepted-row gate and are reported in the attempted-case table rather than included in this conditional accuracy plot.
```

Update body text:

```text
Across the eight solver-accepted rows, ePC-SAFT gives a capture mean absolute error of 3.73 percentage points, compared with 3.78 percentage points for Henry-law closure. The median runtime increases from 8.62 to 9.86 s. The 0.05 percentage-point MAE difference is small; the process-level significance is that an ionic EOS-based CO2 fugacity closure can be embedded in repeated column BVP simulations at near-Henry runtime.
```

Do not write:

```text
ePC-SAFT improved capture prediction
```

unless the next clause immediately quantifies the 0.05 p.p. change.

### 6.4 Section 4.3 / Table 2 — separate numerical convergence from physical acceptance

Revise Table 2.

Current problem: Smooth finite-difference has `Success = Yes` but `107.65%` capture. That should not be a physical success.

Recommended columns:

| Scenario | Method | Numerical status | Physical acceptance | Runtime (s) | Capture (%) | Interpretation |
|---|---|---|---|---:|---:|---|
| Smooth one-bed | Shooting | Converged | Accepted | 4.91 | 90.17 | Fast favorable-case solution |
| Smooth one-bed | Collocation BVP | Converged | Accepted | 7.61 | 89.95 | Reference solution |
| Smooth one-bed | Finite difference | Converged | Rejected | 15.62 | 107.65 | Capture exceeds 100% |
| NCCC 3C thermal pinch | Shooting | Failed gate | Rejected | 62.73 | — | Did not satisfy acceptance gate |
| NCCC 3C thermal pinch | Collocation BVP | Converged | Accepted | 9.40 | 89.40 | Reference validation solution |
| NCCC 3C thermal pinch | Finite difference | Rejected branch | Rejected | 16.61 | 0.00 | Physically wrong zero-capture branch |

If artifacts show a more precise phrase than “Rejected branch,” use it. If the raw artifact conflict remains, add a footnote:

```text
The paper-facing method-contrast artifact reports the finite-difference 3C result as a rejected zero-capture branch; a separate method-slice artifact records a timeout for that workflow. The row is retained only to show that finite-difference did not provide a validation-grade 3C solution.
```

If that footnote is too much for main text, put it in the revision report and use the cleaner table language.

### 6.5 Figure 5 — align with revised Table 2

If Figure 5 marks finite-difference smooth as simply successful, revise the label or caption.

Caption should say:

```text
Runtime and method comparison for representative one-bed cases. Bars indicate numerical runtime; labels mark cases that failed the validation/physical-acceptance gate.
```

If the smooth finite-difference row is shown without a failure marker, add a note in caption:

```text
The smooth finite-difference row converged numerically but is rejected physically because capture exceeds 100%; this distinction is shown in Table 2.
```

### 6.6 Section 4.4 / Table 3 — make full activity path a feasibility boundary

Revise Section 4.4 to remove overclaiming.

Replace:

```text
The strongest proof uses...
```

with:

```text
The most demanding feasibility test uses...
```

Replace:

```text
validated future path
```

with:

```text
feasible but not yet routine path
```

Replace:

```text
right middle ground
```

with:

```text
structured fugacity-coefficient calculation with near-Henry runtime
```

or:

```text
computationally feasible EOS-based CO2 fugacity closure
```

Revise Table 3 to include accuracy and gate/scope information if the values are accepted.

Recommended revised Table 3 columns:

| Path | Scope | Gate/settings | Accuracy metric | Runtime | Interpretation |
|---|---|---|---|---|---|
| Henry-law routine campaign | Eight accepted rows | Common accepted-row gate | MAE 3.78 p.p. | Median 8.62 s | Fast baseline |
| ePC-SAFT fugacity campaign | Eight accepted rows | Common accepted-row gate | MAE 3.73 p.p. | Median 9.86 s | Routine ionic CO2 fugacity closure with concentration-based chemistry |
| Full nine-species activity-coupled proof | Case 3C | Feasibility settings | 89.855% capture; +0.355 p.p. error | 212.809 s; 160.191 s in chemistry solves | Feasible but too slow for routine validation |
| Full nine-species activity-coupled sweep | 1C–7C | Relaxed feasibility settings | MAE 9.459 p.p. if handoff values retained | Mean 351.770 s | Converged sweep; accuracy uneven; acceleration target |

If the author does not want to include seven-row MAE because raw CSV is missing, do not hide the issue. Use:

```text
Seven C rows: all converged under relaxed feasibility settings; handoff MAE 9.459 p.p. [AUTHOR VERIFY: raw CSV not committed]
```

Important: Do not claim the seven-row full activity path is “just as accurate” unless newer committed artifacts support it. Current handoff values show close agreement for Case 3C but uneven seven-row accuracy.

### 6.7 Limitations and Future Work

Update limitations to say:

- The model is one-bed/no-intercooler only.
- Broader plantwide/multi-bed claims require additional data.
- Full activity-coupled ePC-SAFT uses placeholder/diagnostic auxiliary ion parameters for carbonate/hydronium/hydroxide unless better provenance is added.
- The full activity-coupled path used relaxed feasibility settings and should be accelerated and validated under the same gate before being treated as the routine model.
- K20 and 7C are attempted but not accepted under the common gate; 7C has a successful separate overlay scope, so manuscript wording must not conflate scopes.
- `eta_Psi=0.3` is fixed and used consistently, but provenance should be documented.

Remove:

```text
current branch
paper-facing evidence
```

Use:

```text
present implementation
reported validation evidence
```

### 6.8 Conclusion

Revise conclusion to avoid broad improvement and middle-ground language.

Recommended conclusion language:

```text
The accepted-row comparison does not show a large capture-accuracy gain from ePC-SAFT; it shows that a liquid-side ionic EOS-based CO2 fugacity closure can be embedded in repeated packed-column BVP simulations with only modest runtime overhead. The accepted-row capture MAE changes from 3.78 to 3.73 percentage points while median runtime increases from 8.62 to 9.86 s. The full nine-species activity-coupled path is technically feasible, including a close Case 3C result, but its 200–350 s runtime and relaxed feasibility settings make it an acceleration target rather than the routine validation model.
```

Keep the final message focused on engineering value:

- controlled comparison,
- solver diagnostics,
- reproducible benchmark,
- conditional validation accounting,
- future acceleration of activity-coupled thermodynamics.

---

## 7. Repository and Documentation Plan

### 7.1 Add `REPRODUCE.md`

Create a top-level `REPRODUCE.md` unless one already exists.

Minimum structure:

```md
# Reproducing the MEA Absorber Benchmark Results

## Scope
This document reproduces the paper-facing routine benchmark results. The full nine-species activity-coupled path is slower and treated as a feasibility workflow.

## Environment
- Python: [from artifacts or AUTHOR VERIFY]
- Platform used for paper artifacts: [from accepted results]
- Required package manager: uv / pip [verify]
- External dependency: epcsaft [AUTHOR VERIFY public repo/package/commit]

## Required ePC-SAFT dataset
Set or verify the paper-facing dataset:
`MEA_CO2_H2O_ionic_fit`

Example:
```bash
export MEA_EPCSAFT_DATASET_NAME=MEA_CO2_H2O_ionic_fit
```

PowerShell:
```powershell
$env:MEA_EPCSAFT_DATASET_NAME = "MEA_CO2_H2O_ionic_fit"
```

## Routine accepted-row benchmark
[exact command chain]

## Figure and table regeneration
[exact scripts and outputs]

## Validation check
[validate_results.py command]

## Manuscript build
[latexmk/make command]

## Slow full-species feasibility path
Explain that this is optional, slow, and not required for reproducing routine Figure 4. Include command only if documented and safe.

## Expected outputs
List final CSVs, figures, and LaTeX tables.

## Known limitations
- K20 and 7C are attempted but not accepted under the primary accepted-row gate.
- The full activity-coupled raw run CSV may need to be restored/committed.
- External epcsaft package must be available.
```

Use actual commands from repository scripts. If no one-command workflow exists, document the exact ordered chain and mark gaps.

### 7.2 Update README.md

Update README to align with the manuscript.

Must distinguish:

1. Henry-law baseline.
2. Routine `epcsaft_ionic` fugacity campaign.
3. Full `epcsaft_reactive_nine_activity_rebased` feasibility path.

Recommended README paragraph:

```text
The paper-facing routine ePC-SAFT comparison uses concentration-based chemical equilibrium and a liquid-side ionic ePC-SAFT CO2 fugacity closure (`thermo_model=epcsaft_ionic`). It does not use ePC-SAFT activity coefficients inside the routine chemical-equilibrium equations. A separate nine-species activity-coupled ePC-SAFT chemistry-and-fugacity path (`epcsaft_reactive_nine_activity_rebased`) is reported as a timing and feasibility boundary.
```

Remove or generalize local paths such as:

- `C:\Users\Tanner\Documents\git\ePC-SAFT`
- Overleaf local checkout paths
- machine-specific `MEA_EPCSAFT_ROOT` examples without a generic alternative.

Replace with generic examples:

```bash
pip install /path/to/ePC-SAFT
export MEA_EPCSAFT_ROOT=/path/to/ePC-SAFT
```

and note:

```text
[AUTHOR VERIFY: public ePC-SAFT package repository, commit, or archive DOI]
```

### 7.3 Update `docs/workflow_map.md`

Ensure the workflow map says:

- `epcsaft_ionic` is the paper-facing routine fugacity lane.
- `MEA_CO2_H2O_ionic_fit` is required for paper-facing runs.
- `epcsaft_reactive_nine_activity_rebased` is the full slow path.
- Table 3 values are from handoff artifacts unless raw CSV is restored.
- `epcsaft_neutral` is not Figure 4.

### 7.4 Add code-to-paper traceability

Add either:

- `docs/code_to_paper_traceability.md`, and/or
- a LaTeX appendix table.

Recommended columns:

| Manuscript item | Source script | Input artifact | Output artifact | Method | Thermo model | Chemistry model | Gate/scope |
|---|---|---|---|---|---|---|---|
| Table 1 | ... | ... | ... | n/a | n/a | n/a | attempted scope |
| Figure 2 | ... | ... | ... | scipy-bvp | epcsaft_ionic | legacy | accepted 3C |
| Figure 3 | ... | ... | ... | scipy-bvp | epcsaft_ionic | legacy | accepted C cases |
| Figure 4 | ... | ... | ... | scipy-bvp | epcsaft_ionic / ideal_henry | legacy | 8 accepted rows |
| Table 2 | ... | ... | ... | shooting / finite-difference / scipy-bvp | likely ideal/ePC as applicable | legacy | method contrast |
| Figure 5 | ... | ... | ... | same | same | same | method contrast |
| Table 3 | handoff / table | missing raw CSV? | full_ionic_speciation_timing.tex | scipy-bvp | epcsaft_reactive... | activity-coupled | feasibility |

If exact scripts cannot be found, mark `[AUTHOR VERIFY]`.

### 7.5 Restore or document missing raw full-species CSV

The answer files report that `analyses/nccc_validation/results/runs/full_species_ionic_all_c_cases/benchmark_results.csv` is referenced but not present in the checkout.

Codex should:

1. Search for the file or equivalent.
2. If found, update Table 3 provenance and REPRODUCE.
3. If absent, add a clear note in `REPRODUCE.md` and the revision report:

```text
[AUTHOR VERIFY: raw full-species seven-row run CSV referenced by handoff is not committed]
```

Do not fabricate the CSV from the handoff unless explicitly instructed.

### 7.6 External `epcsaft` citation and software metadata

Fix reference [25].

Current placeholder:

```text
Polley, T., 2026. Python package used for epc-saft fugacity-coefficient calculations.
```

Replace with a real software citation if available:

- package name,
- author,
- year,
- version,
- commit hash,
- repository URL,
- archive DOI if available,
- access date if no DOI.

If unavailable, insert:

```text
[AUTHOR VERIFY: public ePC-SAFT package citation/version/commit/DOI]
```

Also consider adding `CITATION.cff` or `.zenodo.json` if the author wants a release DOI. Do not invent DOI.

### 7.7 Code Availability and Data Availability

Update availability sections to include exact paths and commit metadata.

Suggested Data Availability:

```text
The benchmark input data, processed validation tables, plotted data snapshots, and figure-generation scripts are distributed with the project repository at [AUTHOR VERIFY: repository URL and release commit]. The curated paper artifacts are stored under `analyses/nccc_validation/results/final/`.
```

Suggested Code Availability:

```text
The Python code used for the absorber benchmark, including solver settings and thermodynamic driving-force calculations, is available at [AUTHOR VERIFY: repository URL, release tag or commit hash, and archive DOI if available]. The paper-facing ePC-SAFT rows use the `MEA_CO2_H2O_ionic_fit` dataset and the external `epcsaft` package [AUTHOR VERIFY: citation].
```

Use current commit only if the author confirms it is the release commit:

```text
12572b85cb4e722a4c0dde8e18c6d0c969263a3a
```

### 7.8 Generative AI declaration

Update title to:

```text
Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
```

Suggested text:

```text
During the preparation of this work, the authors used ChatGPT and Codex by OpenAI to support grammar refinement, LaTeX formatting, code-editing assistance, and submission-readiness checks. After using these tools, the authors reviewed, edited, and verified the content, calculations, citations, and code as needed and take full responsibility for the content of the publication.
```

Do not list AI tools as authors.

---

## 8. Phrase-Level Cleanup Plan

Search all manuscript `.tex` files for the following phrases.

### 8.1 Replace internal-development language

| Search phrase | Replace with |
|---|---|
| `paper-facing evidence` | `reported validation evidence` |
| `current branch` | `present implementation` |
| `strongest proof` | `most demanding feasibility test` |
| `right middle ground` | `structured fugacity-coefficient calculation with near-Henry runtime` |
| `practical middle-ground thermodynamic model` | `computationally feasible EOS-based CO2 fugacity closure` |
| `full picture` | `coupled transport and thermodynamic description` or delete |
| `paramount` | `necessary` or `central` |
| `routine campaign` | `routine validation campaign` or `routine thermodynamic campaign` |
| `this phenomena` | `this phenomenon` or rewrite |

### 8.2 Replace generic performance claims

Search:

- `improved performance`
- `better performance`
- `more accurate`
- `strong performance`
- `more rigorous`
- `state-of-the-art`
- `cutting-edge`
- `testament`
- `middle ground`

Replace or quantify.

Examples:

Bad:

```text
ePC-SAFT improves capture prediction.
```

Better:

```text
ePC-SAFT changes accepted-row capture MAE from 3.78 to 3.73 percentage points while increasing median runtime from 8.62 to 9.86 s.
```

Bad:

```text
ePC-SAFT is a practical middle ground.
```

Better:

```text
ePC-SAFT provides a structured CO2 fugacity-coefficient closure that remains close to Henry-law runtime for the accepted validation rows.
```

### 8.3 Abbreviation and metadata cleanup

Ensure first-use definitions for:

- MEA = monoethanolamine
- CO2 = carbon dioxide, if needed in prose
- ePC-SAFT = electrolyte perturbed-chain statistical associating fluid theory
- NCCC = National Carbon Capture Center
- PCC = post-combustion carbon capture
- eNRTL = electrolyte nonrandom two-liquid
- BVP = boundary-value problem
- IVP = initial-value problem
- VLE = vapor–liquid equilibrium

In the abstract, define MEA, ePC-SAFT, and NCCC if used.

Do not include bracketed citations in the abstract.

DOI strings are acceptable in the reference list but should not appear as stray prose in the manuscript body.

---

## 9. Optional Engineering Diagnostic

The code can export axial profiles containing:

- `fv_CO2`
- `fl_CO2`
- `DF_CO2`
- `Nl_CO2`
- `Nv_CO2`
- `Psi`
- `Psi_H`
- `E`

If these profiles already exist in committed final artifacts, add a short supplementary figure or appendix table for Case 3C comparing Henry-law and ePC-SAFT axial driving force or flux.

Preferred metric:

```text
(fv_CO2 - fl_CO2)_ePC-SAFT / (fv_CO2 - fl_CO2)_Henry
```

If the profile files are not committed or producing the figure requires nontrivial model changes, do not add the figure. Instead add a limitations sentence:

```text
Future benchmark extensions should report axial fugacity-driving-force and flux profiles to identify where EOS-based fugacity changes the local column response.
```

---

## 10. Quality-Control Checks

### 10.1 Grep checks

Run equivalent commands for the platform.

Bash:

```bash
grep -R "seven accepted" -n docs README.md analyses src || true
grep -R "paper-facing" -n docs README.md analyses src || true
grep -R "current branch" -n docs README.md analyses src || true
grep -R "strongest proof" -n docs README.md analyses src || true
grep -R "right middle ground" -n docs README.md analyses src || true
grep -R "improved performance" -n docs README.md analyses src || true
grep -R "full picture" -n docs README.md analyses src || true
grep -R "paramount" -n docs README.md analyses src || true
grep -R "epcsaft_neutral" -n docs README.md analyses src || true
```

PowerShell:

```powershell
Select-String -Path docs\*,README.md,analyses\*,src\* -Pattern "seven accepted","paper-facing","current branch","strongest proof","right middle ground","improved performance","full picture","paramount","epcsaft_neutral" -Recurse
```

Do not delete legitimate code references to `epcsaft_neutral`; just make sure manuscript/README do not imply it generated Figure 4.

### 10.2 Abstract word count

Count abstract words and report the count. Must be below 250.

### 10.3 Numerical consistency checklist

Verify the manuscript consistently reports:

- accepted NCCC rows: `8 of 10`
- accepted cases: `K18`, `K19`, `1C–6C`
- attempted failed rows: `K20`, `7C`
- routine ePC-SAFT model: `epcsaft_ionic`
- routine chemistry model: `legacy` / concentration-based
- routine fugacity blend: `1.0`
- ePC-SAFT accepted-row capture MAE: `3.73` p.p.
- Henry accepted-row capture MAE: `3.78` p.p.
- ePC-SAFT median runtime: `9.86 s`
- Henry median runtime: `8.62 s`
- full Case 3C predicted capture: `89.855%`
- full Case 3C measured capture: `89.5%`
- full Case 3C error: `+0.355` p.p.
- full Case 3C runtime: `212.809 s`
- full Case 3C chemistry solve time: `160.191 s`
- full C sweep mean runtime: `351.770 s`
- full C sweep MAE: `9.459` p.p. if retained from handoff
- `eta_Psi = 0.3` if discussed

If any value is unsupported in final committed artifacts, mark `[AUTHOR VERIFY]` or remove the numerical claim.

### 10.4 Build manuscript

Find and run the appropriate build command:

```bash
latexmk -pdf main.tex
```

or repository-specific build command.

If LaTeX is unavailable, report:

```text
Build not run: LaTeX tooling unavailable in this environment.
```

### 10.5 Validate result artifacts

Run existing validation scripts if available, such as:

```bash
python analyses/nccc_validation/scripts/validate_results.py
```

Do not write new validation logic unless necessary and minimal.

### 10.6 Do not run slow full-species sweep unless instructed

Because the full path is slow and external-package-dependent, do not rerun it by default. If the raw CSV is missing, mark it as missing and request author action.

---

## 11. Expected Files to Change

Codex will likely edit or add:

- `docs/latex/main.tex` or wherever the abstract is located
- `docs/latex/sections/introduction.tex`
- `docs/latex/sections/model_framework.tex`
- `docs/latex/sections/methods.tex`
- `docs/latex/sections/results.tex`
- `docs/latex/sections/conclusion.tex`, if separate
- `docs/latex/tables/method_case_contrast.tex`
- `docs/latex/tables/full_ionic_speciation_timing.tex`
- new attempted-case table, if added:
  - `docs/latex/tables/nccc_one_bed_attempted_status.tex`
- availability/declaration files if separated:
  - `docs/latex/sections/code_availability.tex`
  - `docs/latex/sections/data_availability.tex`
  - `docs/latex/sections/ai_declaration.tex`
- `README.md`
- `docs/workflow_map.md`
- new `REPRODUCE.md`
- optional `docs/code_to_paper_traceability.md`

Do not modify core model code unless needed only to improve reproducibility/documentation and not numerical behavior.

---

## 12. Final Revision Report Required

At the end of the Codex run, produce this report:

```md
# Revision Report

## Files Changed
[List files]

## Technical Consistency Fixes
- Routine ePC-SAFT naming
- Full activity-coupled path framing
- Accepted/attempted validation accounting
- Validation gate definition
- Table 2 convergence vs physical acceptance
- Table 3 feasibility framing

## Numerical Values Added or Changed
[List each value and supporting artifact]

## Figures/Tables Regenerated
[List commands and outputs]

## Commands Run
[List commands and pass/fail status]

## Remaining AUTHOR VERIFY Items
[List all unresolved items]

## Artifact Conflicts Still Present
- 7C accepted-row timeout vs separate successful overlay row
- full activity-coupled handoff values vs missing raw CSV
- method_slice_3c vs method_case_contrast finite-difference provenance, if unresolved
- `MEA_CO2_H2O_draft` default vs paper-facing `MEA_CO2_H2O_ionic_fit`, if not fully enforced

## Risks for Human Review
[List remaining issues that could trigger reviewer questions]
```

---

## 13. Priority Order if Time Is Limited

If Codex cannot complete everything, implement in this order:

1. Fix routine vs full ePC-SAFT wording everywhere.
2. Fix “seven” to “eight” accepted rows.
3. Replace the abstract.
4. Add validation-gate paragraph.
5. Add attempted-case accounting for K20 and 7C.
6. Revise Table 2 to separate numerical convergence from physical acceptance.
7. Revise Table 3 to make the full activity-coupled path feasibility-only and include accuracy caveat.
8. Explain Figures 2–3 coordinate/terminal drop in captions.
9. Remove internal-development and generic performance language.
10. Add `REPRODUCE.md` and remove local paths from README.
11. Fix Code Availability, Data Availability, software citation, and AI declaration.
12. Add code-to-paper traceability.

---

## 14. One-Sentence Target Outcome

After revision, a reviewer should be able to say:

> The paper cleanly separates the routine ionic ePC-SAFT CO2 fugacity closure from the slower full activity-coupled chemistry path, reports conditional accuracy only for solver-accepted rows, accounts for failed attempts, and frames the contribution as a reproducible absorber-model benchmark rather than an overclaimed thermodynamic accuracy improvement.
