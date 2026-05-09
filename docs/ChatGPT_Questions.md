# Factual Repository/Manuscript Clarification Questions

## Instructions for Codex

Answer only the questions you can verify directly from the repository, committed result files, scripts, manuscript source, or code.

Do not infer intent. Do not invent missing values. Do not rewrite the manuscript yet.

For each answer, include:
- the answer,
- the supporting file path,
- the relevant function, variable, command, table, or artifact name if available.

If the answer is not directly verifiable, write:

> AUTHOR VERIFY: [brief explanation of what is missing]

Pay special attention to:
1. The exact thermodynamic model string used for Figure 4 and the accepted NCCC rows.
2. Whether those rows use `epcsaft_ionic`, `epcsaft_neutral`, or another label.
3. The exact solver settings and acceptance-gate thresholds.
4. Guard/fallback counts for accepted ePC-SAFT rows.
5. The values behind Table 3 for the full nine-species activity-coupled path.
6. Whether H2O fugacity in the routine flux model is computed by ePC-SAFT or by the vapor-pressure expression.
7. The exact scripts and final CSVs that generate each manuscript figure/table.
8. Any inconsistency between README language, manuscript language, and result artifacts.

---

## A. Highest-Priority Clarifications

### 1. Which exact ePC-SAFT lane produced Figure 4?

For the accepted-row comparison in Figure 4, was the ePC-SAFT line generated with:
- `epcsaft_ionic`,
- `epcsaft_neutral`,
- or another label?

Report the exact `thermo_model` string from the result artifact.

### 2. What should the manuscript call the routine ePC-SAFT model?

Which name is technically correct based on the code and result artifacts?
- “ionic ePC-SAFT fugacity-only model”
- “liquid-side ionic ePC-SAFT fugacity closure with concentration-based chemistry”
- “full ionic liquid-side ePC-SAFT fugacity calculation”
- another exact phrase?

### 3. What should the manuscript call the slow 200–350 s path?

Which name is technically correct based on the code and result artifacts?
- “full ionic activity-coupled ePC-SAFT speciation path”
- “full ionic reactive ePC-SAFT speciation path”
- “nine-species activity-coupled ePC-SAFT chemistry and fugacity path”
- another exact phrase?

### 4. Does the full activity-coupled path use activity coefficients in the chemical-equilibrium equations?

Does it compute and use:
- species activity coefficients, `gamma_i`,
- residual chemical potentials,
- fugacity-derived activities,
- activity ratios,
- corrected reaction quotients,
- or another activity-related quantity?

### 5. What is the exact reaction/speciation basis of the full nine-species path?

The manuscript lists nine species:

`CO2, MEA, H2O, MEAH+, MEACOO-, HCO3-, CO3^2-, H3O+, OH-`

Does the slow path solve an expanded chemical-equilibrium system with additional reactions for carbonate, hydronium, and hydroxide, or does it add those species through another ePC-SAFT-compatible speciation closure?

---

## B. Validation and Claims

### 6. What exact acceptance gate defines the eight accepted NCCC rows?

Report the numerical criteria for:
- boundary residual threshold,
- capture-error threshold, if any,
- temperature RMSE threshold, if any,
- solver-status requirement,
- physical-domain checks,
- max runtime or timeout,
- guard/fallback count requirement,
- collocation low-residual override rule, if any.

### 7. Why did K20 fail?

Was the failure due to:
- solver timeout,
- high boundary residual,
- physically invalid branch,
- invalid thermodynamic state,
- ePC-SAFT density/fugacity failure,
- poor capture prediction,
- temperature-profile mismatch,
- or another reason?

### 8. Why did 7C fail?

Was the failure due to:
- solver timeout,
- high boundary residual,
- physically invalid branch,
- invalid thermodynamic state,
- ePC-SAFT density/fugacity failure,
- poor capture prediction,
- temperature-profile mismatch,
- or another reason?

### 9. Is the “seven-row C-case sweep” in Table 3 actually 1C–7C?

Confirm whether the full activity-speciation sweep includes:
- 1C–7C,
- only accepted C rows,
- or another subset.

### 10. What were the capture predictions and errors for the full activity-coupled path across the seven C rows?

Report:
- per-case predicted capture,
- per-case measured capture,
- per-case capture error,
- MAE,
- median runtime,
- mean runtime,
- whether any row used relaxed settings,
- whether any row failed the common gate.

### 11. Should the full activity-coupled path be described as “validated” or only “feasible”?

Based on the code and result artifacts, did the full activity-coupled path use the same validation gate as the routine campaign, or only relaxed feasibility settings?

### 12. Were the 212.809 s and 351.770 s runtimes measured on the same hardware and Python environment as the 8.62 s and 9.86 s routine medians?

Report the recorded platform, Python version, package versions, and timing protocol if available.

### 13. Do the slow-path runtimes include import/setup overhead, or only model solve time?

Clarify what the runtime field measures.

---

## C. Thermodynamic Model Details

### 14. In the routine ionic ePC-SAFT fugacity calculation, what composition vector is passed to ePC-SAFT for the liquid phase?

Is it:
- six true species from concentration-based chemistry: `CO2, MEA, H2O, MEAH+, MEACOO-, HCO3-`,
- apparent species only,
- normalized mole fractions from concentrations,
- or another basis?

### 15. In the routine ePC-SAFT comparison, is the liquid CO2 fugacity based on true molecular CO2 only?

Confirm whether `x_l_CO2` in the ePC-SAFT liquid fugacity expression is the true molecular CO2 mole fraction from the concentration-based speciation solve.

### 16. What is the thermodynamic standard state implicit in the activity-coupled chemical-equilibrium solve?

Does the full activity-coupled path use concentration-based equilibrium constants, activity-based constants, EOS fugacity-based activities, or another reference-state convention?

### 17. Are the ePC-SAFT parameters for CO3^2-, H3O+, and OH- fitted, literature-based, or placeholder/diagnostic?

Report the source/provenance for these parameters.

### 18. What exact ePC-SAFT dataset name should be cited?

For example, should the manuscript cite `MEA_CO2_H2O_ionic_fit`, or another dataset name?

### 19. Does the ePC-SAFT routine campaign ever use fugacity blending?

Confirm the value of `--epcsaft-fugacity-blend` or equivalent for the published ePC-SAFT results.

### 20. Were any guard or fallback fugacities used in the accepted ePC-SAFT rows?

Report guard/fallback counts for:
- routine ionic ePC-SAFT accepted rows,
- full activity-coupled Case 3C,
- full seven-row C sweep.

---

## D. Calibration and Engineering Interpretation

### 21. What is the numerical value of `eta_Psi` in Equation 45?

Report:
- value,
- units or dimensionless status,
- how it was chosen,
- whether it was fitted,
- which data were used,
- whether Henry-law and ePC-SAFT use the same value.

### 22. Are any other model parameters calibrated to NCCC data?

Check for calibration of:
- Henry-law deviation coefficients,
- enhancement-factor scale,
- transfer-coefficient multipliers,
- heat-transfer multipliers,
- `eta_Psi`,
- inlet assumptions,
- any other fitted parameter.

### 23. Is the 45.0 °C lean-inlet assumption for 1C–3C independently justified?

Does the repo or manuscript source contain a sensitivity check, note, or data source supporting the 45.0 °C assumption?

### 24. What is the strongest engineering consequence supported by the result artifacts?

Which claim is best supported?
- ePC-SAFT gives nearly the same capture prediction as Henry-law, so the value is controlled thermodynamic interpretability rather than accuracy gain.
- ionic ePC-SAFT can be embedded inside a full column BVP at routine timescale.
- full activity-coupled ePC-SAFT is feasible but too slow, making acceleration the next target.
- solver diagnostics are as important as thermodynamic model choice.

### 25. Can the code output axial profiles of fugacity driving force or CO2 flux?

Does the current code or result artifact export:
- vapor CO2 fugacity,
- liquid CO2 fugacity,
- CO2 fugacity difference,
- CO2 flux,
- driving-force ratio,
- local enhancement factor,
- axial thermodynamic profiles?

---

## E. Solver and Numerical Method Details

### 26. What exact solver produced the accepted NCCC results?

Is it SciPy `solve_bvp` under the label `scipy-bvp`, `collocation`, `Collocation BVP`, or another exact method string?

### 27. What are the exact solver settings for the accepted rows?

Report:
- initial mesh points,
- maximum nodes,
- solver tolerance,
- boundary-condition tolerance,
- scaling convention,
- initialization profile,
- timeout,
- any case-specific override.

### 28. Why is the collocation tolerance apparently large if `tol = 0.5` is the paper setting?

Is this tolerance scaled? If so, describe the scaling.

### 29. For Table 2, does “Success” mean solver convergence only?

Confirm whether the finite-difference smooth row with 107.65% capture should be described as solver convergence but failed physical acceptance.

### 30. For the NCCC 3C finite-difference row, does 0.00% capture mean a converged but physically wrong branch, or a failed run with a placeholder value?

Clarify how this value is generated and how it should be described.

### 31. For shooting, which IVP integrator and root solver were used for the reported rows?

Report the exact functions or solver names.

---

## F. Figures and Presentation

### 32. What causes the sharp liquid-temperature drop near normalized position 1.0 in Figures 2 and 3?

Is it:
- the lean-liquid inlet boundary condition,
- the coordinate direction,
- interpolation through a boundary point,
- a plotting artifact,
- a real thermal feature?

### 33. Is normalized column position 0 the vapor inlet and 1 the liquid inlet?

Confirm the coordinate convention used in figures.

### 34. Are the temperature taps liquid-phase measurements only?

Confirm whether all markers in Figures 2 and 3 are liquid-temperature taps.

### 35. Should Figure 4 include failed rows in a separate panel or should failures stay in a table?

Based on available artifacts, can failed rows be shown cleanly with failure reasons?

### 36. Should Table 3 include full-path accuracy, not only runtime?

Does the repo contain enough values to add:
- capture prediction,
- capture error,
- MAE for seven C rows,
- runtime?

---

## G. Repository and Reproducibility

### 37. What commit hash should the manuscript cite?

Report the current commit hash or the intended release commit if available.

### 38. Will the repo be archived with Zenodo or another DOI service before submission?

If already configured, report the archive plan or metadata.

### 39. Is the external `epcsaft` package public?

Report whether it is:
- public repository,
- archived package,
- local-only dependency,
- vendored implementation,
- or unavailable.

### 40. Should the README remove local path language before submission?

Identify any local paths, machine-specific instructions, or non-reproducible setup steps.

### 41. Are all curated final CSVs intentionally committed and non-empty?

Check the final results directory and confirm whether paper-facing CSVs are present and populated.

### 42. Which command should a reviewer run to reproduce the paper-facing results?

Report the exact intended command chain to regenerate:
- accepted validation tables,
- Figure 2,
- Figure 3,
- Figure 4,
- Figure 5,
- Table 2,
- Table 3.

### 43. Should the repo add a `REPRODUCE.md` file or strengthen the workflow map?

Based on current repo structure, identify the shortest reproducibility document needed for peer review.

---

## H. Manuscript Language and Claims

### 44. Is first-person plural active voice acceptable and consistent with the manuscript style?

For example:
- “We benchmarked…”
- “We retained concentration-based chemistry…”
- “We used the full activity-coupled path only as a timing boundary…”

### 45. Should the abstract explicitly say the full activity-coupled path is not the routine model?

Would this prevent confusion between the 9.86 s routine ePC-SAFT runtime and the 212.809 s full activity-coupled runtime?

### 46. Should the title mention “ionic” or stay broader?

Current title:

> A Reproducible MEA Absorber Benchmark for ePC-SAFT CO2 Fugacity Driving Forces

Possible revised title:

> A Reproducible MEA Absorber Benchmark for Ionic ePC-SAFT CO2 Fugacity Driving Forces

Which is technically most accurate?

### 47. How aggressive should the “first” claim be?

Assess which version is best supported:
- Strong: “first solved packed-column absorber benchmark to embed a full ionic liquid-side ePC-SAFT fugacity calculation…”
- Safer: “to the authors’ knowledge, the first reproducible MEA absorber benchmark to isolate an ionic ePC-SAFT CO2 fugacity closure under fixed transport, chemistry, and solver settings…”
- Conservative: “a reproducible absorber benchmark that embeds ionic ePC-SAFT fugacity coefficients…”

### 48. Should “improved performance” language be removed?

Given the small MAE difference, should the manuscript avoid saying ePC-SAFT “improves” accuracy unless the improvement is immediately quantified?

### 49. Should “middle ground” be replaced with “controlled sensitivity layer” or another technical phrase?

Identify manuscript phrases such as:
- “middle ground”
- “practical middle ground”
- “right middle ground”

and suggest technically precise replacements.

### 50. Should Sections 3.1 and 3.2 be rewritten aggressively?

These sections currently read like generic tutorials on shooting and finite differences. Should they be replaced with implementation-specific solver details from the code?