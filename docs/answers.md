# Answers To Factual Manuscript Clarification Questions

Repo checked: `MEA-Absorption-Column` on `main` at `12572b85cb4e722a4c0dde8e18c6d0c969263a3a`; full-species C-sweep evidence updated from the committed `full_species_ionic_2017_c_case_sweep.csv` artifact.

## 1. Which exact ePC-SAFT lane produced Figure 4?
Answer: `epcsaft_ionic`.
Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `docs/latex/sections/results.tex`.
Relevant: `thermo_model`; `nccc-one-bed-thermo-benchmark.pdf`.
Confidence: High.

## 2. What should the manuscript call the routine ePC-SAFT model?
Answer: "liquid-side ionic ePC-SAFT fugacity closure with concentration-based chemistry."
Evidence path: `nccc_one_bed_accepted_results.csv`; `src/mea_absorption_column/Thermodynamics/thermo_models.py`.
Relevant: `epcsaft_ionic_fugacity`; `chemical_equilibrium_model=legacy`.
Confidence: High.

## 3. What should the manuscript call the slow 200-350 s path?
Answer: "nine-species activity-coupled ePC-SAFT chemistry and fugacity path."
Evidence path: `docs/full_species_ionic_speciation_handoff.md`; `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`.
Relevant: `epcsaft_reactive_nine_activity_rebased`; `SPECIES_9`.
Confidence: High.

## 4. Does the full activity-coupled path use activity coefficients?
Answer: Yes. It computes species activity coefficients and uses `x_i gamma_i` activities in reaction products.
Evidence path: `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`.
Relevant: `_log_k_from_activity_state`.
Confidence: High.

## 5. What is the exact reaction/speciation basis of the full nine-species path?
Answer: It solves an expanded nine-species equilibrium system including carbonate, hydronium, and hydroxide reactions.
Evidence path: `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`.
Relevant: `SPECIES_9`, `REACTIONS_9`, `REACTION_CONSTANTS_9`.
Confidence: High.

## 6. What exact acceptance gate defines the eight accepted NCCC rows?
Answer: Accepted rows are successful one-bed/no-intercooler rows. Code default gate rejects boundary residual > `1.0`; collocation has no capture gate unless explicitly supplied; no temperature-RMSE or guard-count gate was found. Low-residual collocation can be accepted if residual and capture conditions meet the override.
Evidence path: `src/mea_absorption_column/Run_Model.py`; `analyses/nccc_validation/scripts/generate_nccc_one_bed_artifacts.py`.
Relevant: `_apply_method_success_gates`.
Confidence: High.

## 7. Why did K20 fail?
Answer: K20 failed from maximum mesh nodes exceeded plus domain/profile fallback tied to hydraulics/pressure-drop invalid states.
Evidence path: `nccc_one_bed_all_attempted_results.csv`.
Relevant: K20 rows; `domain_guard_counts`.
Confidence: High.

## 8. Why did 7C fail?
Answer: In the accepted-row attempt artifact, 7C failed by `subprocess_timeout_s=90`. A separate temperature-gallery artifact contains a successful 7C row with high temperature RMSE, so the repo contains two different scopes.
Evidence path: `nccc_one_bed_all_attempted_results.csv`; `c_case_campaign_temperature_overlay_metrics.csv`.
Relevant: 7C rows.
Confidence: High.

## 9. Is the seven-row C-case sweep actually 1C-7C?
Answer: Yes. The committed full-species 2017 C-case sweep contains 1C-7C.
Evidence path: `analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv`.
Relevant: `case_id`; `epcsaft_reactive_nine_activity_rebased`.
Confidence: High.

## 10. What were the full-path capture predictions/errors?
Answer: Committed corrected 2017 C-sweep values: 1C 94.980% (-2.120 p.p.); 2C 88.409% (-3.891); 3C 88.736% (-0.764); 4C 93.747% (+4.847); 5C 93.175% (+6.775); 6C 71.453% (+11.253); 7C 92.914% (+16.514). Mean runtime `171.102 s`.
Evidence path: `analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv`.
Relevant: `epcsaft_reactive_nine_activity_rebased`.
Confidence: High.

## 11. Should the full activity-coupled path be described as validated or feasible?
Answer: Feasible and timing-supported from committed evidence. The committed CSV shows all seven 2017 C rows converged with zero invalid states, zero guard penalties, zero chemistry failures, and residuals below the manuscript tolerance, but the mean runtime is too slow for routine validation sweeps.
Evidence path: `analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv`; `docs/latex/tables/full_ionic_speciation_timing.tex`.
Relevant: `success`; `runtime_s`; residual and guard columns.
Confidence: High.

## 12. Were slow runtimes measured on same environment?
Answer: Yes for the committed slow-path CSV: it records Python `3.13.2`, Windows platform information, package versions, BLAS thread env vars, cache env vars, and the benchmark command for each row.
Evidence path: `analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv`; `nccc_one_bed_accepted_results.csv`.
Relevant: `python_version`, `platform`, `package_versions`.
Confidence: High.

## 13. Do slow-path runtimes include import/setup overhead?
Answer: Subprocess benchmark runtime includes worker startup/import/setup overhead; in-process `run_model` timing measures solver execution only.
Evidence path: `src/mea_absorption_column/benchmark.py`; `src/mea_absorption_column/Run_Model.py`.
Relevant: `_run_solver_settings_subprocess`, `runtime_s`.
Confidence: High.

## 14. What liquid composition vector is passed to routine ePC-SAFT?
Answer: Normalized true-species mole fractions from six-species concentration-based chemistry.
Evidence path: `src/mea_absorption_column/Thermodynamics/thermo_models.py`.
Relevant: `ionic_liquid_composition`, `IONIC_LIQUID_SPECIES_6`.
Confidence: High.

## 15. Is liquid CO2 fugacity based on true molecular CO2 only?
Answer: Yes. `fl_co2 = liquid_x[CO2_INDEX] * phi_l_co2 * P`.
Evidence path: `src/mea_absorption_column/Thermodynamics/thermo_models.py`.
Relevant: `epcsaft_ionic_fugacity`.
Confidence: High.

## 16. What standard state is implicit in activity-coupled solve?
Answer: The rebased path uses `mole_fraction_activity` and calibrates activity constants to the legacy state.
Evidence path: `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`.
Relevant: `epcsaft_reactive_nine_activity_rebased`.
Confidence: High.

## 17. Are CO3^2-, H3O+, and OH- parameters fitted/literature/placeholder?
Answer: Placeholder/diagnostic in the committed parameter summary.
Evidence path: `epcsaft_electrolyte_pure_parameters.csv`.
Relevant: rows for `CO3^2-`, `H3O+`, `OH-`.
Confidence: High.

## 18. What exact ePC-SAFT dataset name should be cited?
Answer: `MEA_CO2_H2O_ionic_fit`.
Evidence path: `epcsaft_electrolyte_config_user_options.csv`.
Relevant: `dataset`.
Confidence: High.

## 19. Does routine campaign use fugacity blending?
Answer: Published accepted rows use `epcsaft_fugacity_blend = 1.0`.
Evidence path: `nccc_one_bed_accepted_results.csv`.
Relevant: `epcsaft_fugacity_blend`.
Confidence: High.

## 20. Were guard/fallback fugacities used?
Answer: Routine accepted ePC-SAFT rows are zero except K19, which has `invalid_state_count=108` and `guard_penalty_count=108`. Full-path handoff reports zero invalid states/domain guards.
Evidence path: `nccc_one_bed_accepted_results.csv`; `docs/full_species_ionic_speciation_handoff.md`.
Relevant: guard/invalid columns.
Confidence: High.

## 21. What is eta_Psi?
Answer: Code value is `0.3`, dimensionless, scaling the `Psi_H` driving-force/enhancement term. Origin is not yet verified.
Evidence path: `src/mea_absorption_column/Transport/Enhancement_Factor.py`; `docs/latex/sections/model_framework.tex`.
Relevant: `Psi_H = Psi / (Psi + H_CO2_mix) * .3`.
Confidence: High for value; AUTHOR VERIFY: provenance.

## 22. Are other model parameters calibrated to NCCC data?
Answer: Accepted rows use global factors `1.0` and no case-specific recovery. Some historical calibration/holdout artifacts remain, but they are not current manuscript-facing evidence.
Evidence path: `primary_validation_gate.csv`; `nccc_one_bed_accepted_results.csv`.
Relevant: `mass_transfer_factor`, `heat_transfer_factor`, `case_specific_recovery`.
Confidence: Medium.

## 23. Is the 45 C assumption justified?
Answer: Author decision: yes, it is a reasonable simple average of nearby inlet temperatures and should be stated as a reconstruction assumption.
Evidence path: manuscript/results context; author response.
Relevant: 1C-3C inlet reconstruction.
Confidence: Medium; AUTHOR VERIFY: exact average source rows.

## 24. Strongest engineering consequence supported?
Answer: Routine ionic ePC-SAFT can be embedded in the full column BVP at near-Henry runtime, with similar capture accuracy; full activity-coupled ePC-SAFT is possible but slow.
Evidence path: `nccc_one_bed_accepted_summary.csv`; `docs/full_species_ionic_speciation_handoff.md`.
Relevant: MAE/runtime summaries.
Confidence: High.

## 25. Can the code output axial fugacity/flux profiles?
Answer: Yes. Profiles include CO2 vapor/liquid fugacity, driving force, flux, enhancement, and related values.
Evidence path: `src/mea_absorption_column/BVP/ABS_Column.py`.
Relevant: `profiles['CO2']`, `molar_flux`.
Confidence: High.

## 26. What exact solver produced accepted NCCC results?
Answer: `scipy-bvp`, paper-facing `Collocation BVP`, implemented with SciPy `solve_bvp`.
Evidence path: `nccc_one_bed_accepted_results.csv`; `Scipy_BVP_Solve.py`.
Relevant: `method`, `solve_bvp`.
Confidence: High.

## 27. Exact solver settings for accepted rows?
Answer: `mesh_points=21`, `max_nodes=1000`, `tol=0.5`, `bc_tol=0.001`, `scaling_mode=legacy_flow_enthalpy`, `transform_mode=bounded_guarded_raw_state`, no continuation path.
Evidence path: `nccc_one_bed_accepted_results.csv`.
Relevant: solver setting columns.
Confidence: High.

## 28. Why is `tol=0.5` large?
Answer: It is applied to scaled solver variables/residuals, not raw physical units.
Evidence path: `Scipy_BVP_Solve.py`; `Run_Model.py`.
Relevant: scaled/solver transform logic.
Confidence: High.

## 29. Does Table 2 "Success" mean solver convergence only?
Answer: Yes for the smooth finite-difference row. It converges numerically but overshoots capture.
Evidence path: `method_case_contrast.csv`.
Relevant: smooth finite-difference row.
Confidence: High.

## 30. Does NCCC 3C finite-difference 0.00% capture mean wrong branch?
Answer: Yes. It is a failed row rejected by strict capture gate despite numerical convergence.
Evidence path: `method_case_contrast.csv`; `nccc_3c_shoot_fd_60s/benchmark_results.csv`.
Relevant: `Rejected by strict capture gate`.
Confidence: High.

## 31. Shooting IVP integrator/root solver?
Answer: Defaults are custom Euler integration and SciPy `root(method='Krylov')`; optional `solve_ivp` modes exist.
Evidence path: `Single_Shoot_Solve.py`.
Relevant: `DEFAULT_SINGLE_SHOOT_SETTINGS`.
Confidence: High.

## 32. Sharp liquid-temperature drop near normalized position 1.0?
Answer: Author decision: explain it as the liquid-inlet/top boundary effect.
Evidence path: coordinate/profile metadata; author response.
Relevant: normalized top boundary.
Confidence: Medium.

## 33. Is normalized position 0 vapor inlet and 1 liquid inlet?
Answer: Yes. Metadata uses bottom-to-top orientation.
Evidence path: `Run_Model.py`.
Relevant: `position_orientation=global_normalized_bottom_to_top`.
Confidence: High.

## 34. Are temperature taps liquid-phase measurements?
Answer: Yes, manuscript and metrics treat them as liquid-temperature taps.
Evidence path: `results.tex`; temperature overlay metrics.
Relevant: `tap_rmse_K`.
Confidence: High.

## 35. Should Figure 4 include failed rows?
Answer: Author decision: no new failure panel; include K20/7C failed-attempt evidence briefly in prose.
Evidence path: `nccc_one_bed_all_attempted_results.csv`; author response.
Relevant: failed-row messages.
Confidence: High.

## 36. Should Table 3 include full-path accuracy?
Answer: Author decision: yes, include accuracy plus timing once CSV evidence is regenerated/committed or accepted.
Evidence path: `docs/full_species_ionic_speciation_handoff.md`; author response.
Relevant: full-path capture/error/runtime values.
Confidence: Medium until CSV is committed.

## 37. What commit hash should manuscript cite?
Answer: Current checked commit: `12572b85cb4e722a4c0dde8e18c6d0c969263a3a`.
Evidence path: `git rev-parse HEAD`.
Relevant: current `main`.
Confidence: High; AUTHOR VERIFY final release commit.

## 38. Zenodo/DOI archive?
Answer: No configured archive metadata found. Author decision: no Zenodo assumed now; cite commit/local package unless release is made.
Evidence path: README/docs search; author response.
Relevant: archive metadata absent.
Confidence: Medium.

## 39. Is external `epcsaft` public?
Answer: Current repo treats it as a local/external companion dependency. Author decision: use commit plus local/external package framing for now.
Evidence path: `README.md`; `analyses/nccc_validation/README.md`.
Relevant: `MEA_EPCSAFT_ROOT`.
Confidence: Medium.

## 40. Should README remove local path language before submission?
Answer: Yes. It contains machine-specific paths.
Evidence path: `README.md`; `docs/workflow_map.md`.
Relevant: `C:\Users\Tanner\...`.
Confidence: High.

## 41. Are curated final CSVs committed and non-empty?
Answer: Final CSVs are tracked and byte-nonempty, but `epcsaft_electrolyte_relative_permittivity_parameters.csv` has zero data rows.
Evidence path: `analyses/nccc_validation/results/final/tables`.
Relevant: final CSV inventory.
Confidence: High.

## 42. Reviewer command chain?
Answer: Commands are spread across `analyses/nccc_validation/README.md`, `analysis.yaml`, scripts, and LaTeX scripts; no single reviewer command exists yet.
Evidence path: `analyses/nccc_validation/README.md`; `analysis.yaml`.
Relevant: benchmark/render/validate scripts.
Confidence: Medium.

## 43. Add REPRODUCE.md?
Answer: Author decision: yes, add `REPRODUCE.md` later as reviewer-facing command chain.
Evidence path: author response.
Relevant: reproducibility plan.
Confidence: High.

## 44. Is first-person plural acceptable?
Answer: Author decision: use direct reviewer-safe prose. Current manuscript is mostly impersonal, so first-person should be used sparingly if at all.
Evidence path: manuscript prose search; author response.
Relevant: style choice.
Confidence: Medium.

## 45. Should abstract say full activity path is not routine?
Answer: Yes. This prevents confusion between `9.86 s` routine ePC-SAFT and the full activity-coupled path, which now has a committed seven-row 2017 C sweep with `171.102 s` mean runtime.
Evidence path: `main.tex`; `results.tex`; author response.
Relevant: abstract/runtime claims.
Confidence: High.

## 46. Should title mention ionic?
Answer: Author decision: keep broad title and explain ionic liquid-side details in abstract/methods.
Evidence path: author response; `main.tex`.
Relevant: title framing.
Confidence: High.

## 47. How aggressive should the first claim be?
Answer: Author decision: cautious "to the authors' knowledge" first claim.
Evidence path: author response.
Relevant: novelty framing.
Confidence: High.

## 48. Remove "improved performance" language?
Answer: Yes, avoid unqualified improvement language because MAE difference is small.
Evidence path: `nccc_one_bed_accepted_summary.csv`.
Relevant: ePC-SAFT MAE `3.725`, Henry MAE `3.775`.
Confidence: High.

## 49. Replace "middle ground"?
Answer: Yes. Replace with technical phrasing such as "structured fugacity-coefficient calculation with near-Henry runtime."
Evidence path: `main.tex`; `results.tex`; author response.
Relevant: "middle ground" occurrences.
Confidence: High.

## 50. Rewrite Sections 3.1 and 3.2 aggressively?
Answer: Author decision: use direct reviewer-safe prose and lightly tighten with implementation-specific solver details, not a full aggressive rewrite.
Evidence path: author response; solver source files.
Relevant: `Single_Shoot_Solve.py`, `Finite_Difference_Solve.py`.
Confidence: High.
