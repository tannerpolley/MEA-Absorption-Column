# Fact-Finding Answers for ChatGPT_Questions.md

Source question file: `docs/ChatGPT_Questions.md`

This document answers the 50 factual clarification questions from the repository, manuscript source, scripts, and committed result artifacts. It does not rewrite the manuscript or resolve conflicts silently.

## Major Inconsistencies Found

- Accepted validation scope is inconsistent across artifacts. `docs/latex/sections/results.tex` describes accepted one-bed ePC-SAFT validation as `K18`, `K19`, and `1C--6C`, while campaign overlay artifacts include `1C--7C`.
- ePC-SAFT dataset naming depends on workflow context. Paper-facing workflow/results cite `MEA_CO2_H2O_ionic_fit`, while the code default in `thermo_models.py` falls back to `MEA_CO2_H2O_draft` unless the dataset environment/configuration is applied.
- Full nine-species slow-path values are present in `docs/full_species_ionic_speciation_handoff.md` and `docs/latex/tables/full_ionic_speciation_timing.tex`, but the referenced raw run CSV under `analyses/nccc_validation/results/runs/full_species_ionic_all_c_cases/benchmark_results.csv` is not present in this checkout.
- Some accepted-result rows reference profile CSV output directories, but those profile CSV directories are absent from this checkout. The code can export the profiles, but the current final artifact set does not contain all referenced dense profile CSVs.

## 1. Which exact ePC-SAFT lane produced Figure 4?

Answer: Figure 4's accepted-row ePC-SAFT comparison uses `thermo_model = epcsaft_ionic`, not `epcsaft_neutral`.

Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_summary.csv`.

Relevant variable/function/script/table/artifact: `thermo_model`; `thermo_label=ePC-SAFT`; `accepted_rows=8`; `nccc_one_bed_accepted_results.csv`.

Confidence: High.

## 2. What should the manuscript call the routine ePC-SAFT model?

Answer: The technically supported phrase is "liquid-side ionic ePC-SAFT fugacity closure with concentration-based chemistry." The routine campaign does not use full activity-coupled chemistry; it keeps the legacy concentration-based chemistry and swaps the CO2 fugacity closure.

Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `src/mea_absorption_column/Thermodynamics/thermo_models.py`; `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`; `docs/latex/sections/methods.tex`.

Relevant variable/function/script/table/artifact: `thermo_model=epcsaft_ionic`; `chemical_equilibrium_model=legacy`; `epcsaft_ionic_fugacity()`; `chemical_equilibrium()`.

Confidence: High.

## 3. What should the manuscript call the slow 200-350 s path?

Answer: The most exact code-backed label is "nine-species activity-rebased ePC-SAFT reactive speciation plus ionic fugacity path." The code label is `epcsaft_reactive_nine_activity_rebased`.

Evidence path: `docs/full_species_ionic_speciation_handoff.md`; `docs/latex/tables/full_ionic_speciation_timing.tex`; `src/mea_absorption_column/benchmark.py`; `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`.

Relevant variable/function/script/table/artifact: `epcsaft_reactive_nine_activity_rebased`; `tab:full-ionic-speciation-timing`; `chemical_equilibrium_model`.

Confidence: Medium-High.

AUTHOR VERIFY: The handoff and LaTeX table contain the slow-path values, but the referenced raw run CSV is not present in this checkout.

## 4. Does the full activity-coupled path use activity coefficients in the chemical-equilibrium equations?

Answer: Yes for the repo-side setup. The rebased activity path computes species activity coefficients `gamma_i`, forms activity-like `x_i * gamma_i` terms, and passes reactions to the native ePC-SAFT solver with `standard_state="mole_fraction_activity"`. The exact internal residual equations are inside the external `epcsaft` package, not this repository.

Evidence path: `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`; `analyses/nccc_validation/results/final/reports/reactive_epcsaft_claim_boundary.md`.

Relevant variable/function/script/table/artifact: `_log_k_from_activity_state()`; `epcsaft_reactive_chemical_equilibrium()`; `epcsaft.solve_reactive_speciation()`; `ReactionDefinition`; `standard_state`.

Confidence: High for repository-side setup; Medium for external native-solver internals.

AUTHOR VERIFY: The native `epcsaft` package internals are not committed in this repository.

## 5. What is the exact reaction/speciation basis of the full nine-species path?

Answer: It solves an expanded nine-species reactive system with five reactions: water autoionization, CO2/HCO3, HCO3/CO3, MEACOO hydrolysis, and MEAH dissociation. The native solver handles charge closure; the absorber/profile export keeps a six-species compatibility view for existing column code.

Evidence path: `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`; `docs/full_species_ionic_speciation_handoff.md`.

Relevant variable/function/script/table/artifact: `SPECIES_9`; `REACTIONS_9`; `REACTION_CONSTANTS_9`; `_reactive_balances("nine")`.

Confidence: High.

## 6. What exact acceptance gate defines the eight accepted NCCC rows?

Answer: The eight accepted one-bed rows are `K18`, `K19`, and `1C--6C` for each of `ideal_henry` and `epcsaft_ionic`; `K20` and `7C` are excluded from that accepted table. The code gate requires solver success unless a low-residual collocation override applies; default boundary residual maximum is `1.0`; `success_capture_error_max_pct` is optional and only applies when set; no temperature-RMSE threshold is implemented in the acceptance script; there is no guard/fallback-count requirement because accepted `K19` contains guard counts. The separate primary validation gate is narrower: six C rows, common settings, no case-specific tuning, mass-transfer and heat-transfer factors equal to `1.0`.

Evidence path: `src/mea_absorption_column/Run_Model.py`; `analyses/nccc_validation/scripts/validate_results.py`; `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `analyses/nccc_validation/results/final/tables/primary_validation_gate.csv`; `analyses/nccc_validation/results/final/tables/validation_evidence_registry.csv`.

Relevant variable/function/script/table/artifact: `_is_successful_result()`; `success_boundary_residual_max`; `success_capture_error_max_pct`; `primary_validation_gate.csv`; `validation_evidence_registry.csv`.

Confidence: High.

## 7. Why did K20 fail?

Answer: In `nccc_one_bed_all_attempted_results.csv`, K20 failed because SciPy BVP exceeded the maximum mesh nodes and then profile export used temperature-only fallback after invalid hydraulics (`Fl_T` nonpositive/nonfinite). The ePC-SAFT K20 row also has physically bad capture (`115.134%`, error `+19.584` percentage points), `invalid_state_count=39092`, `guard_penalty_count=32720`, and domain guard counts for hydraulics and pressure drop. It is not recorded as a timeout or ePC-SAFT fugacity failure.

Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_all_attempted_results.csv`.

Relevant variable/function/script/table/artifact: `message`; `capture_pct`; `capture_error_pct`; `invalid_state_count`; `guard_penalty_count`; `domain_guard_counts`; `first_failed_domain`.

Confidence: High.

## 8. Why did 7C fail?

Answer: It depends on which artifact is treated as controlling. In `nccc_one_bed_all_attempted_results.csv`, 7C fails by subprocess timeout: `Benchmark subprocess exceeded subprocess_timeout_s=90` for both Henry and ePC-SAFT, with no accepted capture/residual values. In `c_case_campaign_temperature_overlay_metrics.csv`, 7C later appears as a successful campaign-overlay row with ePC-SAFT runtime `19.253 s`, capture `72.981%`, capture error `-3.419` percentage points, and temperature RMSE `16.128 K`.

Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_all_attempted_results.csv`; `analyses/nccc_validation/results/final/tables/c_case_campaign_temperature_overlay_metrics.csv`.

Relevant variable/function/script/table/artifact: `subprocess_timeout_s`; `success`; `capture_pct`; `temperature_rmse_K`; `c_case_campaign_temperature_overlay_metrics.csv`.

Confidence: High for the artifact conflict.

AUTHOR VERIFY: Decide which 7C status should control manuscript-facing accepted validation language.

## 9. Is the "seven-row C-case sweep" in Table 3 actually 1C-7C?

Answer: Yes, the full nine-species activity sweep is documented as `1C--7C` from the legacy `C_cases_data.csv` C-case set.

Evidence path: `docs/full_species_ionic_speciation_handoff.md`; `docs/latex/tables/full_ionic_speciation_timing.tex`.

Relevant variable/function/script/table/artifact: `full_species_ionic_all_c_cases`; `C_cases_data`; `epcsaft_reactive_nine_activity_rebased`.

Confidence: Medium.

AUTHOR VERIFY: The handoff references a raw run CSV that is not present in this checkout.

## 10. What were the capture predictions and errors for the full activity-coupled path across the seven C rows?

Answer: The handoff reports `epcsaft_reactive_nine_activity_rebased` values as follows: `1C` predicted `79.258%`, measured `97.1%`, error `-17.842` points, runtime `336.655 s`; `2C` predicted `86.651%`, measured `92.3%`, error `-5.649` points, runtime `289.707 s`; `3C` predicted `89.855%`, measured `89.5%`, error `+0.355` points, runtime `281.664 s`; `4C` predicted `94.678%`, measured `88.9%`, error `+5.778` points, runtime `308.746 s`; `5C` predicted `90.525%`, measured `86.4%`, error `+4.125` points, runtime `444.874 s`; `6C` predicted `70.549%`, measured `60.2%`, error `+10.349` points, runtime `420.357 s`; `7C` predicted `98.515%`, measured `76.4%`, error `+22.115` points, runtime `380.387 s`. MAE is reported as `9.459` percentage points; mean runtime is `351.770 s`; median runtime from the listed rows is `336.655 s`. The run used relaxed feasibility settings: `mesh-points 7`, `tol 10`, `bc-tol 0.5`, `max-nodes 80`, and timeout `900 s`.

Evidence path: `docs/full_species_ionic_speciation_handoff.md`; `docs/latex/tables/full_ionic_speciation_timing.tex`.

Relevant variable/function/script/table/artifact: `epcsaft_reactive_nine_activity_rebased`; `benchmark_results.csv` referenced by handoff; `tab:full-ionic-speciation-timing`.

Confidence: Medium.

AUTHOR VERIFY: Values are from the handoff/table; the referenced raw run CSV is missing from this checkout.

## 11. Should the full activity-coupled path be described as "validated" or only "feasible"?

Answer: Only feasible. The full nine-species path used relaxed feasibility settings and did not use the same validation gate as the routine campaign. The routine validation set uses common settings and accepted rows; the full activity-coupled path is presented as a timing and feasibility boundary.

Evidence path: `docs/latex/tables/full_ionic_speciation_timing.tex`; `docs/full_species_ionic_speciation_handoff.md`; `analyses/nccc_validation/results/final/reports/reactive_epcsaft_claim_boundary.md`; `analyses/nccc_validation/results/final/tables/primary_validation_gate.csv`.

Relevant variable/function/script/table/artifact: `tab:full-ionic-speciation-timing`; `primary_validation_gate.csv`; `reactive_epcsaft_claim_boundary.md`.

Confidence: High.

## 12. Were the 212.809 s and 351.770 s runtimes measured on the same hardware and Python environment as the 8.62 s and 9.86 s routine medians?

Answer: Not directly verifiable. Routine accepted rows record `Python 3.13.2`, `Windows-11-10.0.26200-SP0`, and package versions including `numpy=2.4.4`, `pandas=3.0.2`, `scipy=1.17.1`, and `matplotlib=3.10.9`. The slow-path handoff records commands and timings, but not platform/package versions, and the referenced slow-run CSV is missing.

Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `docs/full_species_ionic_speciation_handoff.md`.

Relevant variable/function/script/table/artifact: `python_version`; `platform`; `package_versions`; `runtime_s`; `full_species_ionic_all_c_cases`.

Confidence: Medium-Low.

AUTHOR VERIFY: Confirm whether the slow-path timings were run on the same hardware and Python/package environment as the routine medians.

## 13. Do the slow-path runtimes include import/setup overhead, or only model solve time?

Answer: For subprocess benchmark runs, `runtime_s` is parent-process elapsed wall time from worker launch to output read, so it includes worker process startup/import/data-load overhead plus solve/output time. For direct `run_model` calls, `runtime_s` is measured around the BVP solve section. The slow-path handoff also reports chemistry solve time separately for the Case 3C proof, e.g. `160.191 s` inside a `212.809 s` total.

Evidence path: `src/mea_absorption_column/benchmark.py`; `src/mea_absorption_column/benchmark_worker.py`; `src/mea_absorption_column/Run_Model.py`; `docs/full_species_ionic_speciation_handoff.md`.

Relevant variable/function/script/table/artifact: `_run_one_case_subprocess()`; `benchmark_worker.main()`; `run_model()`; `runtime_s`; `epcsaft_chemistry_solve_s`.

Confidence: High.

## 14. In the routine ionic ePC-SAFT fugacity calculation, what composition vector is passed to ePC-SAFT for the liquid phase?

Answer: The routine `epcsaft_ionic` lane passes normalized six true-species mole fractions from the concentration-based chemistry solve: `CO2`, `MEA`, `H2O`, `MEAH+`, `MEACOO-`, and `HCO3-`.

Evidence path: `src/mea_absorption_column/BVP/ABS_Column.py`; `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`; `src/mea_absorption_column/Thermodynamics/thermo_models.py`; `docs/workflow_map.md`.

Relevant variable/function/script/table/artifact: `chemical_equilibrium_with_model(... model="legacy")`; `ionic_liquid_composition()`; `epcsaft_ionic_fugacity()`.

Confidence: High.

## 15. In the routine ePC-SAFT comparison, is the liquid CO2 fugacity based on true molecular CO2 only?

Answer: Yes. Liquid CO2 fugacity is based on true molecular CO2 from the speciation solve: `x_true_CO2 * phi_l_CO2 * P`. Vapor CO2 is computed as `y_CO2 * phi_v_CO2 * P`. H2O in the routine flux model remains vapor-pressure based, not ePC-SAFT fugacity based.

Evidence path: `src/mea_absorption_column/Thermodynamics/thermo_models.py`; `src/mea_absorption_column/BVP/ABS_Column.py`.

Relevant variable/function/script/table/artifact: `epcsaft_ionic_fugacity()`; `x_l_CO2`; `phi_l_CO2`; `fv_CO2`; `fl_CO2`; `P_sat_H2O`.

Confidence: High.

## 16. What is the thermodynamic standard state implicit in the activity-coupled chemical-equilibrium solve?

Answer: `epcsaft_reactive_nine_activity_rebased` uses `standard_state="mole_fraction_activity"` with `calibrate_activity_to_legacy=True`. The constants are rebased from the local activity state to reproduce or align with the legacy state; they are not independently regressed predictive activity-basis constants.

Evidence path: `src/mea_absorption_column/Thermodynamics/Chemical_Equilibrium.py`; `analyses/nccc_validation/results/final/reports/reactive_epcsaft_claim_boundary.md`.

Relevant variable/function/script/table/artifact: `standard_state`; `calibrate_activity_to_legacy`; `_log_k_from_activity_state()`; `epcsaft_reactive_chemical_equilibrium()`.

Confidence: High.

## 17. Are the ePC-SAFT parameters for CO3^2-, H3O+, and OH- fitted, literature-based, or placeholder/diagnostic?

Answer: They are placeholder/diagnostic auxiliary carbonate/water-ion parameters in the repo-vendored dataset, not shown as fitted or literature-based parameters. The reports describe `d_born=3 A` for those auxiliary ions as a hydrated-ion-scale assumption.

Evidence path: `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_pure_parameters.csv`; `analyses/nccc_validation/results/final/reports/epcsaft_electrolyte_column_config_matrix.md`.

Relevant variable/function/script/table/artifact: `component`; `source_note`; `CO3^2-`; `H3O+`; `OH-`; `d_born`.

Confidence: High.

## 18. What exact ePC-SAFT dataset name should be cited?

Answer: Cite `MEA_CO2_H2O_ionic_fit` for the paper-facing ePC-SAFT ionic-fit results.

Evidence path: `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_pure_parameters.csv`; `analyses/nccc_validation/analysis.yaml`; `docs/workflow_map.md`; `analyses/nccc_validation/results/final/reports/validation_summary.md`.

Relevant variable/function/script/table/artifact: `MEA_EPCSAFT_DATASET_NAME`; `MEA_CO2_H2O_ionic_fit`; `epcsaft_electrolyte_pure_parameters.csv`.

Confidence: High.

AUTHOR VERIFY: The code default in `thermo_models.py` is `MEA_CO2_H2O_draft`; confirm that manuscript/reproduction commands always select the paper-facing dataset.

## 19. Does the ePC-SAFT routine campaign ever use fugacity blending?

Answer: The published accepted ePC-SAFT rows use `epcsaft_fugacity_blend = 1.0`; no Henry/ePC-SAFT blend is used for those published rows.

Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `src/mea_absorption_column/Run_Model.py`; `src/mea_absorption_column/benchmark.py`; `README.md`.

Relevant variable/function/script/table/artifact: `epcsaft_fugacity_blend`; `--epcsaft-fugacity-blend`.

Confidence: High.

## 20. Were any guard or fallback fugacities used in the accepted ePC-SAFT rows?

Answer: Routine accepted `epcsaft_ionic` rows contain eight rows total. Across those rows, `invalid_state_count=108` and `guard_penalty_count=108`, all from `K19`; `domain_guard_counts` is blank; chemistry best-effort and failed counts are zero. The primary accepted C-case rows `1C--6C` have zero guard penalties. The full Case 3C handoff reports `Invalid states=0` and no domain guards, and the full seven-row sweep handoff reports all rows with `Invalid states=0` and `Domain guards=0`.

Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `analyses/nccc_validation/results/final/tables/nccc_one_bed_all_attempted_results.csv`; `docs/full_species_ionic_speciation_handoff.md`.

Relevant variable/function/script/table/artifact: `invalid_state_count`; `guard_penalty_count`; `domain_guard_counts`; `epcsaft_chemistry_accepted_best_effort_count`; `epcsaft_chemistry_failed_count`.

Confidence: High for routine rows; Medium for full-path rows.

AUTHOR VERIFY: Full-path counts are from the handoff because the referenced raw run CSVs are missing from this checkout.

## 21. What is the numerical value of eta_Psi in Equation 45?

Answer: `eta_Psi = 0.3`. It is dimensionless in the code expression `Psi_H = Psi / (Psi + H_CO2_mix) * .3`. It is used as a fixed global driving-force scale in the enhancement/flux path, and Henry-law and ePC-SAFT use the same enhancement/flux code path.

Evidence path: `src/mea_absorption_column/Transport/Enhancement_Factor.py`; `docs/latex/sections/model_framework.tex`.

Relevant variable/function/script/table/artifact: `Psi_H`; `enhancement_factor`; `H_CO2_mix`; Equation 45 context in `model_framework.tex`.

Confidence: High for value and code use; Low for provenance.

AUTHOR VERIFY: Treat as a fixed dimensionless heuristic/global factor unless the author can provide fitting or legacy-source provenance.

## 22. Are any other model parameters calibrated to NCCC data?

Answer: The final accepted primary rows show no case-specific tuning: `mass_transfer_factor=1.0`, `heat_transfer_factor=1.0`, and `no_case_specific_tuning=True`. I found no final accepted-row evidence that Henry-law deviation coefficients, enhancement scale, transfer/heat multipliers, inlet assumptions, or ePC-SAFT parameters were calibrated to NCCC data. A separate `three_term_global_residual_screen` exists, but its artifacts label it as screening/not final calibration.

Evidence path: `analyses/nccc_validation/results/final/tables/primary_validation_gate.csv`; `analyses/nccc_validation/results/final/tables/calibration_holdout_predictions.csv`; `src/mea_absorption_column/calibration.py`; `analyses/nccc_validation/results/final/reports/validation_summary.md`.

Relevant variable/function/script/table/artifact: `mass_transfer_factor`; `heat_transfer_factor`; `no_case_specific_tuning`; `three_term_global_residual_screen`; `calibration_scope`.

Confidence: High for final accepted artifacts.

AUTHOR VERIFY: `eta_Psi` provenance is not directly documented.

## 23. Is the 45.0 deg C lean-inlet assumption for 1C-3C independently justified?

Answer: Partly supported, but not independently validated. The source-preserving table leaves `1C`, `2C`, `3C`, and `3D` blank for lean inlet temperature; the run-ready model input imputes `45.0 deg C` and flags `lean_solvent_temp_imputed=True`. The manuscript says this is consistent with nearby campaign measured lean temperatures. I found no committed sensitivity check proving the assumption does not affect the conclusions.

Evidence path: `docs/nccc_campaign_case_mapping.md`; `analyses/nccc_validation/scripts/extract_nccc_case_catalog_from_markdown.py`; `docs/latex/tables/nccc_one_bed_case_scope.tex`; `docs/latex/sections/results.tex`.

Relevant variable/function/script/table/artifact: `lean_solvent_temp_C`; `lean_solvent_temp_imputed`; `C_cases_campaign_inputs.csv`; `nccc_one_bed_case_scope.tex`.

Confidence: High for the imputation; Medium-Low for independent justification.

AUTHOR VERIFY: Provide sensitivity evidence or source documentation if the manuscript should claim independent justification.

## 24. What is the strongest engineering consequence supported by the result artifacts?

Answer: The strongest supported claim is that routine ionic ePC-SAFT can be embedded at practical repeated-run timescale and gives nearly the same capture accuracy as Henry-law closure, so its current value is controlled thermodynamic interpretability rather than a clear accuracy gain. The artifacts also support that full activity-coupled ePC-SAFT is feasible but too slow for routine validation sweeps.

Evidence path: `docs/latex/main.tex`; `docs/latex/sections/results.tex`; `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_summary.csv`; `analyses/nccc_validation/results/final/tables/plot_c_case_thermo_summary.csv`; `docs/latex/tables/full_ionic_speciation_timing.tex`.

Relevant variable/function/script/table/artifact: capture MAE; median runtime; `epcsaft_ionic`; `ideal_henry`; `tab:full-ionic-speciation-timing`.

Confidence: High.

## 25. Can the code output axial profiles of fugacity driving force or CO2 flux?

Answer: Yes, the code can export axial profiles containing vapor CO2 fugacity, liquid CO2 fugacity, CO2 driving force, CO2 fluxes, and enhancement fields. Fields include `fv_CO2`, `fl_CO2`, `DF_CO2`, `Nl_CO2`, `Nv_CO2`, `Psi`, `Psi_H`, and `E`.

Evidence path: `src/mea_absorption_column/BVP/ABS_Column.py`; `src/mea_absorption_column/misc/Save_Run_Outputs.py`; `analyses/nccc_validation/README.md`.

Relevant variable/function/script/table/artifact: `property_data`; `save_run_outputs()`; `build_profile_coordinate_frame()`; `run_case_profile.py`; `profile_csv_files`.

Confidence: High for code capability.

AUTHOR VERIFY: Current final artifacts do not contain all dense profile CSV directories referenced by result rows.

## 26. What exact solver produced the accepted NCCC results?

Answer: Accepted NCCC rows use method string `scipy-bvp`, mapped to `scipy_BVP_solve`, which calls SciPy `solve_bvp`. Manuscript/reporting language often calls this `Collocation BVP`, and internal messaging refers to a SciPy collocation-style BVP.

Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `src/mea_absorption_column/Run_Model.py`; `src/mea_absorption_column/BVP/Methods/Scipy_BVP_Solve.py`.

Relevant variable/function/script/table/artifact: `method=scipy-bvp`; `run_model()`; `scipy_BVP_solve()`; `solve_bvp`.

Confidence: High.

## 27. What are the exact solver settings for the accepted rows?

Answer: Accepted rows record `mesh_points=21`, `tol=0.5`, `bc_tol=0.001`, `max_nodes=1000`, `scaling_mode=legacy_flow_enthalpy`, `transform_mode=bounded_guarded_raw_state`, `continuation_stage=direct`, `continuation_success=True`, `co2_capture_guess_pct=95.0`, `h2o_capture_guess_pct=-100.0`, and `epcsaft_fugacity_blend=1.0`. The 2017 C rows also show `vapor_composition_mode=input_o2`. Because accepted rows are direct and no explicit `initial_guess_scaled` is recorded, `scipy_BVP_solve()` falls back to `polynomial_fit(...) / scales` initial profiles.

Evidence path: `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `src/mea_absorption_column/benchmark.py`; `src/mea_absorption_column/BVP/Methods/Scipy_BVP_Solve.py`; `analyses/nccc_validation/README.md`.

Relevant variable/function/script/table/artifact: `mesh_points`; `tol`; `bc_tol`; `max_nodes`; `scales`; `polynomial_fit`; `subprocess_timeout_s`.

Confidence: High for settings in accepted CSV; Medium for timeout.

AUTHOR VERIFY: The accepted CSV does not record the definitive timeout used for those accepted rows; examples show 120 s and 60 s in different workflows.

## 28. Why is the collocation tolerance apparently large if tol = 0.5 is the paper setting?

Answer: The tolerance is applied in scaled/transformed solver space, not directly as raw physical-unit error or capture error. `Run_Model` computes state scales, boundary conditions are expressed in scaled variables, and `ABS_Column` returns scaled derivatives. Accepted rows still report very small boundary residual norms, often around `1e-13`.

Evidence path: `src/mea_absorption_column/Run_Model.py`; `src/mea_absorption_column/misc/Scaling.py`; `src/mea_absorption_column/BVP/ABS_Column.py`; `src/mea_absorption_column/BVP/Methods/Scipy_BVP_Solve.py`; `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`.

Relevant variable/function/script/table/artifact: `get_scaling_factors()`; `scales`; `solver_to_scaled_physical()`; `solve_bvp(tol=tol, bc_tol=bc_tol)`; `boundary_residual_norm`.

Confidence: High.

## 29. For Table 2, does "Success" mean solver convergence only?

Answer: It should not be read as full physical validation. In the smooth finite-difference row, `Success=Yes` with `107.65%` capture means the algebraic solve converged or reached a low residual, but the capture prediction overshoots physically. It is solver/method status, not primary validation acceptance.

Evidence path: `docs/latex/tables/method_case_contrast.tex`; `analyses/nccc_validation/results/final/tables/method_case_contrast.csv`; `analyses/nccc_validation/results/final/reports/solver_method_contrast_srp_3c.md`; `src/mea_absorption_column/Run_Model.py`.

Relevant variable/function/script/table/artifact: `method_case_contrast.csv`; `Success`; `capture_pct`; `interpretation`; `_is_successful_result()`.

Confidence: High.

AUTHOR VERIFY: If the phrase "failed physical acceptance" is meant as a formal gate for the smooth row, the repo only directly supports "not physical validation / overshoots capture."

## 30. For the NCCC 3C finite-difference row, does 0.00% capture mean a converged but physically wrong branch, or a failed run with a placeholder value?

Answer: The paper-facing table and report describe it as a converged or boundary-satisfying but physically wrong zero-capture branch, not a placeholder. However, `method_slice_3c.csv` says the finite-difference run timed out with blank capture, while `method_case_contrast.csv` reports runtime `16.61`, capture `0.0`, and a physically wrong branch interpretation.

Evidence path: `docs/latex/tables/method_case_contrast.tex`; `analyses/nccc_validation/results/final/tables/method_case_contrast.csv`; `analyses/nccc_validation/results/final/tables/method_slice_3c.csv`; `analyses/nccc_validation/results/final/reports/solver_method_contrast_srp_3c.md`.

Relevant variable/function/script/table/artifact: `method_case_contrast.csv`; `method_slice_3c.csv`; `capture_pct`; `interpretation`.

Confidence: Medium.

AUTHOR VERIFY: Resolve the raw provenance conflict between `method_slice_3c.csv` and `method_case_contrast.csv`.

## 31. For shooting, which IVP integrator and root solver were used for the reported rows?

Answer: Code defaults for shooting are custom Euler integration (`eulers`) and `scipy.optimize.root(..., method="Krylov")`. If the CLI passes `--shooting-integrator bdf|radau|rk45|solve_ivp`, the code uses SciPy `solve_ivp` with the selected IVP method. The final method table does not record `integrator`, `root_method`, or `ivp_method` columns for the historical reported rows.

Evidence path: `src/mea_absorption_column/BVP/Methods/Single_Shoot_Solve.py`; `analyses/nccc_validation/README.md`; `analyses/nccc_validation/results/final/tables/method_case_contrast.csv`.

Relevant variable/function/script/table/artifact: `single_shoot_solve()`; `eulers`; `root(..., method="Krylov")`; `solve_ivp`; `--shooting-integrator`.

Confidence: Medium-High for code defaults; Medium for exact historical table rows.

AUTHOR VERIFY: Confirm the exact integrator/root settings used to generate the final published shooting rows if they differed from defaults.

## 32. What causes the sharp liquid-temperature drop near normalized position 1.0 in Figures 2 and 3?

Answer: The most directly supported cause is the top lean-liquid inlet boundary at `z=1`, combined with plotting liquid profiles against bottom-to-top normalized position. It is not interpolation through a tap point; interpolation is used for metrics, while the figure plots `profile["Tl"]` directly.

Evidence path: `src/mea_absorption_column/Run_Model.py`; `src/mea_absorption_column/BVP/Methods/Scipy_BVP_Solve.py`; `analyses/nccc_validation/scripts/render_c_case_campaign_temperature_gallery.py`; `docs/latex/sections/results.tex`.

Relevant variable/function/script/table/artifact: `global_normalized_bottom_to_top`; `profile["Tl"]`; lean-liquid inlet boundary; temperature-profile overlay scripts.

Confidence: Medium-High.

AUTHOR VERIFY: Confirm whether the manuscript should call this a physical inlet-boundary feature rather than a real internal thermal feature.

## 33. Is normalized column position 0 the vapor inlet and 1 the liquid inlet?

Answer: Yes. Normalized position `0` is bottom/vapor inlet and `1` is top/liquid inlet.

Evidence path: `docs/latex/sections/model_framework.tex`; `src/mea_absorption_column/Run_Model.py`; `src/mea_absorption_column/misc/Save_Run_Outputs.py`.

Relevant variable/function/script/table/artifact: `a=0`; `b=H`; `global_normalized_bottom_to_top`; `build_profile_coordinate_frame()`.

Confidence: High.

## 34. Are the temperature taps liquid-phase measurements only?

Answer: Yes for Figures 2 and 3 as written. The manuscript captions call the markers measured liquid-temperature taps, and the plotting code labels them `NCCC liquid taps`.

Evidence path: `docs/latex/sections/results.tex`; `analyses/nccc_validation/scripts/render_c_case_campaign_temperature_gallery.py`; `analyses/nccc_validation/scripts/collect_clean_profiles.py`.

Relevant variable/function/script/table/artifact: `tap_columns`; `NCCC liquid taps`; `temperature_profile` plots.

Confidence: High.

## 35. Should Figure 4 include failed rows in a separate panel or should failures stay in a table?

Answer: Based on available artifacts, failures should stay in a table/report unless reduced to a compact diagnostic panel. The Figure 4 artifact generator filters to successful rows, while failed-row messages exist in `nccc_one_bed_all_attempted_results.csv`. Existing method-contrast plotting can label failed bars, but the detailed failure reasons are cleaner in tables or reports.

Evidence path: `analyses/nccc_validation/scripts/generate_nccc_one_bed_artifacts.py`; `analyses/nccc_validation/results/final/tables/nccc_one_bed_all_attempted_results.csv`; `analyses/nccc_validation/scripts/generate_accuracy_credibility_artifacts.py`; `docs/latex/sections/results.tex`.

Relevant variable/function/script/table/artifact: `results[results["success"]]`; `nccc_one_bed_all_attempted_results.csv`; `failure/message` fields.

Confidence: High.

## 36. Should Table 3 include full-path accuracy, not only runtime?

Answer: Yes, the repository contains enough handoff values to add full-path accuracy, including Case 3C capture `89.855%`, error `+0.355` points, runtime `212.809 s`, seven-row mean runtime `351.770 s`, seven-row MAE `9.459` points, and per-case capture/error values. The current LaTeX table reports timing/scope text only.

Evidence path: `docs/full_species_ionic_speciation_handoff.md`; `docs/latex/tables/full_ionic_speciation_timing.tex`.

Relevant variable/function/script/table/artifact: `full_ionic_speciation_timing.tex`; `epcsaft_reactive_nine_activity_rebased`; seven C-row handoff table.

Confidence: High for availability in handoff; Medium for raw provenance.

AUTHOR VERIFY: The per-case values are not backed by the referenced raw run CSV in this checkout.

## 37. What commit hash should the manuscript cite?

Answer: The current HEAD in this worktree is `12572b85cb4e722a4c0dde8e18c6d0c969263a3a` on branch `answers-v3`, with no upstream configured.

Evidence path: Git metadata read with `git rev-parse HEAD`, `git branch --show-current`, and `git log -1`.

Relevant variable/function/script/table/artifact: current Git commit; current branch.

Confidence: High for current checkout.

AUTHOR VERIFY: Confirm whether this is the intended release/submission commit.

## 38. Will the repo be archived with Zenodo or another DOI service before submission?

Answer: No archive plan is configured in the committed files I inspected. I found no `.zenodo.json`, `CITATION.cff`, or codemeta file. The code availability text cites GitHub/software references but does not provide a DOI.

Evidence path: `docs/latex/sections/code_availability.tex`; `docs/latex/software_references.bib`; `docs/NCE_submission_report.md`; repository file search for `.zenodo*`, `CITATION*`, and `codemeta*`.

Relevant variable/function/script/table/artifact: code availability section; software bibliography entry; missing Zenodo/CITATION metadata.

Confidence: High for current repo state.

AUTHOR VERIFY: Confirm intended archive/DOI plan before submission.

## 39. Is the external epcsaft package public?

Answer: From this repository alone, `epcsaft` is an external/local dependency, not a declared PyPI dependency and not vendored as package code. Parameter datasets are vendored, but package code is expected from a local checkout or installed external package. The repo uses examples such as `uv pip install 'C:\Users\Tanner\Documents\git\ePC-SAFT'` and an import fallback via `MEA_EPCSAFT_ROOT`.

Evidence path: `pyproject.toml`; `README.md`; `docs/workflow_map.md`; `src/mea_absorption_column/Thermodynamics/thermo_models.py`.

Relevant variable/function/script/table/artifact: `MEA_EPCSAFT_ROOT`; `uv pip install`; `import epcsaft`; vendored `epcsaft_datasets`.

Confidence: High for local/external dependency status.

AUTHOR VERIFY: Confirm whether the external `ePC-SAFT` repository/package is public, archived, or intended to be released.

## 40. Should the README remove local path language before submission?

Answer: Yes. The README and workflow map contain machine-specific paths that should be replaced or generalized before submission, including the Overleaf checkout path, local `uv pip install` path for ePC-SAFT, and local `MEA_EPCSAFT_ROOT` examples.

Evidence path: `README.md`; `docs/workflow_map.md`; `analyses/nccc_validation/README.md`.

Relevant variable/function/script/table/artifact: `C:\Users\Tanner\Documents\git\LaTeX-Projects\MEA-Absorption-Column-LaTeX`; `C:\Users\Tanner\Documents\git\ePC-SAFT`; `MEA_EPCSAFT_ROOT`.

Confidence: High.

## 41. Are all curated final CSVs intentionally committed and non-empty?

Answer: Final CSVs are committed/tracked and most are populated, but not all are non-empty. `epcsaft_electrolyte_relative_permittivity_parameters.csv` is header-only with zero data rows.

Evidence path: `analyses/nccc_validation/results/final/tables/`; `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_relative_permittivity_parameters.csv`; Git tracked-file listing.

Relevant variable/function/script/table/artifact: final CSV table set; `epcsaft_electrolyte_relative_permittivity_parameters.csv`.

Confidence: High.

AUTHOR VERIFY: Confirm whether the header-only relative-permittivity table is intentionally committed as an empty provenance table.

## 42. Which command should a reviewer run to reproduce the paper-facing results?

Answer: No single documented command regenerates every paper-facing figure/table, especially Table 3. The shortest evidenced command chain is: `uv sync --group test`; configure `PYTHONPATH=src`; configure/install local ePC-SAFT for ePC-SAFT lanes; run the campaign benchmark shown in `analyses/nccc_validation/README.md`; run `analyses/nccc_validation/scripts/render_c_case_campaign_temperature_gallery.py`; run `analyses/nccc_validation/scripts/generate_nccc_one_bed_artifacts.py`; run `analyses/nccc_validation/scripts/generate_accuracy_credibility_artifacts.py`; run `docs/latex/scripts/sync_latex_figures.ps1`; and run `analyses/nccc_validation/scripts/validate_results.py`. The full nine-species Table 3 values appear static from handoff values rather than regenerated by a documented final-table command.

Evidence path: `README.md`; `docs/workflow_map.md`; `analyses/nccc_validation/README.md`; `analyses/nccc_validation/scripts/generate_nccc_one_bed_artifacts.py`; `analyses/nccc_validation/scripts/render_figures.py`; `docs/latex/scripts/sync_latex_figures.ps1`; `analyses/nccc_validation/scripts/validate_results.py`; `docs/full_species_ionic_speciation_handoff.md`.

Relevant variable/function/script/table/artifact: `uv sync --group test`; `python -m mea_absorption_column.benchmark`; `render_figures.py`; `generate_nccc_one_bed_artifacts.py`; `generate_accuracy_credibility_artifacts.py`; `validate_results.py`.

Confidence: Medium.

AUTHOR VERIFY: Confirm the intended reviewer-facing one-command or ordered workflow, especially for Table 3.

## 43. Should the repo add a REPRODUCE.md file or strengthen the workflow map?

Answer: Yes. The current reproduction instructions are split across README, workflow map, analysis README, sync scripts, and handoff docs, and some contain local machine paths. The shortest peer-review improvement would be a small `REPRODUCE.md` or a strengthened workflow-map section with one command chain, expected outputs, and clear notes about optional ePC-SAFT/full reactive paths.

Evidence path: `README.md`; `docs/workflow_map.md`; `analyses/nccc_validation/README.md`; `docs/full_species_ionic_speciation_handoff.md`.

Relevant variable/function/script/table/artifact: `REPRODUCE.md` missing; workflow map; analysis README; final tables/figures.

Confidence: High.

## 44. Is first-person plural active voice acceptable and consistent with the manuscript style?

Answer: It is acceptable in principle, but not currently consistent as the dominant manuscript style. The manuscript mostly uses impersonal active framing such as "This work..." and "The benchmark..."; first-person appears mostly in instructional/math-method prose, not as the paper's main voice.

Evidence path: `docs/latex/main.tex`; `docs/latex/sections/model_framework.tex`; `docs/latex/sections/methods.tex`.

Relevant variable/function/script/table/artifact: manuscript prose style; abstract; methods sections.

Confidence: Medium.

AUTHOR VERIFY: Confirm target journal/editorial preference before changing voice.

## 45. Should the abstract explicitly say the full activity-coupled path is not the routine model?

Answer: Yes. The abstract currently mentions the routine `9.86 s` ePC-SAFT median but does not distinguish that from the `212.809 s` full activity-coupled boundary. Without that distinction, "full ionic" language can be read as implying that the routine campaign includes activity-coupled speciation.

Evidence path: `docs/latex/main.tex`; `docs/latex/sections/results.tex`; `docs/latex/tables/full_ionic_speciation_timing.tex`.

Relevant variable/function/script/table/artifact: `epcsaft_ionic`; `chemical_equilibrium_model=legacy`; `tab:full-ionic-speciation-timing`; `212.809 s`; `9.86 s`.

Confidence: High.

## 46. Should the title mention "ionic" or stay broader?

Answer: "Ionic" is technically more accurate for the novelty claim, but "liquid-side ionic" is the most precise wording because the vapor side is neutral. The current title remains defensible as a broad ePC-SAFT benchmark title; a revised title with "ionic" better signals the actual contribution.

Evidence path: `docs/latex/main.tex`; `docs/latex/sections/introduction.tex`; `docs/latex/sections/model_framework.tex`; `analyses/nccc_validation/results/final/reports/epcsaft_electrolyte_column_config_matrix.md`.

Relevant variable/function/script/table/artifact: title; `MEA_CO2_H2O_ionic_fit`; `epcsaft_ionic`; vapor-side neutral framework.

Confidence: Medium-High.

AUTHOR VERIFY: Final title strategy is an author/editorial decision.

## 47. How aggressive should the "first" claim be?

Answer: The safer version is best supported: "to the authors' knowledge, the first reproducible MEA absorber benchmark to isolate an ionic ePC-SAFT CO2 fugacity closure under fixed transport, chemistry, and solver settings." The stronger "first solved packed-column absorber benchmark..." wording may be too broad without an exhaustive literature review.

Evidence path: `docs/latex/sections/introduction.tex`; `docs/latex/sections/conclusion.tex`; `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_summary.csv`; `src/mea_absorption_column/Thermodynamics/thermo_models.py`.

Relevant variable/function/script/table/artifact: novelty claim; `epcsaft_ionic_fugacity()`; `nccc_one_bed_accepted_summary.csv`.

Confidence: Medium.

AUTHOR VERIFY: Novelty/literature completeness cannot be proven from the repo alone.

## 48. Should "improved performance" language be removed?

Answer: Yes, or immediately quantified. The supported claim is a slightly lower capture MAE with higher runtime, not a broad performance improvement. One-bed accepted summary supports `3.725` vs `3.775` percentage-point MAE; C-case figure summary supports `4.540` vs `4.696` percentage-point MAE, while temperature RMSE is slightly worse for ePC-SAFT.

Evidence path: `docs/latex/sections/results.tex`; `docs/latex/sections/conclusion.tex`; `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_summary.csv`; `analyses/nccc_validation/results/final/tables/plot_c_case_thermo_summary.csv`.

Relevant variable/function/script/table/artifact: capture MAE; median runtime; `plot_c_case_thermo_summary.csv`; `nccc_one_bed_accepted_summary.csv`.

Confidence: High.

## 49. Should "middle ground" be replaced with "controlled sensitivity layer" or another technical phrase?

Answer: Yes. Replace informal "middle ground" language with a more precise phrase such as "fugacity-only ePC-SAFT sensitivity layer," "controlled electrolyte-EOS fugacity closure," "bounded thermodynamic sensitivity model," or "repeated-run ePC-SAFT fugacity closure." "Controlled sensitivity layer" is acceptable only if tied specifically to fugacity closure.

Evidence path: `docs/latex/main.tex`; `docs/latex/sections/introduction.tex`; `docs/latex/sections/results.tex`; `docs/latex/sections/conclusion.tex`.

Relevant variable/function/script/table/artifact: "middle ground"; "practical middle-ground"; `epcsaft_ionic`; fugacity-only routine lane.

Confidence: High.

## 50. Should Sections 3.1 and 3.2 be rewritten aggressively?

Answer: Yes. The shooting and finite-difference sections read more like generic tutorials than implementation-specific methods, while the code contains concrete solver details: `single_shoot_solve`, custom Euler or optional `solve_ivp`, `root_method="Krylov"`, `finite_difference_solve`, `root(..., method="hybr")`, mesh settings, tolerances, and timeout/gate behavior. The collocation section is more implementation-specific, so Sections 3.1 and 3.2 are stylistically inconsistent with the rest of the methods.

Evidence path: `docs/latex/sections/methods.tex`; `src/mea_absorption_column/BVP/Methods/Single_Shoot_Solve.py`; `src/mea_absorption_column/BVP/Methods/Finite_Difference_Solve.py`; `src/mea_absorption_column/BVP/Methods/Scipy_BVP_Solve.py`; `analyses/nccc_validation/results/final/tables/method_case_contrast.csv`.

Relevant variable/function/script/table/artifact: `single_shoot_solve()`; `finite_difference_solve()`; `scipy_BVP_solve()`; `method_case_contrast.csv`.

Confidence: High.

## Questions for Author

- Should the accepted validation language use `K18`, `K19`, and `1C--6C`, or should it be reconciled with the campaign overlay artifacts that include `1C--7C`?
- Should the manuscript/reproduction workflow explicitly force `MEA_CO2_H2O_ionic_fit` so the code default `MEA_CO2_H2O_draft` cannot be accidentally used?
- Can you provide or restore the raw full nine-species run CSVs referenced by `docs/full_species_ionic_speciation_handoff.md`?
- Were the slow-path timings measured on the same hardware and Python/package environment as the routine accepted rows?
- What is the provenance of `eta_Psi = 0.3`: fixed heuristic, legacy assumption, or fitted parameter?
- Is the `45.0 deg C` lean-inlet imputation for `1C--3C` supported by sensitivity analysis beyond nearby campaign temperature consistency?
- Which release commit and archive/DOI plan should be cited for submission?
- Is the external `epcsaft` package public, archived, or intended to be released with/near this manuscript?
- Is the header-only `epcsaft_electrolyte_relative_permittivity_parameters.csv` intentionally committed?
- What is the intended one-command or ordered reviewer workflow for regenerating all paper-facing figures and tables, especially Table 3?
- Should the manuscript use first-person plural voice consistently, or preserve the current mostly impersonal active style?
- Which final title and novelty-claim strength should be used after the literature/author check?
