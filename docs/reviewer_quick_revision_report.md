# Reviewer quick-revision report

Date: 2026-08-12
Working branch: `codex/reviewer-quick-revisions`

## Preserved before version

The manuscript PDF that existed before this revision pass is preserved at:

`docs/latex/builds/main_before_reviewer_quick_revisions_2026-08-12.pdf`

SHA-256: `09b8ce8226c7e9e083339deb2231adff9eca5b37171f1b9c03d178ea07c16f62`

The preserved PDF was copied before source edits and verified byte-for-byte
against the pre-edit `main.pdf`. The revised source was subsequently built with
the repository's Elsevier support files in an isolated temporary build-support
directory; no support file was added to the manuscript tree.

## Revisions completed in the quick pass

| Reviewer point | Revision | How it addresses the comment |
| --- | --- | --- |
| R1.1--R1.4: theoretical basis, physical versus chemical thermodynamics, workflow, and definition of “fugacity benchmark” | The Introduction now defines the benchmark as a controlled phase-state-to-fugacity replacement. The framework adds an explicit responsibility table and clarifies the existing workflow schematic. | ePC-SAFT is no longer presented as the reaction model. Routine reaction equilibrium supplies species and concentration-basis constants; ePC-SAFT supplies phase residual properties and fugacity coefficients. |
| R1.5--R1.7 and R2.3: parameter completeness and binary interactions | The appendix table now includes all nine stored species, molecular weights, charge, Born diameter, and other selected parameters. It identifies the only nonzero pair value, `k_MEA,H2O = -0.052`, and states that it was not fitted to NCCC absorber rows. | Readers can reproduce the selected artifact and can see its actual scope instead of inferring a complete predictive fit. |
| R1.5--R1.7: provenance limits | The appendix now labels MEAH+/MEACOO- as provisional, bicarbonate as transferred diagnostic, and carbonate/hydronium/hydroxide as placeholders. It also discloses that the traceable lineage of `k_MEA,H2O = -0.052` used a different neutral association-scheme combination from the selected 2B/2B record. | This turns an undocumented provenance weakness into an explicit limitation. It does not resolve the underlying parameter inconsistency. |
| R1.9: transport-property uncertainty | The model-framework discussion now traces viscosity and diffusivity through Reynolds/Schmidt numbers, film coefficients, Hatta number, interfacial area, holdup, and heat transfer. | The manuscript now explains why the thermodynamic comparison remains conditional on fixed transport correlations and states that no transport uncertainty is propagated. |
| R1.10--R1.11: numerical diagnostics and computational expense | Methods now lists exactly what the archived rows contain and do not contain. | The manuscript no longer implies that final adaptive meshes, refinement histories, nonlinear iterations/evaluations, memory, CPU identity, BLAS threads, or controlled hardware timings were recorded. Runtime is described as comparative wall time only. |
| R1.12 and R2.6: other amines | Results and Conclusions now list the solvent-specific chemistry, parameters, properties, transport, kinetics, enhancement, and validation that another amine would require. | The framework is not described as plug-and-play for DEA, MDEA, AMP, PZ, blends, or another solvent. |
| R1.13: limitations | Conclusions now state the limits on parameter provenance, uncertainty propagation, cross-facility validation, operating optimization, reactive chemistry, and other-solvent extension. | The conclusion is balanced against the evidence actually present. |
| R1.14: comparison with other thermodynamic models | The literature review now includes a concise Kent--Eisenberg/eNRTL/CPA/ePC-SAFT comparison table. | It distinguishes each model’s thermodynamic role, strength, and calibration or standard-state burden without claiming a universal accuracy ranking. |
| R2.1: citation numbering | The Elsevier bibliography style was changed from the name-based model to the numeric model. | Visual inspection confirms that the first citation now renders as `[1, 2, 3]`, matching the first three reference-list entries. |
| R2.2: shortcomings of existing absorber comparisons | The Introduction now identifies inconsistent thermodynamic assumptions, solver formulations, acceptance criteria, and validation subsets as the reproducibility gap. | The literature gap is stated as a comparison/reproducibility problem rather than a claim that MEA absorber modeling itself is absent. |
| Reproduction command mismatch found during audit | `REPRODUCE.md` now passes the just-created C-case run directory to the gallery renderer. | The renderer now consumes the run produced by the preceding command instead of its unrelated default directory. |

## ePC-SAFT 0.2 API migration completed in this pass

- The dependency now points to the content-addressed ePC-SAFT wheel with SHA-256 `e14288867d4fb5bc1367dd0de490aeb1551f1613074aced0a8d28432ca762f23`; final release provenance still requires an upstream Engine commit and clean-tree receipt.
- The adapter uses the public `Parameters -> Mixture -> State` API and converts the repository-vendored MEA CSV artifact into a strict parameter document.
- CppAD is treated as the package's sole production derivative authority. Legacy runtime derivative and Born-model switches now fail clearly instead of being silently ignored.
- Neutral, six-species ionic, and nine-species ionic fixed states execute through the new API. The final integration check records package `0.2.0.dev0`, source kind `local_file`, parameter fingerprint `sha256:b4b5c2c255790f64ee20ca0b070a007c2fee4fef22653f578356bb3439d5ccdb`, and a positive ionic CO2 fugacity coefficient.
- The legacy `epcsaft_reactive_*` modes now fail closed. The new typed reactive API requires independently sourced dimensionless reaction constants and an explicit standard-state conversion; the archived locally rebased constants do not satisfy that contract.
- The archived nine-species sweep is labeled as legacy-interface numerical-feasibility evidence throughout the manuscript and reproduction guide.

Focused validation: 22 tests passed across the thermodynamic adapter, fixed electrolyte model, reactive fail-closed boundary, and integration contract.

A full Case 3C `epcsaft_ionic` BVP smoke reached the new thermodynamic path but exceeded its 180 s subprocess limit before solver completion. Therefore, no archived capture, timing, or validation table was relabeled as a new-engine result. Completing and profiling the full campaign is a take-time revision, not a quick wording fix.

## Revisions that need a focused follow-up study

| Reviewer point or audit finding | Effort | Required work |
| --- | --- | --- |
| R1.5--R1.8: provenance-complete parameters and sensitivity | High | Select or refit one internally consistent association/`kij`/ionic parameter set, document primary sources and fit targets, define defensible parameter ranges or covariance, rerun the accepted campaign, and report capture/temperature sensitivity. The current 14-configuration historical matrix is model-form exploration, not parameter uncertainty quantification. |
| R1.10--R1.11: richer solver-cost evidence | Medium to high | Extend result schemas to retain method-specific iterations/evaluations, final mesh, refinement history, CPU time, peak memory, CPU/RAM identity, and thread settings; then rerun all methods on one controlled machine. These quantities cannot be recovered from the existing CSV rows. |
| R2.4: broader validation | High | Reconcile validation scope first, then add an independent facility or literature dataset with traceable input conversion. Current NCCC rows broaden operating coverage but do not provide independent cross-facility validation. |
| R2.5: operating-condition study and optimum | High | Define independent operating variables, constraints, an objective, and validation domain; run a controlled design or optimization study. The existing NCCC observations co-vary naturally and are not a controlled operating sweep. |
| Predictive reactive ePC-SAFT chemistry | High | Source or regress activity-basis reaction constants, document standard states, replace placeholder ionic parameters, migrate to the typed 0.2 chemical-equilibrium API, and rerun the reactive cases. |
| New-engine campaign reproduction | Medium to high | Profile repeated pressure-closed state calls, complete Case 3C, rerun the accepted campaign, compare against archived values, and regenerate every affected table and figure before changing numerical claims. |

## Evidence conflicts that should be resolved before the response letter

- Resolved: the accepted artifact, summary, attempted-status table, and Figure 4 are now generated from the all-attempted CSV using the manuscript's stated gate. Both 7C rows pass; K20 is the only rejected case.
- Resolved: the generated attempted-status table now reports the current K19 guard diagnostics (166 ePC-SAFT and 183 Henry-law events) and the current 7C runtimes and guard counts.
- The new ePC-SAFT fixed-state results are not numerically interchangeable with the archived package lane. Current manuscript performance numbers must remain labeled as archived until the new campaign is complete.

## Post-reconciliation verification

- The current revised manuscript is 31 pages and passes the PDF freshness
  check.
- The NCCC artifact validator passes with nine accepted cases and 18 accepted
  rows: K18, K19, and 1C--7C for both closures; K20 is rejected.
- `tests/test_results_architecture.py` passes all seven tests, including the
  claim-level check that the accepted artifact equals the stated gate applied
  to every attempted row.
- Visual checks passed for the title/abstract, first citation, attempted-status
  table, Figure 4, and full-species timing table. No clipping or overlap was
  observed.
- The remaining LaTeX messages are nonblocking CAS front-matter anchors and
  underfull bibliography lines; they do not affect evidence or legibility.
