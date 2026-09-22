# Language revision batches (editorial review of 2026-09-04)

Seven sequential execution batches, pending independent Astra review and consolidation. Each approved batch is one `codex-implementer` (GPT-5.6 Luna, medium effort) run.
The implementer applies the listed replacements verbatim; it does not compose prose, change numbers,
or touch files outside the batch list. Run batches in order; commit after each passes its checks.

## Author-approved updates and current stage (2026-09-04)

The author requests a fresh Astra audit before manuscript execution. The reviewer first seals an independent full-manuscript audit without reading the prior editorial review or this derived plan; only then does it compare the earlier review, inspect this plan and propose a consolidated plan. This is a language-and-story audit with bounded factual sanity checks against existing evidence. No new analyses, experiments or literature-research expansion are authorized. The parent agent owns later text-edit, build and verification executions. No manuscript batch has been applied or checkpointed in this task.

- Numerical-token changes are allowed when traced to specified rounding, approved additions or deletions, cross-reference renumbering, or pagination. Every unexplained change remains a failure; scientific values must not change incidentally.
- Retain concise energy-convergence evidence, rather than an unresolved-cause discussion. The additional Case 3C runs below establish sub-watt variation at tighter numerical resolution. They do not replace original campaign or sensitivity results, and the approximately 35-second timing remains a separate coarse calculation.
- The Engine optimizations are committed at `7b1e62f0483571f8a194b32441467bdb58b45ce7`. Normal execution now pins restored fast wheel SHA-256 `91632d2812429cbd293aae70fe8d4efb00000efe2377a91546dd7374dca67ee4`; all integration modes reject other wheel hashes. The new refinement results themselves used the preceding fast timing wheel `9b538f4defd5af661cd736af03760a55adb59b231d674f96c1e0f7d67350689d`, as their unchanged identities record. Do not relabel past results as using the newly pinned wheel.
- Original parameter values, enhancement equations and physics remain unchanged. The independently checked explicit enhancement and two-film flux match the manuscript. The retained Engine temperature-boundary test failure remains documented outside the article; no claim of a wholly passing Engine suite is authorized.

### Additional Case 3C convergence evidence

Evidence owner: `analyses/nccc_validation/results/runs/energy_fast_diagnostic_20260904/`. Read `verified_results.json`, both `attempt*/identity.json`, `result.json`, `energy_check.json`, `energy_defect_check.json`, and `solver_polynomial.npz`; `formulation_check.json` records the independent manuscript-equation check. `diagnosis.json` summarizes the limited claim. These are additional verification runs, not replacement study baselines.

| Quantity | Additional refinement 1 | Additional refinement 2 |
| --- | ---: | ---: |
| Initial / final nodes | 41 / 46 | 81 / 121 |
| Requested tolerance | 0.005 | 0.00005 |
| Mesh iterations | 3 | 4 |
| Maximum normalized RMS residual | 0.004782819229884394 | 0.00004962997152392135 |
| Maximum scaled boundary residual | 0 | 0 |
| Component discrepancy (mol/s) | 1.3649525953951525e-11 | 1.3649525953951525e-11 |
| Charge discrepancy (mol/s) | 2.1630375442172056e-15 | 1.951401280148035e-15 |
| Capture (%) | 91.54788008336736 | 91.54831292646224 |
| Energy range at 101 exported positions (W) | 16.39548686146736 | 0.035850100219249725 |
| Energy range at 10,001 positions (W) | 30.32930175680667 | 0.1338401660323143 |
| Signed end-to-end energy difference (W) | 2.197474582120776 | 0.0023996923118829727 |
| BVP wall time (s) | 82.37750029563904 | 248.60570216178894 |

Use 30.33 and 0.134 W for the two dense-sampling table values, and 16.40 and 0.0359 W for the corresponding 101-position values. Explicitly distinguish the sampling grids. Capture differs by less than 0.0001 percentage points between the second additional refinement and the original refined Case 3C calculation. Both new runs converged with positive species concentrations and no invalid-state or penalty evaluations. Do not claim exact zero error, negligible error in every original run, or proof of caloric-model accuracy.

## Shared contract (paste into every spec)

- Repository root: `/home/tnnrpolley21/Workspaces/Engineering/MEA-Absorption-Column`. Branch `codex/fallback-manuscript`.
- Edit only the files named in the batch. Replacements are exact `OLD` → `NEW` strings in LaTeX source under `docs/latex/`.
  If an `OLD` string is not found verbatim, stop and report; do not guess.
- Never change a numeral, unit macro, citation key, label, or equation unless the batch says so.
- Build: `cd <root> && uv run python docs/latex/scripts/latex_workflows.py sync-figures && bash docs/latex/scripts/build_main.sh`.
  The build must succeed; report new LaTeX warnings about undefined references or citations.
- Checks after building (report the raw output of each):
  1. `pdftotext docs/latex/builds/main.pdf /tmp/new.txt; diff <(pdftotext <previous-batch-pdf> -) /tmp/new.txt` — only the intended lines may differ.
  2. `pdftotext docs/latex/builds/main.pdf - | grep -inE '\b(local|locally|history|historical|histories)\b'` must print nothing.
  3. Numeric-token diff: `grep -oE '[0-9]+(\.[0-9]+)?' old.txt | sort | uniq -c` vs new; explain every count change through the explicit rounding list, approved additions/deletions (including the added convergence rows), cross-reference renumbering or pagination. Flag any unexplained change.
  4. `grep -rn -F -f <(printf '%s\n' <batch removed phrases>) docs/latex/sections docs/latex/appendices docs/latex/tables docs/latex/main.tex` must print nothing.
- Copy the previous batch's `builds/main.pdf` to `builds/main_before_batch<N>.pdf` before building (the `builds/` directory is the scratch location already used for the pre-revision copy).
- Report: files changed, `git diff --stat`, check outputs, anything not applied and why. Do not commit; the orchestrator commits.

## Batch 0 (orchestrator, no agent)

`git add -A && git commit -m "Checkpoint before language-revision batches"` so each batch reverts alone.

## Batch 1: back matter and Appendix B artifacts

Files: `sections/code_availability.tex`, `sections/data_availability.tex`, `appendices/appendix_reactive_inputs.tex`, `sections/model_framework.tex`.

1. `code_availability.tex`, replace the whole body after `\section*{Code Availability}` with:
```
The Python absorber model, parameter files, numerical records and reproduction instructions are available in the \href{https://github.com/tannerpolley/MEA-Absorption-Column}{project repository} at tag \texttt{nce-revision-v3}.
The calculations use the nine-species parameterization in \cref{sec:selected-reactive-inputs}, automatic thermodynamic derivatives and differentiated column equations~\cite{Polley2026epcsaft}.
The ePC-SAFT package used for these calculations is not yet publicly released; the repository records the exact build identity of every reported calculation, and the package is available from the authors on request and will be released publicly when ready.
Full-precision parameters, software versions and the commands for every reported calculation are documented with the code.
An archival identifier will be assigned on acceptance.
```
2. `data_availability.tex`, replace the body after `\section*{Data Availability}` with:
```
Experimental capture and packing temperatures are reported by Morgan et al.\ \cite{Morgan2020}.
The repository cited under Code Availability contains the processed seven-case inputs and observations, the Case 3C profiles and refinement, the thermodynamic and transport sensitivity calculations, the six converged operating conditions, the three timing repeats and the per-calculation convergence diagnostics summarized in \cref{tab:reactive-numerical-verification}.
```
3. `appendix_reactive_inputs.tex`:
   - OLD `The selected inputs in the following tables define the baseline parameterization for all seven campaign cases and Case 3C verification; the perturbation studies vary the specified inputs around it.` → NEW `The inputs in the following tables define the baseline parameterization for all seven cases and the Case 3C verification; the perturbation studies vary them around this baseline.`
   - OLD `The tables disclose component, pair, association and reaction coefficients; the accompanying machine-readable parameter and reaction files retain full precision, sources and model choices.` → NEW `The tables give component, pair, association and reaction coefficients; full precision and sources are provided with the code.`
   - OLD `The selected formulation couples the five reactions` → NEW `The formulation couples the five reactions`
   - OLD `on the declared pure-water, infinite-dilution molality reference` → NEW `on the pure-water, infinite-dilution molality reference`
   - OLD `The accompanying parameter record distinguishes literature coefficients, transferred ion values and fitted coefficients with source locators.` → NEW `Literature coefficients, values taken from other ions and fitted coefficients are distinguished below and in the parameter files.`
   - OLD `The selected MEA--water interaction coefficient is` → NEW `The MEA--water interaction coefficient is`
   - OLD `The former comes from the neutral-mixture fit; the latter was selected jointly with R4 in a pressure/speciation calibration grid.` → NEW `The former was fitted to binary MEA--water VLE data; the latter was chosen jointly with R4 from a pressure/speciation calibration grid.`
   - OLD `The selected R2, R4, and R5 temperature correlations incorporate linked reaction-enthalpy shifts adopted from a two-direction fit, with pressure, speciation, and` → NEW `The R2, R4 and R5 temperature correlations incorporate reaction-enthalpy shifts fitted to pressure, speciation and`
   - OLD `Xu data also inform model selection and are not an independent holdout for the entire parameterization.` → NEW `the Xu data also informed model selection and are therefore not independent validation data for the entire parameterization.`
   - OLD `The selected R2, R4 and R5 input correlations replace their source correlations in full, with the standard-state correlation offsets incorporated:` → NEW `The R2, R4 and R5 input correlations, which already include the standard-state offsets, are`
   - OLD `The R2/R4/R5 standard-state correlation offsets must not be added again; the subsequent EOS reference transformation remains required.` → NEW `Only the EOS reference transformation described below is applied to these constants.`
   - OLD `with the adopted molality-conversion offsets:` → NEW `with the molality-conversion offsets:`
   - OLD `Thus the printed temperature correlations alone are not the final EOS-coordinate constants; the selected EOS and declared reference are also required, with each conversion applied once.` → NEW `The effective constants therefore depend on the equation of state and reference state as well as on these correlations.`
   - OLD `The common selected calculation range is` → NEW `The common calculation range is`
   - Delete both lines: `Reproduction uses the retained parameter and reaction JSON files together, preserving species order, units, association topology, reaction standard states, and temperature domains.` and `The reproduction guide specifies the computational environment, commands and runtime selection \texttt{epcsaft\_reactive\_nine}.`
4. `model_framework.tex` line 80: OLD `R2, R4 and R5 use the selected fitted correlations in the parameter document, which replace the corresponding source coefficients in full; their standard-state correlation offsets are already included.` → NEW `R2, R4 and R5 use the fitted correlations in \cref{sec:selected-reactive-inputs}, which replace the corresponding source coefficients in full and already include their standard-state offsets.`

Removed-phrase grep: `parameter document`, `reproduction guide`, `JSON`, `epcsaft\_reactive\_nine`, `REPRODUCE.md`, `holdout`, `failed lower-flow`, `archival identifier has been assigned`.

## Batch 2: Methods (Section 3)

Files: `sections/methods.tex`, `tables/supplementary_run_diagnostics.tex`, `appendices/appendix_reactive_inputs.tex`.

- OLD `with temperature scales of \qty{400}{K}; the retained calculation records give the resulting scales.` → NEW `with temperature scales of \qty{400}{K}.`
- OLD `finite differences serve derivative verification rather than a competing column solution method.` → NEW `finite differences are used only to verify the derivatives.`
- Replace the two lines `Chemical equilibrium normally starts with conservative species amounts generated using reaction-extent fraction 0.001, while the retained campaign Case 1C uses 0.0001 throughout.` and `That exception changes the chemical initial guess, not reaction constants, inlet conditions or tolerances; its exact run is identified in the reproduction guide.` with one line: `The equilibrium solve is initialized from conservative species amounts generated with a reaction-extent fraction of 0.001 (0.0001 for Case 1C); this affects only the initial guess.`
- OLD `Recorded outer mesh iterations, residuals at each iteration, Jacobian evaluations and equilibrium solves describe the numerical work; mesh iterations do not count inner Newton steps.` → NEW `Outer mesh iterations, residuals at each iteration, Jacobian evaluations and equilibrium solves are reported as measures of numerical effort; mesh iterations do not count inner Newton steps.`
- OLD `Invalid reactive states propagate a failure rather than a finite penalty substitution, and failed conditions remain missing rather than receiving zero responses.` → NEW `A calculation in which the reactive equilibrium cannot be solved is reported as unconverged; no substitute value is assigned.` Then append a new line after it: `Every calculation reported in \cref{sec:results} satisfies these criteria; final node counts, mesh iterations, residuals and component/charge discrepancy bounds for each run are listed in \cref{tab:reactive-numerical-verification} and the supplementary run table.`
- OLD `The continuous adopted equations predict zero variation; a nonzero numerical range is reported separately from caloric-model accuracy.` → NEW `The continuous equations predict zero variation; the nonzero numerical range is reported as a diagnostic, separately from caloric-model accuracy.`
- OLD `One-at-a-time thermodynamic perturbations test selected interactions and reaction constants around refined Case 3C.` → NEW `One-at-a-time thermodynamic perturbations test two binary interactions and two reaction constants around refined Case 3C.`
- OLD `A factor-one calculation checks equivalence with the reference; the lower-diffusivity case uses conservative species amounts obtained by continuation from a nearby temperature, and one seven-state directional derivative check per multiplier uses the tolerances stated above.` → NEW `A factor-one calculation checks equivalence with the reference, and the lower-diffusivity case uses conservative species amounts obtained by continuation from a nearby temperature.`
- OLD `Each condition first uses a separately calculated Henry-law profile; the two flow perturbations and colder-inlet condition permit one additional attempt from a retained reactive Case 3C profile, without changing the equations or tolerances.` → NEW `Each condition is initialized from a Henry-law profile; where that initialization did not converge (the two flow perturbations and the colder inlet), the calculation was restarted from the reactive Case 3C profile with unchanged equations and tolerances.`
- OLD `but excludes imports, environment preparation, input loading and final diagnostic serialization.` → NEW `but excludes program start-up, input loading and output writing.`

Supplementary run table: create `tables/supplementary_run_diagnostics.tex` as a `table*` with caption `Convergence diagnostics for every reported calculation: final nodes, mesh iterations, maximum normalized RMS residual, maximum scaled boundary residual, component and charge discrepancy bounds and axial net-enthalpy-flow range.` Populate rows from the values already in `results.tex` (seven cases: 21--22 nodes, 1--2 iterations, RMS $<0.126$, boundary $<\num{2.78e-17}$, discrepancies $<\num{1.54e-11}$/$<\num{2.55e-15}$, enthalpy 106.49--325.13 W; eight thermodynamic runs: 42 nodes, 2 iterations, RMS $<0.044$, 0, $<\num{1.50e-11}$/$<\num{2.40e-15}$; six transport runs: 42 nodes, enthalpy 272.08--273.15 W; six operating runs: 22 nodes, 2 iterations, RMS $<0.184$, 0, $<\num{1.56e-11}$/$<\num{2.21e-15}$, enthalpy 160.21--390.32 W). Label `tab:supplementary-run-diagnostics`; `\input` it at the end of `appendix_reactive_inputs.tex` before `\clearpage`; change the phrase `the supplementary run table` above to `\cref{tab:supplementary-run-diagnostics}`. Where a per-group value is unknown, write `--` and report it. Append the two additional Case 3C verification rows from the evidence table above. Give both their 101-position and 10,001-position energy ranges with explicit sampling labels, either in separate columns or in a compact table note. Include requested tolerances so the refinement is reproducible. Preserve the original group values and identify the new rows as additional verification.

Removed-phrase grep: `retained calculation records`, `retained campaign`, `reproduction guide`, `penalty substitution`, `additional attempt`, `serialization`.

## Batch 3: Results 4.1 to 4.3 prose and captions

File: `sections/results.tex`, `tables/reactive_numerical_verification.tex`.

Rounding list (explicit numeric changes): 91.54839→91.55 (prose only, not captions/tables), 347.86155→347.86, 91.54329→91.54 (R5 and R11), 347.90647→347.91, 0.00510→0.005 (prose), 0.04493→0.045 (prose), sensitivity values to two decimals in R7, R8, R9, R11, R12, R14 (e.g. −0.82244→−0.82, +0.75819→+0.76, 90.72595–92.30657→90.73–92.31, 2.04839→2.05, 0.31049→0.31, 3.75443→3.75, 6.86023→6.86, 0.10730→0.11, 0.08482→0.08, 0.61395→0.61, 0.25063→0.25, 0.82713→0.83, 0.02173→0.02, 0.00967→0.01, 0.10961→0.11, 0.09864→0.10, 0.06029→0.06, 0.05987→0.06, 0.06307→0.06, 0.06721→0.07, 0.01598→0.016, 0.01652→0.017, 0.53137→0.53, 0.39813→0.40, 0.28208→0.28, 0.29224→0.29, 0.08248→0.08, 0.09333→0.09). Figure captions and Table 5 keep full precision.

- R1: `It predicts \qty{91.54839}{\percent} capture against \qty{89.50}{\percent} observed, with sampled peak liquid temperature \qty{347.86155}{K}.` → `It predicts \qty{91.55}{\percent} capture against \qty{89.50}{\percent} observed, with peak liquid temperature \qty{347.86}{K} (\cref{tab:reactive-numerical-verification} gives full precision).`
- Fig. 2 caption: `with the selected parameter set and conventional reaction enhancement` → `with conventional reaction enhancement`.
- R2: `by \num{0.00510} percentage points and peak liquid temperature by \qty{0.04493}{K}` → `by \num{0.005} percentage points and peak liquid temperature by \qty{0.045}{K}`. Replace `Axial signed net-enthalpy-flow variation decreases from \qty{293.219}{W} to \qty{273.006}{W}, although the continuous equations conserve that quantity.` → `Further Case 3C refinement reduces the axial net-enthalpy-flow range to \qty{0.134}{W} on \num{10001} positions, while capture changes by less than \num{0.0001} percentage points relative to the paired refined calculation (\cref{tab:supplementary-run-diagnostics}).`
- Table 5 caption: `Both calculations use the same nonisothermal equations and selected parameter set.` → `Both calculations use the same equations and parameters.` Note: replace from `Mesh iterations are not counts of inner Newton steps.` to the end of the note with `Mesh iterations count outer refinement cycles, not inner Newton steps. Both calculations use boundary tolerance \num{0.001} and a 1000-node limit. Peak memory is reported with the timing study.`
- R3: delete `Its capture is \qty{91.54329}{\percent}.`
- R5: `its coarse calculation has \qty{91.54329}{\percent} capture and peak liquid temperature \qty{347.90647}{K}` → `its coarse calculation has \qty{91.54}{\percent} capture and peak liquid temperature \qty{347.91}{K}`.
- R6: replace the whole paragraph (six lines) with: `All seven calculations converge at the campaign settings (\cref{tab:supplementary-run-diagnostics}), while physical agreement remains uneven. Only Case 3C has the paired refinement, which provides a comparison magnitude for the perturbation responses that follow.`
- R7: apply rounding; `\num{0.00510}-point` → `\num{0.005}-point`.
- Fig. 5 caption: `crosses denote the selected-parameter reference` → `crosses denote the unperturbed reference`.
- R8: `only partly separated from the reference refinement indicator.` → `only partly exceed the \qty{0.045}{K} refinement difference.`; `The reaction-constant effects modestly exceed \qty{0.04493}{K}, while the smaller \COtwo--water effects fall below this reference refinement difference.` → `The reaction-constant effects modestly exceed it, while the \COtwo--water effects fall below it.`; delete the final `All eight calculations have 42 final nodes ...` line.
- R9: `The unperturbed capture agrees with the thermodynamic reference within \num{3e-8} points, and the factor-one control changes it by only \num{1.01e-11} points.` → `The factor-one control reproduces the reference.`
- R10: `smaller than the \qty{0.04493}{K} reference indicator` → `smaller than the \qty{0.045}{K} refinement difference`; delete `The reference and perturbations satisfy the stated residual and positivity checks with 42 final nodes, while numerical net-enthalpy-flow ranges remain \qtyrange{272.08}{273.15}{W}.`
- R10b: `this is not a common limit for every diffusivity formula.` → `the other diffusivity correlations have no documented upper temperature limit.`
- R11: `Six of the seven specified operating conditions meet the numerical inclusion checks, including the coarse baseline with \qty{91.54329}{\percent} capture` → `Six of the seven operating conditions converged, including the coarse baseline with \qty{91.54}{\percent} capture`; apply rounding.
- Fig. 7 caption: `The lower liquid-to-gas condition has no retained result.` → `The lower liquid-to-gas condition did not converge and is omitted.`
- R12: `This is consistent with the capacity and driving-force dependence of the model.` → `Lower loading raises free MEA and lowers the liquid \COtwo{} fugacity, increasing both capacity and driving force.`; `both below the reference refinement indicator.` → `both below the refinement difference.`; apply rounding.
- R13: delete the whole paragraph (three lines).
- R14: `The refined Case 3C capture error of \num{2.04839} percentage points is much larger than its \num{0.00510}-point refinement change` → `The refined Case 3C capture error of \num{2.05} percentage points is much larger than its \num{0.005}-point refinement change`; `limit interpretation without establishing their individual shares of those errors.` → `limit interpretation; their individual contributions to the error are not resolved.`

Removed-phrase grep: `inclusion checks`, `retained conditions`, `retained result`, `reference indicator`, `refinement evidence`, `invalid-state`, `selected parameter set`, `selected-parameter`, `without an established cause`.

## Batch 4: framing (abstract, Introduction paragraph 4, 4.4, Conclusions)

Files: `main.tex` (abstract only), `sections/introduction.tex`, `sections/results.tex` (4.4 only), `sections/conclusion.tex`.

1. Abstract, replace the block between `\begin{abstract}` and `\end{abstract}` with:
```
Electrolyte perturbed-chain statistical associating fluid theory (ePC-SAFT) provides a molecular thermodynamic description of the reactive liquid that can be coupled directly to absorber performance.
A steady aqueous monoethanolamine (MEA) column formulation couples five reactions among nine liquid species to ePC-SAFT activities and fugacity, conventional reaction enhancement and empirical caloric properties.
Evaluation uses seven National Carbon Capture Center (NCCC) cases, a paired Case 3C mesh refinement and thermodynamic, transport and operating perturbations.
The capture mean absolute error is \num{5.85} percentage points; calculated speciation and fugacity profiles describe the thermodynamic coupling, while packing-temperature comparisons identify case-dependent discrepancies.
Refined Case 3C capture is \qty{91.55}{\percent} versus \qty{89.50}{\percent} observed, and mesh refinement changes capture by \num{0.005} points and peak liquid temperature by \qty{0.045}{K}; one coarse solution takes about \qty{35}{s}.
The tested \(\pm5\%\) thermodynamic and \(\pm10\%\) transport changes affect capture by up to \num{0.8} and \num{0.5} points, respectively; lean loadings of 0.225 and 0.275 mol \COtwo{} per mol MEA change capture by \num{+3.8} and \num{-6.9} points relative to 0.25.
The formulation links reactive thermodynamics to observable process behavior and provides a basis for predictive film modeling, thermodynamic-model comparison and constrained absorber optimization.
```
2. `introduction.tex`: replace the three lines from `In the selected MEA parameterization, the neutral-mixture stage refits` through `while retaining capture and temperature accuracy.` with: `Whether this reduces total fitting effort for reactive MEA has not been tested on a common column model; the parameter counts of the present ePC-SAFT set and of a published eNRTL set \cite{Akula2023a} do not by themselves establish an advantage (\cref{sec:selected-reactive-inputs}). A comparison within the same column model would test how much component reuse reduces new data and fitting requirements while retaining capture and temperature accuracy.` Also: `Computational cost is reported as supporting information for coarse Case 3C.` → `Computational cost is measured for the coarse Case 3C calculation.`
3. `results.tex` 4.4: keep R14. Replace everything from the `% R15a` comment to the end of the file with:
```
% R15: limitations and next steps
The formulation supports three follow-on studies.
A liquid-film formulation that calculates transfer from ePC-SAFT bulk and interfacial states, without an empirical enhancement factor, would obtain the film response from the coupled thermodynamic and transport description rather than from a physical-transfer coefficient multiplied by an enhancement correlation.
A matched ePC-SAFT versus eNRTL comparison in this column model, with fixed column configuration, inlet conditions and transport assumptions and internally consistent parameterizations \cite{Akula2023a,Zhang2011}, would separate literature-derived from newly fitted component, binary, ionic and reaction quantities and quantify the calibration data and effort each model requires.
Tests with diethanolamine (DEA), methyldiethanolamine (MDEA), 2-amino-2-methyl-1-propanol (AMP), piperazine (PZ) and blends, combining applicable component, association and ionic parameters with solvent-specific reactions and transport inputs, would show which inputs transfer and which require additional calibration.
The operating responses also motivate constrained optimization of solvent circulation, lean loading and inlet temperature against a process energy or cost objective that combines regeneration, circulation and gas-pressure losses under capture, hydraulic and thermal constraints, followed by assessment of packing height and operating pressure where the model supports them.
```
4. `conclusion.tex`: replace the C2, C3 and C4 blocks (everything from `% C2` to the line before `\clearpage`) with:
```
% C2: numerical behavior and cost
The refined Case 3C calculation predicts \qty{91.55}{\percent} capture against \qty{89.50}{\percent} observed; mesh refinement changes capture and peak temperature by far less than the remaining error, and one coarse solution takes about \qty{35}{s}.

% C3: thermodynamic, transport and operating insights
The MEA--water interaction produces the largest capture response among the tested thermodynamic inputs and the liquid-side transfer coefficient the largest among the transport inputs, but neither closes the observed capture gap within the tested ranges.
Lean loadings of 0.225 and 0.275 mol \COtwo{} per mol MEA change capture by \num{+3.75} and \num{-6.86} points relative to 0.25, identifying solvent loading as the consequential operating variable under the tested conditions.

% C4: contribution and forward program
The formulation gives a direct connection from component-level molecular and ionic descriptions to column-scale observables and operating response.
That connection supports a predictive liquid-film formulation, a direct ePC-SAFT/eNRTL comparison, tests with other amines and constrained column optimization, which together can evaluate how thermodynamic parameter reuse and explicit interphase modeling contribute to solvent selection and absorber design.
```

Removed-phrase grep: `Implementation has begun`, `already underway`, `neutral-mixture stage`, `supporting information`, `scientific value`, `reusable component descriptions`, ` will examine`, ` will hold`, ` will evaluate`.

## Batch 5: Model (Section 2) and Appendix A wording

Files: `sections/model_framework.tex`, `appendices/appendix_properties.tex`.

`model_framework.tex` (line numbers as of 2026-09-04):
- l.2 `Physical System and Model Responsibilities` → `Physical System and Model Structure`
- l.71 `on a declared activity basis` → `on the molality activity basis`
- l.82: delete the line `Before this reference transformation, R4 has ...`.
- l.83 `are identified in \cref{sec:selected-reactive-inputs}; their common selected temperature range is` → `are given in \cref{sec:selected-reactive-inputs}; their common temperature range is`
- l.140 `For the Born contribution, the implementation follows the 2025 update of Figiel et al. \cite{Figiel2025}` → `The Born contribution follows Figiel et al.\ \cite{Figiel2025}`
- l.142 `the selected model choices and numerical inputs are specified in` → `the model choices and numerical inputs are given in`
- l.172 `assembled from inherited, fitted, fixed and derived inputs` → `assembled from literature, fitted, fixed and derived inputs`
- l.174 → `No thermodynamic parameter was fitted to the NCCC cases; some of the thermodynamic measurements used in parameter selection are not independent of the parameterization (\cref{sec:selected-reactive-inputs}).`
- l.182 → `A pressure-drop correlation is evaluated only to confirm that each state lies within the hydraulic operating range; pressure is held constant in the balances.`
- l.215 `; numerical evaluation floors the raw value at \(10^{-12}\epsilon\).` → `.`
- l.264 `$s_{\COtwo}=1.04542981654115$.` → `$s_{\COtwo}=1.0454$ (full precision in \cref{sec:selected-reactive-inputs}).`
- l.266 → `Its calibration basis is not documented; it is treated as a fixed input of the enhancement correlation and does not enter the equilibrium formulation.`
- l.285 → `Numerical evaluation bounds the denominators away from zero and restricts $E$ to $1\leq E\leq10^4$; the holdup correlation is bounded at \(10^{-12}\epsilon\).`
- l.296 → `\Cref{tab:transport-applicability} gives the source ranges of these correlations; \cref{sec:results} compares them with the calculated states.`
- l.355 `the continuous implemented equations give` → `the continuous equations give`
- l.356 `For the implemented additive component-enthalpy correlation` → `For the additive component-enthalpy correlation`
- l.372 `constant in the retained correlation` → `constant in this correlation`
- Add to `appendix_reactive_inputs.tex` after the CO2 dispersion-energy sentence: `The enhancement concentration divisor is $s_{\COtwo}=1.04542981654115$.`

`appendix_properties.tex`:
- l.128 `follows this implemented correlation` → `follows this correlation`
- l.165 `Their coefficients specify the implemented functions; a source calibration range for these vapor formulas has not been recovered.` → `A source calibration range for these vapor formulas is not documented.`
- l.248 `have no recovered calibration range.` → `have no documented calibration range.`
- l.266 `use the implemented vapor-pressure correlation` → `use the vapor-pressure correlation`
- l.274 `The exact coefficients are retained with the property implementation; their original calibration source and range have not been recovered.` → `Their calibration source and range are not documented.`
- Subsection heading `Physical Carbon Dioxide Solubility` → `Physical Carbon Dioxide Solubility (Initialization Only)`.

Removed-phrase grep: `implemented`, `implementation follows`, `domain checking`, `not been recovered`, `declared`, `Responsibilities`, `1.04542981654115` (must appear exactly once, in the appendix).

## Batch 6: Appendix B provenance vocabulary and table captions

Files: `tables/epcsaft_parameter_summary.tex`, `tables/absorber_literature_comparison.tex`, `tables/nccc_one_bed_case_scope.tex`.

`epcsaft_parameter_summary.tex`:
- l.3 caption: `Selected component coefficients.` → `Component coefficients.`; `The CO$_2$ energy is selected by calibration; neutral coefficients otherwise are inherited. MEAH$^+$/MEACOO$^-$ size and energy values are inherited fitted inputs; other ion values are transferred inputs.` → `The CO$_2$ energy was chosen by calibration; the other neutral coefficients are from the literature \cite{Nasrifar2010,Najafloo2018,Held2014}. MEAH$^+$/MEACOO$^-$ size and energy values were fitted by the authors to ion and speciation data in work to be reported separately; other ion values are taken from the corresponding ions in \cite{Held2014,Uyan2015}.`
- l.23 caption: `Selected ionic diameters` → `Ionic diameters`; `The two amine-ion Born diameters are inherited fitted inputs; other Born diameters are transferred.` → `The two amine-ion Born diameters come from the same fit as their segment parameters; other Born diameters are taken from the corresponding ions in \cite{Held2014,Uyan2015}.`
- l.40 caption: `MEA--water is refitted, MEAH$^+$--MEACOO$^-$ is inherited from an ion fit, and the remaining nonzero nonunit values are inherited.` → `MEA--water was fitted to binary VLE data, MEAH$^+$--MEACOO$^-$ comes from the amine-ion fit, and the remaining nonzero nonunit values are from the literature.`
- l.62 `The selected relative permittivity of MEA` → `The relative permittivity of MEA`
- l.70 `The selected model uses` → `The model uses`
- l.89 `for these selected inputs` → `for these inputs`
- l.91 comment and l.92: `only the edges below are active` → `only the site pairs below are active`
- l.95 caption: `Association edges in the selected topology.` → `Active association site pairs.`; `Self-association values are inherited. The reciprocal CO$_2$--water edges are inherited induced-association inputs; MEA--water cross edges are derived by the stated combining rule.` → `Self-association values are from the literature. The reciprocal CO$_2$--water pairs are induced-association inputs from \cite{Schick2023}; MEA--water cross pairs follow the stated combining rule.`
- l.113 → `The MEA molecular coefficients are from Nasrifar and Tafazzol \cite{Nasrifar2010} and Najafloo and Zarei \cite{Najafloo2018}; water and non-amine ion inputs are from Held et al.\ \cite{Held2014} and Uyan et al.\ \cite{Uyan2015}.`
- l.114 `association edges` → `association site pairs`
- l.115 and the following amine-ion sentence (`Six MEAH+/MEACOO− size, dispersion and Born values and their mutual pair coefficient derive from a fit with seven varied parameters and are provisionally calibrated.`) → replace both with: `The six MEAH$^+$/MEACOO$^-$ size, dispersion and Born values and their mutual pair coefficient were fitted by the authors to ion and speciation data in work to be reported separately and have not been independently validated; the other literature values were not refitted.`
- l.117 `was selected jointly with R4` → `was chosen jointly with R4`

`absorber_literature_comparison.tex`: `not stated in the inspected article` → `not stated in the cited article`; `exact code revision not pinned` → `exact code version not specified`.

`nccc_one_bed_case_scope.tex`: order the footnotes a then b and give the unlabeled note a label (report the exact lines changed).

Removed-phrase grep: `inherited`, `transferred`, `provisional`, `lineage`, `edges`, `topology`, `inspected`, `pinned`, `Selected`.

## Batch 7: figure legends (regenerates artifacts)

Files: `analyses/nccc_validation/figures/reactive_column/scripts/render_sensitivity.py`, `analyses/transport_sensitivity/figures/response/scripts/render_figure.py`, `analyses/nccc_validation/figures/reactive_operating/scripts/render.py`.

- `render_sensitivity.py` l.92: `'Selected-parameter reference'` → `'Reference (unperturbed)'`
- `render_figure.py` l.156: `'Prior mesh/tolerance change'` → `'Mesh/tolerance refinement change'`
- `render.py` l.144: `'No accepted result: '+', '.join(...)+'. No zero response is implied.'` → `'Not converged (omitted): '+', '.join(...)+'.'`

Then rerun each script exactly as its figure README states (read the README first; do not change inputs), confirm only legend text changed by diffing the `output/*.svg` text and the provenance JSON, run `sync-figures`, build, and confirm via `pdftotext` that the three legend strings changed and nothing else. Report the provenance diff.

## Close-out (orchestrator)

After approval and execution of the consolidated plan: read the complete final PDF against both independent reviews and the consolidated decisions; review the full checkpoint-to-HEAD diff; run the prohibited-word and batch checks. Keep `docs/latex/scripts/reviewer_checklist.json` unchanged unless the author separately requests an update.
