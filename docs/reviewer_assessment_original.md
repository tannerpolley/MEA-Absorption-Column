I matched every reviewer comment against the current manuscript and response letter. Status judgments are `verified`; the 1–5 scores are `inference`—my assessment, not grades assigned by the reviewers.

Legend: ✅ addressed · ◐ partly addressed · blank = acceptance evidence still missing  
Score: 5 = complete, 4 = strong, 3 = partial, 2 = weak, 1 = absent

### Reviewer 1

| # | Category | Specific reviewer comment | Status | Before | After | Remaining revision/evidence |
|---:|---|---|:---:|:---:|:---:|---|
| 1 | Thermodynamic basis | Justify why ePC-SAFT can be applied to strongly reactive chemical absorption. | ✅ | 1 | 4.5 | Clearly re-scoped as the molecular fugacity closure, not the reaction model. |
| 2 | Model scope | ePC-SAFT is a physical EOS, not a reaction model; the manuscript blurred molecular and chemical thermodynamics. | ✅ | 1 | 5 | Chemistry and molecular thermodynamics are now explicitly separated. |
| 3 | Workflow | Identify which quantities come from ePC-SAFT and which come from reaction equilibrium; provide a schematic workflow. | ✅ | 2 | 4.5 | Responsibility table and workflow now establish the separation. |
| 4 | Terminology | Define “ePC-SAFT fugacity benchmark.” | ✅ | 2 | 5 | The term is explicitly defined and bounded. |
| 5 | Reproducibility | Provide a complete ePC-SAFT parameter table. | ◐ | 2 | 4 | All nine species are represented, but some source provenance remains incomplete. |
| 6 | Binary interactions | State the adopted binary interaction parameters \(k_{ij}\). | ✅ | 1 | 5 | The currently used interaction treatment is disclosed. |
| 7 | Parameter provenance | State whether \(k_{ij}\) values were fitted or taken from literature. | ◐ | 1 | 3 | Missing historical provenance is honestly disclosed, but not recovered. |
| 8 | Sensitivity | Quantify sensitivity to thermodynamic parameters. |  | 1 |  | Requires a frozen parameter family, defensible perturbation ranges or covariance, and complete reruns. |
| 9 | Transport properties | Discuss uncertainty from mass transfer, diffusivity, and viscosity. | ✅ | 2 | 4.5 | Qualitative uncertainty propagation and limitations were added. |
| 10 | Numerical convergence | Report iterations, Jacobian evaluations, nonlinear residual histories, and mesh refinement. |  | 2 |  | Requires an instrumented rerun; acknowledging missing historical diagnostics does not satisfy the requested evidence. |
| 11 | Computational cost | Quantify CPU time, memory, mesh size, and nonlinear iterations on identical hardware. |  | 2 |  | Same-hardware profiling and memory measurements remain needed. |
| 12 | Other amines | Discuss extension to DEA, MDEA, AMP, PZ, and blends, including additional parameterization. | ✅ | 1 | 4 | Required chemistry, molecular parameters, transport data, and validation are described; no second-solvent demonstration yet. |
| 13 | Limitations | Expand limitations concerning parameters, reactive electrolytes, industrial flue gas, and kinetics. | ✅ | 2 | 5 | Limitations and claim boundaries are now explicit. |
| 14 | Model context | Compare eNRTL, Kent–Eisenberg, CPA, and ePC-SAFT. | ✅ | 2 | 5 | A direct comparison and model-selection context were added. |

### Reviewer 2

| # | Category | Specific reviewer comment | Status | Before | After | Remaining revision/evidence |
|---:|---|---|:---:|:---:|:---:|---|
| 1 | Formatting | Correct the reference numbering so citations begin with [1]. | ✅ | 2 | 5 | Completed. |
| 2 | Literature gap | Clarify shortcomings in the existing MEA absorber literature and the manuscript’s contribution. | ✅ | 2 | 4.5 | The literature gap and fixed-chemistry benchmark contribution are clearer. |
| 3 | Reproducibility | Present the parameters needed for peer reproduction. | ◐ | 2 | 3.5 | Parameter values and workflow improved, but a source-complete immutable parameter packet is still needed. |
| 4 | Validation | Validate against more reported MEA absorber results. | ◐ | 2.5 | 3 | The common-gate NCCC set was reconciled to nine accepted cases; independent cross-facility validation remains absent. |
| 5 | Operating study | Evaluate performance across operating conditions and identify optimum operation. |  | 1 |  | Requires a controlled operating envelope, constraints, objective function, sensitivity study, and reproducible optimization. |
| 6 | New amines | Explain how parameters would be fitted for new amines. | ◐ | 1 | 3.5 | Required inputs are described, but no complete calibration and held-out validation workflow has been demonstrated. |

### Category-level gradecard

| Category | Included comments | Addressed? | Before | Current |
|---|---|:---:|:---:|:---:|
| Thermodynamic theory and scope | R1.1–R1.4 | ✅ | 1.5 / 5 — F | 4.8 / 5 — A |
| Parameters, provenance, and sensitivity | R1.5–R1.8; R2.3 | ◐ | 1.4 / 5 — F | 3.2 / 5 — C |
| Transport-property uncertainty | R1.9 | ✅ | 2.0 / 5 — D | 4.5 / 5 — A |
| Numerical convergence and cost | R1.10–R1.11 |  | 2.0 / 5 — D |  |
| Other amines and transferability | R1.12; R2.6 | ◐ | 1.0 / 5 — F | 3.7 / 5 — B |
| Limitations and model context | R1.13–R1.14; R2.2 | ✅ | 2.0 / 5 — D | 4.8 / 5 — A |
| Formatting and references | R2.1 | ✅ | 2.0 / 5 — D | 5.0 / 5 — A |
| Validation breadth | R2.4 | ◐ | 2.5 / 5 — D | 3.0 / 5 — C |
| Operating study and optimization | R2.5 |  | 1.0 / 5 — F |  |
| Fully predictive reactive ePC-SAFT | Cross-cutting scientific endpoint |  | 1.0 / 5 — F |  |

Overall assessment:

- Before reviewer revisions: approximately **1.8/5 — D**
- Current manuscript against every requested item: approximately **3.5/5 — B-/C+**
- Current re-scoped fixed-chemistry benchmark considered on its stated claims alone: approximately **4.2/5 — B+/A-**

The largest completed improvement is the claim boundary: the manuscript no longer presents ePC-SAFT as if it independently predicts reactive MEA chemistry. It now distinguishes reaction equilibrium, molecular fugacity calculations, transport closure, and column simulation.

The blank predictive-reactive row should remain blank until one frozen parameter set produces reactive pressure, vapor composition, and speciation across every admitted state; supplies consistent derivatives; carries immutable wheel and parameter identities; and passes both thermodynamic and column-level validation. `verified`: the canonical plan currently records the August 22 freeze as missed and the fallback manuscript as active, so completion before August 27 remains possible only as new evidence—not something that can yet be checked or promised. See the [canonical revision plan](/home/tnnrpolley21/Workspaces/Engineering/MEA-Absorption-Column/docs/coordination/august_27_predictive_reactive_epcsaft_revision_plan.md:4) and [reviewer response](/home/tnnrpolley21/Workspaces/Engineering/MEA-Absorption-Column/docs/reviewer_response.md:1).
