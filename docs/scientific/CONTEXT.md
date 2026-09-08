# Scientific context

Updated 2026-09-03. Official ePC-SAFT downstream application under Governance
D-038. The current foreground work is the complete living manuscript in
`docs/latex/main.tex`, with result insertions tracked by its existing checklist.

## Question and intended use

When does additional thermodynamic and film detail materially change or improve
predicted capture and axial temperature, and what numerical effort is required
to resolve that difference?

The article retains three controlled studies: Henry/eNRTL/ePC-SAFT
thermodynamics; enhancement-factor/equilibrium-manifold transport; and
boundary-value accuracy and computational cost. Common-property comparisons
precede full-property comparisons. Their changes are conditional on the other
model choices, not a unique additive separation of interacting effects.
Capture and temperature together assess the engineering response.

The abstract and value-independent formulation/methods can be complete before
all figures arrive. Placeholders identify only missing numerical inputs,
figures, or result-specific findings. Source questions and development notes
belong in SOURCE_MAP.md or checklist details, not in the article.

## Scientific and repository boundaries

- ePC-SAFT-project owns generic EOS equations, reactive equilibrium, exact
  derivatives, and caloric implementation.
- MEA-Thermodynamics owns MEA parameter fitting, model selection, adopted
  thermodynamic inputs, and their thermal reconstruction.
- This repository owns absorber integration, film/column comparisons,
  numerical verification, experimental comparison, and the manuscript.

Use a non-editable Engine wheel identified by commit and SHA-256. Do not import
sibling source trees, copy Engine equations into the absorber, or mix parameter
identities. Manuscript-only edits do not authorize wheel installation or model
execution. Final result use requires the repository's immutable integration
check and the evidence appropriate to the claim.

## Current formulation and evidence

The ePC-SAFT formulation uses nine aqueous species with complete material, mass,
and charge constraints. Thermodynamic comparators retain their specified species
and reaction systems on common analytical feed bases; an isolated closure-effect
claim additionally requires common species and reaction thermodynamics. Its CO2 film is a local-equilibrium, isothermal mobility
approximation. The conservative CasADi/IPOPT formulation distinguishes physical
balance quantities, numerical coordinates, constraint derivatives, and the
outer Hessian approximation. It retains explicit SciPy comparisons.

The latest inspected implementation and result owners, exact task/worktree
locations, parameter/Engine identities, caloric inputs, source questions, and
incoming figure requirements are maintained in
[the manuscript source record](../latex/SOURCE_MAP.md).
Read that record and the actual producer files before making a time-sensitive
readiness claim. A file's existence does not establish completed column
results. Keep capture, temperature profiles, parameters, and numerical checks
from one identified run together.

The selected caloric reconstruction supplies nine reference functions from
three neutral anchors, five transformed reaction constraints, and one ionic
reference convention. Its insertion is parameter reporting, not column
validation. Compare its parameter and Engine identities with the final run's
identity; the selected upstream reaction and caloric coefficients have been
inserted together. SOURCE_MAP.md distinguishes their identities from earlier
figures and the later column execution identity.
Six-species electrolyte-NRTL reference states are available, while complete
portable parameter records and the comparison species/reaction mapping,
temperature-tap coordinate provenance,
and the enhancement concentration-factor rationale remain specific source
questions in SOURCE_MAP.md.

## Historical evidence

The August fixed-chemistry submission and supported-negative predictive-transfer
decision remain historical evidence for their exact parameter/run identities.
They do not define the September manuscript's structure or make newer candidate
results valid. The prior finite-rate film formulation is historical, distinct
from the current equilibrium-manifold approximation.
Historical energy-sign-affected column results must not be relabeled as corrected
results. Keep observation, numerical verification, physical validation, and
parameter fitting distinct. Do not infer a better thermal model from capture
agreement alone.

## Repository terminology

| Avoid | Prefer | Meaning | Scope | Exceptions | Evidence |
|---|---|---|---|---|---|
| same bulk thermodynamic state | same thermodynamic model and feed conditions | Full-column paired runs solve their own axial bulk states; a fixed-state local flux comparison is a separate calculation. | docs/latex/sections/introduction.tex; docs/latex/sections/results.tex; docs/latex/sections/conclusion.tex | | Investigator manuscript correction, 2026-09-03; methods comparison-design subsection |

Other scientific terms retain their ordinary definitions. In particular,
fixed-composition Cp excludes equilibrium redistribution, while equilibrium Cp
includes the change in species amounts at fixed pressure and conserved totals.
