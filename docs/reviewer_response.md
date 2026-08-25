# Response to reviewers

Manuscript: *A Reproducible MEA Absorber Benchmark for ePC-SAFT CO2 Fugacity
Driving Forces*

We thank both reviewers for identifying places where the thermodynamic scope,
parameter provenance, numerical evidence, and validation boundary were not
sufficiently explicit. We revised the manuscript to distinguish the physical
equation of state from chemical equilibrium and to state exactly which claims
the retained evidence supports. We did not add results that could not be
reproduced from the frozen artifacts.

## Reviewer 1

### Comments 1--2: theoretical basis and physical versus chemical thermodynamics

**Response.** We agree that ePC-SAFT is not, by itself, a reaction model. The
Introduction and *Model framework* now state that chemical equilibrium supplies
the reaction network, equilibrium constants, and true-species concentrations,
whereas ePC-SAFT supplies mixture density, residual-Helmholtz contributions,
and vapor- and liquid-side CO2 fugacity coefficients for the interfacial
driving force. The routine benchmark therefore tests an electrolyte-EOS
fugacity closure under fixed concentration-based chemistry; it does not claim
predictive reactive ePC-SAFT equilibrium.

### Comment 3: identify predicted quantities and provide a workflow

**Response.** Figure 1 and the new thermodynamic-responsibility table in *Model
framework* now trace apparent component amounts through chemical equilibrium,
the selected phase-equilibrium closure, transport and enhancement calculations,
and the column balances. The table identifies the inputs and outputs owned by
each block and makes clear which blocks are unchanged between the Henry-law and
ePC-SAFT rows.

### Comment 4: define “ePC-SAFT fugacity benchmark”

**Response.** The Introduction now defines this term as the controlled
replacement of the local phase-state-to-fugacity calculation while chemistry,
transport, hydraulics, balances, solver settings, and acceptance gates remain
fixed. The definition explicitly excludes a claim that ePC-SAFT supplies the
reaction network or equilibrium constants.

### Comments 5--7: parameter table, binary interactions, and fitting

**Response.** The revised parameter table reports the stored parameters for all
nine species, including molecular weight, charge, and Born diameter where
applicable. The appendix and parameter-provenance discussion identify the only
nonzero stored binary value, $k_{MEA,H2O}=-0.052$, and state that it was not
fitted to the NCCC absorber rows. They also identify provisional, transferred,
and placeholder ionic records and disclose the association-scheme mismatch in
the lineage of that binary value. These disclosures bound reproducibility and
prevent the selected artifact from being presented as a provenance-complete
predictive parameterization.

### Comment 8: thermodynamic-parameter sensitivity

**Response.** We agree that a quantitative sensitivity study would strengthen
the benchmark. We did not label the existing historical configuration screen
as uncertainty quantification because it does not provide defensible parameter
ranges or covariance. *Results and discussion* and *Conclusions* now state that
thermodynamic parameter uncertainty was not propagated. A valid follow-up must
first select one internally consistent parameter family, establish source- or
fit-supported ranges, and then rerun the accepted column campaign. This remains
outside the evidence frozen for this revision.

### Comment 9: transport-property uncertainty

**Response.** *Model framework* now explains how viscosity and diffusivity
propagate through Reynolds and Schmidt numbers, film coefficients, Hatta
number, interfacial area, holdup, and heat transfer. The text states that these
correlations are held fixed in the thermodynamic comparison and that no
transport-correlation uncertainty is propagated.

### Comments 10--11: numerical diagnostics and computational expense

**Response.** *Numerical methods* now distinguishes persisted evidence from
diagnostics that were not archived. The reported rows retain starting mesh,
maximum-node limit, tolerances, residual information, convergence message,
wall runtime, validation error, Python version, platform, and package versions.
They do not retain final adaptive node counts, refinement histories,
method-specific iterations or evaluations, CPU time, peak memory, processor,
RAM, BLAS threads, or system load. Runtime is consequently described as
comparative wall time, not a controlled hardware benchmark. Recovering the
requested missing quantities requires a same-hardware rerun with an expanded
result schema; they cannot be reconstructed reliably from the archived rows.

### Comment 12: extension to other amines

**Response.** *Results and discussion* and *Conclusions* now state that the
balance and solver framework can be reused, but another solvent is not a
parameter-file substitution. DEA, MDEA, AMP, PZ, or a blend requires a declared
reaction network and standard state, traceable pure/binary/association/
electrolyte parameters, solvent-specific properties, transport and kinetics,
an enhancement model, and independent thermodynamic and absorber validation.

### Comment 13: limitations

**Response.** The revised conclusion now bounds the study to the selected MEA
parameter set, fixed transport and enhancement correlations, one-bed NCCC
cases, and archived numerical diagnostics. It identifies the absence of
thermodynamic and transport uncertainty propagation, independent cross-facility
validation, controlled operating optimization, predictive nine-species
chemistry, and demonstrated extension to another amine.

### Comment 14: comparison with other thermodynamic models

**Response.** The Introduction now includes a concise comparison of
Kent--Eisenberg, eNRTL, CPA, and ePC-SAFT. It distinguishes their primary
thermodynamic roles, practical strengths, and calibration or standard-state
requirements without asserting a universal accuracy ranking.

## Reviewer 2

### Comment 1: reference numbering

**Response.** The Elsevier numeric bibliography style is now used. The first
in-text citation and reference-list entries begin at 1.

### Comment 2: shortcomings in prior absorber comparisons

**Response.** The revised Introduction identifies the specific reproducibility
gap: studies often change thermodynamics, solver formulation, acceptance
criteria, and validation subsets together, making the source of a performance
difference difficult to isolate. We no longer imply that MEA absorber modeling
itself is absent from the literature.

### Comment 3: parameters for reproduction

**Response.** The expanded appendix table and generated parameter-provenance
tables expose the selected pure, ionic, binary-interaction, and relative-
permittivity records. The manuscript also identifies provisional and
inconsistent records instead of implying a complete predictive fit.

### Comment 4: validation against other reported results

**Response.** We reconciled every attempted one-bed NCCC row under one stated
acceptance gate: K18, K19, and 1C--7C are included for both closures, while K20
remains visible as rejected. Temperature-profile evidence is reported where
measured taps are available. We agree that an independent facility or
literature dataset would strengthen external validity, but it requires a
source-traceable conversion of inputs and was not added without a completed
validation packet. The conclusion now states this cross-facility limitation.

### Comment 5: operating-condition study and optimization

**Response.** The NCCC observations span different conditions, but their inputs
co-vary and do not constitute a controlled design. The model also lacks a
plant-level energy or economic objective. *Results and discussion* therefore
does not infer an optimum and specifies what a defensible study would require:
predeclared decision variables, hydraulic and process constraints, an energy
or cost objective, a validation domain, and uncertainty analysis.

### Comment 6: fitting another amine

**Response.** The revised manuscript gives the same solvent-extension boundary
described in our response to Reviewer 1, Comment 12. The numerical framework is
reusable, but chemistry, thermodynamics, properties, transport, kinetics,
enhancement, and validation must be established for the new solvent.

## Revision evidence boundary

The revised manuscript retains the archived fixed-chemistry Henry-law versus
ePC-SAFT fugacity comparison. A separate nine-species calculation is labeled
only as numerical-feasibility evidence because its reaction constants were
locally rebased rather than independently sourced. A newer upstream
pressure/speciation candidate was not admitted into this revision: its frozen
application contract, observation admission set, immutable artifact receipt,
and downstream column acceptance evidence are incomplete. Accordingly, the
manuscript makes no predictive-reactive-ePC-SAFT claim.
