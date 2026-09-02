# Reactive-film formulation and numerical method

Status: issue 16 architecture and gate evidence. Placeholder-dependent results
are `provisional_concept_only` and are not manuscript evidence.

## Species and reactions

The liquid vector is ordered as CO2, MEA, H2O, MEAH+, MEACOO-, HCO3-,
CO3^2-, H3O+, and OH-. A finite reaction has a reversible net rate

\[
r_j = r_{j,+}(\mathbf f,T)-r_{j,-}(\mathbf f,T), \tag{RF-1}
\]

where \(\mathbf f\) is the single species-fugacity basis supplied by ePC-SAFT.
The solver accepts multiple stoichiometric columns and never imposes an exact
equilibrium constraint on a finite-rate reaction at the same node.

The Work Package A source contract classifies F1, F2, and F3 as finite, but its
numeric kinetic coefficients are rejected or unavailable. The implementation
therefore tests reversible architecture with manufactured rates and does not
promote those rates to a physical MEA model.

## Effective-Fick film

For film coordinate \(z\), positive from interface to bulk,

\[
N_i=-D_i\frac{dC_i}{dz}, \qquad
\frac{dN_i}{dz}=\sum_j \nu_{ij}r_j. \tag{RF-2}
\]

At the bulk edge, \(C_i=C_{i,b}\). At the interface, only CO2 crosses the
mathematical boundary; every other species flux is zero. CO2 closes against
the gas film as

\[
N_{CO_2}(0)=k_g\left[f_{CO_2,g,b}-f_{CO_2,l}(0)\right]. \tag{RF-3}
\]

The boundary Jacobian consumes the installed provider's exact fixed-\(T,P\)
charged-composition tangent. For a CO2 concentration perturbation with every
other concentration fixed,

\[
\frac{\partial f_{CO_2}}{\partial (C_{CO_2}/C_{CO_2,b})}
=\frac{f_{CO_2}}{C_{CO_2}/C_{CO_2,b}}
\frac{\partial\ln f_{CO_2}}{\partial\ln C_{CO_2}}. \tag{RF-4}
\]

No downstream EOS equation, production finite difference, projection, or
fallback supplies RF-4.

## Numerical method

The existing collocation solver advances concentration ratios and fluxes by
reaction-strength continuation. For charged states it eliminates one charged
species from both vectors: local electroneutrality reconstructs its
concentration and zero current reconstructs its flux. The boundary Jacobian
uses RF-4 directly. Mesh doubling and three initial-flux factors test refinement
and branch agreement; provider rejections and solver failures retain their
typed error and stopping gate.

For provisional numerical method development, the analysis records the Work
Package A temperature, discrete MEA source-label, loading, finite-rate
coefficient, and diffusion-input status on every row without using those source
limitations as an unconditional exception. The calculation reports separately
whether a row was declared, reached the governing film equations, returned a
result, and passed scientific input admission.

The retained column-derived states use manufactured relative-fugacity rates and
retained diffusivities. They may establish numerical reachability and
repeatability only. Scientific adoption still requires source-admitted kinetics,
transport inputs, state domain, parameter identity, and Review Pass.

## Acceptance and stopping rules

Numerically accepted provisional solutions require positive finite concentrations, interface closure,
species conservation, stoichiometric invariants, electroneutrality, zero
current, mesh agreement, and initialization/branch agreement. Provider domain
failures remain typed failed rows.

Maxwell-Stefan comparison starts only after the common reversible chemistry,
thermodynamic, and transport basis passes. A reactive-film flux is never
followed by E, Psi_H, eta_psi, or another fitted multiplier. Rate-data and
column comparisons require source-complete observations and admitted inputs;
otherwise the workflow stops without a manuscript claim.
