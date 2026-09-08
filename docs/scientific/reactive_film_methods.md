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

Issue 41 now retains the source-rate evidence separately from that architecture
test. Putta2016 F1/F2 concentration and activity forms, source domains, source
units, and the 20-cell aggregate AARD comparison are preserved; the printed
third-order `s^-2` unit is rejected by dimensional reconstruction. The
Gondal2015 F3 coefficient and row-level rate observations remain unavailable,
and the declared Luo/WWC/SDC and literature apparatus split has no retained row
IDs or uncertainty weights. The immutable provider bundle supplies `K(T)` on
its aqueous-molality standard state, but Issue 40 admits no true-species row:
all five packet candidates remain `basis_unresolved`. Consequently `ln Q`,
detailed-balance residuals, reaction timescales, rate fitting, and physical
film adoption remain unevaluated. The retained result is a supported-negative
source-evidence record, not a predictive kinetic model.

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

Before a physical-basis run, the analysis enforces the Work Package A
temperature, exact MEA-molarity, and loading domain and requires admitted
finite-rate coefficients and diffusion inputs. The retained column states stop
at this preflight gate; rejected inputs never enter the film solver.

## Acceptance and stopping rules

Admitted solutions require positive finite concentrations, interface closure,
species conservation, stoichiometric invariants, electroneutrality, zero
current, mesh agreement, and initialization/branch agreement. Provider domain
failures remain typed failed rows.

Maxwell-Stefan comparison starts only after the common reversible chemistry,
thermodynamic, and transport basis passes. A reactive-film flux is never
followed by E, Psi_H, eta_psi, or another fitted multiplier. Rate-data and
column comparisons require source-complete observations and admitted inputs;
otherwise the workflow stops without a manuscript claim.
