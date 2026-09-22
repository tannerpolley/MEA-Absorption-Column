# Scientific context

Updated 2026-09-08. This project studies when thermodynamic and film detail
changes predicted capture and axial temperature, and what numerical effort
resolves those differences. The manuscript revision is submitted; the active
objective is exploratory analysis and research, with possible later promotion.

## Terms

- **Formulation:** the balances, state variables, property relations and interphase
  equations solved together. Changing formulation may change physics as well as numerics.
- **Thermodynamic model:** the equations and identified parameter/reaction inputs
  used to obtain phase properties, speciation and chemical potentials.
- **Film model:** the interphase transport closure, including enhancement,
  a frozen conductance profile, or explicitly coupled film equations.
- **Numerical method:** the discretization, integration and nonlinear solution
  choices applied to a specified formulation.
- **Research configuration:** explicit choices of formulation, models, method,
  input basis, initial conditions and numerical settings. A convenient example
  is not the only permitted case.
- **Retained attempt:** inputs, executable identity, returned values, diagnostics
  and failures saved together. Solver convergence does not establish physical accuracy.
- **Promotion:** a reviewed finding selected by the investigator for a durable
  scientific claim. Exploratory notebooks may document preliminary and negative
  results without implying promotion.
- **Submitted revision:** the unchanged document and evidence preserved on its
  archive branch, separate from ongoing research.

## Ownership and scientific comparisons

The Engine owns generic thermodynamic equations, equilibrium and derivatives;
MEA-Thermodynamics owns fitting and parameter adoption; this repository owns
absorber integration, transport/column studies and numerical comparisons.
Identified candidate inputs may be explored without claiming prior validation.
Preserve units, species order, charge, conservation, finite-domain requirements,
failures and exact input identities.

Compare model changes on explicitly stated common input bases. A controlled
closure comparison must hold its other equations and inputs fixed. Capture and
axial temperature assess different aspects of the response. Estimated mobilities,
uncertain thermal references, incomplete convergence and observation ambiguities
remain visible. No selected manuscript case constrains the research program.
