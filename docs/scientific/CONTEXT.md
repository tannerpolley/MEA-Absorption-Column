# Scientific context

Status: official ePC-SAFT downstream application under Governance D-037.

## Question and intended use

Evaluate and use MEA thermodynamic models inside absorber-column calculations,
validate column predictions against retained campaign evidence, and maintain the
column manuscript. This repository does not own generic ePC-SAFT equations or
adopt MEA thermodynamic parameters.

## Repository boundary

- `ePC-SAFT-project` owns the generic Engine, GREPE, exact derivatives, and
  regression mechanics.
- `MEA-Thermodynamics` owns MEA chemistry hypotheses, parameter fitting, model
  selection, and thermodynamic parameter adoption.
- This repository owns absorber integration, column validation, process
  analyses, and its manuscript.

The repositories remain separate siblings. This application consumes a
non-editable Engine wheel identified by Engine commit and wheel SHA-256. A
predictive MEA parameter document enters only after MEA-Thermodynamics freezes
and identifies it; no runtime sibling-source import or nested Git repository is
permitted.

## Claim boundary

Existing fixed-chemistry column evidence remains distinct from a future
predictive MEA lane. A candidate Engine wheel or parameter document does not
become a manuscript result until the applicable column validation is complete.
