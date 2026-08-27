# Scientific context

Status: official ePC-SAFT downstream application under Governance D-038.

Current program: `FALLBACK_ACTIVE` fixed-chemistry submission with a gated
predictive-transfer path. This is the sole current integration/manuscript
program. The August 27 plan and reactive-VLE upstream handoff are retained as
dated history, not live plans. Native GitHub issues are the only work queue.

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

## Current decision

The August 27 submission is the fixed-chemistry manuscript and reviewer
response. Closed MEA-Thermodynamics issue #70 reached a supported-negative
decision: no predictive MEA parameter set was adopted or authorized for column
transfer.
The immutable refusal is recorded at upstream commit
`0ce38270150fbb5d8dcbafc34988d18f04a59f7c` by decision record SHA-256
`47da30a7cd75c95c53677766738e338f5af31069f6ad81fd855853bcd2083779`
and transfer-refusal SHA-256
`6a14cfa988660596fcbeb7516b7123c45da558ef00311b72306fdcb1b859e835`.

Engine issues #30, #31, #44, and #48 and MEA-Thermodynamics issues #61--#67,
#13, and #14 are closed historical prerequisites. MEA-Thermodynamics #68
remains open for its upstream manuscript work, not as a column transfer.
Closing the prerequisites did not create an accepted predictive parameter set.
Future capability and application work remains in Engine #79,
MEA-Thermodynamics #72, and local issues #3 and #16.

## Evidence and claim ladder

1. **Numerical feasibility:** the archived activity-rebased nine-species sweep
   shows that the calculation returned under its stated settings; it is not
   predictive chemistry.
2. **Fixed-chemistry comparison:** the Henry-law and `epcsaft_ionic` NCCC lanes
   compare driving-force closures while retaining concentration-based
   chemistry. This is the current manuscript claim.
3. **Predictive thermodynamics:** this lane may exist only after the transfer
   gate below accepts one immutable MEA parameter/result identity and Engine
   wheel receipt. No such lane is currently accepted.
4. **Absorber validation:** column convergence, NCCC comparison, sensitivity,
   and independent validation follow transfer; they are not MEA calibration or
   reserved thermodynamic validation.

Existing NCCC evidence validates only its recorded fixed-chemistry lanes. It
must not be reclassified as validation of a future thermodynamic candidate or
used to fit thermodynamic parameters.

## Predictive-transfer gate

A future predictive lane requires all of the following through GitHub issues:

- an adopted, immutable MEA parameter document plus hash-identified
  data/result and promotion decision from MEA-Thermodynamics;
- a non-editable Engine wheel receipt naming the Engine commit, wheel filename,
  wheel SHA-256, and admitted capability identity;
- downstream mapping, integration, convergence, and column-validation evidence
  generated without sibling-source imports or case-specific fitting; and
- CSE Review Pass for the stated predictive and absorber-validation claims.

Until every item passes, the supported-negative refusal remains the current
downstream decision and the fixed-chemistry manuscript boundary is unchanged.

## Locked Stage A runtime

The retained Stage A results use Engine commit
`d88f703974fa8d6e7be54ca3cbd51b6f0f78a372` and wheel SHA-256
`81f21a6226de1fb68ca992c17f25e1a4ff7b791d3806220fefca31f7ad615f80`.
That lock reproduces Stage A only. A newer upstream wheel, including one used
for diagnostic fitting, is neither this locked runtime nor evidence that a
predictive parameter set was accepted.

## Manuscript boundary

Existing fixed-chemistry column evidence remains distinct from a future
predictive MEA lane. A candidate Engine wheel or parameter document does not
become a manuscript result until the applicable column validation is complete.
