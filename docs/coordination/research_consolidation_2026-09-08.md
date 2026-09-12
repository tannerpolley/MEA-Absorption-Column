# Research consolidation — 8 September 2026

The investigator authorized preserving the submitted revision remotely, pruning
redundant branches and consolidating research. No manuscript edits or numerical
reproduction were requested. The active objective is selectable thermodynamics,
film models, numerical methods and case/configuration choices, documented in
analysis and Quarto notebooks rather than manuscript prose.

## Preserved source

- Submitted revision: branch `codex/fallback-manuscript`, commit `eef9dce`.
  ZIP bytes, 33 revised-source files and 1,451 integrity entries checked.
- Coupled solver worktree: commit `79f8b6cae8b65a4f0bf3d9f05aaba7d476dda5f4`,
  tag `archive/coupled-solver-2026-09-08`. Contains all 98 changed/new files;
  original worktree and branch remain untouched. Not a validation claim.
- Exploratory film tip: tag `archive/exploratory-film-2026-09-08` at `edd9aac`.
- CasADi comparison: source `c65e399`, distinct analysis files imported.
- Issue16 reachability: source `c042eb4`, notebook and bounded evidence imported.

Main and overhaul histories were merged. Shared runtime conflicts retained the
submitted nine-species implementation and input identity; older film research,
source evidence and negative results were retained without relabelling their
scientific scope. The newer standalone manifold-film implementation and its
analytic checks were retained together. The cleanup from `501e86c` was not applied
wholesale: its removal of continuation and alternative numerical paths conflicts
with the requested research flexibility. Selected implicit enhancement now fails
explicitly instead of substituting a different model.

The twelve-state coupled builder requires a different thermodynamic action and
caloric interface. It remains retrievable at its exact checkpoint. Independent
conservative solver modules were incorporated without substituting its physics
into the seven-state formulation. `research.run_conserved` accepts a compatible
study-supplied problem and selects a method explicitly.

## Branch dispositions

Removed redundant pointers: quarto-latex-pilot, r18-absorber-sensitivity,
absorber-default-runtime, enrtl-column-comparison and transport-sensitivity
(all with the `codex/` prefix). Removed bvp-derivative-trials and
exploratory-chemical-potential-film after separately authorized removal of their
clean worktrees. The latter exact tip is preserved by its archive tag.

MDEA remains deferred research on its own branch. Issue47's distinct differential
film formulation remains a reference branch/worktree. The dirty Section 4.3 solver
worktree remains untouched and independently preserved. No publisher PDFs or Engine
binaries were uploaded. Binary prerequisites and original identities remain explicit.

## Checks and limits

No manuscript column results were rerun. Archive integrity, non-numerical dependency
inspection, mocked configuration/dispatch checks, independent analytic method/film
checks and source syntax checks were used. The HTML notebook was rendered with
execution disabled. Submitted manuscript files remain byte-identical to the archive.
Restored Issue40 analysis imports now pass through the existing thermodynamics
adapter; old run/source hashes remain unchanged and do not identify that edit.

The full-column twelve-state comparison, eNRTL implementation, MDEA validation and
exact numerical replay with every earlier Engine wheel remain separate work.
No new physical accuracy or timing claim follows from consolidation.

The prior implicit 10-point observed-capture gate for shooting/finite differences is now opt-in. Failed collocation final-iterate acceptance also requires explicit selection. Numerical completion and experimental agreement remain distinct.

## Formulation runner verification — 12 September 2026

Issue 52's approved implementation now supplies immutable one-case configuration,
resolved SI inputs, verified worker/wheel identity, native profile retention and
the named twelve-state graph with shared transport and native thermodynamic
callbacks. Graph preparation is separate from a full-column solve. Capture
agreement cannot grant numerical or physical acceptance; unsuccessful multistart
candidates and preparation timeouts retain their failure meaning.

The initial check set passed 146 focused tests and the final Engine integration
check. After four runner corrections, 55 configuration/benchmark checks passed.
The native assembly/film check set passed 12 tests, including A2/caloric actions
and the 19-by-12 node directional Jacobian. Two live positive-transform tests
exceeded 45 seconds in the changed tree; the ordinary single-bed comparison
passed separately. The positive flow-pressure failure also reproduces at
`26fa66c`: hydraulic domain violations precede a singular collocation Jacobian
and rejection with boundary residual 10503.3. It is not introduced by this work.
Selection now requires both existing result acceptance and every required solver
stage; the complete four-case acceptance/stage check failed before replacement
and passed afterward. The corrected configuration/benchmark set passed 55 tests.

A fresh-process comparison used base `26fa66c` and the changed source with the
same prepared interpreter, pinned Engine, Case 3C, ideal Henry model, single bed,
SciPy BVP, 11 initial nodes, tolerance 1, boundary tolerance 0.01 and 120 maximum
nodes. Both converged in one iteration on 11 nodes. Capture was
91.4625148309739% at the base and 91.46251483078366% with the changes, a difference
of 1.90e-10 percentage points; both boundary residual norms were
5.6616491335583666e-14. This checks preservation of that numerical path, not
physical accuracy or the conserved-column solution.

Change accounting against `26fa66c` reports 3,276 gross executable additions and
121 removals, within the approved 3,500-line ceiling. New equations remain in the
coupled column and existing transport owners; generic thermodynamic equations
and derivatives remain in the immutable Engine. Submitted evidence is unchanged.
