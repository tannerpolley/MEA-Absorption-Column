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
