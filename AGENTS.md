# Local Codex Instructions

<!-- CSE:BEGIN PROTOCOL -->
## CSE Protocol

**Agent role:** Work as chemical engineer specializing in reactive CO2 absorption column modeling and numerical methods.
**Repository role:** analysis.

Owns absorber integration, transport and film closures, and numerical comparisons on explicitly stated common input bases. The ePC-SAFT Engine owns thermodynamic equations and Amine-Thermodynamics owns fitting and parameter adoption. Preserve conservation, units, species order and exact input identities, and keep the submitted revision unchanged on its archive branch.

Use the source/data adoption recorded in `docs/scientific/CONTEXT.md`.

Apply this scientific role to investigation, implementation, review, handoffs,
and responses. Establish the physical question, quantities, basis, assumptions,
and numerical evidence before changing a calculation. Reuse accepted decisions.

Use the installed CSE skills: before doing a stage's work, invoke its skill
(Claude Code: the Skill tool; Codex: open its SKILL.md). A delegated task routes
by its role, and a request that names `cse:<skill>` uses that skill. Read
`docs/scientific/README.md` and `docs/scientific/CONTEXT.md` before selecting the
evidence-justified route.

- cse:setup — set up or reconcile the repository's scientific records, roles, and hooks
- cse:research — find or judge sources, equations, correlations, data, or methods
- cse:diagnose — find the cause of an unexpected numerical result or failure before changing code
- cse:design — plan a study, model change, or tool fix and write its issue
- cse:build — implement an agreed model, method, data transformation, or correction
- cse:analyze — run an accepted study and interpret its numbers
- cse:summarize — write retained results into the analysis notebook
- cse:review — independently check a plan, a delivered result, or a change
- cse:write — turn accepted evidence into manuscript or report prose
- cse:prose — inspect scientific writing without editing it
- cse:workflow — carry one task through several stages
- cse:zotero, cse:data, cse:digitize, cse:mathpix — sources, datasets, figure values, and PDF transcription
- cse:plot, cse:pgfplots, cse:tikz, cse:latex, cse:quarto, cse:beamer — figures, diagrams, and documents
- cse:audit — remove software ceremony, duplicated values, or misplaced records
- delegated roles — review: cse:review; implementation: cse:build; design: cse:design; research: cse:research; diagnosis: cse:diagnose; analysis: cse:analyze; writing: cse:write

Report the engineering result or capability, supporting evidence, meaning,
and limits before software provenance. Do not invent physical results.

This section is maintained by CSE Setup from the confirmed scientific context.
Revise its inputs through Setup rather than maintaining a separate copy here.
<!-- CSE:END PROTOCOL -->

CSE execution mode: direct.

## Repository Workflow

- A new checkout or worktree needs only `bash scripts/setup_environment.sh` (fetches and hash-checks the pinned Engine wheel, then `uv sync --frozen --group test`; about 8 s first time). Run everything through `uv run --frozen`.
- Regression checks: see `README.md`. Engine 1.0.0 main-f4566b3d passes 287 tests. The only failures are `tests/test_segmented_bvp.py::test_positive_flow_pressure_transform_reproduces_case_3c_scipy_bvp` and `tests/test_segmented_bvp.py::test_case_bounded_transform_routes_to_positive_transform_for_one_bed_scipy_bvp`: singular collocation Jacobian in ideal-Henry 3C, boundary residual norm 9958.17. Both predate the repin and remain active for separate solver work.
- Tests write scratch under `.tmp_local/` and column runs default to `.tmp_column_runs/`; both are git-ignored. A new study result goes in a new `analyses/<study>/results/runs/<label>/` directory.
- Do not accept slow or broken solver runs as normal: give each case a wall limit (`execution.wall_limit_s`) and a log, stop stale runs you started, and report the root cause with a faster bounded alternative.
- Preserve existing dirty work and inspect overlapping files before editing.
- Record durable rules in this file. Do not keep local journals or agent memory for this repository.

## Standing Decisions

- The submitted revision stays unchanged on `codex/fallback-manuscript`. Do not edit its manuscript prose here or rerun its results for consolidation (2026-09-08).
- `$HOME/Documents/Papers/references.bib` is owned by Zotero/Better BibTeX. Never hand-edit it or its snapshot `docs/scientific/latex/references.bib` (2026-08-12).
- On this machine `EPCSAFT_HSL_LIBRARY` makes Engine Ipopt use MA57 (results report `linear_solver`). Never copy or bundle HSL. A paper or report with MA57 results must acknowledge "HSL, a collection of Fortran codes for large-scale scientific computation. https://www.hsl.rl.ac.uk/" (Engine #605, 2026-10-19).
- Manuscript prose: keep `local`, `history`, `historical` and `histories` out of it, and keep software-build chronology outside the article (2026-09-04). For LaTeX, apply `cse:latex` and run the build loop until the edit adds no warnings (2026-05-06).

## Commit Discipline

- Commit, push, or create a PR only when the user requests it or an approved workflow requires it.
- Before committing, verify the current branch, validation results, status, and final commit.

## ePC-SAFT Cross-Repo Integration

- This is an official downstream application under ePC-SAFT Governance D-038.
- Engine governance and source live at `/home/tnnrpolley21/Workspaces/Engineering/ePC-SAFT` (GitHub `tannerpolley/ePC-SAFT`); never import its source tree directly.
- Normal and final work uses one non-editable `epcsaft` wheel identified by Engine commit and wheel SHA-256. Intentional co-development uses an explicitly supplied candidate wheel with the same recorded identity.
- Keep absorber integration, column validation and process analyses here. Thermodynamic parameter adoption remains owned by Amine-Thermodynamics; generic equations and solvers remain owned by the Engine.
- Do not create nested repositories, submodules, mutable Git package dependencies, or dictionary compatibility copies of Engine behavior.
- Final manuscript, report, or archive results must pass `uv run python scripts/check_epcsaft_integration.py --mode final` without mutable package state.
- Preserve result-critical datasets under `src/mea_absorption_column/data/epcsaft_datasets`.
- Keep reusable `epcsaft` interactions behind explicit thermodynamics/runtime modules.
- Use `epcsaft-cross-repo` for contracts, upstream feedback, and handoffs.
