# Local Codex Instructions

<!-- CSE:BEGIN PROTOCOL -->
## CSE Protocol

**Agent role:** Work as chemical engineer specializing in reactive CO2 absorption column modeling and numerical methods.
**Repository role:** analysis.

Owns absorber integration, transport and film closures, and numerical comparisons on explicitly stated common input bases. The ePC-SAFT Engine owns thermodynamic equations and MEA-Thermodynamics owns fitting and parameter adoption. Preserve conservation, units, species order and exact input identities, and keep the submitted revision unchanged on its archive branch.

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

## Startup Reads

- Read `docs/.codex-journal/user_preferences.md` when it exists.
- Read `docs/.codex-journal/project_memory.md` when it exists.

## Memory Policy

- Keep user preferences and durable project facts concise, date-stamped, and deduplicated.
- Do not update memory for routine Q&A or small one-off work.
- Do not store secrets, add placeholder entries, or create new agent memory, including `.codex`, `$HOME/.codex/projects`, or Claude auto memory.

## Repository Workflow

- Prefer `Local` for foreground solver inspection and `Worktree` for isolated background implementation.
- Prefer uv-managed commands. Use `.venv/bin/python` only for interpreter-specific debugging.
- Use `.codex/environments/environment.toml` actions when available.
- For LaTeX/manuscript work, apply the `cse:latex` skill plus repository-local policy.
- Preserve the user's existing dirty worktree and inspect overlapping files before editing.

## Commit Discipline

- Commit, push, or create a PR only when the user requests it or an approved workflow requires it.
- Before committing, verify the current branch, validation results, status, and final commit.

## ePC-SAFT Cross-Repo Integration

- This is an official downstream application under ePC-SAFT Governance D-038.
- Engine governance and source live at `/home/tnnrpolley21/Workspaces/Engineering/ePC-SAFT-project`; do not use the retired `/ePC-SAFT` path or a sibling source import.
- Normal and final work uses one non-editable `epcsaft` wheel identified by Engine commit and wheel SHA-256. Intentional co-development uses an explicitly supplied candidate wheel with the same recorded identity.
- Keep absorber integration, column validation, process analyses, and this repository's manuscript here. Thermodynamic parameter adoption remains owned by MEA-Thermodynamics; generic equations and solvers remain owned by ePC-SAFT-project.
- Do not create nested repositories, submodules, mutable Git package dependencies, or dictionary compatibility copies of Engine behavior.
- Final manuscript, report, or archive results must pass `uv run python scripts/check_epcsaft_integration.py --mode final` without mutable package state.
- Preserve result-critical datasets under `src/mea_absorption_column/data/epcsaft_datasets`.
- Keep reusable `epcsaft` interactions behind explicit thermodynamics/runtime modules.
- Use `epcsaft-cross-repo` for contracts, upstream feedback, and handoffs.
