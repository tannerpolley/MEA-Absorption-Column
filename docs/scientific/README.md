# Scientific project records

This is a CSE project. Use the installed CSE skills and read this file and
[CONTEXT.md](CONTEXT.md) before selecting the evidence-justified CSE route.
Active work is research in `analyses/`; the submitted manuscript is not edited
here. Use Research for an unresolved source and Diagnose for an unexplained
numerical result.

For source discovery, use `cse-zotero wiki-context mea-absorption-column --query "<terms>" --json`; repeat with `--global` if the project view is incomplete. Use `cse-zotero wiki-evidence <KEY>` to inspect exact source passages.

## Established owners

| Content | Authoritative location |
|---|---|
| Current question, boundaries, and terminology | [CONTEXT.md](CONTEXT.md) |
| Article formulation and numbered equations | [model_framework.tex](latex/sections/model_framework.tex) |
| Article methods and comparison design | [methods.tex](latex/sections/methods.tex) |
| Scientific evidence and result-insertion requirements | [SOURCE_MAP.md](latex/SOURCE_MAP.md), with exact values in the named analysis files |
| Manuscript and PDF | [main.tex](latex/main.tex), [builds/main.pdf](latex/builds/main.pdf) |
| Writing style and source organization | [writing/STYLE.md](writing/STYLE.md), [writing/RULES.md](writing/RULES.md) |
| Reusable absorber implementation and inputs | `src/mea_absorption_column/` |
| NCCC comparisons | `analyses/nccc_validation/` |
| Reactive-film study | `analyses/reactive_film_evidence/` |
| Root analysis website and ordered notebook membership | `analyses/index.qmd`, `analyses/_cse-manuscript.json` |

The existing article sections supply the formulation and methods records.
Do not create a competing manuscript, duplicate equation documents, a new
analysis set, or empty decision records merely to match a template.
The finite-rate [reactive_film_methods.md](reactive_film_methods.md) records
an earlier formulation; it is not the current equilibrium-manifold article.
Consequential historical decisions remain in their existing documents and Git
history. The current manuscript author notes belong in SOURCE_MAP.md.

## Manuscript commands

The submitted revision, its reviewer records and their checklists live on branch
`codex/fallback-manuscript`. To rebuild this source from the repository root:

```bash
TEXMFHOME="$HOME/texmf" bash docs/scientific/latex/scripts/build_main.sh
python3 docs/scientific/latex/scripts/check_main_pdf_fresh.py
```

The build renders the manuscript without executing the absorber model.
It verifies/projects the existing Zotero bibliography through the prescribed
repository command; source metadata changes require the approved CSE Sources
workflow. The root Quarto site at `analyses/` renders only registered
analysis pages with execution disabled. An optional page PDF uses the same root
renderer with `bash analyses/render.sh <analysis>/notebook.qmd --to pdf`.
