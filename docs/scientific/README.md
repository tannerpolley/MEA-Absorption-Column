# Scientific project records

This is a CSE project. Use the installed CSE skills and read this file and
[CONTEXT.md](CONTEXT.md) before selecting the evidence-justified CSE route.
For the current manuscript, use CSE Write and CSE Visualize LaTeX; use Research
for an unresolved source and Diagnose for an unexplained numerical result.

## Established owners

| Content | Authoritative location |
|---|---|
| Current question, boundaries, and terminology | [CONTEXT.md](CONTEXT.md) |
| Article formulation and numbered equations | [model_framework.tex](../latex/sections/model_framework.tex) |
| Article methods and comparison design | [methods.tex](../latex/sections/methods.tex) |
| Scientific evidence and result-insertion requirements | [SOURCE_MAP.md](../latex/SOURCE_MAP.md), with exact values in the named analysis files |
| Manuscript and PDF | [main.tex](../latex/main.tex), [builds/main.pdf](../latex/builds/main.pdf) |
| Missing manuscript content | [Existing checklist definitions](../latex/scripts/manuscript_checklist.json) |
| Reviewer assessments and additional revision work | [Live assessment definitions](../latex/scripts/reviewer_checklist.json); `/reviewers` on the existing checklist server  [Original comments](../reviewer_comments.txt) and [individual prior scores](../reviewer_assessment_original.md). |
| Writing style and source organization | [writing/STYLE.md](writing/STYLE.md), [writing/RULES.md](writing/RULES.md) |
| Reusable absorber implementation and inputs | `src/mea_absorption_column/` |
| NCCC comparisons | `analyses/nccc_validation/` |
| Reactive-film study | `analyses/reactive_film_evidence/` |

The existing article sections supply the formulation and methods records.
Do not create a competing manuscript, duplicate equation documents, a new
analysis set, or empty decision records merely to match a template.
The finite-rate [reactive_film_methods.md](reactive_film_methods.md) records
an earlier formulation; it is not the current equilibrium-manifold article.
Consequential historical decisions remain in their existing documents and Git
history. The current manuscript author notes belong in SOURCE_MAP.md.

## Manuscript commands

Run from the repository root:

```bash
TEXMFHOME="$HOME/texmf" bash docs/latex/scripts/build_main.sh
python3 docs/latex/scripts/check_main_pdf_fresh.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s docs/latex/scripts -p test_manuscript_checklist.py
python3 docs/latex/scripts/manuscript_checklist.py --json
```

The existing server is at `http://127.0.0.1:37543/`.
Reuse its listener; if absent, run
`python3 docs/latex/scripts/manuscript_checklist.py --serve --port 37543`.
The page reads the LaTeX sources and checklist definitions on each refresh.
Checked means scoped content exists; scientific review notes remain separate.

The build renders the manuscript without executing the absorber model.
It verifies/projects the existing Zotero bibliography through the prescribed
repository command; source metadata changes require the approved CSE Sources
workflow. The raw-LaTeX CAS/XeLaTeX format is the investigator's selected
format. Quarto remains available for existing analysis notebooks only.

Repository CSE hooks are registered in `.codex/hooks.json` and call the stable
runtime under `$HOME/.codex/hooks/`. Runtime maintenance is shared across
repositories and is separate from manuscript setup. No personal terminology
configuration is required for this project.
