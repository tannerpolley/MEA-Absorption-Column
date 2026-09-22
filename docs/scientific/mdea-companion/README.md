# MDEA research companion

This directory separates exploratory MDEA work from the submission-safe MEA manuscript.

- `manuscript/` is an exact, source-complete copy of the frozen MEA manuscript at tag
  `nce-mea-only-pre-mdea-2026-08-25`. It is the candidate manuscript that may later receive
  promoted MDEA material.
- `research-journal/` records questions, required inputs, source admission, methods, and
  candidate results. It starts with no admitted MDEA result.

MDEA material may move from the journal into the candidate manuscript only after every gate
in `research-journal/sections/promotion_log.tex` is satisfied. The tagged MEA manuscript is
never edited for MDEA exploration.

Build the candidate with:

```bash
bash docs/scientific/mdea-companion/manuscript/scripts/build_main.sh
```

Build the research journal with:

```bash
cd docs/scientific/mdea-companion/research-journal
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=builds main.tex
```
