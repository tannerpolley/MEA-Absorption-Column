# Manuscript style

Apply CSE Write and the manuscript writing/source-organization protocol in
CSE Visualize LaTeX. The investigator selected the existing one-column CAS
article and raw-LaTeX build; preserve that choice.

Lead each paragraph with its scientific quantity, mechanism, comparison, or
result. Name the calculation or physical process that performs an action and
use a precise verb. Let the next sentence explain that statement, its basis,
or its consequence. Keep logical links explicit and vary sentence length.
Avoid a long list of subordinate clauses when two connected sentences are
clearer. Keep equations, conditions, units, and limitations beside the claims
they govern. Distinguish observations, calculations, and interpretation.

In source, place one complete prose sentence on each physical line and retain
blank lines as paragraph boundaries. Precede each prose paragraph with a stable
comment such as `% P:introduction-05`. Preserve existing IDs through edits;
assign new IDs without renumbering existing paragraphs. Derive sentence IDs
from paragraph ID and order only when reviewing. Equations, captions, figures,
and tables retain their semantic commands and labels.

Paragraph comments never change rendered headings. Keep generated cohesion or
sentence reports under `docs/latex/builds/`; do not create a second prose map.
Separate substantive wording edits from source-only reflow and verify that the
reflow preserves the rendered text. Do not create commits unless requested.
