# Manuscript writing rules

- Use one current manuscript: `docs/latex/main.tex`, with its existing section,
  appendix, table, and figure files. Preserve CHECKLIST and equation labels.
- Cite source-dependent claims and identify parameter definitions, units,
  conventions, domains, and exact source locators in the established source
  record. Do not hand-edit Zotero-owned bibliography records.
- Write complete value-independent prose now. Keep bracketed replacements for
  unavailable results and inputs; never invent outcomes to remove them.
- Apply the installed CSE terminology and banned-language checks to edited
  prose, using `docs/scientific/CONTEXT.md` for scoped repository terminology.
- Avoid throat-clearing, vague importance claims, filler adverbs, rhetorical
  contrasts, em dashes, and stacked sentence fragments.
- Keep workflow, file ownership, execution history, and source-reconciliation
  notes outside the rendered article. Scientific assumptions and limits stay
  beside the equations or claims they qualify.
- Keep verification, experimental validation, and parameter fitting distinct.
  Compare capture and temperature together and interpret model differences
  only at the numerical accuracy demonstrated by their matching runs.
- Rebuild through the existing manuscript command. Check references, labels,
  figure paths, prose, and the complete PDF. Document review in the existing
  `docs/latex/QA_REPORT.md`; add no parallel review database.
