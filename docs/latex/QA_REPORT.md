# Presentation-only integration — 2026-09-03

The raw Elsevier CAS manuscript remains the sole article source. The comparison baseline is commit `4cd362c83e0a6b439655a18819a862da25e782cf`. The older visual pilot supplied table geometry and drawing styles; its manuscript prose, results, bibliography, and Quarto helpers were not imported.

## Content preservation

The eleven main, section, and appendix source files match the baseline after reversing the explicit quantity and derivative aliases and removing layout-only float commands. All numerical operating-condition, pure/binary-parameter, and solver-comparison rows match. Table captions and explanatory cells retain their original wording. Both bibliography files are byte-for-byte unchanged. An independent source comparison found no content drift.

Figure 1 is a document-native redraw of the existing raster. Its labels and directed relationships are retained, including the enhancement-factor labels that belong to this baseline. The later scientific revision must update those labels together with the governing formulation.

## Presentation changes

- `siunitx` quantities and numeric columns; five ordinary derivatives use `derivative` without changing their symbols.
- Six table layouts use booktabs rules, aligned numeric columns where applicable, and attached captions; the operating-condition notes use `threeparttable`.
- Figure 1 uses shared TikZ styles and explicit connector geometry.
- Real figure/table floats and `placeins` keep captions attached and Figure 1 beside the model introduction.

## Build and review

Build command: `TEXMFHOME=/home/tnnrpolley21/texmf bash docs/latex/scripts/build_main.sh`.
The explicit TeX root selects the already-installed Elsevier CAS files; no package was installed or copied into the manuscript.
The baseline has 33 pages; the integrated manuscript has 34 pages, including the full-page vector flowchart near its introduction.
The build and PDF freshness check pass, and references and citations resolve.
The baseline title-box warning of 117.08 pt remains; no new overfull box warning remains.

The CSE visual runner retains color/grayscale pages and geometry findings under `builds/presentation-release-qa/`.
Its conservative automated result reports 123 warnings and two equation text-overlap signals. Full-size color/grayscale review of pages 1–16 found the equation signals on pages 8 and 13 to be extraction-geometry false positives: accents, superscripts, and subscripts remain visibly separated. Figure 1 is contained and its branches are clear. Page 3 retains whitespace before the dedicated figure page.
Source comparison and visual acceptance apply only to this presentation checkpoint, not to later scientific edits.

Full-size color/grayscale review also covered pages 17–34. All redesigned tables fit without clipping, collisions, or detached notes. Remaining presentation limits are grayscale ambiguity in the unchanged solver bar chart and sparse continuation pages (20, 27, 29, 31, 34). These remain publication-polish items for the scientific revision, which will change pagination; the selected TikZ/table integration preserves content and is accepted with these limits.
