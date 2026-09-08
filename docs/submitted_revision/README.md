# Submitted revision — 4 September 2026

The investigator identified this as the revision sent to the journal. The branch
`codex/fallback-manuscript` preserves the submitted source plus its supporting
records. Do not use this branch for new research or manuscript editing.

## Retrieve without calculating

Clone the repository at `codex/fallback-manuscript`, then run from its root:

```bash
sha256sum --check docs/submitted_revision/SHA256SUMS
unzip docs/submitted_revision/revision_submission_2026-09-04.zip -d submitted-revision
```

The ZIP contains the submitted PDF, revised source with bibliography and figures,
reviewer response, cover letter, and original manuscript/source. Its SHA-256 is
`bbca890e2cc607e7ef60add78ec3820e1b222c892df45302969bcbbca4c76f84`.
The revised PDF SHA-256 is
`d3e8ceee222bea03422a0a86871e3598526a34195d34e079a21b3f74c00ce13f`.
All revised-source files match the repository source at archive creation.
The extracted source can be compiled with XeLaTeX/BibTeX using its own bibliography;
it does not require the machine-owned Zotero bibliography or a column calculation.

## Trace the results

`REPRODUCE.md` identifies the studies, commands, parameters, numerical limitations
and executable identities. `docs/code_to_paper_traceability.md` maps the model to
the paper. Numerical tables, profiles, failed attempts, original identities and
plotting scripts remain under `analyses/nccc_validation` and
`analyses/transport_sensitivity`. The operating notebook is under
`analyses/nccc_validation/figures/reactive_operating/notebook.qmd`.
The checksum list covers tracked source and evidence; verification does not execute
scientific code. Submitted figure bytes in the ZIP take precedence over later
plot serialization in analysis output directories.

## Optional future numerical reproduction

The existing commands in `REPRODUCE.md` are opt-in calculations. None were rerun
when this archive was assembled. Rebuilding a PDF, plotting retained values and
recalculating a column are separate operations.

The current dependency requires CPython 3.13 on Linux x86-64 and the exact Engine
wheel SHA-256 `91632d2812429cbd293aae70fe8d4efb00000efe2377a91546dd7374dca67ee4`.
The wheel path in the dependency/lock is machine-specific. Obtain the exact wheel
from its owner, verify its hash, then make an explicit path-only dependency/lock
adjustment in a separate reproduction checkout. Earlier runs identify different
wheels and source hashes; use their recorded inputs for an exact replay. Engine
binaries and publisher PDFs are not redistributed in this public repository.
The retained tables and submitted document are independently retrievable, but
this archive does not claim a fresh numerical replay or that the current wheel
reproduces every earlier build bit-for-bit.
