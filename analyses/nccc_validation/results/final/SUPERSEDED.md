# Bundle-1 source claims superseded by bundle 3

Issue #78, owner decision 2026-10-29: the earlier issue16, issue33, issue35,
issue42, issue36 and downstream issue30 records in this directory remain
byte-unchanged evidence of their declared inputs. They must not be read as
current admission or row-level uncertainty evidence for loaded Amundsen2009
observations. Earlier input bytes remain available through each result's source
commit; the live identity files now pin bundle 3.

Amine-Thermodynamics #152/#194 corrected the uncertainty interpretation:
Amundsen2009 pp. 3096–3100 provides general loaded-system estimates with
unnamed exceptions at the highest temperatures and loadings. Those estimates
cannot be assigned to individual loaded rows. Bundle 3 at commit
`a6de0317333e9b4bbd65537d285bfe3a384101c7` does not admit the 68 loaded density
or 75 loaded viscosity rows. The 35 unloaded density and 35 unloaded viscosity
rows remain admitted. Viscosity admission is distinguished by loading state.

Current transport and release checks are retained in
[../runs/issue78_bundle3/](../runs/issue78_bundle3/). The research notebook
records before/after counts and unavailable reruns. No new physical film
flux, propagated uncertainty, or column validation follows from this repin.
The issue33 density-source choice belongs to #83 and remains outside #78.

The original result validators replay the original source-stage protocol;
they do not validate the repinned inputs against the archived result bytes.
Use `check_issue78_bundle3.py` for the current bounded transport and release
checks. Its release summary reports the missing parameter-bundle ZIP
explicitly; successful guard checks do not imply archive verification.
