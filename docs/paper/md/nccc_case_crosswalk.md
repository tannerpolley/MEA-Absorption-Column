# NCCC K-Case And Appendix-Style Crosswalk

This folder contains source markdown extracted from the local literature export used for the NCCC
intercooling and validation review. The canonical machine-readable case table is:

- `data/reference/nccc_master_cases.csv`
- `analyses/nccc_validation/data/input/nccc_master_cases.csv`

The Morgan et al. 2018 supporting-information table names the NCCC runs as `K1` through `K23`.
The Appendix-C-style A-D labels used by the temperature-profile plotting workflow are a plotting
nomenclature, not a replacement for the source K identifiers. The crosswalk below maps K rows by
bed/intercooler group and the existing Appendix-C plotting order.

| 2018 K case | Appendix-style case | Group | Beds | Intercoolers |
| --- | --- | --- | ---: | ---: |
| K1 | 1A | A: three beds with two intercoolers | 3 | 2 |
| K2 | 2A | A: three beds with two intercoolers | 3 | 2 |
| K3 | 3A | A: three beds with two intercoolers | 3 | 2 |
| K4 | 4A | A: three beds with two intercoolers | 3 | 2 |
| K5 | 5A | A: three beds with two intercoolers | 3 | 2 |
| K6 | 6A | A: three beds with two intercoolers | 3 | 2 |
| K7 | 7A | A: three beds with two intercoolers | 3 | 2 |
| K8 | 8A | A: three beds with two intercoolers | 3 | 2 |
| K9 | 9A | A: three beds with two intercoolers | 3 | 2 |
| K10 | 10A | A: three beds with two intercoolers | 3 | 2 |
| K11 | 11A | A: three beds with two intercoolers | 3 | 2 |
| K12 | 12A | A: three beds with two intercoolers | 3 | 2 |
| K13 | 1B | B: three beds without intercoolers | 3 | 0 |
| K14 | 13A | A: three beds with two intercoolers | 3 | 2 |
| K15 | 14A | A: three beds with two intercoolers | 3 | 2 |
| K16 | 15A | A: three beds with two intercoolers | 3 | 2 |
| K17 | 1D | D: two beds with 0 intercooler(s) | 2 | 0 |
| K18 | 1C | C: one bed without intercoolers | 1 | 0 |
| K19 | 2C | C: one bed without intercoolers | 1 | 0 |
| K20 | 3C | C: one bed without intercoolers | 1 | 0 |
| K21 | 2D | D: two beds with 0 intercooler(s) | 2 | 0 |
| K22 | 3D | D: two beds with 1 intercooler(s) | 2 | 1 |
| K23 | 4D | D: two beds with 1 intercooler(s) | 2 | 1 |

Notes:

- `A` rows are the 15 three-bed/two-intercooler K cases: K1-K12 and K14-K16.
- The 2018 K set contains one three-bed/no-intercooler absorber row, K13, mapped here to `1B`.
- The 2018 K set contains three one-bed rows, K18-K20, mapped here to `1C`-`3C`; the separate
  seven one-bed C-case dataset remains a different validation source.
- `D` rows cover the two-bed cases, with K22-K23 carrying one intercooler.
