# NCCC Campaign Case Mapping

This note records how the two markdown paper exports under `docs/papers/md/` combine into one NCCC case catalog for the MEA absorber manuscript.

## Source Files

- `Morgan et al. - 2018 - Development of a Rigorous Modeling Framework for Solvent-Based CO2 Capture.md`
  - Provides the 2014 NCCC `K1`-`K23` steady-state case set.
  - These are the cases already represented in `NCCC_Data_mole_based.csv`.
- `Morgan et al.md`
  - Provides the 2017 NCCC SDoE and alternative-height case set.
  - `A` cases: 15 first-iteration three-bed cases with intercooling.
  - `B` cases: 3 second-iteration three-bed cases with intercooling.
  - `C` cases: 7 one-bed cases with no intercooling.
  - `D` cases: 4 two-bed cases with no intercooling.

The two naming systems are separate campaign identifiers. The `C` and `D` cases should not be renamed as `K` cases unless a deliberate alias table is added, because the papers use different source IDs for different campaign years.

## Combined Catalogs

The extraction script is:

```powershell
.\.venv\Scripts\python.exe analyses\nccc_validation\scripts\extract_nccc_case_catalog_from_markdown.py
```

It writes:

- `src/mea_absorption_column/data/NCCC_2014_cases.csv`
  - Source-preserving 2014 `K` campaign table.
- `src/mea_absorption_column/data/NCCC_2017_cases.csv`
  - Source-preserving 2017 `A`/`B`/`C`/`D` campaign table.
- `src/mea_absorption_column/data/NCCC_2014_model_inputs_mass.csv`
  - Run-ready 2014 mass-basis inputs for `benchmark --nccc-dataset 2014 --data-type mass`.
- `src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv`
  - Run-ready 2017 mass-basis inputs for `benchmark --nccc-dataset 2017 --data-type mass`.
- `src/mea_absorption_column/data/NCCC_combined_case_catalog.csv`
  - 52 total cases.
- `src/mea_absorption_column/data/NCCC_no_intercooler_case_options.csv`
  - 17 cases with no intercooling.

## No-Intercooler Case Set

The no-intercooler set combines:

- 2014 `K` cases:
  - `K13`: three beds, no intercooling.
  - `K17`, `K21`: two beds, no intercooling.
  - `K18`, `K19`, `K20`: one bed, no intercooling.
- 2017 `C` cases:
  - `1C`-`7C`: one bed, no intercooling.
- 2017 `D` cases:
  - `1D`-`4D`: two beds, no intercooling.

This is the right source set for a manuscript table of available NCCC no-intercooler operating points. It is not automatically the validated model-results set; each row still needs a current bounded model run before it can be claimed as a successful validation case.

## Modeling Defaults

The 2017 source table leaves the lean solvent inlet temperature blank for `1C`, `2C`, `3C`, and `3D`. The source-preserving CSV keeps those values blank. The run-ready 2017 model-input CSV sets those four cases to `45.0 C` (`318.15 K`) and flags them with `lean_solvent_temp_imputed=True`, because the measured lean inlet temperatures across the campaign vary only slightly around that value.

The `D` cases are two-bed, no-intercooler cases. They should be run from `NCCC_2017_model_inputs_mass.csv` with `--nccc-dataset 2017 --data-type mass`; do not convert them to one-bed `C` cases.

The older `src/mea_absorption_column/data/C_cases_campaign_inputs.csv` should be treated as a legacy seven-case input file. The year-specific `NCCC_2017_model_inputs_mass.csv` is the corrected appendix-derived source for rerunning 2017 `C`/`D` cases.
