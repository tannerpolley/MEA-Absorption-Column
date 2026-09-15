# Corrected 3C input source

This companion preserves the source reconciliation for `case_3c.json` without
changing that retained input or its SHA-256. It was recovered from repository
commit `30b6231`, path `analyses/nccc_validation/r18_sensitivity.md`, blob
`b52b5239280d4143f35662bcf35ae2932ee488c5`.

The reviewed source was Morgan et al. (2020), *Applied Energy* 262, 114533,
DOI `10.1016/j.apenergy.2020.114533`. Table A1, pp. 18–19, identifies run 3C
as a 2017 MEA campaign case with one bed and no intercooler. Table 6, p. 15,
independently repeats its mass flows, loading, dry CO2 fraction, and capture.
Table C2, p. 27, supplies the absorber temperatures. The archived review
records visual inspection of pp. 18, 19, 20, and 27; this recovery did not
reacquire or reread the paper.

| Quantity | Source value | Use in `case_3c.json` |
|---|---:|---|
| Liquid flow | 7517 kg/h | Converted to the retained wet molar feed |
| Gas flow | 2013 kg/h | Interpreted as total wet mass; the moisture basis is unspecified |
| Lean loading | 0.25 mol CO2/mol MEA | Retained |
| MEA mass fraction | 0.30 on a CO2-free solvent basis | Retained |
| Dry inlet CO2 | 0.093 | Retained before the stated saturation assumption |
| Dry inlet O2 | 0.090 | Retained before the stated saturation assumption |
| Gas inlet temperature | 43.6 °C | Converted exactly to 316.75 K |
| Liquid inlet temperature | Not available | 318.15 K is an explicit imputation |
| Pressure | 109.5 kPa top; 110.9 kPa gas inlet | The boundary uses 110900 Pa |
| Capture | 89.5 ± 1.2% | Observation only; not a numerical acceptance condition |

The paper defines gas compositions on a water-free basis and recommends
estimating inlet water by saturation at the inlet temperature and pressure.
That is an assumption, not a humidity measurement. The exact origin of the
older `C_cases_data.csv` gas values and flow conversion remains unresolved;
they are not substituted for this input.
