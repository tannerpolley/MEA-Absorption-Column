# Experimental four-species vapor input

This is an unaccepted PC-SAFT candidate, not the default column gas model.
Species order is CO2, water, nitrogen, oxygen. No electrolyte, permittivity,
reaction, or bulk phase-equilibrium model is selected.

CO2/water dispersion, water diameter correlation, their binary correction,
water self-association and reciprocal CO2/water association are copied from
the retained reactive parameter document
`sha256:00049473d53c7e8088ef3e2dbbc6a1bab058f6dc4de963ee98936b4cd9bda25e`.
Unused solvation coefficients and ionic model inputs are removed. The original
reactive document is unchanged.

Nitrogen is the Engine's retained Gross–Sadowski 2001 Table 2 record; its
63–126 K source fit range remains intact. Oxygen is transcribed from
[Thermopack's pinned component record](https://github.com/thermotools/thermopack/blob/d68c794c7342bfc6938eb424a1fbb88b7780b738/fluids/Oxygen.json),
whose byte SHA-256 is
`ee757893099745d92732aaca3738181890222cd452b8d920950474c8d43a59c7`.
The original oxygen fit range is not reported there. Row-level sources,
units, qualifications and hashes are retained in the parameter file.

The other five binary corrections are explicitly **unfitted predictive zeros**,
not claims that their fitted values vanish. Cross dispersion remains
`epsilon_ij = sqrt(epsilon_i * epsilon_j) * (1-k_ij)` for every pair.
N2 and O2 have no association sites. Compatibility and sensitivity of the
unfitted corrections remain to be evaluated.

Historically, Engine `27cb36f` (also A1 `40b217c`) parsed this input but
`Mixture(parameters)` raises `invalid-domain: temperature empty-intersection`.
The resolver intersects historical source fit ranges before processing the
uniformly declared candidate-extrapolation domain. Thus the EOS is never
evaluated. Supporting this experiment requires the Engine to distinguish an
empty jointly qualified source range from a valid, explicitly declared
experimental evaluation range, while preserving the original source evidence.
No source ranges were relabeled or removed to bypass that admission check.

Installed immutable candidate `9164294` corrects this distinction. The actual
four-gas callback now matches native f/rho/H/Cp and all six input derivative
directions at 293.15, 313.15 and 393.15 K. Temperature derivative checks at
the boundaries use inward second-order differences for validation only;
outside-domain evaluation is still rejected. The original nitrogen source
range remains 63–126 K. PR209 is not assumed merged by this adoption.

`reference-thermochemistry.json` retains identical CO2/water component references
from the provisional liquid reference paired with `MEA_reactive_epcsaft_2026_09_03`.
The adopted reaction refit does not change the shared CO2/water EOS parameters.
N2/O2 use NIST WebBook Shomate ideal-gas
Cp, represented by degree-8 polynomials in T−353.15 K over 293.15–393.15 K,
with elemental formation enthalpies zero at 298.15 K. Both source equations
and conversion errors are recorded. Tests compare Cp and its enthalpy integral
against the source equations; no separate phase-dependent CO2/water reference
shift is introduced. The inherited water reference remains physically
provisional: at 298.15 K it implies ideal-gas Cp 44.918 J/mol/K, versus
[NIST's 33.60 J/mol/K](https://cccbdb.nist.gov/exp2x.asp?casno=7732185&charge=0).
Matching liquid Cp does not by itself validate vapor Cp.
