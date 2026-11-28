# Case 3C net enthalpy flow under collocation refinement

The runs use the nine-species reactive model, temperature states, native
Jacobian, dry-saturated inlet gas and the reported dry-gas mass flow. Case 3C
inlet inputs are 109,500 Pa, 318.15 K liquid, 316.75 K gas, 30 mass percent MEA
and 0.25 mol CO₂ per mol MEA. All runs use the pinned Engine 0.1.2 wheel.

Each dense refinement starts from the retained paired solution for its own
dataset. The archived diagnostic used this initialization for its 41-node
attempt. The original successful dense-run commands and seeds are unavailable.
The fit-A paired replay uses the archived coarse seed and reproduces all the
paired CSV exports byte for byte. Each run has an adjacent solver log and
retains its input identity, execution command and collocation polynomial.

The energy diagnostic evaluates vapor enthalpy flow minus countercurrent
liquid enthalpy flow using the existing caloric correlations and the polynomial
states. The 101-point evaluations agree with the exported phase enthalpy flows.
The exact ranges, endpoint differences and extrema locations are in
`energy_check.json` in each run. The mesh-node range alone misses interpolation
excursions between nodes.

The NCE calculations reproduce the archived 30.33 W and 0.134 W dense ranges.
At the finest setting, NCE capture differs from its archived paired run by
−0.0000393 percentage points. Fit A reaches 0.128 W and changes capture by
+0.000723 percentage points from its paired run, above the archived comparison
of 0.0001 percentage points. These differences describe numerical refinement
for Case 3C. The finest fit-A capture is 94.20469% against the reported
89.5% observation, a difference of +4.70 percentage points.
