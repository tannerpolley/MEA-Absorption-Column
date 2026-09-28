# Physical acceptance on the adopted record (Engine #149)

Question: does the twelve-state absorber, on the MEA record `868a5018…` adopted by
Engine #61, meet the physical criteria frozen by Engine #91
(`docs/coordination/greenfield-migration.md`)? Numerical qualification comes first
and is replayed: node checks in `analyses/greenfield_node_qualification/`, and the
K1/K2 cosine ladder in `analyses/bvp_solution_methods/results/replay_149/`.

Inputs: the record and its Engine reaction records, the physical ideal-gas records,
the four-gas vapor record, case 3C, and the wheel pinned in
`integration/epcsaft_contract.json` (`48a639e7…`). Column quantities come from the
replayed cosine 65-node attempt. `results/physical-checks.csv` has one row per
check; `results/summary.json` records identities and hashes.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen python analyses/physical_acceptance_149/scripts/physical_checks.py \
  analyses/bvp_solution_methods/results/replay_149/upwind_cos_n65/attempt.json
```

## Result (wheel `48a639e7…`, record `868a5018…`, case 3C)

| Criterion (#91) | Model | Reference | Error | Limit | Outcome |
|---|---|---|---|---|---|
| C3 vapor Cp, V1 (316.75 K) | 30.5689 J/(mol K) | 30.5663 (JANAF + EOS residual) | +8.6e-5 | 2 % | pass |
| C3 vapor Cp, V2 (329.571 K) | 30.2317 | 30.2324 | −2.5e-5 | 2 % | pass |
| C5 liquid Cp, α 0, 45 °C | 3.354 kJ/(kg K) | 3.7195 (Hilliard G.2) | −9.8 % | 3 % | fail: model limit |
| C5 liquid Cp, α 0.358, 45 °C | 3.025 | 3.3675 | −10.2 % | 3 % | fail: model limit |
| C5 liquid Cp, α 0.358, 80 °C | 2.963 | 3.4707 | −14.6 % | 3 % | fail: model limit |
| C6 released heat, 80 °C, α 0.047 → 0.090 | 88.35 kJ/mol CO2 | 90.904 (Kim–Svendsen) | −2.55 kJ/mol | 10 kJ/mol | pass |
| Capture (65 cosine nodes) | 88.84 % | 89.5 % | −0.66 pp | 5 pp | pass |
| Liquid temperature taps | RMSE 3.29 K | Morgan 2020 Table C2 | | 6 K | pass |
| Temperature peak | 347.59 K at z = 5.32 m | 348.05 K (tap at z = 4.8 m) | −0.46 K | 5 K | pass |
| Density, 50 °C, α 0.3 / 0.4 | 1.193 / 1.265 g/cm³ | 1.058 / 1.083 (Amundsen) | +12.7 / +16.8 % | 1.6 % | fail: model limit |
| Density, 70 °C, α 0.3 / 0.4 | 1.177 / 1.247 | 1.0464 / 1.0719 | +12.5 / +16.3 % | 1.6 % | fail: model limit |

- **C3.** The reference is Σ yᵢ Cp°ᵢ from the tabulated NIST-JANAF values (cubic
  spline through 200, 298.15, 400 and 500 K), which are independent of the Shomate
  records the model uses, plus the model's EOS residual (0.28 and 0.43 J/(mol K)).
  The action-based Cp_V equals the state's ideal + residual Cp to 5e-15. The water
  Shomate fit, extrapolated below its 500 K range, stays within 0.03 % of JANAF here.
- **C5 cause: the EOS liquid-water heat capacity.** Pure liquid water from the record
  has Cp 64.71 J/(mol K) at 318.15 K and 66.19 at 353.15 K against IAPWS-95 75.31 and
  75.61 (−14.1 % and −12.5 %). The ideal part is right (33.68 against JANAF 33.69);
  the EOS residual is 31.0 J/(mol K) where IAPWS-95 minus JANAF requires 41.6. Water
  is 70 % (unloaded) and 65 % (loaded) of the solution mass, so this deficit alone is
  −0.41 and −0.38 kJ/(kg K) at 45 °C, more than the whole shortfall (−0.37, −0.34).
  At 80 °C, loaded, it is −0.34 of the −0.51 kJ/(kg K) shortfall; the remaining
  −0.17 kJ/(kg K) belongs to the reacting solution and is not resolved further here.
  This is the #140 / MEA #103 limit of #61's model, not a numerical failure. The
  unloaded row uses loading 1e-5 (the callback needs a positive CO2 feed; 1e-6
  stalls on this wheel in the trace-inventory regime fixed later by Engine #172).
- **C6.** The absorber's own caloric chain gives 88.353 kJ/mol CO2; MEA #116's
  evaluator gives 88.349 on the same record (different wheel, bubble-point pressure
  instead of 101325 Pa). The limit is #82's calibration-lineage paired-heat limit
  (10 kJ/mol); the Kim–Svendsen 80 °C pair is calibration data.
- **Capture and temperatures** use the replayed cosine 65-node solution. Capture
  lies 0.12–0.17 pp above the Richardson limit of the #176 ladder (88.67–88.72 %), so
  the comparison does not depend on the remaining mesh error. Morgan 2020 Appendix C
  measures tap position x from the top, so z = (1 − x) × 6 m. Residuals (model −
  tap) at z = 0, 1.2, 2.4, 3.6, 4.8 m are −5.86, +4.08, +0.75, −1.42, −0.82 K. The
  source labels the taps "absorber temperature" with no phase; the liquid inlet
  temperature 318.15 K is imputed.
- **Density cause: the ion volumes of the record.** Matched states are the Amundsen
  2009 30 mass % rows whose T and loading lie inside the 65-node liquid range
  (318.15–347.59 K, α 0.25–0.401). Unloaded solution (0.994 vs 0.998 g/cm³) and pure
  water (0.988 vs IAPWS 0.988) are right. The record models MEAH⁺ and MEACOO⁻ as
  single segments (m = 1, packing diameter 3.07 and 3.11 Å, about 29–30 Å³ each),
  while MEA has m σ³ = 86 Å³ and CO2 45 Å³. Each absorbed CO2 (2 MEA + CO2 →
  MEAH⁺ + MEACOO⁻) therefore removes about 157 Å³, about 95 cm³/mol, of hard-core
  volume. The model's apparent volume change is −77 cm³/mol CO2 against Amundsen's
  +3. Falsifying probe (diagnostic, record not refit, VLE not claimed): segment
  counts that give MEAH⁺ the MEA volume (m = 2.97) and MEACOO⁻ the MEA + CO2 volume
  (m = 4.33) reduce the errors to −1.0, −1.6, −1.2 and −2.1 %.
  Column-state EOS densities (1.14–1.27 g/cm³) are reported as diagnostics. The
  column uses the EOS density for liquid volume, velocity and holdup, so its liquid
  volumetric flow is 12–17 % low; the effect on capture is not evaluated here (a new
  column campaign is a non-goal).

## Claim limits

Case 3C, one record, one wheel. C5 and density fail because of physical-parameter
limits of the adopted record (water's EOS residual Cp; single-segment ion volumes).
Both remain failures; neither is waived. Fixing them means re-parameterizing the MEA
record (MEA-Thermodynamics owns it) and then replaying this study. Capture and
temperature agreement is a conditional prediction with the assumed mobilities and
imputed liquid inlet temperature; no capture fitting was done.
