# Greenfield node qualification (Engine #148, contract N1–N6, C1, C2, C4)

Question: do the twelve-state column's Engine-derived node quantities and exact
actions satisfy the frozen #91 numerical criteria on the current Engine
interface? This is numerical verification, not physical validation.

Inputs: MEA exploratory record `868a5018…` (not adopted) with its Engine
reaction and neutral-reference records, the shared physical ideal-gas records
(`src/mea_absorption_column/data/epcsaft_datasets/MEA_greenfield_exploratory/`),
the four-gas vapor record, case 3C, and the wheel pinned in
`integration/epcsaft_contract.json`. `results/summary.json` records every
identity and hash; `results/node-checks.csv` holds one row per check.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen python analyses/greenfield_node_qualification/scripts/qualify_nodes.py
```

## Result (wheel `48a639e7…`, 58 s against the 600 s node budget)

| Check | States | Criterion | Largest passing defect | Outcome |
|---|---|---|---|---|
| N1 elements, mass, charge | S1, S2 | 1e-12 × scale | 8.5e-14 | 12/12 pass |
| N1 row residual, liquid pressure root | S1, S2 | 1e-7; 1e-8 | 4.3e-14; 1.2e-14 | pass |
| N1 f = a R T ρ0 vs x φ P (CO2, water) | S1, S2 | 1e-10 | 7.0e-11 | 4/4 pass |
| N2 Gibbs–Duhem, v, e_MEA, e_H2O | 7 loadings | 1e-12 × scale | 1.4e-14 | 21/21 pass; ion-dropped controls fail |
| N3 tangent ladder | 7 loadings × 9 species | 5e-6 | 6.8e-8 | 63/63 resolved, pass |
| N4 outer actions e_T, e_P, e_CO2, e_MEA, e_H2O | 7 × 5 × 9 | 5e-5 | e_T 5.2e-9; e_P 8.4e-6; feeds 9.7e-8 | 315/315 pass; chain-term controls fail |
| N5 projections, t^T M t ≥ 0, g > 0 | 47 quadrature states | 1e-12 × ‖M‖∞ | 4.9e-21 | 282/282 pass |
| N6 19×12 node Jacobian along the retained direction, step 1e-3 | 3C, S2 node | 1e-5 | 6.1e-8 | 38/38 rows resolved, pass |
| C1 apparent-feed Euler, H_L and H_V | S1, S2, V1, V2 | 5e-6 | 9.9e-15 | 4/4 pass; one-term-omitted controls fail |
| C2 first order dH_L (T, P, feeds), dH_V (T, feeds) | S1, S2, V1, V2 | 5e-6 | dH_L/dP 1.2e-7; others 1.3e-10 | 20/20 pass |
| C2 second order Cp_V, h̄_CO2, h̄_H2O | V1, V2 × 6 directions | 5e-5 | 7.3e-8 | 36/36 pass |
| C4 pure-water Δh_vap vs IAPWS-95 | 313.15, 329.4, 348.5 K | 2 % | 0.42 % | 3/3 pass |

dH_L/dP is 1.577e-3 J/Pa at S1 and 1.431e-3 J/Pa at S2, the order of the liquid
volume N/ρ as (∂H/∂P)_T = V(1 − Tα) requires. No refusal remains in the consumed
node quantities.

Every finite-difference component was resolved (the two finest Richardson
estimates agree to 10 %). Each zeroed or omitted-term negative control fails
its own criterion. S3 (360 K, 106.4 kPa, loading 0.45) evaluates. The mixture
diagnostic h̄_H2O^V − h̄_H2O^L is 42.8 kJ/mol (V1/S1) and 42.4 kJ/mol (V2/S2);
no reference exists for it.

## Start policy

A failed cold start falls back to native continuation from the last accepted
state, else the case anchor loading. On wheel `684b213a…` the Engine cold start
stalled on about 1 % of absorber states (3 of 264 solves; tannerpolley/ePC-SAFT#157).
With the #157 fix in this wheel no solve needed the fallback (0 of 294); it is kept
as harmless. Continuation from S1 to S2 still reproduces the cold-solved S2 amounts
(largest relative amount difference 0, criterion 1e-8).

## Claim limits

Numerical verification only, at S1/S2/V1/V2, their loadings and the 3C/S2 node, on
an exploratory record. The CO2 ideal-gas Shomate record is extrapolated below 298 K
and water's below 500 K; both reproduce JANAF Cp within 0.03 % over 298–400 K. C4
tests the pure-water EOS residual; the ideal parts cancel. The K1–K2 column checks
are in `analyses/bvp_solution_methods/results/k1_k2_148/`; the physical checks C3,
C5 and C6 belong to Engine #149.
