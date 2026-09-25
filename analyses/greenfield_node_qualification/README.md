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

## Result (wheel `684b213a…`, 34 s against the 600 s node budget)

| Check | States | Criterion | Largest passing defect | Outcome |
|---|---|---|---|---|
| N1 elements, mass, charge | S1, S2 | 1e-12 × scale | 8.5e-14 | 12/12 pass |
| N1 row residual, liquid pressure root | S1, S2 | 1e-7; 1e-8 | 2.4e-14; 8.9e-15 | pass |
| N1 f = a R T ρ0 vs x φ P (CO2, water) | S1, S2 | 1e-10 | 4.8e-11 | 4/4 pass |
| N2 Gibbs–Duhem, v, e_MEA, e_H2O | 7 loadings | 1e-12 × scale | 1.3e-14 | 21/21 pass; ion-dropped controls fail |
| N3 tangent ladder | 7 loadings × 9 species | 5e-6 | 6.9e-8 | 63/63 resolved, pass |
| N4 outer actions e_P, e_CO2, e_MEA, e_H2O | 7 × 4 × 9 | 5e-5 | 1.0e-5 | 252/252 pass; chain-term controls fail |
| N4 e_T | 7 loadings | — | — | typed refusal `ReferenceUnavailable` (#147) |
| N5 projections, t^T M t ≥ 0, g > 0 | 47 quadrature states | 1e-12 × ‖M‖∞ | 4.6e-21 | 282/282 pass |
| N6 19×12 node Jacobian | 3C, S2 node | 1e-5 | — | typed refusal `ReferenceUnavailable` (#147) |
| C1 apparent-feed Euler, H_L and H_V | S1, S2, V1, V2 | 5e-6 | 7.0e-15 | 4/4 pass; one-term-omitted controls fail |
| C2 first order dH_L (T, feeds), dH_V (T, feeds) | S1, S2, V1, V2 | 5e-6 | 3.0e-11 | 18/18 pass |
| C2 second order Cp_V, h̄_CO2, h̄_H2O | V1, V2 × 6 directions | 5e-5 | 7.3e-8 | 36/36 pass |
| C2 dH_L/dP | S1, S2 | — | — | typed refusal `ReferenceUnavailable` (#147) |
| C4 pure-water Δh_vap vs IAPWS-95 | 313.15, 329.4, 348.5 K | 2 % | 0.42 % | 3/3 pass |

Every finite-difference component was resolved (the two finest Richardson
estimates agree to 10 %). Each zeroed or omitted-term negative control fails
its own criterion. S3 (360 K, 106.4 kPa, loading 0.45) evaluates. The mixture
diagnostic h̄_H2O^V − h̄_H2O^L is 42.8 kJ/mol (V1/S1) and 42.4 kJ/mol (V2/S2);
no reference exists for it.

## Start policy and an Engine finding

The Engine's default cold start fails on about 1 % of absorber states
(budget_exhausted after 200 iterations, "contraction absent"), and a 1 Pa
change flips the outcome, for example S1 at λ = −0.2 and P + 200 Pa, or the
S2-temperature anchor state. The liquid then follows native continuation from
the last accepted state, else the case anchor loading; 3 of 264 solves used
it. Continuation from S1 to S2 reproduces the cold-solved S2 amounts
(largest relative amount difference 0, criterion 1e-8; 1.7e-16 in a separate replay). The stall itself is an Engine
driver defect, reported as tannerpolley/ePC-SAFT#157.

## Claim limits

Numerical verification only, at S1/S2/V1/V2 and their loadings, on an
exploratory record. The CO2 ideal-gas Shomate record is extrapolated below 298 K. C4 tests the pure-water EOS residual; the ideal parts
cancel. N4 e_T, dH_L/dP, N6 and the K1–K2 column checks wait for Engine #147;
the physical checks C3, C5 and C6 belong to Engine #149.
