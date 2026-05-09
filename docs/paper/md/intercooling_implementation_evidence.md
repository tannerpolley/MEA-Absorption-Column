# Intercooling Implementation Evidence And Modeling Direction

This note collects the evidence used for the test branch `codex/intercooler-temperature-evidence`.
It is intended as a project note, not manuscript text.

## Local Sources Reviewed

- Morgan et al. 2018, `docs/paper/md/morgan_2018_supporting_information.md`: source of the
  NCCC K1-K23 absorber operating table, gas-side/liquid-side capture comparisons, and temperature
  profile figure captions for the NCCC validation rows.
- Moore et al. 2021, `docs/paper/md/moore_2021_advanced_absorber_heat_integration.md`: rate-based
  absorber heat-integration paper. It treats intercooling as heat removal distributed through heat
  exchange packing and compares that behavior against simple intercooling and optimized temperature
  profiles.
- Zotero collection `Absorption Column Modeling`: checked for rate-based absorber and intercooling
  context. Relevant indexed items include Morgan et al. 2018 (`RCZFHMRH`), Chinen et al. 2018
  (`KJ688HMG`), Moore et al. 2021 (`LY8GZ649`), Kvamsdal and Rochelle 2008 (`C96CLVL4`), Moioli
  et al. 2012 (`VIWZ7BCQ`), Shahid et al. 2019 (`FAPLYGDV`), and related rate-based absorber work.

## External Sources Checked

- DOE OSTI record for Moore et al. 2021 reports that heat exchange packing can remove absorber heat
  to a cooling fluid and that 10-20% cooled column length was sufficient in their modeled MEA case
  to reduce column height by about 15%: https://www.osti.gov/pages/biblio/1770528
- Aspen Plus/RadFrac documentation describes interstage heater/cooler and pumparound specifications
  in terms of flow rate, outlet temperature, temperature change, vapor fraction, or heat duty, and
  allows returning a pumparound on-stage or above-stage. This supports treating discrete intercoolers
  as side draws/coolers/returns rather than as hidden height multipliers:
  https://1library.net/article/heater-cooler-specifications-aspen-reference-manual-operation-models.yrm3pdoq
- AspenTech describes Rate-Based Distillation Column Modeling as the detailed RadFrac path for more
  reliable column simulations over a wider operating range:
  https://www.aspentech.com/en/products/pages/distillation-modeling-in-aspen-plus/

## Engineering Interpretation

The current project has two physically different intercooler concepts:

- Discrete liquid cooling between packed beds. This is closest to a side-draw cooler or pumparound
  return. It should preserve liquid composition and molar flow unless an explicit side stream is
  modeled, reset the liquid thermal state through a cooler energy balance, and impose vapor
  continuity upward.
- Distributed heat removal over a finite packed zone. This is closer to heat exchange packing or a
  cooling jacket within a bed section. It should appear as an energy sink term in the liquid/gas
  energy equations over a selected axial interval, optionally coupled to a coolant temperature ODE.

The old "instant reset at a bed boundary" version is useful as a first diagnostic, but it has two
limitations that hurt accuracy and convergence:

- It imposes a hard internal thermal boundary condition, which can create sharp temperature kinks
  and a stiff collocation residual at the bed interface.
- It does not represent finite heat-transfer area, coolant approach temperature, or duty limits, so
  it can overcool or undercool the liquid without exposing the missing heat-exchanger assumptions.

## Recommended Implementation Ladder

1. Keep the current single-bed and staged-bed baseline as the reference path.
2. Prefer direct temperature-state solving for intercooled cases so the BVP variables are bounded
   temperatures rather than enthalpies that must be inverted during Newton iterations.
3. Represent a discrete intercooler as an explicit side-cooler map between beds:
   preserve liquid molar flows, compute outlet liquid temperature from either a measured target,
   approach temperature, or heat duty, then return the cooled liquid to the next bed.
4. Use continuation on the intercooler strength: no reset, weak reset, full measured target or duty.
5. Add a distributed-cooling option for smoother profiles:
   `q_cool = U_a (T_liquid - T_coolant)` over a finite axial zone, with a constant coolant
   temperature first and a coolant ODE later if coolant flow data are available.
6. Treat measured NCCC intercooled temperature profiles as validation evidence only when the K-case
   operating row, bed count, and intercooler count are matched through `data/reference/nccc_master_cases.csv`.

## Direct Temperature-State Status

This branch now avoids precomputing inlet and boundary enthalpies when
`solver_settings={"thermal_state_mode": "temperature"}` is selected. In that mode, `run_model(...)`
passes liquid and vapor temperatures directly into `abs_column(...)`; `abs_column(...)` then evaluates
enthalpies algebraically for flux calculations while returning `dT_l/dz` and `dT_v/dz` as the solved
thermal differentials. The legacy enthalpy-state path remains available and unchanged by default.
