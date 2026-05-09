# SRP LHC Design Handoff

Date: 2026-05-09

This note captures what was learned from the legacy branches about the file called `LHC_design_w_SRP_cases.csv` and the old SRP run path. The short version is that this is best treated as a quick diagnostic DOE around an SRP-like case that once behaved well in the legacy model, not as a validated SRP campaign dataset.

## Current File

- Current branch file: `src/mea_absorption_column/data/LHC_design_w_SRP_cases.csv`
- Shape: 25 data rows plus one header row.
- Legacy name: `LHC_design_w_SRP_cases.csv`
- The file is already present on the current branch.

The columns are:

```text
Run,Tl,Tv,L,V,alpha,w_MEA,y_H2O,y_CO2,...
```

The remaining columns are old property-correlation coefficients for VLE, surface tension, molar volume, and viscosity.

## What The File Actually Is

The legacy generator is:

```text
legacy/main-legacy:src/mea_absorption_column/data/create_LHC_design.py
```

and older branches use:

```text
legacy/Multiple_Solving_Methods:MEA_Absorption_Column/data/create_LHC_design.py
```

The generator uses `scipy.stats.qmc.LatinHypercube` and samples around these means:

```text
Tl=314 K
Tv=320 K
L=29
V=3.5
alpha=0.28 mol CO2/mol MEA
w_MEA=0.300
y_H2O=0.013
y_CO2=0.175
```

with these half-widths:

```text
Tl +/- 3 K
Tv +/- 3 K
L +/- 3
V +/- 0.5
alpha +/- 0.02
w_MEA +/- 0.020
y_H2O +/- 0.003
y_CO2 +/- 0.01
```

So this file is a Latin-hypercube design sweep. It should be described as a legacy SRP-like DOE or diagnostic design, not as measured SRP validation data.

## Important Legacy SRP Details

Several legacy branches had an SRP conversion path:

```text
legacy/Multiple_Solving_Methods:MEA_Absorption_Column/Convert_Data/Convert_SRP_Data.py
legacy/Added_Pressure_Drop:MEA_Absorption_Column/Convert_Data/Convert_SRP_Data.py
legacy/Enhancement_Factor_Update:MEA_Absorption_Column/Convert_Data/Convert_SRP_Data.py
legacy/Solve_Enthalpy:MEA_Absorption_Column/Convert_Data/Convert_SRP_Data.py
```

The old SRP converter expects this eight-variable order:

```text
Tl_z, Tv_0, L or Fl_z, V or Fv_0, alpha, w_MEA, y_H2O, y_CO2
```

In mole mode it treats `L` and `V` as liquid and vapor molar flow totals. It then reconstructs liquid and vapor component flows from `alpha`, `w_MEA`, `y_H2O`, and `y_CO2`.

Key hardcoded legacy assumptions:

```text
P = 109180 Pa
alpha_O2_N2 = 0.085
```

## The "Good" SRP Case Was Often Hardcoded

The old SRP code did not always actually use the LHC row. In several branches, it read the row and then overwrote `X` with a hardcoded case.

In:

```text
legacy/Multiple_Solving_Methods:MEA_Absorption_Column/BVP/Run_Model.py
```

the SRP branch does this:

```python
X = df.iloc[run, :8].to_numpy()
X = [314, 320, 29.0, 3.52, 0.279, 0.325, 0.013, 0.100]
```

That means the "good SRP case" in that branch was:

```text
Tl = 314 K
Tv = 320 K
L = 29.0
V = 3.52
alpha = 0.279
w_MEA = 0.325
y_H2O = 0.013
y_CO2 = 0.100
```

In:

```text
legacy/Solve_Enthalpy:MEA_Absorption_Column/BVP/Run_Model.py
```

another hardcoded SRP case appears:

```python
X = [314, 322.717468719381, 29.0, 3.653690241, 0.279, 0.325, 0.0856390806700391, 0.0598166656728878]
```

That one has much higher vapor water and lower CO2. Before using either as a paper-facing claim, verify which hardcoded case was the one that originally "worked better."

## Column Geometry And Packing

The later legacy constants define:

```text
SRP:
  D = 0.467 m
  H = 12.0 m
```

This is visible in:

```text
legacy/main-legacy:src/mea_absorption_column/config/Constants.py
legacy/Solve_Temperature:MEA_Absorption_Column/Parameters.py
```

Older branches sometimes define:

```text
SRP:
  D ~= 0.427 m
  H = 6.0 m
```

and one converter hardcodes:

```text
D = 0.43 m
H = 6.0 m
```

So the geometry is not consistent across all legacy branches. The current repo constants already contain the later SRP values:

```text
src/mea_absorption_column/config/Constants.py
SRP D = 0.467 m
SRP H = 12.0 m
```

The packing setup in the later/current constants includes `MellapakPlus252Y` and `IMTP-40`. The current `convert_data` path defaults to `MellapakPlus252Y`, which matches the likely later SRP-style branch. Do not assume IMTP-40 unless the old specific run you are reproducing proves it used that packing.

## Current Runner Adapter Needed

The current benchmark runner does not accept the legacy LHC file directly. Current `convert_data` expects the first columns to mean:

```text
L_G, vapor_flow_rate, CO2_loading, w_MEA, y_CO2, T_liq, T_vap, P, Beds
```

The legacy LHC file instead has:

```text
Tl, Tv, L, V, alpha, w_MEA, y_H2O, y_CO2
```

For a current diagnostic adapter, the intended mapping should be:

```text
L_G = L / V
vapor_flow_rate = V
CO2_loading = alpha
w_MEA = w_MEA
y_CO2 = y_CO2
T_liq = Tl
T_vap = Tv
P = 109180 Pa
Beds = 1
D = 0.467 m
H = 12.0 m
y_H2O = y_H2O
```

If the current model should reproduce the old SRP path more faithfully, also verify whether the current `vapor_composition_mode` should use the explicit `y_H2O` column instead of reconstructing water from the current default ratio. The current converter will use `y_H2O` if the column exists.

## What Was Checked

Branch/file searches were run without checking out old branches.

Useful commands:

```powershell
git branch --list 'legacy/*' --format '%(refname:short)'
git ls-tree -r --name-only legacy/main-legacy
git grep -n -i --max-count 20 'SRP' legacy/main-legacy -- '*.py' '*.csv' '*.txt'
git show legacy/main-legacy:src/mea_absorption_column/data/create_LHC_design.py
git show legacy/Multiple_Solving_Methods:MEA_Absorption_Column/Convert_Data/Convert_SRP_Data.py
git show legacy/Multiple_Solving_Methods:MEA_Absorption_Column/BVP/Run_Model.py
git show legacy/Solve_Enthalpy:MEA_Absorption_Column/BVP/Run_Model.py
git show legacy/main-legacy:src/mea_absorption_column/config/Constants.py
```

Legacy `.xlsx` blobs were also checked for the text `SRP` using `openpyxl` from Git blobs. No SRP text was found inside those workbooks.

## Recommended Next-Agent Plan

1. Treat the LHC file as a DOE around a legacy SRP-like case, not a validation dataset.
2. Decide which legacy hardcoded SRP case is the actual seed case to reproduce.
3. Build a small adapter DataFrame using the mapping above.
4. First run one case with `ideal_henry` and one solver, bounded by subprocess timeout.
5. Only after that works, run the 25-row DOE across solvers and thermodynamic models.
6. Keep output in `analyses/nccc_validation/results/runs/<run_id>/`; do not promote it into `results/final` unless it becomes paper-facing.
7. If a manuscript claim is made, call it a "legacy SRP-like diagnostic design" or "legacy LHC diagnostic case set," not SRP validation data.

## Claim Boundary

Safe language:

```text
Legacy SRP-like DOE cases were used as a diagnostic stress test for solver and thermodynamic-model behavior.
```

Avoid:

```text
SRP validation cases
measured SRP campaign data
SRP experimental validation
```

unless a real SRP source table with documented geometry, packing, inlet conditions, and measured outputs is found.
