# Issue #22 CasADi comparison

Status: **supported-negative; do not retain a CasADi runtime path**.

The probe used the live seven-state absorber RHS and the immutable ePC-SAFT
wheel recorded in `issue22_casadi_probe.json`. CasADi 3.8.0 and IPOPT were
available. The public ePC-SAFT fixed-`T,P` chemical-potential tangent was
available with shape `9 x 7`; no ePC-SAFT equation was copied downstream.

## What was tested

The first pass wraps the unchanged SciPy RHS as a CasADi `Callback` with only
`eval()`. CasADi can evaluate the value, but direct collocation cannot form a
checked derivative and IPOPT is not attempted. This is the Issue #22 negative
gate, not a claim that CasADi itself cannot solve the equations.

The second pass adds a disposable full central-difference Jacobian callback.
That adapter is intentionally marked non-admissible: it is diagnostic proof of
the missing interface, not a candidate production derivative. It transcribes
the same RHS into a five-point trapezoidal collocation problem with 35 state
unknowns, 28 defects, and 7 boundary equations. IPOPT solves all three
diagnostic cases:

| Case | SciPy BVP (s) | SciPy nodes | IPOPT diagnostic (s) | IPOPT iterations | Defect inf-norm | Boundary inf-norm |
|---|---:|---:|---:|---:|---:|---:|
| K18 | 22.303 | 26 | 25.051 | 26 | 1.13e-10 | 1.49e-08 |
| 1C | 11.109 | 23 | 12.430 | 14 | 1.69e-09 | 6.15e-09 |
| 5C | 22.198 | 24 | 22.380 | 24 | 7.06e-12 | 1.46e-08 |

The EOS derivative was **not** used to produce those IPOPT Jacobians. The
separate `_eos_derivative_probe()` calls the public fixed-`T,P` tangent and
records its availability. The live RHS callback then calls the ordinary value
path; the diagnostic Jacobian finite-differences that complete value path,
including each EOS value evaluation. The existing reactive equilibrium call
also returns values/evidence, not a derivative object consumed by CasADi.

These timings are not an adoption comparison: the CasADi run is a coarse
five-point diagnostic transcription and uses finite differences plus IPOPT's
limited-memory Hessian. It establishes feasibility and exposes the remaining
derivative work. The measured runtime is not 2x faster than SciPy in any case.

## Exact work required for an admissible CasADi route

CasADi's callback seam needs a value contract plus a derivative contract. For
this RHS the contract is:

| Layer | Required output | Current situation | Required change |
|---|---|---|---|
| Callback seam | `rhs_solver(z, y_solver)` and exact `7 x 8` Jacobian ordered as `[z, y_solver[0:7]]` | Value-only callback; the diagnostic `7 x 8` matrix is central finite difference | Implement `has_jacobian()`/`get_jacobian()` around a checked analytic/AD Jacobian; fail closed if absent |
| State transform | Derivative of `solver_to_scaled_physical` and its inverse | Raw mode is identity; positive flow/pressure mode is not represented by the NumPy helper inside an MX graph | Encode the transform symbolically and include its chain rule |
| Thermal inversion | Derivatives of `Tl(Fl,Hlf)` and `Tv(Fv,Hvf)` | Each state evaluation calls SciPy `root`/`least_squares` | Either expose implicit sensitivities, `dT/dq = -G_q/G_T`, or promote temperature and enthalpy relations to algebraic constraints |
| Chemistry | Total derivative of `x_true(Fl,Tl,P)` and `Cl_true` | Current path hides a nonlinear Python equilibrium solve; the available ePC-SAFT tangent is only fixed-`T,P` composition information | Differentiate the selected chemistry residual with an implicit solve, or promote species/chemistry variables and residuals into the NLP |
| ePC-SAFT | Values and supplied CppAD derivatives in declared coordinate order, units, phase, branch, and fingerprints | Current retained state provides a fixed-`T,P` `9 x 7` composition tangent, not all temperature/pressure/property sensitivities needed by the RHS | Extend/use the public derivative capsule for every ePC-SAFT quantity entering fugacity, density, chemistry, and transport; never finite-difference the wheel |
| Properties | Partials of Henry coefficient, density, enthalpy, heat capacity, vapor pressure, viscosity, diffusivity, and conductivity | Implemented as NumPy/Python scalar functions, with `float()` and array coercions | Give CasADi expressions or exact partials at the module seam; preserve units and scaling |
| Hydraulics/transport | Partials through velocity, area, holdup, flooding, transfer coefficients, pressure drop, and fluxes | Value-only correlations with hard domain checks and piecewise branches | Encode the accepted branch equations and provide their chain-rule Jacobians; domain violations must be bounded/rejected, not penalty values inside a derivative graph |
| Film coupling | Derivative of the five-node interpolation, or derivatives of the film map if coupled | Current film conductance/bulk fugacity are outer fixed-point inputs; the film calculation contains quadrature and a scalar root | Keep them frozen as explicit parameters for a column-only comparison, or differentiate the integral/root implicitly before claiming a coupled route |
| Boundary conditions | Exact derivatives of the 7 endpoint constraints | Simple endpoint equations; direct collocation can express them symbolically | Preserve the existing mixed bottom/top ordering and scaling |
| NLP second order | Hessian strategy declared | Diagnostic used limited-memory Hessian because only first derivatives existed | Prefer sparse AD through the residual; otherwise explicitly validate a limited-memory IPOPT contract |

The central reason the existing public ePC-SAFT tangent is insufficient is that
the absorber RHS is a composition of maps, not just a fugacity evaluation:

```text
y_solver
  -> flows/compositions and (Tl,Tv) from enthalpy inversion
  -> properties and hydraulics
  -> selected chemistry branch x_true
  -> fugacity/VLE and transfer/flux correlations
  -> seven balances
```

The required RHS Jacobian is the total derivative through every arrow. A
fixed-`T,P` composition tangent covers only one internal ePC-SAFT block; it does
not differentiate the Python chemistry solve, temperature roots, pressure
dependence, or the downstream correlations.

## Recommended implementation route

1. Define one explicit `column_rhs_with_jacobian` module at the current RHS
   seam. Its result should contain the seven scaled physical RHS values, a
   finite exact `7 x 8` Jacobian, coordinate/unit metadata, and the selected
   branch/fingerprint identities.
2. Make the chemistry and thermal subproblems differentiable by implicit
   sensitivities or explicit algebraic NLP variables. Do not call SciPy roots,
   `np.interp`, `float`, guard penalties, or hidden finite differences from an
   MX evaluation.
3. Wrap that module in the CasADi callback and validate value/Jacobian parity
   against independent centered directional differences at nonsingular
   interior states. Use the wheel's supplied CppAD derivatives at the ePC-SAFT
   seam and reject missing/failed derivative payloads.
4. Run mesh/order refinement, branch/initialization sensitivity, all physical
   residuals, IPOPT primal/dual infeasibility, complementarity, bounds,
   iterations, and timings against the accepted SciPy route. Only then apply
   Issue #22's adoption rule.

The full machine-readable evidence, including every retained five-node
outer-film row and both missing-derivative and diagnostic-FD statuses, is in
`issue22_casadi_probe.json`. The CasADi callback mechanics used here are
documented by the [official CasADi documentation](https://web.casadi.org/docs/).
