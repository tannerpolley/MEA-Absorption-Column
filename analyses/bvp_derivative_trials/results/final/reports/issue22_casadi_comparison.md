# Issue #22 CasADi comparison

Status: **supported-negative; do not retain a CasADi runtime path**.

The probe ran on repository commit `4cd362c` with CasADi 3.8.0 and IPOPT
available. It used the immutable ePC-SAFT wheel from Engine commit
`8007a70815efcd277f06eddc4df9fffdd8cdca48`, SHA-256
`f6e5b51dad79741c759393688f7daa547c9eb73f5944d5f877b3b32b1e56714a`, and
the retained reactive parameter and reaction-system identities. The public
ePC-SAFT fixed-
`T,P` tangent was available with shape `9 x 7`; no ePC-SAFT equation was
copied or reimplemented.

| Case | SciPy runtime (s) | SciPy boundary norm | SciPy dense ODE residual | SciPy nodes | Outer updates | Final conductance change | Final bulk-fugacity change |
|---|---:|---:|---:|---:|---:|---:|---:|
| K18 | 24.963 | 4.00e-14 | 6.66e-02 | 26 | 15 | 2.681e-02 | 2.139e-02 |
| 1C | 18.288 | 1.06e-13 | 3.65e-02 | 23 | 15 | 2.083e-02 | 1.370e-02 |
| 5C | 22.104 | 1.79e-14 | 1.14e-01 | 24 | 3 | 1.734e-02 | 2.087e-02 |

For each case, the unchanged seven-state column RHS evaluated through a
CasADi callback. A five-point trapezoidal direct-collocation defect was then
formed, but CasADi reported a missing callback Jacobian for every case before
an NLP or IPOPT iteration could be created. IPOPT was therefore not attempted;
enabling CasADi finite differences would violate the Issue #22 checked-
derivative gate.

The current equilibrium-manifold film is not a spatial film BVP: it evaluates
a scalar resistance integral and gas-film interface root while calling nested
Python reactive-state solves. A fully coupled CasADi transcription would need
an additional checked derivative contract through those callbacks and through
the complete absorber RHS. The supplied ePC-SAFT tangent alone is insufficient.

The candidate therefore meets neither adoption route: it resolves no admitted
SciPy failure and has no measured IPOPT runtime. The complete machine-readable
evidence, including every retained five-node outer row, is in
`issue22_casadi_probe.json`.
