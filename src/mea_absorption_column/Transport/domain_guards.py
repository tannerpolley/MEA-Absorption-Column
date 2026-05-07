from __future__ import annotations

import math

import numpy as np

from mea_absorption_column.BVP.robust_core import record_domain_guard


class DomainGuardError(ValueError):
    def __init__(self, domain: str, reason: str):
        super().__init__(f"{domain}: {reason}")
        self.domain = domain
        self.reason = reason


def require_positive(domain: str, diagnostics: dict | None = None, **values):
    bad = [
        name
        for name, value in values.items()
        if not _is_finite_positive(value)
    ]
    if bad:
        _raise(domain, f"expected positive finite values for {', '.join(bad)}", diagnostics)


def require_fraction_between(domain: str, name: str, value: float, lower: float, upper: float, diagnostics=None):
    value = float(value)
    if not math.isfinite(value) or not (lower < value < upper):
        _raise(domain, f"{name} must satisfy {lower} < {name} < {upper}; got {value!r}", diagnostics)


def require_finite(domain: str, diagnostics: dict | None = None, **values):
    bad = [
        name
        for name, value in values.items()
        if not np.all(np.isfinite(np.asarray(value, dtype=float)))
    ]
    if bad:
        _raise(domain, f"expected finite values for {', '.join(bad)}", diagnostics)


def _is_finite_positive(value) -> bool:
    arr = np.asarray(value, dtype=float)
    return bool(np.all(np.isfinite(arr)) and np.all(arr > 0.0))


def _raise(domain: str, reason: str, diagnostics: dict | None):
    record_domain_guard(diagnostics, domain, reason)
    if diagnostics is not None and not bool(diagnostics.get("_strict_domain_guards", True)):
        return
    raise DomainGuardError(domain, reason)
