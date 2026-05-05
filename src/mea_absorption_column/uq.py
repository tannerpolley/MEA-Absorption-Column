from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UQPlan:
    reference_runtime_s: float
    cached_runtime_s: float
    surrogate_runtime_s: float


def estimate_two_tier_throughput(
    plan: UQPlan,
    samples: int,
    reference_fraction: float = 0.1,
    cache_fraction: float = 0.4,
) -> dict[str, float | int]:
    if samples < 1:
        raise ValueError("samples must be positive.")
    if reference_fraction < 0.0 or cache_fraction < 0.0 or reference_fraction + cache_fraction > 1.0:
        raise ValueError("runtime fractions must be non-negative and sum to at most 1.")

    reference_samples = int(round(samples * reference_fraction))
    cached_samples = int(round(samples * cache_fraction))
    surrogate_samples = samples - reference_samples - cached_samples
    total_runtime = (
        reference_samples * plan.reference_runtime_s
        + cached_samples * plan.cached_runtime_s
        + surrogate_samples * plan.surrogate_runtime_s
    )
    return {
        "samples": int(samples),
        "reference_samples": int(reference_samples),
        "cached_samples": int(cached_samples),
        "surrogate_samples": int(surrogate_samples),
        "estimated_total_runtime_s": float(total_runtime),
        "estimated_samples_per_second": float(samples / total_runtime) if total_runtime > 0.0 else float("inf"),
    }
