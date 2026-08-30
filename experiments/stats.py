"""Small statistics helpers (NumPy only) for finite-shot experiments.

Kept dependency-free on purpose: no scipy. The normal-approximation and Wilson
intervals below are standard and adequate for shot counts in the thousands.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

Z_95 = 1.959963984540054  # standard normal quantile for a two-sided 95% interval


@dataclasses.dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    method: str

    def as_dict(self) -> dict:
        return {"point": self.point, "ci_low": self.low, "ci_high": self.high, "method": self.method}


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion -- well behaved near 0/1."""
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0, "wilson")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), "wilson")


def mean_interval(samples: np.ndarray, z: float = Z_95) -> Interval:
    """Normal-approximation CI for the mean of bounded samples (e.g. +/-1 eigenvalues)."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    return Interval(mean, mean - z * se, mean + z * se, "normal-approx")


def chi_square_uniform(counts: dict, n_categories: int) -> tuple[float, int]:
    """Pearson chi-square statistic against a uniform distribution, plus dof."""
    total = sum(counts.values())
    expected = total / n_categories
    stat = sum((counts.get(k, 0) - expected) ** 2 / expected for k in range(n_categories)) \
        if all(isinstance(k, int) for k in counts) \
        else sum((v - expected) ** 2 / expected for v in counts.values())
    return float(stat), n_categories - 1


# chi-square 0.1% upper-tail critical values, dof 1..8 (no scipy needed)
CHI2_0001 = {1: 10.828, 2: 13.816, 3: 16.266, 4: 18.467, 5: 20.515, 6: 22.458, 7: 24.322, 8: 26.124}
