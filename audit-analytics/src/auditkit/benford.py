"""Benford's law tests (first-two digits), following Nigrini's conformity thresholds.

Benford's law: in many naturally occurring financial data sets, the leading digits are not uniform; a first
digit of 1 appears about 30% of the time and 9 about 5%. Invented numbers, or numbers pushed just under a
threshold, often break this pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Nigrini (2012) mean absolute deviation thresholds for the first-two-digits test
MAD_LIMITS = [(0.0012, "close conformity"), (0.0018, "acceptable conformity"),
              (0.0022, "marginal conformity"), (float("inf"), "nonconformity")]


def first_two_digits(x: pd.Series) -> pd.Series:
    x = x.abs()
    x = x[x >= 10]
    return (x / 10 ** np.floor(np.log10(x) - 1)).astype(int).clip(10, 99)


def expected_first_two() -> pd.Series:
    d = np.arange(10, 100)
    return pd.Series(np.log10(1 + 1 / d), index=d)


@dataclass
class BenfordResult:
    n: int
    mad: float
    conclusion: str
    chi2: float
    p_value: float
    table: pd.DataFrame  # index digit pair; columns count, observed, expected, z


def benford_test(amounts: pd.Series) -> BenfordResult:
    d = first_two_digits(amounts)
    n = len(d)
    exp = expected_first_two()
    count = d.value_counts().reindex(exp.index, fill_value=0)
    obs = count / n
    # z-statistic with continuity correction (Nigrini)
    z = (np.abs(obs - exp) - 1 / (2 * n)).clip(lower=0) / np.sqrt(exp * (1 - exp) / n)
    mad = float(np.mean(np.abs(obs - exp)))
    conclusion = next(label for limit, label in MAD_LIMITS if mad <= limit)
    chi2, p = stats.chisquare(count, exp * n)
    table = pd.DataFrame({"count": count, "observed": obs, "expected": exp, "z": z})
    return BenfordResult(n, mad, conclusion, float(chi2), float(p), table)
