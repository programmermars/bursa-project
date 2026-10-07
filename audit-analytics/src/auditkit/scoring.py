"""Scoring: how well each test finds the planted irregularities, and a combined risk ranking.

precision   share of flagged records that are a planted irregularity of the type the test targets
hit rate    share of flagged records that are *any* planted irregularity (a test can find other problems)
recall      share of planted irregularities of the target type that the test flagged
"""
from __future__ import annotations

import pandas as pd

from .checks import CHECKS, Check
from .generate import Dataset


def score_checks(exceptions: pd.DataFrame, truth: pd.DataFrame, checks: list[Check] = CHECKS) -> pd.DataFrame:
    all_bad = set(zip(truth.table, truth.record_id))
    rows = []
    for c in checks:
        e = exceptions[exceptions.test == c.name]
        flagged = set(zip(e.table, e.record_id))
        tgt = truth[truth.anomaly.isin(c.targets)]
        target = set(zip(tgt.table, tgt.record_id))
        tp = len(flagged & target)
        rows.append({"test": c.name, "area": c.area, "objective": c.objective, "flagged": len(flagged),
                     "planted": len(target), "found": tp,
                     "precision": tp / len(flagged) if flagged else None,
                     "hit_rate": len(flagged & all_bad) / len(flagged) if flagged else None,
                     "recall": tp / len(target) if target else None})
    return pd.DataFrame(rows)


def risk_ranking(exceptions: pd.DataFrame) -> pd.DataFrame:
    """One row per flagged record, highest risk first.

    risk_score = sum over signal groups of the strongest test weight in that group, so two tests that measure
    the same thing (e.g. Benford spike and just-below-limit) do not double-count.
    """
    if exceptions.empty:
        return pd.DataFrame(columns=["table", "record_id", "risk_score", "tests", "reasons"])
    g = exceptions.groupby(["table", "record_id"])
    score = exceptions.groupby(["table", "record_id", "group"]).weight.max().groupby(["table", "record_id"]).sum()
    out = pd.DataFrame({
        "risk_score": score,
        "tests": g.test.nunique(),
        "reasons": g.apply(lambda x: " | ".join(f"{t}: {r}" for t, r in zip(x.test, x.reason)), include_groups=False),
    }).reset_index()
    return out.sort_values(["risk_score", "tests"], ascending=False, kind="stable").reset_index(drop=True)


def precision_at(ranking: pd.DataFrame, truth: pd.DataFrame, ns=(25, 50, 100)) -> dict[int, float]:
    bad = set(zip(truth.table, truth.record_id))
    keys = list(zip(ranking.table, ranking.record_id))
    return {n: sum(k in bad for k in keys[:n]) / min(n, len(keys)) for n in ns if keys}


def coverage(exceptions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """For each planted irregularity type: how many exist and how many were flagged by at least one test."""
    flagged = set(zip(exceptions.table, exceptions.record_id))
    t = truth.assign(found=[k in flagged for k in zip(truth.table, truth.record_id)])
    return t.groupby("anomaly").agg(planted=("record_id", "count"), found=("found", "sum")) \
            .assign(recall=lambda d: d.found / d.planted).sort_values("recall").reset_index()


def evaluate(ds: Dataset, exceptions: pd.DataFrame) -> dict:
    ranking = risk_ranking(exceptions)
    return {"by_test": score_checks(exceptions, ds.truth), "coverage": coverage(exceptions, ds.truth),
            "ranking": ranking, "precision_at": precision_at(ranking, ds.truth)}
