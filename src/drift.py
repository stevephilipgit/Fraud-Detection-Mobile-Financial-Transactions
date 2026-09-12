"""Prototype data-drift detection — compares a reference (training-like)
dataset against current incoming data.

Numeric features  : Population Stability Index (PSI), quantile-based bins
                    computed on the REFERENCE distribution.
Categorical       : per-category share comparison; drift score = the largest
                    absolute share change across categories.

IMPORTANT — prototype thresholds (documented heuristics, NOT universal
industry standards; interpretations vary by domain and use case):

    PSI < 0.10           -> OK        (no significant drift)
    0.10 <= PSI < 0.25   -> WARNING   (moderate drift, investigate)
    PSI >= 0.25          -> DRIFT     (significant distribution change)

    categorical: max share change < 0.05 -> OK, < 0.15 -> WARNING, else DRIFT

This module only REPORTS drift. It never retrains and never modifies
artifacts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src import config

EPSILON = 1e-6  # avoids division by zero / log(0) in PSI

STATUS_OK = "OK"
STATUS_WARNING = "WARNING"
STATUS_DRIFT = "DRIFT"


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI between two numeric distributions.

    Bin edges are quantiles of the reference distribution; current values are
    assigned to those same bins (values outside the reference range fall into
    the outermost bins, mirroring how production scoring would see extremes).
    """
    reference = pd.Series(reference).dropna().astype(float)
    current = pd.Series(current).dropna().astype(float)
    if reference.empty or current.empty:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference column
        const = reference.iloc[0]
        return float((current != const).mean())

    edges[0], edges[-1] = -np.inf, np.inf  # open outer bins
    ref_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_counts = np.histogram(current, bins=edges)[0] / len(current)

    ref_counts = np.clip(ref_counts, EPSILON, None)
    cur_counts = np.clip(cur_counts, EPSILON, None)

    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def categorical_drift(reference: pd.Series, current: pd.Series) -> Dict[str, Any]:
    """Largest absolute per-category share change between two categorical
    distributions. Returns the score plus per-category shares for reporting."""
    ref = pd.Series(reference).fillna("<MISSING>").astype(str)
    cur = pd.Series(current).fillna("<MISSING>").astype(str)

    ref_shares = ref.value_counts(normalize=True)
    cur_shares = cur.value_counts(normalize=True)
    categories = sorted(set(ref_shares.index) | set(cur_shares.index))

    max_change = 0.0
    detail: Dict[str, Dict[str, float]] = {}
    for cat in categories:
        r = float(ref_shares.get(cat, 0.0))
        c = float(cur_shares.get(cat, 0.0))
        detail[cat] = {"reference": round(r, 4), "current": round(c, 4)}
        max_change = max(max_change, abs(r - c))

    return {"score": float(max_change), "shares": detail}


def _status_numeric(psi: float) -> str:
    if psi < config.PSI_OK:
        return STATUS_OK
    if psi < config.PSI_WARNING:
        return STATUS_WARNING
    return STATUS_DRIFT


def _status_categorical(change: float) -> str:
    if change < config.CATEGORICAL_DRIFT_THRESHOLD:
        return STATUS_OK
    if change < 3 * config.CATEGORICAL_DRIFT_THRESHOLD:  # 0.15
        return STATUS_WARNING
    return STATUS_DRIFT


def compare_distributions(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare reference vs current data and return a readable result table:

        Feature | Reference | Current | Drift Score | Status

    Reference / Current columns show the mean (numeric) or the majority
    category share (categorical) to make the table human-readable.
    """
    if numeric_features is None:
        numeric_features = [
            c for c in reference_df.columns
            if pd.api.types.is_numeric_dtype(reference_df[c])
        ]
    if categorical_features is None:
        categorical_features = [
            c for c in reference_df.columns
            if c not in numeric_features
        ]

    results: List[Dict[str, Any]] = []

    for feature in numeric_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref, cur = reference_df[feature], current_df[feature]
        score = psi_numeric(ref, cur)
        results.append(
            {
                "Feature": feature,
                "Reference": round(float(pd.to_numeric(ref, errors="coerce").mean()), 4),
                "Current": round(float(pd.to_numeric(cur, errors="coerce").mean()), 4),
                "Drift Score": round(score, 4),
                "Status": _status_numeric(score),
            }
        )

    for feature in categorical_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref, cur = reference_df[feature], current_df[feature]
        outcome = categorical_drift(ref, cur)
        top_cat = max(outcome["shares"], key=lambda k: outcome["shares"][k]["reference"])
        results.append(
            {
                "Feature": feature,
                "Reference": f"{top_cat} {outcome['shares'][top_cat]['reference']:.0%}",
                "Current": f"{top_cat} {outcome['shares'][top_cat]['current']:.0%}",
                "Drift Score": round(outcome["score"], 4),
                "Status": _status_categorical(outcome["score"]),
            }
        )

    return pd.DataFrame(results)

