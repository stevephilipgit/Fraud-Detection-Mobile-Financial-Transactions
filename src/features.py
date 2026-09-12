"""Deterministic feature engineering — EXACT reproduction of the finalized
notebook's `engineer_features` (fraud_capstone_final.ipynb, Module 5.1).

Feature formulas (do NOT change — must match the saved model artifacts):

    log_amount        = np.log1p(amount)
    errorBalanceOrig  = oldbalanceOrg - amount - newbalanceOrig
    errorBalanceDest  = oldbalanceDest + amount - newbalanceDest
    isDestMerchant    = nameDest.str.startswith('M').astype(int)

The feature column order BEFORE encoding is read from
artifacts/feature_metadata.json (key: 'raw_feature_columns'), with the
notebook's order kept as a documented fallback constant.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src import config

# Notebook feature order (before encoding) — fallback if metadata is missing.
DEFAULT_FEATURE_ORDER: List[str] = [
    "type",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "log_amount",
    "errorBalanceOrig",
    "errorBalanceDest",
    "isDestMerchant",
]

ENGINEERED_FEATURES = ["log_amount", "errorBalanceOrig", "errorBalanceDest", "isDestMerchant"]


def get_feature_order() -> List[str]:
    """Feature column order before encoding, from feature_metadata.json.

    Single source of truth so the order is never hardcoded in multiple places.
    """
    try:
        import json

        with open(config.FEATURE_METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        order = list(metadata["raw_feature_columns"])
        if order:
            return order
    except (OSError, KeyError, ValueError):
        pass
    return list(DEFAULT_FEATURE_ORDER)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's exact row-wise feature engineering.

    Mirrors fraud_capstone_final.ipynb Module 5.1 verbatim. Deterministic —
    no fitting, no learned state.
    """
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["isDestMerchant"] = df["nameDest"].str.startswith("M").astype(int)
    return df


def validate_transaction(raw: Dict[str, Any]) -> List[str]:
    """Validate one raw transaction dict. Returns a list of error messages
    (empty list = valid).

    Checks presence, numeric types, non-negative balances/amount, and a known
    transaction type. The saved OneHotEncoder uses handle_unknown='ignore',
    which would silently produce an all-zero encoding for an unknown type —
    rejecting here instead prevents that silent degradation.
    """
    errors: List[str] = []

    for field in config.RAW_TRANSACTION_FIELDS:
        if field not in raw or raw[field] is None or (isinstance(raw[field], float) and np.isnan(raw[field])):
            errors.append(f"Missing required field: '{field}'")
    if errors:
        return errors

    for field in config.NUMERIC_TRANSACTION_FIELDS:
        try:
            value = float(raw[field])
        except (TypeError, ValueError):
            errors.append(f"Field '{field}' must be numeric, got: {raw[field]!r}")
            continue
        if value < 0:
            errors.append(f"Field '{field}' must be non-negative, got: {value}")

    tx_type = str(raw["type"]).upper()
    if tx_type not in config.VALID_TRANSACTION_TYPES:
        errors.append(
            f"Field 'type' must be one of {config.VALID_TRANSACTION_TYPES}, got: {raw['type']!r}"
        )

    if not isinstance(raw["nameDest"], str) or len(raw["nameDest"].strip()) == 0:
        errors.append("Field 'nameDest' must be a non-empty string (recipient ID)")

    return errors


def prepare_single(raw: Dict[str, Any]) -> pd.DataFrame:
    """Raw transaction dict -> single-row DataFrame with engineered features,
    columns ordered exactly as the model expects (pre-encoding order).

    The returned frame contains ONLY the model-relevant columns in
    `get_feature_order()` order (raw step / nameOrig are excluded).
    """
    errors = validate_transaction(raw)
    if errors:
        raise ValueError("Invalid transaction: " + "; ".join(errors))

    row = {
        "type": str(raw["type"]).upper(),
        "amount": float(raw["amount"]),
        "oldbalanceOrg": float(raw["oldbalanceOrg"]),
        "newbalanceOrig": float(raw["newbalanceOrig"]),
        "oldbalanceDest": float(raw["oldbalanceDest"]),
        "newbalanceDest": float(raw["newbalanceDest"]),
        "nameDest": str(raw["nameDest"]),
    }
    single = pd.DataFrame([row])
    engineered = engineer_features(single)
    return engineered[get_feature_order()]
