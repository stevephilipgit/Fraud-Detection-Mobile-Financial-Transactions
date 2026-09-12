"""Single-transaction inference against the saved notebook artifacts.

The flow implemented here is the architectural contract of the application:

    raw transaction
        -> feature engineering (features.engineer_features, notebook-exact)
        -> saved fitted preprocessor  (transform() ONLY — never fit)
        -> saved XGBoost model        (predict_proba() ONLY — never fit)
        -> probability
        -> saved threshold (from model_metadata.json)
        -> Fraud / Legitimate decision

Inference never fits preprocessing, never fits an encoder, never trains,
never runs SMOTE, never retrains XGBoost.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src import config, features


class FraudDetector:
    """Loads the saved artifacts once and scores raw transactions."""

    def __init__(self) -> None:
        self._loaded = False
        self.model = None
        self.preprocessor = None
        self.feature_metadata: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        self.threshold: Optional[float] = None
        self.model_version: Optional[str] = None

    # -- artifact loading ---------------------------------------------------
    def load(self) -> "FraudDetector":
        """Load model, preprocessor and metadata from artifacts/."""
        self.model = joblib.load(config.MODEL_PATH)
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)

        with open(config.FEATURE_METADATA_PATH, encoding="utf-8") as f:
            self.feature_metadata = json.load(f)
        with open(config.MODEL_METADATA_PATH, encoding="utf-8") as f:
            self.model_metadata = json.load(f)

        # Threshold comes from the notebook's exported metadata — never
        # hardcoded in application code.
        threshold = self.model_metadata.get("selected_threshold")
        if threshold is None:
            raise KeyError(
                "model_metadata.json does not contain 'selected_threshold'; "
                "refusing to fall back to a hardcoded threshold."
            )
        self.threshold = float(threshold)

        # Model version: metadata key if present, otherwise the centrally
        # recorded version from config (never invented per-request).
        self.model_version = self.model_metadata.get("model_version", config.MODEL_VERSION)

        self._loaded = True
        return self

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # -- scoring -------------------------------------------------------------
    def predict_raw(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Score ONE raw transaction and return the full result payload.

        Raises ValueError on invalid input (see features.validate_transaction).
        """
        self.ensure_loaded()
        start = time.perf_counter()

        # 1. exact notebook feature engineering (validation happens inside)
        model_input = features.prepare_single(raw)

        # 2. saved fitted preprocessor — transform() only
        encoded = self.preprocessor.transform(model_input)

        # 3. saved model — predict_proba() only
        probabilities = np.asarray(self.model.predict_proba(encoded))
        if probabilities.ndim != 2 or probabilities.shape[0] != 1 or probabilities.shape[1] < 2:
            raise ValueError(f"Model returned invalid predict_proba shape: {probabilities.shape}")
        proba = float(probabilities[:, 1][0])
        if not 0.0 <= proba <= 1.0:
            raise ValueError(f"Model returned invalid fraud probability: {proba}")

        # 4. saved threshold from model_metadata.json
        prediction = int(proba >= self.threshold)
        decision = "FRAUD" if prediction == 1 else "LEGITIMATE"

        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "fraud_probability": proba,
            "prediction": prediction,
            "decision": decision,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "model_name": self.model_metadata.get("model_name"),
            "latency_ms": latency_ms,
            "engineered_features": _serialize_model_input(model_input),
            "raw_transaction": dict(raw),
        }


# Module-level shared instance (loaded lazily on first use).
_detector: Optional[FraudDetector] = None


def get_detector() -> FraudDetector:
    """Return the shared FraudDetector singleton, loading artifacts on first use."""
    global _detector
    if _detector is None:
        _detector = FraudDetector().load()
    return _detector


def predict_transaction(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience wrapper: score one raw transaction dict."""
    return get_detector().predict_raw(raw)


def _serialize_model_input(model_input: pd.DataFrame) -> Dict[str, Any]:
    """Return one model-input row as JSON-friendly values.

    The categorical `type` column must remain a string; only numeric model
    inputs are converted to floats for stable display/logging.
    """
    row = model_input.iloc[0]
    serialized: Dict[str, Any] = {}
    for col in features.get_feature_order():
        value = row[col]
        serialized[col] = value if isinstance(value, str) else float(value)
    return serialized
