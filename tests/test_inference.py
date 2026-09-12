import json

import pytest

from src import config, inference


def valid_raw(tx_type="TRANSFER"):
    return {
        "type": tx_type,
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameDest": "C123456789",
    }


def test_artifacts_load_and_threshold_matches_metadata():
    detector = inference.FraudDetector().load()
    metadata = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))

    assert detector.model is not None
    assert detector.preprocessor is not None
    assert detector.threshold == float(metadata["selected_threshold"])


@pytest.mark.parametrize("tx_type", ["TRANSFER", "CASH_OUT", "PAYMENT"])
def test_valid_transaction_predicts_for_categorical_types(tx_type):
    result = inference.FraudDetector().load().predict_raw(valid_raw(tx_type))

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision"] in {"FRAUD", "LEGITIMATE"}
    assert result["prediction"] in {0, 1}
    assert isinstance(result["threshold"], float)
    assert result["model_version"]
    assert result["engineered_features"]["type"] == tx_type


def test_missing_input_is_rejected():
    raw = valid_raw()
    raw.pop("amount")

    with pytest.raises(ValueError, match="Missing required field"):
        inference.FraudDetector().load().predict_raw(raw)


def test_invalid_probability_is_rejected(monkeypatch):
    detector = inference.FraudDetector().load()

    class BadModel:
        def predict_proba(self, encoded):
            return [[0.0, 1.2]]

    detector.model = BadModel()

    with pytest.raises(ValueError, match="invalid fraud probability"):
        detector.predict_raw(valid_raw())
