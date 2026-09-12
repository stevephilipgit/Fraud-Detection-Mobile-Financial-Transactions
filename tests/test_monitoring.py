from src import config, logging_utils, monitoring


def test_compute_summary_without_labels():
    logs = [
        {"fraud_probability": 0.2, "predicted_class": 0, "model_version": "v1", "actual_label": None},
        {"fraud_probability": 0.8, "predicted_class": 1, "model_version": "v1", "actual_label": None},
        {"fraud_probability": 0.6, "predicted_class": 1, "model_version": "v1", "actual_label": None},
    ]

    summary = monitoring.compute_summary(logs)

    assert summary["sample_count"] == 3
    assert summary["fraud_prediction_count"] == 2
    assert summary["legitimate_prediction_count"] == 1
    assert summary["average_probability"] == 0.533333
    assert summary["labels_available"] is False


def test_compute_summary_with_labels():
    logs = [
        {"fraud_probability": 0.9, "predicted_class": 1, "actual_label": 1},
        {"fraud_probability": 0.7, "predicted_class": 1, "actual_label": 0},
        {"fraud_probability": 0.2, "predicted_class": 0, "actual_label": 0},
        {"fraud_probability": 0.1, "predicted_class": 0, "actual_label": 1},
    ]

    summary = monitoring.compute_summary(logs)

    assert summary["labels_available"] is True
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["tn"] == 1
    assert summary["fn"] == 1
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["f1"] == 0.5


def test_prediction_logging_jsonl_fallback(monkeypatch):
    fallback_path = config.LOGS_DIR / "test_predictions_fallback.jsonl"
    if fallback_path.exists():
        fallback_path.unlink()
    monkeypatch.setattr(logging_utils.database, "is_available", lambda: False)
    monkeypatch.setattr(logging_utils.config, "JSONL_FALLBACK_PATH", fallback_path)

    status = logging_utils.log_prediction(
        {
            "fraud_probability": 0.42,
            "prediction": 1,
            "threshold": 0.44,
            "model_version": "xgboost_v1",
            "latency_ms": 3.5,
        },
        transaction_reference="C123",
    )

    assert status["status"] == "jsonl"
    assert fallback_path.exists()
    assert fallback_path.stat().st_size > 0
    fallback_path.unlink()
