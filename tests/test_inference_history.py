"""Tests for the Monitoring / Drift / inference-history redesign.

Database-backed tests run against an in-memory SQLite database created from the
project's real SQLAlchemy metadata; database._connection and database.get_engine
are monkeypatched, so no MySQL server is required for these unit tests.
"""
import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from src import config, database, monitoring, upload_inference

REQUIRED = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "nameDest",
]


@pytest.fixture
def db(monkeypatch):
    """Fresh in-memory SQLite DB with the project's real table metadata."""
    from contextlib import contextmanager

    engine = create_engine("sqlite://")
    database.metadata.create_all(engine)

    @contextmanager
    def fake_connection():
        with engine.begin() as conn:
            yield conn

    monkeypatch.setattr(database, "_connection", fake_connection)
    monkeypatch.setattr(database, "get_engine", lambda: engine)
    monkeypatch.setattr(database, "is_available", lambda: True)
    return engine


def _raw_df(n=3):
    rows = []
    for i in range(n):
        rows.append({
            "type": "PAYMENT", "amount": 100.0 + i, "oldbalanceOrg": 200.0,
            "newbalanceOrig": 100.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
            "nameDest": f"M{i}",
        })
    return pd.DataFrame(rows)


def _scored_results(n=3, fraud_every=2):
    results = []
    for i in range(n):
        prob = 0.9 if i % fraud_every == 0 else 0.01
        results.append({
            "row_index": i,
            "error": None,
            "result": {
                "fraud_probability": prob,
                "prediction": int(prob >= 0.44),
                "decision": "FRAUD" if prob >= 0.44 else "LEGITIMATE",
                "threshold": 0.44,
                "model_version": "xgboost_v1",
                "latency_ms": 1.0,
                "raw_transaction": {
                    "type": "PAYMENT", "amount": 100.0 + i,
                    "oldbalanceOrg": 200.0, "newbalanceOrig": 100.0,
                    "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
                    "nameDest": f"M{i}",
                },
                "engineered_features": {"log_amount": 1.0},
            },
        })
    return results


# ---------------------------------------------------------------------------
# Migration / schema
# ---------------------------------------------------------------------------
def test_02_idempotent_migration(db):
    """prediction_logs.batch_id is added when missing and the migration is idempotent."""
    with db.begin() as conn:
        conn.execute(text("DROP TABLE prediction_logs"))
        conn.execute(text(
            "CREATE TABLE prediction_logs ("
            "prediction_id INTEGER PRIMARY KEY, timestamp DATETIME, "
            "transaction_reference TEXT, raw_transaction TEXT, engineered_features TEXT, "
            "fraud_probability FLOAT, predicted_class INTEGER, threshold FLOAT, "
            "model_version VARCHAR(64), actual_label INTEGER, latency_ms FLOAT)"
        ))
    assert "batch_id" not in [c["name"] for c in inspect(db).get_columns("prediction_logs")]
    assert database._ensure_column("prediction_logs", "batch_id", "INT NULL") is True
    assert "batch_id" in [c["name"] for c in inspect(db).get_columns("prediction_logs")]
    # second run is a no-op
    assert database._ensure_column("prediction_logs", "batch_id", "INT NULL") is False


# ---------------------------------------------------------------------------
# Batch creation / persistence
# ---------------------------------------------------------------------------
def test_01_create_inference_batch(db):
    batch_id = database.create_inference_batch(
        source=config.BATCH_SOURCE_UPLOADED, label="test.csv",
        content_hash="a" * 64, dataset_hash="b" * 64,
        row_count=3, fraud_count=1, avg_fraud_probability=0.31, has_ground_truth=False,
    )
    batches = database.fetch_inference_batches()
    assert len(batches) == 1
    assert batches[0]["batch_id"] == batch_id
    assert batches[0]["source"] == "uploaded_csv"
    assert batches[0]["has_ground_truth"] == 0


def test_03_batch_creation_and_persistence(db):
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="batch.csv",
        raw_df=_raw_df(3), results=_scored_results(3), labels=pd.Series([0, 1, 0]),
    )
    assert status["status"] == "saved"
    batches = database.fetch_inference_batches()
    assert len(batches) == 1
    assert batches[0]["row_count"] == 3
    assert batches[0]["fraud_count"] == 2  # probs 0.9/0.01/0.9 -> 2 fraud predictions
    assert batches[0]["has_ground_truth"] == 1
    preds = database.fetch_batch_predictions(batches[0]["batch_id"])
    assert len(preds) == 3
    assert [p["actual_label"] for p in preds] == [0, 1, 0]
    assert all(p["batch_id"] == batches[0]["batch_id"] for p in preds)


def test_04_batch_lookup_by_content_hash(db):
    upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    found = database.get_batch_by_content_hash("c" * 64, source=config.BATCH_SOURCE_UPLOADED)
    assert found is not None and found["label"] == "a.csv"
    assert database.get_batch_by_content_hash("z" * 64) is None


def test_05_content_hash_duplicate_reused(db):
    first = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(3), results=_scored_results(3),
    )
    again = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(3), results=_scored_results(3),
    )
    assert first["status"] == "saved" and again["status"] == "reused"
    assert again["batch_id"] == first["batch_id"]
    assert len(database.fetch_inference_batches()) == 1


def test_06_dataset_hash_duplicate_reused(db):
    """Same records in a different row order (different bytes) reuse the batch."""
    first = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(3), results=_scored_results(3),
    )
    second = upload_inference.persist_scored_batch(
        content_hash="e" * 64, dataset_hash="d" * 64, label="b.csv",
        raw_df=_raw_df(3), results=_scored_results(3),
    )
    assert second["status"] == "reused"
    assert second["batch_id"] == first["batch_id"]
    assert len(database.fetch_inference_batches()) == 1


def test_07_prediction_rows_carry_batch_id(db):
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    preds = database.fetch_batch_predictions(status["batch_id"])
    assert len(preds) == 2
    assert all(p["batch_id"] == status["batch_id"] for p in preds)


def test_08_unlabeled_batch(db):
    upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="u.csv",
        raw_df=_raw_df(2), results=_scored_results(2), labels=None,
    )
    batches = database.fetch_inference_batches()
    assert batches[0]["has_ground_truth"] == 0
    preds = database.fetch_batch_predictions(batches[0]["batch_id"])
    assert all(p["actual_label"] is None for p in preds)


def test_09_no_label_leakage_in_persisted_rows(db):
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="l.csv",
        raw_df=_raw_df(3), results=_scored_results(3), labels=pd.Series([0, 1, 0]),
    )
    preds = database.fetch_batch_predictions(status["batch_id"])
    for p in preds:
        raw = p["raw_transaction"]
        assert "isFraud" not in raw
        assert set(raw.keys()) == set(REQUIRED)


# ---------------------------------------------------------------------------
# Historical retrieval
# ---------------------------------------------------------------------------
def test_10_historical_retrieval(db):
    b1 = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d1" * 32, label="one.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    b2 = upload_inference.persist_scored_batch(
        content_hash="e" * 64, dataset_hash="d2" * 32, label="two.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    batches = database.fetch_inference_batches()
    assert [b["batch_id"] for b in batches] == [b2["batch_id"], b1["batch_id"]]
    preds = database.fetch_batch_predictions(b1["batch_id"])
    assert len(preds) == 2
    assert all(p["batch_id"] == b1["batch_id"] for p in preds)


def test_20_no_batches_empty_state(db):
    assert database.fetch_inference_batches() == []
    assert database.fetch_latest_batch(source=config.BATCH_SOURCE_UPLOADED) is None


# ---------------------------------------------------------------------------
# Monitoring scope / metrics
# ---------------------------------------------------------------------------
def _insert_legacy_row():
    with database._connection() as conn:
        conn.execute(
            text("INSERT INTO prediction_logs (timestamp, transaction_reference, "
                 "fraud_probability, predicted_class, threshold, model_version, "
                 "actual_label, latency_ms) VALUES (:ts, 'smoke', 0.5, 1, 0.44, "
                 "'xgboost_v1', NULL, 1)"),
            {"ts": datetime.datetime(2026, 1, 1)},
        )


def test_11_monitoring_scope_excludes_legacy(db):
    _insert_legacy_row()
    upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="new.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    default_rows = database.fetch_scoped_predictions(include_legacy=False)
    assert len(default_rows) == 2
    assert all(r["batch_id"] is not None for r in default_rows)
    legacy_rows = database.fetch_scoped_predictions(include_legacy=True)
    assert len(legacy_rows) == 3
    assert any(r["batch_id"] is None for r in legacy_rows)


def test_12_monitoring_scope_sources(db):
    fc = database.get_or_create_fraud_check_batch()
    database.insert_batch_predictions([
        {"timestamp": datetime.datetime.now(), "transaction_reference": "M1",
         "raw_transaction": {"type": "PAYMENT"}, "engineered_features": {},
         "fraud_probability": 0.1, "predicted_class": 0, "threshold": 0.44,
         "model_version": "xgboost_v1", "actual_label": None, "latency_ms": 1.0,
         "batch_id": fc},
    ])
    hr = database.create_inference_batch(
        source=config.BATCH_SOURCE_HISTORICAL_REPLAY, label="replay",
        row_count=0, fraud_count=0,
    )
    database.insert_batch_predictions([
        {"timestamp": datetime.datetime.now(), "transaction_reference": "M2",
         "raw_transaction": {"type": "PAYMENT"}, "engineered_features": {},
         "fraud_probability": 0.1, "predicted_class": 0, "threshold": 0.44,
         "model_version": "xgboost_v1", "actual_label": None, "latency_ms": 1.0,
         "batch_id": hr},
    ])
    upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="new.csv",
        raw_df=_raw_df(1), results=_scored_results(1),
    )
    rows = database.fetch_scoped_predictions(include_legacy=False)
    assert {r["batch_source"] for r in rows} == {"fraud_check", "uploaded_csv"}


def test_13_monitoring_metrics_with_labels(db):
    """Predictions match labels exactly: TP=2, TN=2, accuracy 1.0."""
    results = _scored_results(4, fraud_every=1)
    results[2]["result"]["fraud_probability"] = 0.01
    results[2]["result"]["prediction"] = 0
    results[3]["result"]["fraud_probability"] = 0.01
    results[3]["result"]["prediction"] = 0
    labels = pd.Series([1, 1, 0, 0])
    upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="l.csv",
        raw_df=_raw_df(4), results=results, labels=labels,
    )
    rows = database.fetch_scoped_predictions(include_legacy=False)
    summary = monitoring.compute_summary(rows)
    assert summary["labels_available"] is True
    assert summary["accuracy"] == 1.0
    assert summary["tp"] == 2 and summary["tn"] == 2
    assert summary["fp"] == 0 and summary["fn"] == 0


def test_22_volume_and_source_breakdown(db):
    fc = database.get_or_create_fraud_check_batch()
    database.insert_batch_predictions([
        {"timestamp": datetime.datetime.now(), "transaction_reference": "M1",
         "raw_transaction": {"type": "PAYMENT"}, "engineered_features": {},
         "fraud_probability": 0.1, "predicted_class": 0, "threshold": 0.44,
         "model_version": "xgboost_v1", "actual_label": None, "latency_ms": 1.0,
         "batch_id": fc},
    ])
    upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="new.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    _insert_legacy_row()
    rows = database.fetch_scoped_predictions(include_legacy=True)
    volume = monitoring.daily_volume(rows)
    assert volume["predictions"].sum() == 4
    breakdown = monitoring.source_breakdown(rows)
    assert set(breakdown["source"]) >= {"uploaded_csv", "fraud_check", "legacy"}


# ---------------------------------------------------------------------------
# Snapshot / fn regression
# ---------------------------------------------------------------------------
def test_14_snapshot_insertion_and_fn_regression(db):
    """The fn payload key used to crash values(**payload); execute-time params fix it."""
    with pytest.raises(TypeError):
        database.model_monitoring.insert().values(fn=1)  # characterization

    summary = monitoring.compute_summary([
        {"fraud_probability": 0.9, "predicted_class": 1, "model_version": "xgboost_v1", "actual_label": 0},
        {"fraud_probability": 0.1, "predicted_class": 0, "model_version": "xgboost_v1", "actual_label": 1},
    ])
    status = monitoring.save_snapshot(summary)
    assert status["status"] == "mysql"
    with db.begin() as conn:
        rows = conn.execute(text("SELECT * FROM model_monitoring")).fetchall()
    assert len(rows) == 1
    assert rows[0]._mapping["fn"] == 1  # FN correctly stored
    assert rows[0]._mapping["fp"] == 1
    assert rows[0]._mapping["tp"] == 0


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def test_17_drift_reference_frame():
    from src import features

    df = pd.read_csv(config.INFERENCE_REFERENCE_CSV)
    ref = features.engineer_features(df)
    assert len(ref) == 11142
    for col in ("log_amount", "errorBalanceOrig", "errorBalanceDest",
                "isDestMerchant", "type"):
        assert col in ref.columns


def test_18_drift_current_batch_frame(db):
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="drift.csv",
        raw_df=_raw_df(3), results=_scored_results(3),
    )
    frame = upload_inference.load_batch_frame(status["batch_id"])
    assert len(frame) == 3
    for col in ("log_amount", "errorBalanceOrig", "errorBalanceDest",
                "isDestMerchant", "type"):
        assert col in frame.columns


def test_19_min_drift_rows_gate():
    ok, msg = upload_inference.meets_min_rows(config.MIN_DRIFT_ROWS)
    assert ok is True and msg == ""
    ok, msg = upload_inference.meets_min_rows(config.MIN_DRIFT_ROWS - 1)
    assert ok is False
    assert str(config.MIN_DRIFT_ROWS) in msg


# ---------------------------------------------------------------------------
# Data protection / MySQL-down behavior / Fraud Check batch
# ---------------------------------------------------------------------------
def test_15_transactions_table_untouched_by_persistence(db):
    upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="x.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    with db.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar() == 0


def test_16_mysql_unavailable(db, monkeypatch):
    monkeypatch.setattr(database, "is_available", lambda: False)
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="x.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    assert status["status"] == "unavailable"
    assert "MySQL" in status["message"]
    with db.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM prediction_logs")).scalar() == 0
        assert conn.execute(text("SELECT COUNT(*) FROM inference_batches")).scalar() == 0


def test_21_fraud_check_daily_batch(db):
    id1 = database.get_or_create_fraud_check_batch()
    id2 = database.get_or_create_fraud_check_batch()
    assert id1 == id2  # one batch per calendar day
    batches = database.fetch_inference_batches(
        sources=(config.BATCH_SOURCE_FRAUD_CHECK,)
    )
    assert len(batches) == 1
    assert batches[0]["source"] == "fraud_check"


# ---------------------------------------------------------------------------
# Delete / cleanup
# ---------------------------------------------------------------------------
def test_23_delete_existing_batch(db):
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    result = database.delete_inference_batch(status["batch_id"])
    assert result["deleted"] is True
    assert result["batch_found"] is True
    assert result["prediction_rows_deleted"] == 2
    assert database.fetch_inference_batches() == []
    assert database.fetch_batch_predictions(status["batch_id"]) == []


def test_24_delete_removes_only_that_batch(db):
    b1 = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d1" * 32, label="one.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    b2 = upload_inference.persist_scored_batch(
        content_hash="e" * 64, dataset_hash="d2" * 32, label="two.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    result = database.delete_inference_batch(b1["batch_id"])
    assert result["deleted"] is True
    batches = database.fetch_inference_batches()
    assert [b["batch_id"] for b in batches] == [b2["batch_id"]]
    assert len(database.fetch_batch_predictions(b2["batch_id"])) == 2
    assert database.fetch_batch_predictions(b1["batch_id"]) == []


def test_25_delete_nonexistent_batch(db):
    result = database.delete_inference_batch(9999)
    assert result["deleted"] is False
    assert result["batch_found"] is False
    assert "does not exist" in result["message"]


def test_26_repeated_delete_safe(db):
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    first = database.delete_inference_batch(status["batch_id"])
    second = database.delete_inference_batch(status["batch_id"])
    assert first["deleted"] is True
    assert second["deleted"] is False
    assert second["batch_found"] is False


def test_27_delete_preserves_legacy(db):
    _insert_legacy_row()
    _insert_legacy_row()
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    database.delete_inference_batch(status["batch_id"])
    assert database.count_legacy_prediction_logs() == 2


def test_28_delete_legacy_only_null_rows(db):
    _insert_legacy_row()
    _insert_legacy_row()
    _insert_legacy_row()
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    result = database.delete_legacy_prediction_logs()
    assert result["rows_deleted"] == 3
    assert database.count_legacy_prediction_logs() == 0
    # batch and its predictions are untouched
    assert len(database.fetch_batch_predictions(status["batch_id"])) == 2
    assert len(database.fetch_inference_batches()) == 1


def test_29_delete_rolls_back_on_failure(db, monkeypatch):
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    bid = status["batch_id"]

    class ExplodingBatches:
        def select(self):
            return database.inference_batches.select()

        def delete(self):
            raise RuntimeError("forced failure during batch delete")

    original = database.inference_batches
    monkeypatch.setattr(database, "inference_batches", ExplodingBatches())
    with pytest.raises(RuntimeError):
        database.delete_inference_batch(bid)
    monkeypatch.setattr(database, "inference_batches", original)

    # rollback: both the predictions and the batch record still exist
    assert len(database.fetch_batch_predictions(bid)) == 2
    assert database.fetch_latest_batch() is not None


def test_30_transactions_untouched_by_delete(db):
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="a.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    database.delete_inference_batch(status["batch_id"])
    with db.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar() == 0


def test_31_delete_queries_parameterized():
    stmt = database.prediction_logs.delete().where(
        database.prediction_logs.c.batch_id == 5
    )
    compiled = str(stmt.compile())
    assert "5" not in compiled  # value bound as a parameter, never inlined
    assert ":batch_id" in compiled


def test_32_monitoring_excludes_deleted_batch(db):
    b1 = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d1" * 32, label="one.csv",
        raw_df=_raw_df(2), results=_scored_results(2),
    )
    b2 = upload_inference.persist_scored_batch(
        content_hash="e" * 64, dataset_hash="d2" * 32, label="two.csv",
        raw_df=_raw_df(1), results=_scored_results(1),
    )
    database.delete_inference_batch(b1["batch_id"])
    rows = database.fetch_scoped_predictions(include_legacy=True)
    assert len(rows) == 1
    assert all(r["batch_id"] == b2["batch_id"] for r in rows)


def test_33_drift_excludes_deleted_batch(db):
    status = upload_inference.persist_scored_batch(
        content_hash="c" * 64, dataset_hash="d" * 64, label="drift.csv",
        raw_df=_raw_df(3), results=_scored_results(3),
    )
    assert len(database.fetch_inference_batches(
        sources=(config.BATCH_SOURCE_UPLOADED,))) == 1
    database.delete_inference_batch(status["batch_id"])
    assert database.fetch_inference_batches(
        sources=(config.BATCH_SOURCE_UPLOADED,)) == []


def test_34_count_legacy(db):
    _insert_legacy_row()
    _insert_legacy_row()
    assert database.count_legacy_prediction_logs() == 2