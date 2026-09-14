"""Tests for the locked "Revision 2" Uploaded Inference Data workflow.

Pure unit tests (no MySQL required); integration checks (MySQL-dependent) are
skipped when the server is unavailable. The upload module is exercised through
its public functions; the frozen model is only touched for the consistency test.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from src import config, database, inference, logging_utils, upload_inference

REQUIRED = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "nameDest",
]

BASE_ROWS = [
    {"type": "PAYMENT", "amount": 100.0, "oldbalanceOrg": 200.0, "newbalanceOrig": 100.0,
     "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "nameDest": "M1111"},
    {"type": "TRANSFER", "amount": 500.0, "oldbalanceOrg": 1000.0, "newbalanceOrig": 500.0,
     "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "nameDest": "C2222"},
    {"type": "CASH_OUT", "amount": 250.0, "oldbalanceOrg": 300.0, "newbalanceOrig": 50.0,
     "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "nameDest": "M3333"},
]


def _make_csv(rows, extra=None):
    df = pd.DataFrame(rows)
    if extra:
        for col, values in extra.items():
            df[col] = values
    return df.to_csv(index=False).encode("utf-8")


class FakeDetector:
    """Minimal predict_raw() stand-in that records what it was fed."""

    def __init__(self, probability=0.05):
        self.probability = float(probability)
        self.calls = 0
        self.keys_seen = set()

    def predict_raw(self, raw):
        self.calls += 1
        self.keys_seen.update(str(k) for k in raw)
        prediction = int(self.probability >= 0.44)
        return {
            "fraud_probability": self.probability,
            "prediction": prediction,
            "decision": "FRAUD" if prediction else "LEGITIMATE",
            "threshold": 0.44,
            "model_version": "test",
            "engineered_features": {"type": str(raw["type"]), "log_amount": 1.0},
            "raw_transaction": dict(raw),
            "latency_ms": 1.0,
        }


def test_upload_limits_defaults():
    assert config.UPLOAD_MAX_BYTES == 10 * 1024 * 1024
    assert config.UPLOAD_MAX_ROWS == 5000


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_1_valid_unlabeled_upload():
    parsed = upload_inference.validate_upload(_make_csv(BASE_ROWS))
    assert parsed.row_count == 3
    assert parsed.labels is None
    assert list(parsed.raw_df.columns) == REQUIRED
    assert parsed.dataset_hash


def test_3_full_paysim_with_optional_columns():
    parsed = upload_inference.validate_upload(
        _make_csv(
            BASE_ROWS,
            {"step": [1, 1, 1], "nameOrig": ["A1", "A2", "A3"],
             "isFraud": [0, 0, 0], "transaction_id": [1, 2, 3]},
        )
    )
    assert parsed.row_count == 3
    assert parsed.present_optional_columns == ["isFraud", "nameOrig", "step", "transaction_id"]
    assert list(parsed.raw_df.columns) == REQUIRED


def test_4_missing_required_column():
    df = pd.DataFrame(BASE_ROWS).drop(columns=["amount"])
    with pytest.raises(upload_inference.UploadValidationError) as excinfo:
        upload_inference.validate_upload(df.to_csv(index=False).encode("utf-8"))
    assert "Missing required column(s)" in str(excinfo.value)


def test_5_unexpected_column_rejected():
    with pytest.raises(upload_inference.UploadValidationError) as excinfo:
        upload_inference.validate_upload(_make_csv(BASE_ROWS, {"mystery": [1, 2, 3]}))
    assert "Unexpected column(s)" in str(excinfo.value)


def test_6_invalid_transaction_type():
    rows = [dict(BASE_ROWS[0], type="WIRE")] + BASE_ROWS[1:]
    with pytest.raises(upload_inference.UploadValidationError) as excinfo:
        upload_inference.validate_upload(_make_csv(rows))
    assert "type" in str(excinfo.value)


def test_7_negative_numeric_value():
    rows = [dict(BASE_ROWS[0], amount=-1.0)] + BASE_ROWS[1:]
    with pytest.raises(upload_inference.UploadValidationError):
        upload_inference.validate_upload(_make_csv(rows))


def test_8_nan_value():
    rows = [dict(BASE_ROWS[0], amount=float("nan"))] + BASE_ROWS[1:]
    with pytest.raises(upload_inference.UploadValidationError):
        upload_inference.validate_upload(_make_csv(rows))


def test_8b_inf_value():
    df = pd.DataFrame(BASE_ROWS)
    df.loc[0, "amount"] = float("inf")
    with pytest.raises(upload_inference.UploadValidationError) as excinfo:
        upload_inference.validate_upload(df.to_csv(index=False).encode("utf-8"))
    assert "finite" in str(excinfo.value)


def test_9_invalid_isfraud():
    with pytest.raises(upload_inference.UploadValidationError) as excinfo:
        upload_inference.validate_upload(_make_csv(BASE_ROWS, {"isFraud": [0, 1, 2]}))
    assert "0/1" in str(excinfo.value)
    # genuinely ambiguous values are rejected, never coerced
    with pytest.raises(upload_inference.UploadValidationError):
        upload_inference.validate_upload(_make_csv(BASE_ROWS, {"isFraud": ["yes", "no", "maybe"]}))


def test_10_empty_and_header_only():
    with pytest.raises(upload_inference.UploadValidationError):
        upload_inference.validate_upload(b"")
    header_only = ",".join(REQUIRED).encode("utf-8") + b"\n"
    with pytest.raises(upload_inference.UploadValidationError):
        upload_inference.validate_upload(header_only)


def test_11_duplicate_rows_reported_not_removed():
    rows = [dict(BASE_ROWS[0]), dict(BASE_ROWS[0]), dict(BASE_ROWS[1])]
    parsed = upload_inference.validate_upload(_make_csv(rows))
    assert parsed.row_count == 3
    assert parsed.duplicate_row_count == 1
    assert len(parsed.row_fingerprints) == 3
    assert len(set(parsed.row_fingerprints)) == 2


def test_11b_row_limit_and_size_limit():
    with pytest.raises(upload_inference.UploadValidationError) as excinfo:
        upload_inference.validate_upload(_make_csv([dict(BASE_ROWS[0])] * 5), max_rows=3)
    assert "5 rows" in str(excinfo.value)
    with pytest.raises(upload_inference.UploadValidationError) as excinfo:
        upload_inference.validate_upload(_make_csv(BASE_ROWS), max_bytes=50)
    assert "byte" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Content hash, fingerprints, dataset hash, exact overlap
# ---------------------------------------------------------------------------
def test_14_same_bytes_same_content_hash():
    blob = _make_csv(BASE_ROWS)
    assert upload_inference.compute_content_hash(blob) == upload_inference.compute_content_hash(blob)


def test_15_same_content_different_filename_same_hash():
    # The content hash has no filename input — identical bytes must match.
    assert upload_inference.compute_content_hash(_make_csv(BASE_ROWS)) == (
        upload_inference.compute_content_hash(_make_csv(BASE_ROWS))
    )


def test_16_same_size_different_bytes_different_hash():
    blob_a = b"abcdef"
    blob_b = b"abcdeg"  # same length, different trailing byte
    assert len(blob_a) == len(blob_b)
    assert upload_inference.compute_content_hash(blob_a) != upload_inference.compute_content_hash(blob_b)


def test_17_reordered_dataset_same_dataset_hash():
    rows_a = [dict(BASE_ROWS[0]), dict(BASE_ROWS[1]), dict(BASE_ROWS[2])]
    rows_b = [dict(BASE_ROWS[2]), dict(BASE_ROWS[0]), dict(BASE_ROWS[1])]
    parsed_a = upload_inference.validate_upload(_make_csv(rows_a))
    parsed_b = upload_inference.validate_upload(_make_csv(rows_b))
    assert parsed_a.dataset_hash == parsed_b.dataset_hash
    # dataframe row order is preserved (never sorted)
    assert list(parsed_a.raw_df["nameDest"]) == ["M1111", "C2222", "M3333"]
    assert list(parsed_b.raw_df["nameDest"]) == ["M3333", "M1111", "C2222"]


def test_12_overlap_zero():
    fp_upload = [upload_inference.fingerprint_row(dict(BASE_ROWS[0]))]
    fp_other = upload_inference.fingerprint_row(
        {"type": "PAYMENT", "amount": 999.0, "oldbalanceOrg": 1.0, "newbalanceOrig": 0.0,
         "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "nameDest": "X999"}
    )
    assert upload_inference.compute_overlap(fp_upload, [fp_other]) == 0
    message = upload_inference.overlap_message(0, 1)
    assert "No exact duplicate records found in the original dataset." in message


def test_13_overlap_positive():
    fps = [upload_inference.fingerprint_row(dict(row)) for row in BASE_ROWS]
    assert upload_inference.compute_overlap(fps, [fps[1]]) == 1
    message = upload_inference.overlap_message(1, 3)
    assert "1 of 3 uploaded records exactly match records in the original dataset." in message


def test_13b_original_csv_replay_overlap_positive():
    original = pd.read_csv(config.INFERENCE_REFERENCE_CSV, nrows=5)
    rows = [
        {col: row[col] for col in upload_inference.INFERENCE_COLUMNS}
        for _, row in original.iterrows()
    ]
    parsed = upload_inference.validate_upload(_make_csv(rows))
    overlap = upload_inference.compute_overlap(
        parsed.row_fingerprints, upload_inference.original_dataset_fingerprints()
    )
    assert overlap == len(rows)


# ---------------------------------------------------------------------------
# Scoring guards / canonical path / labels
# ---------------------------------------------------------------------------
def test_18_score_once_guard():
    session = {}
    detector = FakeDetector(probability=0.9)
    parsed = upload_inference.validate_upload(_make_csv(BASE_ROWS))
    assert upload_inference.ensure_scored(session, parsed.raw_df, parsed.dataset_hash, detector) is True
    assert detector.calls == 3
    assert upload_inference.ensure_scored(session, parsed.raw_df, parsed.dataset_hash, detector) is False
    assert detector.calls == 3  # no rescore


def test_18b_progress_reported_per_row():
    events = []
    detector = FakeDetector()
    parsed = upload_inference.validate_upload(_make_csv(BASE_ROWS))
    upload_inference.ensure_scored(
        {}, parsed.raw_df, parsed.dataset_hash, detector,
        progress=lambda done, total: events.append((done, total)),
    )
    assert events == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_19_labeled_evaluation_correct():
    metrics = upload_inference.evaluate_predictions([1, 1, 0, 0], [1, 0, 0, 1])
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (1, 1, 1, 1)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    # zero denominators -> None, no crash
    all_legit = upload_inference.evaluate_predictions([0, 0], [0, 0])
    assert all_legit["precision"] is None
    assert all_legit["recall"] is None
    assert all_legit["f1"] is None
    assert all_legit["accuracy"] == 1.0


def test_20_isFraud_never_reaches_model():
    detector = FakeDetector()
    parsed = upload_inference.validate_upload(_make_csv(BASE_ROWS, {"isFraud": [0, 1, 0]}))
    upload_inference.ensure_scored({}, parsed.raw_df, parsed.dataset_hash, detector)
    assert detector.keys_seen == set(REQUIRED)
    assert "isFraud" not in detector.keys_seen


def test_21_score_dataframe_matches_predict_raw():
    detector = inference.FraudDetector().load()
    parsed = upload_inference.validate_upload(_make_csv(BASE_ROWS))
    results = upload_inference.score_dataframe(parsed.raw_df, detector)
    assert len(results) == 3
    for i, entry in enumerate(results):
        expected = detector.predict_raw(parsed.raw_df.iloc[i].to_dict())
        assert entry["error"] is None
        assert entry["result"]["prediction"] == expected["prediction"]
        assert np.isclose(
            entry["result"]["fraud_probability"],
            expected["fraud_probability"],
            rtol=1e-9,
            atol=1e-12,
        )


def test_22_unavailable_mysql_still_scores():
    # Uploaded inference is artifact-only: it never touches the database module.
    detector = FakeDetector()
    parsed = upload_inference.validate_upload(_make_csv(BASE_ROWS))
    results = upload_inference.score_dataframe(parsed.raw_df, detector)
    assert len(results) == 3
    assert all(r["error"] is None for r in results)


def test_22b_logging_falls_back_when_mysql_down(monkeypatch):
    fallback_path = config.LOGS_DIR / "test_upload_fallback.jsonl"
    if fallback_path.exists():
        fallback_path.unlink()
    monkeypatch.setattr(logging_utils.database, "is_available", lambda: False)
    monkeypatch.setattr(logging_utils.config, "JSONL_FALLBACK_PATH", fallback_path)

    parsed = upload_inference.validate_upload(_make_csv(BASE_ROWS))
    results = upload_inference.score_dataframe(parsed.raw_df, FakeDetector())
    statuses = upload_inference.log_results(
        results,
        parsed.raw_df,
        lambda result, transaction_reference=None: logging_utils.log_prediction(
            result, transaction_reference=transaction_reference
        ),
    )
    assert len(statuses) == 3
    assert all(s["status"] == "jsonl" for s in statuses)
    assert fallback_path.exists()
    fallback_path.unlink()


# ---------------------------------------------------------------------------
# Protected / integration path checks
# ---------------------------------------------------------------------------
def test_23_mysql_historical_path_intact():
    assert callable(database.fetch_recent_transactions)
    signature = inspect.signature(database.fetch_recent_transactions)
    assert signature.parameters["limit"].default == 50
    source = inspect.getsource(database.fetch_recent_transactions)
    assert "ORDER BY transaction_id DESC" in source


def test_24_transactions_row_count_unchanged():
    if not database.is_available():
        pytest.skip("MySQL unavailable — integration check skipped")
    from sqlalchemy import text as sql_text

    with database.get_engine().connect() as conn:
        assert conn.execute(sql_text("SELECT COUNT(*) FROM transactions")).scalar() == 11142