"""Uploaded Inference Data — session-scoped CSV batch scoring.

This module implements the LOCKED "Revision 2" design for scoring arbitrary
uploaded CSVs through the frozen inference artifacts:

    upload bytes
        -> SHA-256 content hash (authoritative upload identity)
        -> validation (file-level, row-level, label-level)
        -> row fingerprints (deterministic canonical 7-field representation)
        -> order-independent dataset hash
        -> exact-overlap vs the original model-development dataset
        -> frozen inference path (FraudDetector.predict_raw ONLY)
        -> results
        -> optional labeled evaluation (isFraud AFTER prediction only)
        -> optional prediction logging (existing logging_utils)

Never fits, retrains, resamples or changes the threshold. `isFraud`, when
present, is separated before scoring and used ONLY for post-prediction
evaluation. Uploaded data is session-scoped and is never written to MySQL.
"""
from __future__ import annotations

import datetime
import hashlib
import io
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

from src import config, database, features

# ---------------------------------------------------------------------------
# CSV column contract (locked)
# ---------------------------------------------------------------------------
INFERENCE_COLUMNS: tuple = tuple(config.RAW_TRANSACTION_FIELDS)  # the 7 required fields
NUMERIC_INFERENCE_COLUMNS: tuple = tuple(config.NUMERIC_TRANSACTION_FIELDS)
KNOWN_OPTIONAL_COLUMNS: tuple = ("step", "nameOrig", "isFraud", "transaction_id")
ALLOWED_COLUMNS: set = set(INFERENCE_COLUMNS) | set(KNOWN_OPTIONAL_COLUMNS)

# Canonical fingerprint serialization (locked contract).
_FIELD_DELIMITER = "\x1f"          # ASCII unit separator
_FINGERPRINT_ENCODING = "utf-8"


class UploadValidationError(ValueError):
    """Raised when an uploaded CSV fails validation (whole-file rejection)."""


@dataclass
class UploadResult:
    """Result of validating one uploaded CSV.

    `raw_df` contains ONLY the 7 required inference columns in the uploaded
    row order (never sorted). `labels` (when present) is positionally aligned
    to `raw_df`. Nothing here is ever passed to MySQL.
    """

    content_hash: str
    raw_df: pd.DataFrame
    labels: Optional[pd.Series] = None
    present_optional_columns: List[str] = field(default_factory=list)
    row_count: int = 0
    duplicate_row_count: int = 0
    row_fingerprints: List[str] = field(default_factory=list)
    dataset_hash: str = ""


# ---------------------------------------------------------------------------
# Upload identity — SHA-256 of the ACTUAL file bytes (never name+size)
# ---------------------------------------------------------------------------
def compute_content_hash(file_bytes: bytes) -> str:
    """Authoritative identity of an upload = hash of the raw uploaded bytes.

    Two files with the same name and size but different bytes MUST hash
    differently; the filename is never part of the identity.
    """
    return hashlib.sha256(file_bytes).hexdigest()


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Strict CSV read. UTF-8 (BOM tolerated); locale numerics NOT accepted.

    Values such as ``1,234.56`` stay strings and subsequently fail numeric
    validation — no locale parsing is performed.
    """
    text = file_bytes.decode("utf-8-sig")  # strips UTF-8 BOM when present
    return pd.read_csv(io.StringIO(text))


# ---------------------------------------------------------------------------
# Validation — whole-file rejection on any invalid row, with row-level detail
# ---------------------------------------------------------------------------
def _validate_columns(columns: Sequence[str]) -> List[str]:
    """Required vs known-optional vs unexpected columns (exact names)."""
    errors: List[str] = []
    cols = [str(c) for c in columns]
    missing = [c for c in INFERENCE_COLUMNS if c not in cols]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing))
    duplicates = sorted({c for c in cols if cols.count(c) > 1})
    if duplicates:
        errors.append("Duplicate column name(s): " + ", ".join(duplicates))
    unexpected = [c for c in cols if c not in ALLOWED_COLUMNS]
    if unexpected:
        errors.append("Unexpected column(s) (remove them from the file): " + ", ".join(unexpected))
    return errors


def _validate_row(raw: Mapping[str, Any]) -> List[str]:
    """Row validation: the existing canonical contract + finite-number guard.

    Uses features.validate_transaction unchanged (never re-implemented here);
    the extra guard rejects non-finite numerics (inf), which the canonical
    validator intentionally does not treat as "negative".
    """
    errors = list(features.validate_transaction(dict(raw)))
    for col in NUMERIC_INFERENCE_COLUMNS:
        try:
            value = float(raw[col])
        except (TypeError, ValueError):
            continue  # already reported as non-numeric by validate_transaction
        if not np.isfinite(value):
            errors.append(f"Field '{col}' must be finite, got: {raw[col]!r}")
    return errors


def _is_valid_label_value(value: Any) -> bool:
    """isFraud must be a numeric/bool 0 or 1. Strings are NEVER coerced."""
    if value is None:
        return False
    if isinstance(value, (str, bytes)):
        return False  # no coercion of "0", "1", "yes", "no", ...
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(parsed):
        return False
    return parsed in (0.0, 1.0)


def _validate_labels(series: pd.Series) -> List[str]:
    invalid = series[~series.map(_is_valid_label_value)]
    if len(invalid):
        sample = ", ".join(str(i) for i in list(invalid.index)[:10])
        return [
            "isFraud must contain only numeric 0/1 values; "
            f"invalid value(s) at row(s): {sample}."
        ]
    return []


def validate_upload(
    file_bytes: bytes,
    max_bytes: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> UploadResult:
    """Validate an uploaded CSV and build the clean scoring dataset.

    Raises UploadValidationError (whole-file rejection) when the file is
    empty, not parseable, over the size/row limits, missing/unexpected
    columns, or contains any invalid row / invalid label. No row is ever
    silently dropped or modified, and no partial file is ever scored.
    """
    max_bytes = config.UPLOAD_MAX_BYTES if max_bytes is None else max_bytes
    max_rows = config.UPLOAD_MAX_ROWS if max_rows is None else max_rows

    content_hash = compute_content_hash(file_bytes)

    if len(file_bytes) == 0:
        raise UploadValidationError("The uploaded file is empty (0 bytes).")
    if len(file_bytes) > max_bytes:
        raise UploadValidationError(
            f"The uploaded file is {len(file_bytes)} bytes, exceeding the "
            f"{max_bytes}-byte upload limit."
        )

    try:
        df = _read_csv(file_bytes)
    except Exception as exc:  # includes UnicodeDecodeError and CSV parse errors
        raise UploadValidationError(f"Could not parse the file as a UTF-8 CSV: {exc}") from exc

    if df.empty:
        raise UploadValidationError("The CSV contains no data rows (header only or empty).")
    if len(df) > max_rows:
        raise UploadValidationError(
            f"The uploaded CSV has {len(df)} rows, exceeding the {max_rows}-row upload limit."
        )

    column_errors = _validate_columns(df.columns)
    if column_errors:
        raise UploadValidationError("Invalid CSV columns:\n- " + "\n- ".join(column_errors))

    # Row-level validation — one invalid row rejects the whole file.
    row_errors: List[tuple] = []
    for idx, row in df.iterrows():
        raw = {col: row[col] for col in INFERENCE_COLUMNS}
        errors = _validate_row(raw)
        if errors:
            row_errors.append((int(idx), errors))

    label_errors = _validate_labels(df["isFraud"]) if "isFraud" in df.columns else []

    if row_errors or label_errors:
        lines = ["Upload rejected — the file contains invalid rows and nothing was scored."]
        for row_number, errors in row_errors[:10]:
            lines.append(f"  row {row_number}: " + "; ".join(errors))
        if len(row_errors) > 10:
            lines.append(f"  ... and {len(row_errors) - 10} more invalid row(s).")
        for message in label_errors[:5]:
            lines.append("  " + message)
        raise UploadValidationError("\n".join(lines))

    duplicate_row_count = int(df.duplicated().sum())

    # Separate the label BEFORE scoring; keep ONLY the 7 model fields.
    raw_df = df[list(INFERENCE_COLUMNS)].copy().reset_index(drop=True)
    labels: Optional[pd.Series] = None
    if "isFraud" in df.columns:
        labels = df["isFraud"].map(lambda v: int(float(v))).astype("int64").reset_index(drop=True)

    fingerprints = [fingerprint_row(dict(raw_df.iloc[i])) for i in range(len(raw_df))]
    present_optional = sorted(c for c in KNOWN_OPTIONAL_COLUMNS if c in df.columns)
    dataset_hash = compute_dataset_hash(fingerprints, present_optional)
    return UploadResult(
        content_hash=content_hash,
        raw_df=raw_df,
        labels=labels,
        present_optional_columns=present_optional,
        row_count=len(raw_df),
        duplicate_row_count=duplicate_row_count,
        row_fingerprints=fingerprints,
        dataset_hash=dataset_hash,
    )


# ---------------------------------------------------------------------------
# Row fingerprint — exact-record identity (deterministic canonical contract)
# ---------------------------------------------------------------------------
def _canonical_numeric(value: Any) -> str:
    """Fixed 2-decimal serialization — explicitly defined, locale-independent."""
    return f"{float(value):.2f}"


def fingerprint_row(record: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical representation of the 7 raw inference fields.

    Locked canonicalization:
      - field order: type, amount, oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest, nameDest
      - type: strip whitespace + ASCII uppercase
      - nameDest: strip whitespace, case preserved
      - numerics: float(v) then fixed 2-decimal formatting
      - delimiter: ASCII unit separator (\\x1f), encoding UTF-8
      - hash: SHA-256 hex digest

    Callers must only pass validated rows (finite numerics). This function
    never mutates any dataframe — it builds a separate canonical string.
    """
    parts = [
        str(record["type"]).strip().upper(),
        _canonical_numeric(record["amount"]),
        _canonical_numeric(record["oldbalanceOrg"]),
        _canonical_numeric(record["newbalanceOrig"]),
        _canonical_numeric(record["oldbalanceDest"]),
        _canonical_numeric(record["newbalanceDest"]),
        str(record["nameDest"]).strip(),
    ]
    payload = _FIELD_DELIMITER.join(parts)
    return hashlib.sha256(payload.encode(_FINGERPRINT_ENCODING)).hexdigest()


# ---------------------------------------------------------------------------
# Dataset hash — ORDER-INDEPENDENT dataset identity
# ---------------------------------------------------------------------------
def compute_dataset_hash(row_fingerprints: Sequence[str], schema_columns: Sequence[str]) -> str:
    """Deterministic, order-independent identity of an uploaded dataset.

    Sorts the row fingerprints and the schema column list before hashing, so
    [A, B, C] and [C, A, B] produce the SAME dataset hash. The actual dataframe
    row order is never touched — sorting applies only to the hash inputs.
    """
    sorted_fps = sorted(row_fingerprints)
    sorted_cols = sorted(str(c) for c in schema_columns)
    payload = (
        "v1"
        + _FIELD_DELIMITER
        + _FIELD_DELIMITER.join(sorted_fps)
        + _FIELD_DELIMITER
        + _FIELD_DELIMITER.join(sorted_cols)
    )
    return hashlib.sha256(payload.encode(_FINGERPRINT_ENCODING)).hexdigest()


# ---------------------------------------------------------------------------
# Exact-overlap vs the original model-development dataset
# ---------------------------------------------------------------------------
_ref_fingerprints: Optional[Set[str]] = None


def original_dataset_fingerprints() -> Set[str]:
    """Fingerprints of all records in Fraud_Analysis_Dataset.csv.

    Computed lazily once and reused across uploads (read-only; the CSV is
    never modified). Used ONLY to report exact-record overlap.
    """
    global _ref_fingerprints
    if _ref_fingerprints is None:
        df = pd.read_csv(config.INFERENCE_REFERENCE_CSV)
        fingerprints: Set[str] = set()
        for _, row in df.iterrows():
            raw = {col: row[col] for col in INFERENCE_COLUMNS}
            fingerprints.add(fingerprint_row(raw))
        _ref_fingerprints = fingerprints
    return set(_ref_fingerprints)


def compute_overlap(
    upload_fingerprints: Iterable[str], reference_fingerprints: Iterable[str]
) -> int:
    """Number of distinct uploaded fingerprints that exactly match the reference."""
    reference = set(reference_fingerprints)
    return sum(1 for fp in set(upload_fingerprints) if fp in reference)


def overlap_message(overlap: int, total: int) -> str | None:
    """Neutral 'Uploaded Inference Data' wording. NOT a novelty proof."""
    if overlap > 0:
        message = (
            f"{overlap} of {total} uploaded records exactly match records in the "
            "original dataset."
        )
    else:
        message = "No exact duplicate records found in the original dataset."
    return (
        message
        + " Exact hash matching detects exact duplicate records only. "
        "It does not prove statistical/distributional novelty."
    )


# ---------------------------------------------------------------------------
# Scoring — canonical frozen path ONLY (FraudDetector.predict_raw)
# ---------------------------------------------------------------------------
def score_dataframe(
    raw_df: pd.DataFrame,
    detector: Any,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Score every uploaded row in the uploaded order through predict_raw().

    `detector` is anything exposing `predict_raw(raw) -> dict` (typically
    src.inference.FraudDetector). `progress(done, total)` is called per row so
    the UI can show a progress indicator for larger uploads.
    Purely additive: predict_raw semantics are never modified by this module.
    """
    total = len(raw_df)
    results: List[Dict[str, Any]] = []
    for i in range(total):
        if progress is not None:
            progress(i, total)
        record = raw_df.iloc[i]
        # ONLY the 7 required raw fields ever reach the detector.
        raw = {col: record[col] for col in INFERENCE_COLUMNS}
        try:
            result = detector.predict_raw(raw)
        except Exception as exc:  # surfaced per row; validation should prevent this
            results.append({"row_index": i, "result": None, "error": str(exc)})
        else:
            results.append({"row_index": i, "result": result, "error": None})
    if progress is not None:
        progress(total, total)
    return results


def ensure_scored(
    session: MutableMapping,
    raw_df: pd.DataFrame,
    dataset_hash: str,
    detector: Any,
    progress: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """Score a dataset at most once per session (rerun/rescore guard).

    `session` is any mapping with Streamlit-compatible keys
    (`inference_scored_dataset_hash`, `inference_results`), e.g.
    `st.session_state`. Returns True only when scoring was actually executed.
    """
    if (
        session.get("inference_scored_dataset_hash") == dataset_hash
        and session.get("inference_results") is not None
    ):
        return False
    results = score_dataframe(raw_df, detector, progress=progress)
    session["inference_results"] = results
    session["inference_scored_dataset_hash"] = dataset_hash
    return True


# ---------------------------------------------------------------------------
# Labeled evaluation — isFraud used ONLY after prediction
# ---------------------------------------------------------------------------
def evaluate_predictions(
    predicted_classes: Iterable[Any], actual_labels: Iterable[Any]
) -> Dict[str, Any]:
    """Confusion metrics computed strictly AFTER model prediction.

    Returns accuracy/precision/recall/F1 (None when the denominator is zero),
    plus TP/TN/FP/FN and sample_count. Never re-scored; never fed to the model.
    """
    preds = [int(p) for p in predicted_classes]
    actual = [int(a) for a in actual_labels]
    if len(preds) != len(actual):
        raise ValueError("predicted_classes and actual_labels must have equal length.")
    tp = sum(p == 1 and a == 1 for p, a in zip(preds, actual))
    fp = sum(p == 1 and a == 0 for p, a in zip(preds, actual))
    tn = sum(p == 0 and a == 0 for p, a in zip(preds, actual))
    fn = sum(p == 0 and a == 1 for p, a in zip(preds, actual))
    total = len(preds)

    def _safe_div(numerator: float, denominator: float) -> Optional[float]:
        return round(numerator / denominator, 6) if denominator else None

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        _safe_div(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    return {
        "sample_count": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Optional prediction logging (existing logging_utils, guard helpers)
# ---------------------------------------------------------------------------
def log_results(
    results: Sequence[Mapping[str, Any]],
    raw_df: pd.DataFrame,
    log_fn: Callable[..., Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Log scored results via a per-row logging callable.

    `log_fn(result, transaction_reference=...)` is typically
    `logging_utils.log_prediction` (MySQL-first / JSONL fallback). Rows that
    failed scoring are skipped. Returns the status dicts for every logged row.
    """
    statuses: List[Dict[str, Any]] = []
    for entry in results:
        if entry.get("error") is not None or entry.get("result") is None:
            continue
        reference = str(raw_df.iloc[int(entry["row_index"])]["nameDest"])
        statuses.append(log_fn(entry["result"], transaction_reference=reference))
    return statuses


def was_logged(session: MutableMapping, dataset_hash: str) -> bool:
    """True when the current upload's predictions were already logged once."""
    return session.get("inference_logged_dataset_hash") == dataset_hash


# ---------------------------------------------------------------------------
# Inference-batch persistence (scored uploads become inference history)
# ---------------------------------------------------------------------------
def persist_scored_batch(
    content_hash: str,
    dataset_hash: str,
    label: str,
    raw_df: pd.DataFrame,
    results: List[Dict[str, Any]],
    labels: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Persist a scored uploaded batch as inference history.

    Returns {"status": "saved"|"reused"|"unavailable", "batch_id", "message"}:

    - MySQL unavailable  -> status "unavailable"; NOTHING is written and the
      batch is NOT converted into a historical batch (never a silent fallback).
    - content_hash match -> the existing batch is reused (no duplicate rows).
    - dataset_hash match (same records, different order) -> existing batch reused.
    - otherwise a new batch is created and every scored row is persisted with
      batch_id and (when provided) actual_label — ground truth associated
      AFTER prediction; isFraud never reaches the model.
    """
    if not database.is_available():
        return {
            "status": "unavailable",
            "batch_id": None,
            "message": (
                "Prediction completed, but inference history could not be saved "
                "because MySQL is unavailable. Start MySQL and retry."
            ),
        }

    scored = [r for r in results if r.get("result") is not None]
    if not scored:
        return {
            "status": "unavailable",
            "batch_id": None,
            "message": "No scored rows to persist.",
        }

    existing = database.get_batch_by_content_hash(
        content_hash, source=config.BATCH_SOURCE_UPLOADED
    )
    if existing:
        return {
            "status": "reused",
            "batch_id": existing["batch_id"],
            "message": (
                f"This dataset was already scored as inference batch "
                f"#{existing['batch_id']}."
            ),
        }

    existing = database.get_batch_by_dataset_hash(
        dataset_hash, source=config.BATCH_SOURCE_UPLOADED
    )
    if existing:
        return {
            "status": "reused",
            "batch_id": existing["batch_id"],
            "message": (
                f"These records were already scored as inference batch "
                f"#{existing['batch_id']} (different row order) — reusing it."
            ),
        }

    probabilities = [float(r["result"]["fraud_probability"]) for r in scored]
    fraud_count = sum(int(r["result"]["prediction"]) for r in scored)
    avg = (sum(probabilities) / len(probabilities)) if probabilities else None

    batch_id = database.create_inference_batch(
        source=config.BATCH_SOURCE_UPLOADED,
        label=label,
        content_hash=content_hash,
        dataset_hash=dataset_hash,
        row_count=len(scored),
        fraud_count=fraud_count,
        avg_fraud_probability=avg,
        has_ground_truth=labels is not None,
    )

    entries = []
    for r in scored:
        i = int(r["row_index"])
        res = r["result"]
        entry = {
            "timestamp": datetime.datetime.now(),
            "transaction_reference": str(raw_df.iloc[i]["nameDest"]),
            "raw_transaction": res.get("raw_transaction"),
            "engineered_features": res.get("engineered_features"),
            "fraud_probability": float(res["fraud_probability"]),
            "predicted_class": int(res["prediction"]),
            "threshold": float(res["threshold"]),
            "model_version": res.get("model_version"),
            "latency_ms": (
                float(res["latency_ms"]) if res.get("latency_ms") is not None else None
            ),
            "batch_id": batch_id,
            "actual_label": None,
        }
        if labels is not None:
            value = labels.iloc[i] if hasattr(labels, "iloc") else labels[i]
            entry["actual_label"] = int(value)
        entries.append(entry)

    database.insert_batch_predictions(entries)
    return {
        "status": "saved",
        "batch_id": batch_id,
        "message": f"Inference batch #{batch_id} saved ({len(entries)} predictions).",
    }


def load_batch_frame(batch_id: int) -> pd.DataFrame:
    """Engineered model-input frame for a persisted batch (Drift 'current')."""
    raws = database.fetch_batch_raw_transactions(batch_id)
    if not raws:
        return pd.DataFrame(columns=list(INFERENCE_COLUMNS))
    return features.engineer_features(pd.DataFrame(raws))


def meets_min_rows(row_count: int):
    """Drift availability gate (prototype heuristic minimum)."""
    minimum = config.MIN_DRIFT_ROWS
    if int(row_count) >= minimum:
        return True, ""
    return False, (
        f"Latest batch has {row_count} rows — at least {minimum} are required "
        "for a meaningful drift comparison."
    )