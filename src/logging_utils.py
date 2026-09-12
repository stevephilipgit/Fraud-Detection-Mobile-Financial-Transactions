"""Prediction logging with a MySQL-first, JSONL-fallback strategy.

Every prediction is logged. Preferred destination: MySQL `prediction_logs`.
If MySQL is unavailable, the entry is appended to a local JSONL file — and
the fallback is NEVER silent: the returned status tells the application (and
therefore the Streamlit UI) exactly where the prediction was logged.
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Dict

from src import config, database

# Status values returned in the log result
STATUS_MYSQL = "mysql"
STATUS_JSONL = "jsonl"
STATUS_FAILED = "failed"


def _mysql_message() -> str:
    return "MySQL saved successfully"


def _fallback_message(path: str) -> str:
    return f"MySQL unavailable — saved to local JSONL fallback ({path})"


def log_prediction(
    result: Dict[str, Any],
    transaction_reference: str = None,
    actual_label: int = None,
) -> Dict[str, Any]:
    """Log one prediction result (as returned by inference.predict_transaction).

    Returns a dict describing the logging outcome:

        {
            'status': 'mysql' | 'jsonl' | 'failed',
            'message': human-readable status string,
            'destination': mysql table name or JSONL file path,
        }
    """
    timestamp = datetime.datetime.now()

    entry = {
        "timestamp": timestamp,
        "transaction_reference": transaction_reference,
        "raw_transaction": result.get("raw_transaction"),
        "engineered_features": result.get("engineered_features"),
        "fraud_probability": float(result["fraud_probability"]),
        "predicted_class": int(result["prediction"]),
        "threshold": float(result["threshold"]),
        "model_version": result["model_version"],
        "actual_label": actual_label,
        "latency_ms": float(result.get("latency_ms", 0.0)) if result.get("latency_ms") is not None else None,
    }

    # Preferred destination: MySQL
    if database.is_available():
        try:
            database.create_tables()
            database.insert_prediction_log(entry)
            return {
                "status": STATUS_MYSQL,
                "message": _mysql_message(),
                "destination": "mysql.prediction_logs",
            }
        except database.DatabaseUnavailableError as exc:
            # MySQL was reachable but the write failed — still fall back,
            # but say so explicitly.
            fallback = _write_jsonl(entry)
            fallback["message"] = (
                f"MySQL write failed ({exc.__class__.__name__}) — "
                f"saved to local JSONL fallback ({fallback['destination']})"
            )
            return fallback

    # MySQL unavailable -> local JSONL fallback (never silent)
    return _write_jsonl(entry)


def _write_jsonl(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Append one entry to the JSONL fallback file. Raises on disk failure so
    a failed log is still visible to the caller."""
    serializable = dict(entry)
    serializable["timestamp"] = entry["timestamp"].isoformat()
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.JSONL_FALLBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(serializable, default=str) + "\n")
    except OSError as exc:
        return {
            "status": STATUS_FAILED,
            "message": f"Logging FAILED — MySQL unavailable and JSONL write error: {exc}",
            "destination": str(config.JSONL_FALLBACK_PATH),
        }
    return {
        "status": STATUS_JSONL,
        "message": _fallback_message(str(config.JSONL_FALLBACK_PATH)),
        "destination": str(config.JSONL_FALLBACK_PATH),
    }


def read_jsonl_fallback() -> list:
    """Read back all entries from the JSONL fallback file (used by monitoring
    and by tests)."""
    if not config.JSONL_FALLBACK_PATH.exists():
        return []
    entries = []
    with open(config.JSONL_FALLBACK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
