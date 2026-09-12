"""Lightweight prototype monitoring (deliberately NOT enterprise MLOps).

Tracks, from prediction logs:
  - total predictions
  - fraud predictions / legitimate predictions
  - fraud prediction percentage
  - average fraud probability

When actual labels become available (prediction_logs.actual_label), also:
  - precision, recall, F1
  - TP, FP, TN, FN

Snapshots can be persisted to the MySQL `model_monitoring` table; if MySQL is
unavailable the snapshot is written to a local JSONL file and the caller is
told explicitly (never a silent fallback).
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Dict, Iterable, List, Optional

from src import config, database

MONITORING_FALLBACK_PATH = config.LOGS_DIR / "monitoring_fallback.jsonl"


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def compute_summary(logs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate prediction-log rows into a monitoring summary.

    Each log row needs at least 'fraud_probability' and 'predicted_class';
    'actual_label' (0/1 or None) is optional and enables confusion-matrix and
    precision/recall/F1 computation.
    """
    rows = list(logs)
    total = len(rows)
    probabilities = [float(r["fraud_probability"]) for r in rows]
    fraud_count = sum(int(r["predicted_class"]) == 1 for r in rows)

    summary: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(),
        "model_version": rows[0].get("model_version", config.MODEL_VERSION) if rows else config.MODEL_VERSION,
        "sample_count": total,
        "fraud_prediction_count": fraud_count,
        "legitimate_prediction_count": total - fraud_count,
        "fraud_percentage": round(_safe_div(fraud_count * 100.0, total) or 0.0, 4),
        "average_probability": round(sum(probabilities) / total, 6) if total else 0.0,
        "labels_available": False,
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": None,
        "fp": None,
        "tn": None,
        "fn": None,
    }

    labelled = [r for r in rows if r.get("actual_label") is not None]
    if labelled:
        tp = sum(int(r["predicted_class"]) == 1 and int(r["actual_label"]) == 1 for r in labelled)
        fp = sum(int(r["predicted_class"]) == 1 and int(r["actual_label"]) == 0 for r in labelled)
        tn = sum(int(r["predicted_class"]) == 0 and int(r["actual_label"]) == 0 for r in labelled)
        fn = sum(int(r["predicted_class"]) == 0 and int(r["actual_label"]) == 1 for r in labelled)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None

        summary.update(
            {
                "labels_available": True,
                "labelled_count": len(labelled),
                "precision": round(precision, 4) if precision is not None else None,
                "recall": round(recall, 4) if recall is not None else None,
                "f1": round(f1, 4) if f1 is not None else None,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    return summary


def _snapshot_to_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": summary["timestamp"],
        "model_version": summary["model_version"],
        "sample_count": summary["sample_count"],
        "fraud_prediction_count": summary["fraud_prediction_count"],
        "fraud_percentage": summary["fraud_percentage"],
        "average_probability": summary["average_probability"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
        "tp": summary["tp"],
        "fp": summary["fp"],
        "tn": summary["tn"],
        "fn": summary["fn"],
    }


def save_snapshot(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a monitoring snapshot. MySQL preferred, JSONL fallback —
    the destination is always reported back to the caller."""
    row = _snapshot_to_row(summary)
    if database.is_available():
        try:
            database.create_tables()
            database.insert_monitoring_record(row)
            return {"status": "mysql", "message": "Monitoring snapshot saved to MySQL model_monitoring"}
        except database.DatabaseUnavailableError as exc:
            fallback = _write_jsonl(row)
            fallback["message"] = (
                f"MySQL write failed ({exc.__class__.__name__}) — "
                f"saved to local JSONL fallback ({fallback['destination']})"
            )
            return fallback
    return _write_jsonl(row)


def _write_jsonl(row: Dict[str, Any]) -> Dict[str, Any]:
    serializable = dict(row)
    serializable["timestamp"] = row["timestamp"].isoformat()
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MONITORING_FALLBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(serializable, default=str) + "\n")
    except OSError as exc:
        return {
            "status": "failed",
            "message": f"Monitoring snapshot FAILED — JSONL write error: {exc}",
            "destination": str(MONITORING_FALLBACK_PATH),
        }
    return {
        "status": "jsonl",
        "message": f"MySQL unavailable — monitoring snapshot saved to local JSONL fallback ({MONITORING_FALLBACK_PATH})",
        "destination": str(MONITORING_FALLBACK_PATH),
    }


def collect_logs() -> List[Dict[str, Any]]:
    """Gather prediction logs from MySQL, falling back to the JSONL file."""
    if database.is_available():
        try:
            return database.fetch_prediction_logs(limit=500)
        except database.DatabaseUnavailableError:
            pass
    from src import logging_utils

    return logging_utils.read_jsonl_fallback()
