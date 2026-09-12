"""MySQL access layer (SQLAlchemy + PyMySQL — both already installed).

Credentials come exclusively from environment variables (DB_HOST, DB_PORT,
DB_NAME, DB_USER, DB_PASSWORD) — never hardcoded.

Tables (kept deliberately simple):
  - transactions      : raw transaction data source for the application
  - prediction_logs   : one row per prediction made by the application
  - model_monitoring  : periodic monitoring snapshots

All functions degrade gracefully: if MySQL is unavailable they raise
`DatabaseUnavailableError` so callers (e.g. logging_utils) can fall back to
JSONL explicitly instead of failing silently.
"""
from __future__ import annotations

import datetime
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import SQLAlchemyError

from src import config


class DatabaseUnavailableError(RuntimeError):
    """Raised when MySQL cannot be reached or a query fails."""


def get_database_url() -> str:
    return (
        f"mysql+pymysql://{config.DB_USER}:{config.DB_PASSWORD}"
        f"@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
    )


_engine = None


def get_engine():
    """Lazy SQLAlchemy engine for the fraud_detection database."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return _engine


def is_available() -> bool:
    """True if a connection to MySQL can be established right now."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


@contextmanager
def _connection():
    try:
        with get_engine().begin() as conn:
            yield conn
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"MySQL unavailable or failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
metadata = MetaData()

transactions = Table(
    "transactions",
    metadata,
    Column("transaction_id", Integer, primary_key=True, autoincrement=True),
    # Original dataset fields
    Column("step", Integer, nullable=True),
    Column("type", String(16), nullable=False),
    Column("amount", Float, nullable=False),
    Column("nameOrig", String(64), nullable=True),
    Column("oldbalanceOrg", Float, nullable=False),
    Column("newbalanceOrig", Float, nullable=False),
    Column("nameDest", String(64), nullable=False),
    Column("oldbalanceDest", Float, nullable=False),
    Column("newbalanceDest", Float, nullable=False),
    # Nullable: future application transactions have no known label
    Column("isFraud", Integer, nullable=True),
    Column("ingested_at", DateTime, nullable=False),
)

prediction_logs = Table(
    "prediction_logs",
    metadata,
    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("transaction_reference", String(64), nullable=True),
    # Full audit trail for each prediction (JSON): the raw transaction as
    # entered by the user and the engineered features fed to the model.
    Column("raw_transaction", JSON, nullable=True),
    Column("engineered_features", JSON, nullable=True),
    Column("fraud_probability", Float, nullable=False),
    Column("predicted_class", Integer, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("actual_label", Integer, nullable=True),
    Column("latency_ms", Float, nullable=True),
)

model_monitoring = Table(
    "model_monitoring",
    metadata,
    Column("monitoring_id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("sample_count", Integer, nullable=False),
    Column("fraud_prediction_count", Integer, nullable=False),
    Column("fraud_percentage", Float, nullable=False),
    Column("average_probability", Float, nullable=False),
    Column("precision", Float, nullable=True),
    Column("recall", Float, nullable=True),
    Column("f1", Float, nullable=True),
    Column("tp", Integer, nullable=True),
    Column("fp", Integer, nullable=True),
    Column("tn", Integer, nullable=True),
    Column("fn", Integer, nullable=True),
)


def create_tables() -> None:
    """Create the three tables if they do not exist."""
    with _connection() as conn:
        metadata.create_all(get_engine())


def init_database() -> None:
    """Initialize the application database safely and idempotently.

    1. Connect to the MySQL server-level endpoint (no database selected).
    2. CREATE DATABASE IF NOT EXISTS bia_fraud_detection
    3. Reconnect to the application database.
    4. Ensure transactions / prediction_logs / model_monitoring exist
       (CREATE TABLE IF NOT EXISTS via create_tables).

    Never drops tables and never deletes rows, so it is safe to call
    repeatedly (application startup, loader, tests). Raises
    DatabaseUnavailableError when MySQL cannot be reached.
    """
    global _engine

    server_url = (
        f"mysql+pymysql://{config.DB_USER}:{config.DB_PASSWORD}"
        f"@{config.DB_HOST}:{config.DB_PORT}/"
    )
    server_engine = create_engine(server_url, pool_pre_ping=True)
    try:
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}`"))
        server_engine.dispose()
    except SQLAlchemyError as exc:
        server_engine.dispose()
        raise DatabaseUnavailableError(f"MySQL unavailable or failed: {exc}") from exc

    # A previously created engine may still hold a pool from before the
    # database existed; dispose it so the next get_engine() connects fresh.
    if _engine is not None:
        _engine.dispose()
        _engine = None

    create_tables()


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------
def insert_transaction(tx: Dict[str, Any]) -> None:
    """Insert one raw transaction row. isFraud may be None."""
    payload = {col.name: tx.get(col.name) for col in transactions.columns}
    payload.setdefault("ingested_at", datetime.datetime.now())
    with _connection() as conn:
        conn.execute(transactions.insert().values(**payload))


def insert_transactions(rows: Iterable[Dict[str, Any]], batch_size: int = 1000) -> int:
    """Bulk insert transactions (e.g. from CSV load). Returns inserted count."""
    now = datetime.datetime.now()
    count = 0
    batch: List[Dict[str, Any]] = []
    with _connection() as conn:
        for row in rows:
            payload = {col.name: row.get(col.name) for col in transactions.columns}
            payload["ingested_at"] = row.get("ingested_at", now)
            batch.append(payload)
            count += 1
            if len(batch) >= batch_size:
                conn.execute(transactions.insert(), batch)
                batch = []
        if batch:
            conn.execute(transactions.insert(), batch)
    return count


def fetch_recent_transactions(limit: int = 50) -> List[Dict[str, Any]]:
    """Most recent transaction rows for batch scoring (returned chronological)."""
    with _connection() as conn:
        result = conn.execute(
            text("SELECT * FROM transactions ORDER BY transaction_id DESC LIMIT :lim"),
            {"lim": int(limit)},
        )
        rows = [dict(r._mapping) for r in result]
    rows.reverse()  # chronological order
    return rows


def insert_prediction_log(entry: Dict[str, Any]) -> None:
    payload = {col.name: entry.get(col.name) for col in prediction_logs.columns}
    payload.setdefault("timestamp", datetime.datetime.now())
    with _connection() as conn:
        conn.execute(prediction_logs.insert().values(**payload))


def fetch_prediction_logs(limit: int = 500) -> List[Dict[str, Any]]:
    with _connection() as conn:
        result = conn.execute(
            text("SELECT * FROM prediction_logs ORDER BY prediction_id DESC LIMIT :lim"),
            {"lim": int(limit)},
        )
        return [dict(r._mapping) for r in result]


def insert_monitoring_record(record: Dict[str, Any]) -> None:
    payload = {col.name: record.get(col.name) for col in model_monitoring.columns}
    payload.setdefault("timestamp", datetime.datetime.now())
    with _connection() as conn:
        conn.execute(model_monitoring.insert().values(**payload))


def table_exists(name: str) -> bool:
    try:
        return inspect(get_engine()).has_table(name)
    except SQLAlchemyError:
        return False

