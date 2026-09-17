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
from typing import Any, Dict, Iterable, List, Optional

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
    func,
    inspect,
    select,
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
    # Inference batch this prediction belongs to (NULL = legacy/pre-redesign)
    Column("batch_id", Integer, nullable=True),
)

inference_batches = Table(
    "inference_batches",
    metadata,
    Column("batch_id", Integer, primary_key=True, autoincrement=True),
    Column("source", String(32), nullable=False),
    Column("label", String(255), nullable=True),
    Column("content_hash", String(64), nullable=True),
    Column("dataset_hash", String(64), nullable=True),
    Column("row_count", Integer, nullable=False),
    Column("fraud_count", Integer, nullable=False),
    Column("avg_fraud_probability", Float, nullable=True),
    Column("has_ground_truth", Integer, nullable=False),
    Column("created_at", DateTime, nullable=False),
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
    2. CREATE DATABASE IF NOT EXISTS fraud_detection
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
    ensure_schema_updates()


def ensure_schema_updates() -> None:
    """Additive, idempotent schema updates that create_all cannot perform.

    metadata.create_all() creates missing TABLES but never adds columns to
    existing ones, so prediction_logs.batch_id is added explicitly when
    missing. Never drops tables and never modifies existing rows — existing
    predictions keep batch_id = NULL and become legacy/unattributed records.
    """
    _ensure_column("prediction_logs", "batch_id", "INT NULL")


def _ensure_column(table_name: str, column_name: str, ddl: str) -> bool:
    """Add a column to an existing table if missing. Returns True if added."""
    engine = get_engine()
    existing = [c["name"] for c in inspect(engine).get_columns(table_name)]
    if column_name in existing:
        return False
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}"))
    return True


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
        # Execute-time parameters (NOT values(**payload)): the payload key
        # `fn` collides with the first positional parameter of SQLAlchemy's
        # generative wrapper, which raised TypeError before any INSERT.
        conn.execute(model_monitoring.insert(), [payload])


def table_exists(name: str) -> bool:
    try:
        return inspect(get_engine()).has_table(name)
    except SQLAlchemyError:
        return False


# ---------------------------------------------------------------------------
# Inference batches (persisted inference activity)
# ---------------------------------------------------------------------------
def get_batch_by_content_hash(
    content_hash: str, source: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    stmt = inference_batches.select().where(
        inference_batches.c.content_hash == content_hash
    )
    if source is not None:
        stmt = stmt.where(inference_batches.c.source == source)
    stmt = stmt.order_by(inference_batches.c.created_at.desc()).limit(1)
    with _connection() as conn:
        row = conn.execute(stmt).first()
    return dict(row._mapping) if row else None


def get_batch_by_dataset_hash(
    dataset_hash: str, source: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    stmt = inference_batches.select().where(
        inference_batches.c.dataset_hash == dataset_hash
    )
    if source is not None:
        stmt = stmt.where(inference_batches.c.source == source)
    stmt = stmt.order_by(inference_batches.c.created_at.desc()).limit(1)
    with _connection() as conn:
        row = conn.execute(stmt).first()
    return dict(row._mapping) if row else None


def create_inference_batch(
    source: str,
    label: Optional[str] = None,
    content_hash: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    row_count: int = 0,
    fraud_count: int = 0,
    avg_fraud_probability: Optional[float] = None,
    has_ground_truth: bool = False,
) -> int:
    """Create one inference-batch row and return its batch_id."""
    payload = {
        "source": source,
        "label": label,
        "content_hash": content_hash,
        "dataset_hash": dataset_hash,
        "row_count": int(row_count),
        "fraud_count": int(fraud_count),
        "avg_fraud_probability": avg_fraud_probability,
        "has_ground_truth": int(bool(has_ground_truth)),
        "created_at": datetime.datetime.now(),
    }
    with _connection() as conn:
        result = conn.execute(inference_batches.insert().values(**payload))
    return int(result.inserted_primary_key[0])


def insert_batch_predictions(entries: List[Dict[str, Any]]) -> int:
    """Persist scored predictions using execute-time parameters (safe for any
    column name, unlike values(**payload))."""
    entries = [dict(e) for e in entries]
    if not entries:
        return 0
    with _connection() as conn:
        conn.execute(prediction_logs.insert(), entries)
    return len(entries)


def refresh_batch_counts(batch_id: int) -> None:
    """Recompute row_count / fraud_count / average probability from prediction_logs."""
    stmt = select(
        prediction_logs.c.fraud_probability,
        prediction_logs.c.predicted_class,
    ).where(prediction_logs.c.batch_id == batch_id)
    with _connection() as conn:
        rows = conn.execute(stmt).fetchall()
        count = len(rows)
        fraud = sum(1 for r in rows if int(r[1]) == 1)
        avg = (sum(float(r[0]) for r in rows) / count) if count else None
        conn.execute(
            inference_batches.update()
            .where(inference_batches.c.batch_id == batch_id)
            .values(row_count=count, fraud_count=fraud, avg_fraud_probability=avg)
        )


def fetch_inference_batches(
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """All inference batches, newest first (optionally filtered by source)."""
    stmt = inference_batches.select().order_by(
        inference_batches.c.created_at.desc(), inference_batches.c.batch_id.desc()
    )
    if sources:
        stmt = stmt.where(inference_batches.c.source.in_(list(sources)))
    with _connection() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt)]


def fetch_latest_batch(source: Optional[str] = None) -> Optional[Dict[str, Any]]:
    stmt = inference_batches.select()
    if source is not None:
        stmt = stmt.where(inference_batches.c.source == source)
    stmt = stmt.order_by(
        inference_batches.c.created_at.desc(), inference_batches.c.batch_id.desc()
    ).limit(1)
    with _connection() as conn:
        row = conn.execute(stmt).first()
    return dict(row._mapping) if row else None


def fetch_batch_predictions(batch_id: int) -> List[Dict[str, Any]]:
    stmt = (
        prediction_logs.select()
        .where(prediction_logs.c.batch_id == batch_id)
        .order_by(prediction_logs.c.prediction_id)
    )
    with _connection() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt)]


def fetch_batch_raw_transactions(batch_id: int) -> List[Dict[str, Any]]:
    """Raw (pre-encoding) transaction dicts for one batch, in scoring order."""
    stmt = (
        select(prediction_logs.c.raw_transaction)
        .where(prediction_logs.c.batch_id == batch_id)
        .order_by(prediction_logs.c.prediction_id)
    )
    with _connection() as conn:
        rows = [dict(r._mapping)["raw_transaction"] for r in conn.execute(stmt)]
    return [r for r in rows if r]


def fetch_scoped_predictions(
    include_legacy: bool = False,
    sources: Optional[List[str]] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Prediction rows joined with their batch source.

    Default scope: real application inference (uploaded_csv, fraud_check).
    Legacy rows (batch_id IS NULL, pre-redesign) and historical replays are
    excluded unless explicitly requested.
    """
    if sources is None:
        sources = list(config.MONITORING_DEFAULT_SOURCES)
    j = prediction_logs.join(
        inference_batches,
        prediction_logs.c.batch_id == inference_batches.c.batch_id,
        isouter=True,  # keep legacy rows (batch_id NULL) when include_legacy=True
    )
    cols = list(prediction_logs.columns) + [
        inference_batches.c.source.label("batch_source"),
        inference_batches.c.label.label("batch_label"),
        inference_batches.c.created_at.label("batch_created_at"),
    ]
    stmt = (
        select(*cols)
        .select_from(j)
        .order_by(prediction_logs.c.prediction_id.desc())
        .limit(int(limit))
    )
    if not include_legacy:
        stmt = stmt.where(inference_batches.c.source.in_(sources))
    with _connection() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt)]


def get_or_create_fraud_check_batch() -> int:
    """One fraud-check batch per calendar day (avoids a batch per click)."""
    day_start = datetime.datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    stmt = (
        inference_batches.select()
        .where(inference_batches.c.source == config.BATCH_SOURCE_FRAUD_CHECK)
        .where(inference_batches.c.created_at >= day_start)
        .order_by(inference_batches.c.created_at.desc())
        .limit(1)
    )
    with _connection() as conn:
        row = conn.execute(stmt).first()
    if row:
        return int(row._mapping["batch_id"])
    return create_inference_batch(
        source=config.BATCH_SOURCE_FRAUD_CHECK,
        label=f"Fraud Check — {day_start:%Y-%m-%d}",
        row_count=0,
        fraud_count=0,
    )


# ---------------------------------------------------------------------------
# Inference-batch cleanup (batch-scoped deletes only)
# ---------------------------------------------------------------------------
def count_legacy_prediction_logs() -> int:
    """Prediction records without an inference batch (batch_id IS NULL)."""
    with _connection() as conn:
        return int(
            conn.execute(
                select(func.count())
                .select_from(prediction_logs)
                .where(prediction_logs.c.batch_id.is_(None))
            ).scalar()
        )


def delete_legacy_prediction_logs() -> Dict[str, Any]:
    """Delete ONLY prediction rows without an inference batch.

    Targets rows WHERE batch_id IS NULL exclusively — never inference
    batches and never the source transactions table.
    """
    with _connection() as conn:
        result = conn.execute(
            prediction_logs.delete().where(prediction_logs.c.batch_id.is_(None))
        )
        deleted = int(result.rowcount or 0)
    return {
        "deleted": deleted > 0,
        "rows_deleted": deleted,
        "message": f"Deleted {deleted} legacy prediction records.",
    }


def delete_inference_batch(batch_id: int) -> Dict[str, Any]:
    """Delete one inference batch and all its predictions in a single
    transaction (commit on success, rollback on any failure).

    Strictly batch-scoped: prediction_logs WHERE batch_id = :batch_id, then
    inference_batches WHERE batch_id = :batch_id. The transactions table and
    the source dataset are never touched.
    """
    with _connection() as conn:
        batch = conn.execute(
            inference_batches.select().where(
                inference_batches.c.batch_id == batch_id
            )
        ).first()
        if batch is None:
            return {
                "deleted": False,
                "batch_found": False,
                "prediction_rows_deleted": 0,
                "message": f"Inference batch #{batch_id} does not exist.",
            }
        result = conn.execute(
            prediction_logs.delete().where(prediction_logs.c.batch_id == batch_id)
        )
        pred_count = int(result.rowcount or 0)
        conn.execute(
            inference_batches.delete().where(inference_batches.c.batch_id == batch_id)
        )
    return {
        "deleted": True,
        "batch_found": True,
        "prediction_rows_deleted": pred_count,
        "message": f"Inference batch #{batch_id} deleted ({pred_count} predictions).",
    }

