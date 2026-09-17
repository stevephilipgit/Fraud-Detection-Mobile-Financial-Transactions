"""Central configuration for the fraud detection application.

Paths, environment-variable names and prototype defaults live here so that no
other module hardcodes them. The saved notebook artifacts are the single
source of truth for the model; this module only points at them.
"""
import os
from pathlib import Path

# Optional local development convenience. Credentials still come from
# environment variables; this only loads them from an untracked .env file.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is listed for the app env
    load_dotenv = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

MODEL_PATH = ARTIFACTS_DIR / "fraud_model_xgboost.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
FEATURE_METADATA_PATH = ARTIFACTS_DIR / "feature_metadata.json"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

LOGS_DIR.mkdir(exist_ok=True)

# Local JSONL fallback destination for prediction logging (used only when
# MySQL is unavailable — the fallback is never silent, see logging_utils).
JSONL_FALLBACK_PATH = LOGS_DIR / "predictions_fallback.jsonl"

# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------
# The finalized notebook's export cell does not embed a model version, so the
# application records a single consistent version centrally here.
# If model_metadata.json ever contains a 'model_version' key it takes
# precedence (see inference.load_model_version).
MODEL_VERSION = "xgboost_v1"

# ---------------------------------------------------------------------------
# Raw transaction fields accepted by the application
# ---------------------------------------------------------------------------
VALID_TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

RAW_TRANSACTION_FIELDS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "nameDest",
]

NUMERIC_TRANSACTION_FIELDS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# ---------------------------------------------------------------------------
# Uploaded Inference Data — session-scoped CSV batch scoring (demo bounds)
# ---------------------------------------------------------------------------
# Demo-oriented upper bounds only. Larger uploads are simply rejected because
# the frozen inference path deliberately scores one transaction at a time.
UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
UPLOAD_MAX_ROWS = 5000               # 5,000 rows

# The original model-development dataset used by the exact-overlap check.
INFERENCE_REFERENCE_CSV = PROJECT_ROOT / "Fraud_Analysis_Dataset.csv"

# ---------------------------------------------------------------------------
# MySQL connection (via environment variables — never hardcoded credentials)
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "bia_fraud_detection")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ---------------------------------------------------------------------------
# Monitoring / drift prototype thresholds (documented heuristics, NOT
# universal industry standards)
# ---------------------------------------------------------------------------
PSI_OK = 0.10          # PSI < 0.10          -> no significant drift
PSI_WARNING = 0.25     # 0.10 <= PSI < 0.25  -> moderate drift, investigate
                       # PSI >= 0.25         -> significant drift
CATEGORICAL_DRIFT_THRESHOLD = 0.05  # max absolute share change per category

# ---------------------------------------------------------------------------
# Inference batches / Monitoring scope / Drift
# ---------------------------------------------------------------------------
BATCH_SOURCE_UPLOADED = "uploaded_csv"
BATCH_SOURCE_FRAUD_CHECK = "fraud_check"
BATCH_SOURCE_HISTORICAL_REPLAY = "historical_replay"
BATCH_SOURCE_LEGACY = "legacy"

# Sources shown in the default (production-style) Monitoring scope.
MONITORING_DEFAULT_SOURCES = (BATCH_SOURCE_UPLOADED, BATCH_SOURCE_FRAUD_CHECK)

# Minimum rows for a meaningful drift comparison (prototype heuristic).
MIN_DRIFT_ROWS = 30
