"""Presentation-only helpers for the Streamlit interface.

This module contains NO business logic: only user-facing terminology, value
formatting and one small CSS block. It is imported by ``app.py`` exclusively —
nothing under ``src/`` imports it, and it never touches the model, the
database, monitoring or drift calculations.

Terminology policy (single source of truth — do not duplicate these strings
inline in the pages, or the wording will drift apart again):

    Transactions Analyzed            count of scored/scored rows
    Fraudulent Transactions Detected count of rows predicted as fraud
    Fraud Detection Rate             share of analyzed rows detected as fraud
    Average Fraud Probability        mean fraud probability

An individual prediction is reported only as ``FRAUD DETECTED`` or
``NO FRAUD DETECTED``.
"""
from __future__ import annotations

import datetime as _datetime
import logging
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Application identity
# ---------------------------------------------------------------------------
APP_TITLE = "Fraud Detection Analytics"
APP_SUBTITLE = "Machine-learning fraud screening for financial transactions"

# ---------------------------------------------------------------------------
# Primary KPI terminology — used everywhere, never varied
# ---------------------------------------------------------------------------
KPI_TRANSACTIONS = "Transactions Analyzed"
KPI_FRAUD = "Fraudulent Transactions Detected"
KPI_RATE = "Fraud Detection Rate"
KPI_AVG_PROB = "Average Fraud Probability"

# Individual prediction outcomes
PREDICTION_FRAUD = "FRAUD DETECTED"
PREDICTION_LEGITIMATE = "NO FRAUD DETECTED"

# Supporting outcome words (per-record detail columns)
OUTCOME_FRAUD = "Fraud"
OUTCOME_LEGITIMATE = "Legitimate"
NOT_AVAILABLE = "—"

# Model-performance keys (evaluation against supplied labels)
PERF_ACCURACY = "Accuracy"
PERF_PRECISION = "Precision"
PERF_RECALL = "Recall"
PERF_F1 = "F1 Score"

TECHNICAL_DETAILS_TITLE = "Technical Details"

# Help text / tooltips for interpretation-heavy metrics
HELP_PRECISION = (
    "Of the transactions the model flagged as fraud, how many were actually "
    "fraudulent."
)
HELP_RECALL = (
    "Of the fraudulent transactions in the labeled data, how many the model "
    "successfully detected."
)
HELP_F1 = "Balance between precision and recall (harmonic mean)."
HELP_ACCURACY = "Share of labeled transactions the model classified correctly."
HELP_FRAUD_RATE = (
    "Share of analyzed transactions the model flagged as fraud."
)
HELP_AVG_PROB = (
    "Mean fraud probability across analyzed transactions. The model's decision "
    "threshold is applied per transaction, not to this average."
)


# ---------------------------------------------------------------------------
# Label maps: internal/technical name -> user-facing name
# ---------------------------------------------------------------------------
FIELD_LABELS: Dict[str, str] = {
    "type": "Transaction Type",
    "amount": "Transaction Amount",
    "nameOrig": "Sender Account",
    "nameDest": "Recipient Account",
    "oldbalanceOrg": "Sender Balance Before",
    "newbalanceOrig": "Sender Balance After",
    "oldbalanceDest": "Recipient Balance Before",
    "newbalanceDest": "Recipient Balance After",
    "log_amount": "Log Transaction Amount",
    "errorBalanceOrig": "Sender Balance Difference",
    "errorBalanceDest": "Recipient Balance Difference",
    "isDestMerchant": "Recipient Is Merchant",
    "isFraud": "Actual Fraud Status",
    "step": "Time Step (hours)",
    "transaction_id": "Transaction Identifier",
}

# Drift table labels (model features compared against the reference dataset)
FEATURE_LABELS: Dict[str, str] = {
    "type": "Transaction Type",
    "amount": "Transaction Amount",
    "oldbalanceOrg": "Sender Balance Before",
    "newbalanceOrig": "Sender Balance After",
    "oldbalanceDest": "Recipient Balance Before",
    "newbalanceDest": "Recipient Balance After",
    "log_amount": "Log Transaction Amount",
    "errorBalanceOrig": "Sender Balance Difference",
    "errorBalanceDest": "Recipient Balance Difference",
    "isDestMerchant": "Recipient Is Merchant",
}

SOURCE_LABELS: Dict[str, str] = {
    "uploaded_csv": "Uploaded CSV",
    "fraud_check": "Fraud Check",
    "historical_replay": "Historical Replay",
    "legacy": "Legacy Records",
}

DRIFT_STATUS_LABELS: Dict[str, str] = {
    "OK": "No Significant Drift",
    "WARNING": "Possible Drift",
    "DRIFT": "Significant Drift",
}

TRANSACTION_TYPE_LABELS: Dict[str, str] = {
    "CASH_IN": "Cash In",
    "CASH_OUT": "Cash Out",
    "DEBIT": "Debit",
    "PAYMENT": "Payment",
    "TRANSFER": "Transfer",
}


def pretty_source(value: Any) -> str:
    """Inference-batch source id -> readable Data Source name."""
    if value is None:
        return NOT_AVAILABLE
    text = str(value)
    return SOURCE_LABELS.get(text, text.replace("_", " ").title())


def pretty_feature(value: Any) -> str:
    """Model/engineered feature name -> readable feature label."""
    if value is None:
        return NOT_AVAILABLE
    text = str(value)
    return FEATURE_LABELS.get(text, FIELD_LABELS.get(text, text.replace("_", " ")))


def pretty_status(value: Any) -> str:
    """Drift status code -> readable status sentence."""
    if value is None:
        return NOT_AVAILABLE
    text = str(value).upper()
    return DRIFT_STATUS_LABELS.get(text, str(value).title())


def pretty_transaction_type(value: Any) -> str:
    """Raw transaction type -> readable label (the underlying value is unchanged)."""
    if value is None:
        return NOT_AVAILABLE
    text = str(value).upper()
    return TRANSACTION_TYPE_LABELS.get(text, text.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Value formatting (presentation only — stored values are never modified)
# ---------------------------------------------------------------------------
def fmt_amount(value: Any) -> str:
    """``581421.854321`` -> ``581,421.85`` (no currency symbol: the source
    data is synthetic and currency-agnostic)."""
    if value is None:
        return NOT_AVAILABLE
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_count(value: Any) -> str:
    """``11142`` -> ``11,142``."""
    if value is None:
        return NOT_AVAILABLE
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_percent(ratio: Any, decimals: int = 2) -> str:
    """``0.9887`` -> ``98.87%`` (a 0–1 ratio)."""
    if ratio is None:
        return NOT_AVAILABLE
    try:
        return f"{float(ratio):.{decimals}%}"
    except (TypeError, ValueError):
        return str(ratio)


def fmt_percent_value(value: Any, decimals: int = 2) -> str:
    """``98.87`` -> ``98.87%`` (a value already expressed in percent)."""
    if value is None:
        return NOT_AVAILABLE
    try:
        return f"{float(value):,.{decimals}f}%"
    except (TypeError, ValueError):
        return str(value)


def fmt_datetime(value: Any) -> str:
    """``2026-09-16 19:20`` -> ``16 Sep 2026, 7:20 PM`` (locale-independent)."""
    if value is None:
        return NOT_AVAILABLE
    if not isinstance(value, (_datetime.datetime, _datetime.date)):
        return str(value)
    if not isinstance(value, _datetime.datetime):
        return f"{value:%d %b %Y}"
    hour12 = value.hour % 12 or 12
    meridiem = "AM" if value.hour < 12 else "PM"
    return f"{value:%d %b %Y}, {hour12}:{value:%M} {meridiem}"


# ---------------------------------------------------------------------------
# Shared UI building blocks
# ---------------------------------------------------------------------------
_THEME_CSS = """
<style>
  /* Restrained typography and spacing only — no colours, gradients,
     animations or external fonts, so light and dark themes both stay legible. */
  .block-container { padding-top: 2.4rem; padding-bottom: 3rem; }
  h1 { font-size: 1.85rem; font-weight: 650; letter-spacing: -0.01em;
       margin-bottom: 0.1rem; padding-bottom: 0; }
  h2 { font-size: 1.22rem; font-weight: 600; margin-top: 0.3rem; }
  h3 { font-size: 1.03rem; font-weight: 600; }
  [data-testid="stCaptionContainer"] p { font-size: 0.83rem; line-height: 1.5; }
  [data-testid="stMetricLabel"] p { font-size: 0.9rem; font-weight: 500;
       white-space: normal; overflow: visible; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; }
  [data-testid="stExpander"] summary { font-size: 0.88rem; font-weight: 500; }
  [data-testid="stSidebar"] h2 { font-size: 1.05rem; }
  hr { margin: 1.05rem 0; opacity: 0.22; }
</style>
"""


def inject_theme() -> None:
    """Apply the small presentation stylesheet once per rerun."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    """Consistent page header: title, one-line explanation, divider."""
    st.title(title)
    st.caption(subtitle)
    st.divider()


def section(title: str, description: Optional[str] = None) -> None:
    """Consistent section heading, optionally with an explanatory line."""
    st.subheader(title)
    if description:
        st.caption(description)


def empty_state(message: str, hint: Optional[str] = None) -> None:
    """Informative empty state: what is missing and what to do next."""
    st.info(message if not hint else f"{message}\n\n{hint}")


def technical_details(rows: Dict[str, Any], title: str = TECHNICAL_DETAILS_TITLE,
                      expanded: bool = False, note: Optional[str] = None) -> None:
    """Collapsed-by-default expander holding internal/technical values.

    This is the only place where raw identifiers and hashes are shown, so the
    main interface never reads like a developer console.
    """
    with st.expander(title, expanded=expanded):
        if note:
            st.caption(note)
        frame = pd.DataFrame(
            [{"Field": str(k), "Value": "" if v is None else str(v)}
             for k, v in rows.items()]
        )
        st.dataframe(frame, hide_index=True, width="stretch")


def technical_error(context: str, exc: BaseException) -> None:
    """Show an actionable message while retaining exception details in server logs."""
    logging.getLogger(__name__).error(
        context, exc_info=(type(exc), exc, exc.__traceback__)
    )
    st.error(context)


def metric_row(items: Iterable[Dict[str, Any]]) -> None:
    """Render a row of KPIs with a consistent column count and formatting."""
    items = list(items)
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        column.metric(
            item["label"],
            item["value"],
            help=item.get("help"),
            border=True,
        )


def humanise_fields(message: str) -> str:
    """Replace quoted internal field names inside a message with readable labels.

    Validation messages are produced by src/features.py and
    src/upload_inference.py and quote the raw column names. This only rewrites
    those quoted names for display; the message text itself is untouched.
    """
    if not message:
        return message
    text = str(message)
    for raw, label in sorted(FIELD_LABELS.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(f"'{raw}'", f"'{label}'")
    return text
