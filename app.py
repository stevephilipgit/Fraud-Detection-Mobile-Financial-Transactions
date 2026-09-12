"""Streamlit UI for the fraud detection prototype.

Contains NO model training code. All predictions go through
src/inference.py (saved artifacts only):
    User -> Streamlit -> inference.py -> features.py
         -> preprocessor.transform() -> XGBoost.predict_proba()
         -> threshold -> decision -> log prediction -> display

Pages:
  - Fraud Check      : score one manually-entered transaction
  - Batch Scoring    : score recent transactions from MySQL
  - Monitoring/Drift : prediction-log stats + data-drift table

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable when Streamlit runs from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, database, drift, features, inference, logging_utils, monitoring  # noqa: E402

st.set_page_config(page_title="Fraud Detection Prototype", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading saved model artifacts…")
def load_detector():
    return inference.FraudDetector().load()


def _ensure_db_initialized() -> bool:
    """Attempt MySQL initialization once per Streamlit session.

    Never crashes the app: when MySQL is unavailable (or credentials are not
    configured yet) the app keeps the existing JSONL fallback behaviour.
    """
    if "db_ready" in st.session_state:
        return st.session_state["db_ready"]
    try:
        database.init_database()
        st.session_state["db_ready"] = True
    except database.DatabaseUnavailableError:
        st.session_state["db_ready"] = False
    return st.session_state["db_ready"]


def page_fraud_check() -> None:
    st.header("🛡️ Fraud Check")
    st.caption(
        "Enter the RAW transaction below. Engineered features "
        "(log_amount, errorBalanceOrig, errorBalanceDest, isDestMerchant) "
        "are calculated internally — never entered manually."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        tx_type = st.selectbox("Transaction type", config.VALID_TRANSACTION_TYPES, index=4)
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0, format="%.2f")
        name_dest = st.text_input("Destination account (nameDest)", value="C123456789")
    with col2:
        oldbalance_org = st.number_input(
            "Origin balance before (oldbalanceOrg)", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
        newbalance_orig = st.number_input(
            "Origin balance after (newbalanceOrig)", min_value=0.0, value=4000.0, step=100.0, format="%.2f")
    with col3:
        oldbalance_dest = st.number_input(
            "Destination balance before (oldbalanceDest)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
        newbalance_dest = st.number_input(
            "Destination balance after (newbalanceDest)", min_value=0.0, value=0.0, step=100.0, format="%.2f")

    raw = {
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "nameDest": name_dest,
    }

    if st.button("Check transaction", type="primary"):
        errors = features.validate_transaction(raw)
        if errors:
            st.error("Input validation failed: " + "; ".join(errors))
            return

        try:
            result = load_detector().predict_raw(raw)
        except Exception as exc:  # artifact/model problems must be visible
            st.error(f"Inference failed: {exc}")
            return

        # ---- display result ---------------------------------------------
        if result["decision"] == "FRAUD":
            st.error("## 🚨 Prediction: FRAUD")
        else:
            st.success("## ✅ Prediction: LEGITIMATE")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
        m2.metric("Threshold", f"{result['threshold']:.2f}")
        m3.metric("Decision", result["decision"])
        m4.metric("Model Version", result["model_version"])

        # ---- log prediction ---------------------------------------------
        log_status = logging_utils.log_prediction(result, transaction_reference=raw["nameDest"])
        st.info(f"**Logging:** {log_status['message']}")

        # ---- collapsible technical details ------------------------------
        with st.expander("Technical details (engineered features & latency)"):
            eng = pd.DataFrame([result["engineered_features"]]).T.rename(columns={0: "value"})
            st.dataframe(eng, use_container_width=True)
            st.caption(f"Inference latency: {result['latency_ms']:.1f} ms")


def page_batch_scoring() -> None:
    st.header("📦 Batch Scoring")
    st.caption("Scores the most recent transactions stored in MySQL through the exact same inference pipeline.")

    limit = st.slider("How many recent transactions", 5, 200, 25)

    if not database.is_available():
        st.warning(
            "MySQL is unavailable — batch scoring reads transactions from MySQL, "
            "so it cannot run right now. Check DB_* environment variables."
        )
        return

    rows = database.fetch_recent_transactions(limit=limit)
    if not rows:
        st.info("No transactions found in MySQL. Load the dataset first: `python scripts/load_csv_to_mysql.py`")
        return

    detector = load_detector()
    scored = []
    for row in rows:
        raw = {k: row[k] for k in ("type", "amount", "oldbalanceOrg", "newbalanceOrig",
                                   "oldbalanceDest", "newbalanceDest", "nameDest")}
        try:
            res = detector.predict_raw(raw)
            scored.append({"row": row, "raw": raw, "result": res, "error": None})
        except ValueError as exc:
            scored.append({"row": row, "raw": raw, "result": None, "error": str(exc)})

    df = pd.DataFrame(
        [
            {
                "transaction_id": s["row"].get("transaction_id"),
                "reference": s["row"].get("nameDest"),
                "type": s["row"]["type"],
                "amount": s["row"]["amount"],
                "fraud_probability": None if s["error"] else round(s["result"]["fraud_probability"], 4),
                "decision": s["result"]["decision"] if not s["error"] else f"INVALID ({s['error']})",
            }
            for s in scored
        ]
    )
    st.dataframe(df, use_container_width=True)

    fraud_n = int((df["decision"] == "FRAUD").sum())
    st.metric("Flagged as fraud", f"{fraud_n} / {len(df)}")

    if st.button("Log these batch predictions"):
        statuses = [logging_utils.log_prediction(s["result"], transaction_reference=s["row"].get("nameDest"))
                    for s in scored if not s["error"]]
        mysql_n = sum(s["status"] == "mysql" for s in statuses)
        jsonl_n = sum(s["status"] == "jsonl" for s in statuses)
        st.info(f"Logged {len(statuses)} predictions — {mysql_n} to MySQL, {jsonl_n} to JSONL fallback.")


def page_monitoring_drift() -> None:
    st.header("📈 Monitoring & Drift")

    logs = monitoring.collect_logs()
    if not logs:
        st.info("No prediction logs yet. Make some predictions on the Fraud Check page first.")
    else:
        summary = monitoring.compute_summary(logs)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total predictions", summary["sample_count"])
        m2.metric("Fraud predictions", summary["fraud_prediction_count"])
        m3.metric("Fraud percentage", f"{summary['fraud_percentage']:.2f}%")
        m4.metric("Avg probability", f"{summary['average_probability']:.4f}")

        if summary["labels_available"]:
            st.subheader("When actual labels are available")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{summary['precision']:.4f}")
            c2.metric("Recall", f"{summary['recall']:.4f}")
            c3.metric("F1", f"{summary['f1']:.4f}")
            c4.metric("TP/FP/TN/FN",
                      f"{summary['tp']}/{summary['fp']}/{summary['tn']}/{summary['fn']}")
        else:
            st.caption("No actual labels available yet — confusion-matrix metrics appear once labels are backfilled.")

        if st.button("Save monitoring snapshot"):
            st.info(monitoring.save_snapshot(summary)["message"])

    st.subheader("Data drift — reference (CSV) vs recent MySQL transactions")
    if not database.is_available():
        st.warning("MySQL unavailable — the current-data side of the drift table comes from recent MySQL transactions.")
        return

    reference_df = pd.read_csv(config.PROJECT_ROOT / "Fraud_Analysis_Dataset.csv").head(2000)
    current_rows = database.fetch_recent_transactions(limit=500)
    current_df = pd.DataFrame(current_rows)

    model_features = features.get_feature_order()
    table = drift.compare_distributions(
        reference_df, current_df,
        numeric_features=[f for f in model_features if f not in ("type", "isDestMerchant")],
        categorical_features=["type"],
    )
    st.dataframe(table, use_container_width=True)
    st.caption(
        "Drift thresholds are documented prototype heuristics (PSI: <0.10 OK, <0.25 WARNING, else DRIFT; "
        "categorical max share change: <0.05 OK, <0.15 WARNING, else DRIFT). They are NOT universal "
        "industry standards. This prototype only reports drift — no automatic retraining."
    )


def main() -> None:
    st.sidebar.title("Fraud Detection Prototype")
    # Startup initialization (safe): never crashes Streamlit when MySQL is down.
    _ensure_db_initialized()
    page = st.sidebar.radio("Navigation", ["Fraud Check", "Batch Scoring", "Monitoring & Drift"])
    st.sidebar.caption(f"Model version: {config.MODEL_VERSION}")
    if page == "Fraud Check":
        page_fraud_check()
    elif page == "Batch Scoring":
        page_batch_scoring()
    else:
        page_monitoring_drift()


main()

