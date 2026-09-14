"""Streamlit UI for the fraud detection prototype.

Contains NO model training code. All predictions go through
src/inference.py (saved artifacts only):
    User -> Streamlit -> inference.py -> features.py
         -> preprocessor.transform() -> XGBoost.predict_proba()
         -> threshold -> decision -> log prediction -> display

Pages:
  - Fraud Check      : score one manually-entered transaction
  - Batch Scoring    : MySQL historical data + session-scoped CSV upload inference
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

from src import config, database, drift, features, inference, logging_utils, monitoring, upload_inference  # noqa: E402

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
    tab_historical, tab_upload = st.tabs(["MySQL Historical Data", "Uploaded Inference Data"])
    # Tabs are UI separation ONLY — Streamlit runs both tab bodies on every
    # rerun. Behaviour isolation is enforced with explicit session-state guards.
    with tab_historical:
        _page_batch_scoring_mysql()
    with tab_upload:
        _page_batch_scoring_upload()


_UPLOAD_SESSION_KEYS = (
    "inference_upload_content_hash",
    "inference_raw_df",
    "inference_labels",
    "inference_meta",
    "inference_row_fingerprints",
    "inference_upload_dataset_hash",
    "inference_results",
    "inference_scored_dataset_hash",
    "inference_logged_dataset_hash",
)


def _page_batch_scoring_mysql() -> None:
    st.caption(
        "Scores the most recent transactions stored in MySQL. These are the "
        "original model-development records (historical/replay data)."
    )

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


def _page_batch_scoring_upload() -> None:
    st.caption(
        "Upload any PaySim-style CSV and score every row with the frozen saved "
        "model. Uploaded data is temporary and session-scoped — it is never "
        "written to MySQL and the original `transactions` table is untouched."
    )
    st.info(
        "Uploaded data is called **Uploaded Inference Data**. An exact-record "
        "overlap check reports whether any uploaded row exactly matches the "
        "original model-development dataset — it does not prove that the data "
        "is 'unseen' or distributionally novel."
    )

    uploaded = st.file_uploader("Upload a transaction CSV", type=["csv"])
    if uploaded is None:
        for key in _UPLOAD_SESSION_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        st.caption("No file uploaded yet.")
        return

    file_bytes = bytes(uploaded.getvalue())
    content_hash = upload_inference.compute_content_hash(file_bytes)

    if (
        st.session_state.get("inference_upload_content_hash") == content_hash
        and "inference_raw_df" in st.session_state
    ):
        # Same uploaded bytes -> reuse parsed state (content hash is the
        # authoritative upload identity; the filename is display-only).
        raw_df = st.session_state["inference_raw_df"]
        labels = st.session_state.get("inference_labels")
        fingerprints = st.session_state["inference_row_fingerprints"]
        dataset_hash = st.session_state["inference_upload_dataset_hash"]
        meta = dict(st.session_state["inference_meta"])
        meta["name"] = uploaded.name
        st.session_state["inference_meta"] = meta
    else:
        # Genuinely different upload -> validate once and replace the active dataset.
        with st.spinner("Parsing and validating the uploaded CSV…"):
            try:
                parsed = upload_inference.validate_upload(file_bytes)
            except upload_inference.UploadValidationError as exc:
                st.error("Upload rejected — no rows were scored.\n\n" + str(exc))
                return
        raw_df = parsed.raw_df
        labels = parsed.labels
        fingerprints = parsed.row_fingerprints
        dataset_hash = parsed.dataset_hash
        meta = {
            "name": uploaded.name,
            "rows": parsed.row_count,
            "duplicates": parsed.duplicate_row_count,
            "optional_columns": parsed.present_optional_columns,
            "content_hash": parsed.content_hash,
        }
        st.session_state["inference_upload_content_hash"] = parsed.content_hash
        st.session_state["inference_raw_df"] = raw_df
        st.session_state["inference_labels"] = labels
        st.session_state["inference_meta"] = meta
        st.session_state["inference_row_fingerprints"] = fingerprints
        st.session_state["inference_upload_dataset_hash"] = dataset_hash
        # New dataset: previous scoring / logging guards no longer apply.
        st.session_state["inference_results"] = None
        st.session_state["inference_scored_dataset_hash"] = None
        st.session_state["inference_logged_dataset_hash"] = None

    c1, c2, c3 = st.columns(3)
    c1.metric("File", meta["name"])
    c2.metric("Rows", meta["rows"])
    c3.metric("Label column", "Yes (isFraud)" if labels is not None else "No")
    st.caption(
        f"Duplicate rows inside the upload: {meta['duplicates']} (reported, never removed). "
        f"Optional source columns: {', '.join(meta['optional_columns']) or 'none'}."
    )

    overlap = upload_inference.compute_overlap(
        fingerprints, upload_inference.original_dataset_fingerprints()
    )
    st.info(upload_inference.overlap_message(overlap, len(fingerprints)))

    if st.button("Score Dataset", type="primary"):
        detector = load_detector()
        progress = st.progress(0.0, text="Scoring uploaded rows…")
        scored_now = upload_inference.ensure_scored(
            st.session_state,
            raw_df,
            dataset_hash,
            detector,
            progress=lambda done, total: progress.progress(
                (done / total) if total else 1.0,
                text=f"Scored {done} of {total} rows",
            ),
        )
        if not scored_now:
            st.caption(
                "These results were already computed for this upload — nothing was re-scored."
            )

    results = st.session_state.get("inference_results")
    if results is None:
        st.caption("No results yet — click **Score Dataset**.")
        return

    valid = [r for r in results if r.get("result") is not None]
    if not valid:
        st.error("Scoring failed for every uploaded row.")
        return

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total scored", f"{len(valid)} / {len(results)}")
    fraud_n = sum(int(r["result"]["prediction"]) for r in valid)
    r2.metric("Predicted fraud", fraud_n)
    r3.metric("Predicted legitimate", len(valid) - fraud_n)
    probabilities = [float(r["result"]["fraud_probability"]) for r in valid]
    r4.metric("Avg fraud probability", f"{sum(probabilities) / len(probabilities):.4f}")

    display = pd.DataFrame(
        [
            {
                "row": int(r["row_index"]) + 1,
                "reference": raw_df.iloc[int(r["row_index"])]["nameDest"],
                "type": raw_df.iloc[int(r["row_index"])]["type"],
                "amount": raw_df.iloc[int(r["row_index"])]["amount"],
                "fraud_probability": round(float(r["result"]["fraud_probability"]), 4),
                "decision": r["result"]["decision"],
            }
            for r in valid
        ]
    )
    st.dataframe(display, use_container_width=True)
    threshold = float(valid[0]["result"]["threshold"])
    st.caption(
        f"Probability min / max: {min(probabilities):.6f} / {max(probabilities):.6f} — "
        f"threshold {threshold:.2f}."
    )

    if labels is not None:
        st.subheader("Evaluation vs uploaded labels")
        st.caption(
            "Ground-truth labels were provided by the uploaded file and were "
            "not used as model inputs."
        )
        if len(valid) != len(results):
            st.caption("Evaluation skipped: some rows failed to score.")
        else:
            predicted_classes = [int(r["result"]["prediction"]) for r in valid]
            metrics = upload_inference.evaluate_predictions(predicted_classes, list(labels))
            e1, e2 = st.columns(2)
            e1.metric("TP / FP", f"{metrics['tp']} / {metrics['fp']}")
            e2.metric("TN / FN", f"{metrics['tn']} / {metrics['fn']}")
            e3, e4, e5 = st.columns(3)
            e3.metric("Accuracy", "—" if metrics["accuracy"] is None else f"{metrics['accuracy']:.4f}")
            e4.metric("Precision", "—" if metrics["precision"] is None else f"{metrics['precision']:.4f}")
            e5.metric("Recall", "—" if metrics["recall"] is None else f"{metrics['recall']:.4f}")
            e6, e7 = st.columns(2)
            e6.metric("F1", "—" if metrics["f1"] is None else f"{metrics['f1']:.4f}")
            e7.metric("Sample count", metrics["sample_count"])

    if st.button("Log Predictions", type="primary"):
        if upload_inference.was_logged(st.session_state, dataset_hash):
            st.info("These predictions have already been logged for this upload.")
        else:
            def _log_uploaded(result, transaction_reference=None):
                return logging_utils.log_prediction(result, transaction_reference=transaction_reference)

            statuses = upload_inference.log_results(results, raw_df, _log_uploaded)
            st.session_state["inference_logged_dataset_hash"] = dataset_hash
            mysql_n = sum(s["status"] == "mysql" for s in statuses)
            jsonl_n = sum(s["status"] == "jsonl" for s in statuses)
            st.info(
                f"Logged {len(statuses)} predictions for this upload — "
                f"{mysql_n} to MySQL, {jsonl_n} to JSONL fallback."
            )


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

