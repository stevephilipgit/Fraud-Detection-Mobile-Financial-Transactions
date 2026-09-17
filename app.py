"""Streamlit interface for the fraud detection application.

Presentation only. Every prediction still runs through src/inference.py using
the saved artifacts:

    input -> Streamlit -> inference.py -> features.py
          -> preprocessor.transform() -> XGBoost.predict_proba()
          -> saved threshold -> decision -> logging -> display

Pages:
  - Fraud Check              screen one manually entered transaction
  - Upload Transaction Data  score a CSV of transactions
  - Transaction History      previously analyzed batches (view / delete)
  - Monitoring               prediction activity and model performance
  - Data Drift Analysis      reference data vs recent inference data

Terminology, formatting and layout live here and in ui_presentation.py. No
model, database, monitoring, drift, validation or logging logic is implemented
or modified in this file.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable when Streamlit runs from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import (  # noqa: E402
    config,
    database,
    drift,
    features,
    inference,
    logging_utils,
    monitoring,
    upload_inference,
)
import ui_presentation as ui  # noqa: E402

st.set_page_config(
    page_title=ui.APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_theme()

PAGES = (
    "Fraud Check",
    "Upload Transaction Data",
    "Transaction History",
    "Monitoring",
    "Data Drift Analysis",
)


@st.cache_resource(show_spinner="Loading the fraud detection model…")
def load_detector():
    return inference.FraudDetector().load()


@st.cache_data(show_spinner="Loading the reference dataset…")
def load_reference_frame() -> pd.DataFrame:
    """Full original dataset with engineered features — the fixed Drift reference."""
    df = pd.read_csv(config.INFERENCE_REFERENCE_CSV)
    return features.engineer_features(df)


def _ensure_db_initialized() -> bool:
    """Attempt database initialization once per Streamlit session.

    Never crashes the app: when the database is unavailable (or credentials are
    not configured yet) the app keeps the existing JSONL fallback behaviour.
    """
    if "db_ready" in st.session_state:
        return st.session_state["db_ready"]
    try:
        database.init_database()
        st.session_state["db_ready"] = True
    except database.DatabaseUnavailableError:
        st.session_state["db_ready"] = False
    return st.session_state["db_ready"]


# ---------------------------------------------------------------------------
# Presentation helpers (formatting and grouping only — no business logic)
# ---------------------------------------------------------------------------
def _technical_value(value) -> str:
    """Compact technical rendering: 1314.5 -> '1,314.5', 0.0 -> '0'."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.6g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fraud_rate(fraud_count, row_count):
    """Share of analyzed transactions detected as fraud, in percent (or None)."""
    try:
        row_count = int(row_count or 0)
        if row_count <= 0:
            return None
        return 100.0 * float(fraud_count or 0) / row_count
    except (TypeError, ValueError):
        return None


def _batch_option_label(batch) -> str:
    name = batch.get("label") or ui.pretty_source(batch.get("source"))
    return (
        f"Batch #{batch['batch_id']} — {name} — "
        f"{ui.fmt_count(batch.get('row_count'))} transactions — "
        f"{ui.fmt_datetime(batch.get('created_at'))}"
    )


def _batch_table_frame(batches) -> pd.DataFrame:
    rows = []
    for batch in batches:
        average = batch.get("avg_fraud_probability")
        rows.append(
            {
                "Batch": int(batch["batch_id"]),
                "Data Source": ui.pretty_source(batch.get("source")),
                "File": batch.get("label") or ui.NOT_AVAILABLE,
                "Transactions Analyzed": int(batch.get("row_count") or 0),
                "Fraudulent Transactions Detected": int(batch.get("fraud_count") or 0),
                "Fraud Detection Rate": _fraud_rate(
                    batch.get("fraud_count"), batch.get("row_count")
                ),
                "Average Fraud Probability": (
                    None if average is None else float(average) * 100.0
                ),
                "Actual Labels Available": "Yes" if batch.get("has_ground_truth") else "No",
                "Created": ui.fmt_datetime(batch.get("created_at")),
            }
        )
    return pd.DataFrame(rows)


_BATCH_COLUMNS = {
    "Batch": st.column_config.NumberColumn("Batch", format="#%d", width="small"),
    "Data Source": st.column_config.TextColumn("Data Source", width="small"),
    "File": st.column_config.TextColumn("File", width="medium"),
    "Transactions Analyzed": st.column_config.NumberColumn(
        ui.KPI_TRANSACTIONS, format="%,d", width="small"
    ),
    "Fraudulent Transactions Detected": st.column_config.NumberColumn(
        ui.KPI_FRAUD, format="%,d", width="small"
    ),
    "Fraud Detection Rate": st.column_config.NumberColumn(
        ui.KPI_RATE, format="%.2f%%", help=ui.HELP_FRAUD_RATE, width="small"
    ),
    "Average Fraud Probability": st.column_config.NumberColumn(
        ui.KPI_AVG_PROB, format="%.2f%%", help=ui.HELP_AVG_PROB, width="small"
    ),
    "Actual Labels Available": st.column_config.TextColumn(
        "Actual Labels Available",
        width="small",
        help="Whether the data supplied with this batch included actual outcomes.",
    ),
    "Created": st.column_config.TextColumn("Created", width="medium"),
}


def _result_row(raw_row, probability, prediction) -> dict:
    """One row of a transaction-results table (formatted for display)."""
    raw = raw_row or {}
    return {
        "Recipient Account": raw.get("nameDest"),
        "Transaction Type": ui.pretty_transaction_type(raw.get("type")),
        "Transaction Amount": raw.get("amount"),
        "Fraud Probability": None if probability is None else float(probability) * 100.0,
        "Prediction": prediction,
    }


_RESULT_COLUMNS = {
    "Recipient Account": st.column_config.TextColumn("Recipient Account", width="medium"),
    "Transaction Type": st.column_config.TextColumn("Transaction Type", width="small"),
    "Transaction Amount": st.column_config.NumberColumn(
        "Transaction Amount",
        format="%,.2f",
        width="small",
        help="Amount as recorded in the scored transaction. The source data is "
             "currency-agnostic, so no currency symbol is shown.",
    ),
    "Fraud Probability": st.column_config.NumberColumn(
        "Fraud Probability",
        format="%.2f%%",
        width="small",
        help="Model output for this transaction, before the decision threshold is applied.",
    ),
    "Prediction": st.column_config.TextColumn("Prediction", width="medium"),
    "Actual Outcome": st.column_config.TextColumn(
        "Actual Outcome",
        width="small",
        help="The true outcome supplied with the data, shown only when labels were available.",
    ),
}


# ---------------------------------------------------------------------------
# Page 1 — Fraud Check
# ---------------------------------------------------------------------------
def page_fraud_check() -> None:
    ui.page_header(
        "Fraud Check",
        "Analyze a transaction and estimate its likelihood of fraud.",
    )

    with st.container(border=True):
        ui.section("Transaction Details")
        col_type, col_amount, col_recipient = st.columns(3)
        with col_type:
            tx_type = st.selectbox(
                "Transaction Type",
                config.VALID_TRANSACTION_TYPES,
                index=4,
                format_func=ui.pretty_transaction_type,
            )
        with col_amount:
            amount = st.number_input(
                "Transaction Amount", min_value=0.0, value=1000.0,
                step=100.0, format="%.2f",
            )
        with col_recipient:
            name_dest = st.text_input("Recipient Account", value="C123456789")

    sender_column, recipient_column = st.columns(2)
    with sender_column:
        with st.container(border=True):
            ui.section("Sender Account Balances")
            oldbalance_org = st.number_input(
                "Balance Before Transaction", min_value=0.0, value=5000.0,
                step=100.0, format="%.2f", key="sender_balance_before",
            )
            newbalance_orig = st.number_input(
                "Balance After Transaction", min_value=0.0, value=4000.0,
                step=100.0, format="%.2f", key="sender_balance_after",
            )
    with recipient_column:
        with st.container(border=True):
            ui.section("Recipient Account Balances")
            oldbalance_dest = st.number_input(
                "Balance Before Transaction", min_value=0.0, value=0.0,
                step=100.0, format="%.2f", key="recipient_balance_before",
            )
            newbalance_dest = st.number_input(
                "Balance After Transaction", min_value=0.0, value=0.0,
                step=100.0, format="%.2f", key="recipient_balance_after",
            )

    # The raw transaction contract is unchanged: exactly these seven fields.
    raw = {
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "nameDest": name_dest,
    }

    st.write("")
    if st.button("Check Transaction", type="primary"):
        errors = features.validate_transaction(raw)
        if errors:
            st.error(
                "This transaction could not be analyzed. Please correct the "
                "following and try again."
            )
            st.markdown("\n".join(f"- {ui.humanise_fields(e)}" for e in errors))
            st.session_state.pop("fraud_check_result", None)
        else:
            try:
                st.session_state["fraud_check_result"] = load_detector().predict_raw(raw)
                st.session_state["fraud_check_logged"] = False
            except Exception as exc:  # artifact/model problems must stay visible
                ui.technical_error(
                    "This transaction could not be analyzed because the prediction "
                    "model is unavailable. Please try again.",
                    exc,
                )
                st.session_state.pop("fraud_check_result", None)
                return

    # The prediction stays visible across reruns; saving remains explicit.
    result = st.session_state.get("fraud_check_result")
    if result is None:
        return

    probability = float(result["fraud_probability"])
    is_fraud = result["decision"] == "FRAUD"

    with st.container(border=True):
        st.caption("Prediction")
        if is_fraud:
            st.error(f"### {ui.PREDICTION_FRAUD}")
        else:
            st.success(f"### {ui.PREDICTION_LEGITIMATE}")
        ui.metric_row([
            {
                "label": "Fraud Probability",
                "value": ui.fmt_percent(probability),
                "help": "The model's estimated likelihood that this transaction is "
                        "fraudulent. The decision threshold is applied separately "
                        "and is listed under Technical Details.",
            },
        ])
        if is_fraud:
            st.caption(
                "The fraud probability is at or above the model's decision "
                "threshold, so this transaction would be flagged for review."
            )
        else:
            st.caption(
                "The fraud probability is below the model's decision threshold, "
                "so no fraud alert would be raised for this transaction."
            )

        st.divider()
        if st.session_state.get("fraud_check_logged"):
            st.info("This prediction is already in your transaction history.")
        elif st.button("Add to Transaction History", type="secondary"):
            batch_id = None
            if database.is_available():
                batch_id = database.get_or_create_fraud_check_batch()
            log_status = logging_utils.log_prediction(
                result,
                transaction_reference=result["raw_transaction"]["nameDest"],
                batch_id=batch_id,
            )
            if batch_id is not None:
                database.refresh_batch_counts(batch_id)
            st.session_state["fraud_check_logged"] = True
            if log_status.get("status") == logging_utils.STATUS_MYSQL:
                st.success("Saved to your transaction history.")
            elif log_status.get("status") == logging_utils.STATUS_JSONL:
                st.warning(
                    "The database is unavailable, so this prediction was written to "
                    "a local fallback file. It will not appear in Monitoring or "
                    "Data Drift Analysis."
                )
            else:
                st.error("This prediction could not be saved.")
            ui.technical_details({
                "Logging destination": log_status.get("destination"),
                "Raw logging response": log_status.get("message"),
            })

    # ---- technical details (collapsed; the main view stays business-facing) ---
    engineered = result.get("engineered_features") or {}
    eng_rows = [
        {"Feature": ui.pretty_feature(name), "Internal Name": name,
         "Value": _technical_value(value)}
        for name, value in engineered.items()
    ]
    raw_rows = [
        {"Field": ui.pretty_feature(name), "Internal Name": name,
         "Value": _technical_value(value)}
        for name, value in (result.get("raw_transaction") or {}).items()
    ]
    with st.expander(ui.TECHNICAL_DETAILS_TITLE, expanded=False):
        st.caption(
            "Values exactly as used by the prediction pipeline, before display "
            "formatting. Engineered features are derived automatically from the "
            "submitted balances and amount — they are never entered manually."
        )
        if eng_rows:
            st.markdown("**Model Inputs (Engineered Features)**")
            st.dataframe(pd.DataFrame(eng_rows), hide_index=True, width="stretch")
        if raw_rows:
            st.markdown("**Transaction as Submitted**")
            st.dataframe(pd.DataFrame(raw_rows), hide_index=True, width="stretch")
        st.markdown("**Model Information**")
        st.dataframe(
            pd.DataFrame([
                {"Field": "Model Version", "Value": str(result["model_version"])},
                {"Field": "Decision Threshold", "Value": ui.fmt_percent(result["threshold"])},
                {"Field": "Decision Label", "Value": str(result["decision"])},
                {"Field": "Predicted Class", "Value": str(result["prediction"])},
                {"Field": "Inference Latency", "Value": f"{result['latency_ms']:.1f} ms"},
            ]),
            hide_index=True,
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Page 2 — Upload Transaction Data
# ---------------------------------------------------------------------------
_UPLOAD_SESSION_KEYS = (
    "inference_upload_content_hash",
    "inference_raw_df",
    "inference_labels",
    "inference_meta",
    "inference_row_fingerprints",
    "inference_upload_dataset_hash",
    "inference_results",
    "inference_scored_dataset_hash",
    "inference_persisted_hash",
    "inference_persistence",
)


def page_upload() -> None:
    ui.page_header(
        "Upload Transaction Data",
        "Upload a CSV file containing transactions to analyze.",
    )
    st.caption(
        f"Supported file: CSV  |  Maximum size: "
        f"{config.UPLOAD_MAX_BYTES // (1024 * 1024)} MB  |  Maximum rows: "
        f"{ui.fmt_count(config.UPLOAD_MAX_ROWS)}"
    )

    uploaded = st.file_uploader("Upload a transaction CSV", type=["csv"])
    if uploaded is None:
        for key in _UPLOAD_SESSION_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        ui.empty_state(
            "No file uploaded yet.",
            "Choose a CSV file to begin. Every row is analyzed with the saved "
            "fraud detection model.",
        )
        return

    file_bytes = bytes(uploaded.getvalue())
    content_hash = upload_inference.compute_content_hash(file_bytes)

    if (
        st.session_state.get("inference_upload_content_hash") == content_hash
        and "inference_raw_df" in st.session_state
    ):
        # Same uploaded bytes -> reuse parsed state (the content hash is the
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
        with st.spinner("Validating the uploaded file…"):
            try:
                parsed = upload_inference.validate_upload(file_bytes)
            except upload_inference.UploadValidationError as exc:
                st.error("This file could not be analyzed. Nothing was scored.")
                st.markdown(ui.humanise_fields(str(exc)))
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
        # New dataset: previous scoring / persistence guards no longer apply.
        st.session_state["inference_results"] = None
        st.session_state["inference_scored_dataset_hash"] = None
        st.session_state["inference_persisted_hash"] = None
        st.session_state["inference_persistence"] = None

    st.success("File validated successfully")
    st.caption(f"File: **{meta['name']}**")
    ui.metric_row([
        {"label": "Transactions in File", "value": ui.fmt_count(meta["rows"])},
        {"label": "Duplicate Transactions", "value": ui.fmt_count(meta["duplicates"]),
         "help": "Rows that repeat an earlier row in the same file. They are reported but never removed."},
        {"label": "Actual Labels Available",
         "value": "Yes" if labels is not None else "No",
         "help": "Whether the file included actual outcomes, which are used only to measure model performance."},
    ])

    ui.section(
        "Reference Data Check",
        "Compares uploaded records with the original model-development dataset "
        "by exact record match.",
    )
    overlap = upload_inference.compute_overlap(
        fingerprints, upload_inference.original_dataset_fingerprints()
    )
    overlap_text = upload_inference.overlap_message(overlap, len(fingerprints))
    if overlap_text:
        st.info(overlap_text)

    if st.button("Analyze Transactions", type="primary"):
        detector = load_detector()
        progress = st.progress(0.0, text="Analyzing transactions…")
        scored_now = upload_inference.ensure_scored(
            st.session_state,
            raw_df,
            dataset_hash,
            detector,
            progress=lambda done, total: progress.progress(
                (done / total) if total else 1.0,
                text=f"Analyzed {done} of {total} transactions",
            ),
        )
        if not scored_now:
            st.caption(
                "These results were already computed for this upload — nothing was "
                "re-analyzed."
            )

    results = st.session_state.get("inference_results")
    if results is None:
        ui.empty_state(
            "Analysis has not been run yet.",
            "Select **Analyze Transactions** to score the uploaded rows.",
        )
        return

    # ---- automatic persistence: a scored upload becomes transaction history ---
    persisted = st.session_state.get("inference_persistence")
    if st.session_state.get("inference_persisted_hash") != content_hash:
        persisted = upload_inference.persist_scored_batch(
            content_hash=content_hash,
            dataset_hash=dataset_hash,
            label=uploaded.name,
            raw_df=raw_df,
            results=results,
            labels=labels,
        )
        st.session_state["inference_persisted_hash"] = content_hash
        st.session_state["inference_persistence"] = persisted

    if persisted and persisted.get("status") == "saved":
        st.success(
            f"Saved to your transaction history as batch #{persisted['batch_id']}."
        )
    elif persisted and persisted.get("status") == "reused":
        st.info(
            f"This dataset was already analyzed as batch #{persisted['batch_id']}. "
            "The existing batch is reused, so no duplicate records were created."
        )
    elif persisted and persisted.get("status") == "unavailable":
        st.warning(
            "This analysis could not be saved to your transaction history, so it "
            "will not appear in Monitoring or Data Drift Analysis."
        )
        ui.technical_details({
            "Persistence response": persisted.get("message"),
        })

    valid = [entry for entry in results if entry.get("result") is not None]
    if not valid:
        st.error(
            "No transactions could be analyzed. Please check the file contents and "
            "try again."
        )
        return

    st.success("Analysis complete")
    probabilities = [float(entry["result"]["fraud_probability"]) for entry in valid]
    fraud_count = sum(int(entry["result"]["prediction"]) for entry in valid)
    ui.metric_row([
        {"label": ui.KPI_TRANSACTIONS, "value": ui.fmt_count(len(valid))},
        {"label": ui.KPI_FRAUD, "value": ui.fmt_count(fraud_count)},
        {"label": ui.KPI_RATE,
         "value": ui.fmt_percent_value(_fraud_rate(fraud_count, len(valid))),
         "help": ui.HELP_FRAUD_RATE},
        {"label": ui.KPI_AVG_PROB,
         "value": ui.fmt_percent(sum(probabilities) / len(probabilities)),
         "help": ui.HELP_AVG_PROB},
    ])
    if len(valid) != len(results):
        st.caption(
            f"{ui.fmt_count(len(results) - len(valid))} of "
            f"{ui.fmt_count(len(results))} transactions could not be analyzed and "
            "are excluded from the figures above."
        )

    display_rows = []
    for entry in valid:
        index = int(entry["row_index"])
        source_row = raw_df.iloc[index]
        row = _result_row(
            {
                "nameDest": source_row["nameDest"],
                "type": source_row["type"],
                "amount": source_row["amount"],
            },
            entry["result"]["fraud_probability"],
            ui.PREDICTION_FRAUD
            if int(entry["result"]["prediction"]) == 1
            else ui.PREDICTION_LEGITIMATE,
        )
        display_rows.append({"Row": index + 1, **row})

    ui.section("Transaction Results", f"{ui.fmt_count(len(valid))} analyzed transactions.")
    st.dataframe(
        pd.DataFrame(display_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Row": st.column_config.NumberColumn(
                "Row", format="%d", width="small",
                help="Position of this transaction in the uploaded file.",
            ),
            **_RESULT_COLUMNS,
        },
    )
    st.caption(
        f"Fraud probability ranges from {ui.fmt_percent(min(probabilities))} to "
        f"{ui.fmt_percent(max(probabilities))} across this file."
    )

    ui.section(
        "Model Performance",
        "These metrics require actual outcomes. Actual outcomes come from the "
        "uploaded file and are used only after prediction — never as model inputs.",
    )
    if labels is None:
        st.caption(
            "Not available for this upload because the file did not include actual "
            "outcomes. Include an isFraud column with 0/1 values to measure model "
            "performance."
        )
    elif len(valid) != len(results):
        st.caption(
            "Not available because some rows in this file could not be analyzed."
        )
    else:
        predicted_classes = [int(entry["result"]["prediction"]) for entry in valid]
        metrics = upload_inference.evaluate_predictions(predicted_classes, list(labels))
        ui.metric_row([
            {"label": ui.PERF_ACCURACY, "value": ui.fmt_percent(metrics["accuracy"]),
             "help": ui.HELP_ACCURACY},
            {"label": ui.PERF_PRECISION, "value": ui.fmt_percent(metrics["precision"]),
             "help": ui.HELP_PRECISION},
            {"label": ui.PERF_RECALL, "value": ui.fmt_percent(metrics["recall"]),
             "help": ui.HELP_RECALL},
            {"label": ui.PERF_F1, "value": ui.fmt_percent(metrics["f1"]),
             "help": ui.HELP_F1},
        ])
        with st.expander("Detailed Evaluation Metrics", expanded=False):
            st.caption(
                f"Computed from {ui.fmt_count(metrics['sample_count'])} labeled "
                "transactions."
            )
            ui.metric_row([
                {"label": "True Positives", "value": ui.fmt_count(metrics["tp"]),
                 "help": "Fraudulent transactions correctly detected as fraud."},
                {"label": "False Positives", "value": ui.fmt_count(metrics["fp"]),
                 "help": "Legitimate transactions incorrectly flagged as fraud."},
                {"label": "True Negatives", "value": ui.fmt_count(metrics["tn"]),
                 "help": "Legitimate transactions correctly identified as legitimate."},
                {"label": "False Negatives", "value": ui.fmt_count(metrics["fn"]),
                 "help": "Fraudulent transactions that were not detected."},
            ])

    ui.technical_details({
        "Upload identity (content hash)": meta.get("content_hash"),
        "Dataset hash": dataset_hash,
        "Duplicate rows reported": meta.get("duplicates"),
        "Optional source columns present": ", ".join(meta.get("optional_columns") or []) or "none",
        "Exact overlaps with reference data": overlap,
        "Decision threshold applied": ui.fmt_percent(valid[0]["result"]["threshold"]),
        "Model version": valid[0]["result"]["model_version"],
    }, note="Scored uploads are saved to your transaction history automatically.")


# ---------------------------------------------------------------------------
# Page 3 — Transaction History
# ---------------------------------------------------------------------------
def page_transaction_history() -> None:
    ui.page_header(
        "Transaction History",
        "Previously analyzed transaction batches.",
    )

    if not database.is_available():
        st.warning(
            "Your transaction history is stored in a database that cannot be "
            "reached right now, so it cannot be displayed. Fraud Check and Upload "
            "still work."
        )
        ui.technical_details({
            "Requirement": "A reachable database connection",
            "Connection settings": "DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD",
        })
        return

    # Deletion results are shown after the rerun triggered by the delete action.
    delete_message = st.session_state.pop("batch_delete_message", None)
    if delete_message:
        if delete_message.get("deleted"):
            st.success(
                f"Batch #{delete_message.get('batch_id')} was deleted, including "
                f"{ui.fmt_count(delete_message.get('prediction_rows_deleted'))} saved "
                "transaction results."
            )
        else:
            st.error("This batch could not be deleted. Nothing was changed.")
            ui.technical_details({"Response": delete_message.get("message")})
    legacy_message = st.session_state.pop("legacy_delete_message", None)
    if legacy_message:
        st.success(
            f"{ui.fmt_count(legacy_message.get('rows_deleted'))} legacy records were "
            "deleted."
        )
    legacy_error = st.session_state.pop("legacy_delete_error", None)
    if legacy_error:
        st.error("The legacy records could not be deleted. Nothing was changed.")
        ui.technical_details({"Response": legacy_error.get("message")})

    batches = database.fetch_inference_batches()

    if not batches:
        ui.empty_state(
            "No inference history yet.",
            "Upload a CSV file or run a Fraud Check to create your first analysis "
            "batch.",
        )
    else:
        ui.section(
            "Analyzed Batches",
            "Each analyzed file or fraud check is stored as one batch.",
        )
        st.dataframe(
            _batch_table_frame(batches),
            hide_index=True,
            width="stretch",
            column_config=_BATCH_COLUMNS,
        )

        options = {_batch_option_label(batch): batch for batch in batches}
        selected = st.selectbox("Select an inference batch", list(options))
        batch = options[selected]
        _batch_detail_card(batch, database.fetch_batch_predictions(batch["batch_id"]))

    _legacy_section()


def _batch_detail_card(batch, predictions) -> None:
    """Summary, transaction results and (secondary) cleanup for one batch."""
    average = batch.get("avg_fraud_probability")
    with st.container(border=True):
        st.subheader(
            f"Batch #{batch['batch_id']} · "
            f"{batch.get('label') or ui.pretty_source(batch.get('source'))}"
        )
        st.caption(
            f"{ui.pretty_source(batch.get('source'))} · "
            f"Created {ui.fmt_datetime(batch.get('created_at'))} · "
            f"Actual labels "
            f"{'available' if batch.get('has_ground_truth') else 'not available'}"
        )
        ui.metric_row([
            {"label": ui.KPI_TRANSACTIONS, "value": ui.fmt_count(batch.get("row_count"))},
            {"label": ui.KPI_FRAUD, "value": ui.fmt_count(batch.get("fraud_count"))},
            {"label": ui.KPI_RATE,
             "value": ui.fmt_percent_value(
                 _fraud_rate(batch.get("fraud_count"), batch.get("row_count"))
             ),
             "help": ui.HELP_FRAUD_RATE},
            {"label": ui.KPI_AVG_PROB,
             "value": ui.NOT_AVAILABLE if average is None else ui.fmt_percent(average),
             "help": ui.HELP_AVG_PROB},
        ])

        if not predictions:
            st.caption("This batch contains no saved transaction results.")
        else:
            rows = []
            for prediction in predictions:
                raw = dict(prediction.get("raw_transaction") or {})
                raw.setdefault("nameDest", prediction.get("transaction_reference"))
                row = _result_row(
                    raw,
                    prediction["fraud_probability"],
                    ui.PREDICTION_FRAUD
                    if int(prediction["predicted_class"]) == 1
                    else ui.PREDICTION_LEGITIMATE,
                )
                actual = prediction.get("actual_label")
                row["Actual Outcome"] = (
                    ui.NOT_AVAILABLE
                    if actual is None
                    else (ui.OUTCOME_FRAUD if int(actual) == 1 else ui.OUTCOME_LEGITIMATE)
                )
                rows.append(row)

            st.markdown("**Transaction Results**")
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config=_RESULT_COLUMNS,
            )

        ui.technical_details({
            "Batch identifier": batch["batch_id"],
            "Data source (internal)": batch.get("source"),
            "Upload content hash": batch.get("content_hash"),
            "Dataset hash": batch.get("dataset_hash"),
            "Stored fraud count": batch.get("fraud_count"),
            "Stored average probability": batch.get("avg_fraud_probability"),
            "Created at (raw)": batch.get("created_at"),
        })

        # ---- destructive action: visually secondary and always confirmed ------
        st.divider()
        st.caption(
            "Deleting a batch removes only that batch and its own saved results. "
            "Your uploaded file and the original training data are never affected."
        )
        if st.button(
            "Delete Batch", type="secondary", key=f"ask_delete_{batch['batch_id']}"
        ):
            st.session_state["confirm_delete_batch_id"] = batch["batch_id"]

        if st.session_state.get("confirm_delete_batch_id") == batch["batch_id"]:
            st.warning(
                "Delete this analysis batch?\n\n"
                "This will permanently remove the saved predictions for this batch. "
                "Your original training data will not be affected."
            )
            cancel_column, confirm_column, _ = st.columns([1, 1, 4])
            if cancel_column.button("Cancel", key="cancel_batch_delete"):
                st.session_state.pop("confirm_delete_batch_id")
                st.rerun()
            if confirm_column.button(
                "Delete Batch", type="secondary", key="confirm_batch_delete"
            ):
                try:
                    result = database.delete_inference_batch(batch["batch_id"])
                except Exception as exc:
                    result = {
                        "deleted": False,
                        "batch_id": batch["batch_id"],
                        "prediction_rows_deleted": 0,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                st.session_state.pop("confirm_delete_batch_id")
                st.session_state["batch_delete_message"] = result
                st.rerun()


def _legacy_section() -> None:
    """Legacy record management — deliberately secondary (collapsed expander)."""
    with st.expander("Legacy Records", expanded=False):
        st.caption(
            "Legacy records are older predictions that were created before batch "
            "tracking was introduced. Use Include Legacy Records in Monitoring "
            "to include them there. They are not used in Data Drift Analysis."
        )
        try:
            legacy_count = database.count_legacy_prediction_logs()
        except Exception as exc:
            ui.technical_error(
                "The number of legacy records could not be determined right now.",
                exc,
            )
            return

        if not legacy_count:
            st.caption("No legacy records are stored.")
            return

        st.caption(
            f"{ui.fmt_count(legacy_count)} legacy records are stored. Deleting them "
            "affects only these records — analyzed batches and the original training "
            "data are not touched."
        )
        if st.button("Delete Legacy Records", type="secondary", key="ask_delete_legacy"):
            st.session_state["confirm_delete_legacy"] = True

        if st.session_state.get("confirm_delete_legacy"):
            st.warning(
                f"Delete {ui.fmt_count(legacy_count)} legacy records?\n\n"
                "This permanently removes only records that do not belong to an "
                "analyzed batch. Analyzed batches and the original training data "
                "are not affected."
            )
            cancel_column, confirm_column, _ = st.columns([1, 1, 4])
            if cancel_column.button("Cancel", key="cancel_legacy_delete"):
                st.session_state.pop("confirm_delete_legacy")
                st.rerun()
            if confirm_column.button(
                "Delete Legacy Records", type="secondary", key="confirm_legacy_delete"
            ):
                try:
                    result = database.delete_legacy_prediction_logs()
                except Exception as exc:
                    result = {
                        "deleted": False,
                        "rows_deleted": 0,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                st.session_state.pop("confirm_delete_legacy")
                if result.get("deleted"):
                    st.session_state["legacy_delete_message"] = result
                else:
                    st.session_state["legacy_delete_error"] = result
                st.rerun()


# ---------------------------------------------------------------------------
# Page 4 — Monitoring
# ---------------------------------------------------------------------------
def page_monitoring() -> None:
    ui.page_header(
        "Fraud Detection Monitoring",
        "Monitor transaction volume, fraud detection rates, and model performance.",
    )

    include_legacy = st.toggle(
        "Include Legacy Records",
        value=False,
        help="Legacy records are older predictions that were created before batch "
             "tracking was introduced.",
    )
    if include_legacy:
        st.caption(
            "Legacy records are older predictions created before batch tracking was "
            "introduced. While this option is on, they are included in every figure "
            "on this page."
        )

    try:
        rows = database.fetch_scoped_predictions(include_legacy=include_legacy)
    except Exception as exc:
        ui.technical_error(
            "Monitoring data could not be loaded because the database is not "
            "reachable right now.",
            exc,
        )
        return

    if not rows:
        ui.empty_state(
            "No prediction data available.",
            "Run a Fraud Check or upload transaction data to start monitoring.",
        )
        return

    summary = monitoring.compute_summary(rows)

    ui.section(
        "Prediction Activity",
        "What the model has analyzed so far and how often it detects fraud.",
    )
    ui.metric_row([
        {"label": ui.KPI_TRANSACTIONS, "value": ui.fmt_count(summary["sample_count"])},
        {"label": ui.KPI_FRAUD,
         "value": ui.fmt_count(summary["fraud_prediction_count"])},
        {"label": ui.KPI_RATE,
         "value": ui.fmt_percent_value(summary["fraud_percentage"]),
         "help": ui.HELP_FRAUD_RATE},
        {"label": ui.KPI_AVG_PROB,
         "value": ui.fmt_percent(summary["average_probability"]),
         "help": ui.HELP_AVG_PROB},
    ])
    st.caption(
        f"{ui.fmt_count(summary['legitimate_prediction_count'])} analyzed transactions "
        "were assessed as legitimate."
    )

    batches = database.fetch_inference_batches()
    if batches:
        latest = batches[0]
        st.info(
            f"Most recent batch: #{latest['batch_id']} — "
            f"{ui.fmt_count(latest.get('row_count'))} transactions analyzed, "
            f"{ui.fmt_count(latest.get('fraud_count'))} fraudulent transactions "
            f"detected ({ui.fmt_datetime(latest.get('created_at'))})."
        )

    ui.section("Transaction Volume Over Time")
    volume = monitoring.daily_volume(rows)
    if volume.empty:
        st.caption("No dated activity to chart yet.")
    else:
        chart = volume.rename(
            columns={"date": "Date", "predictions": ui.KPI_TRANSACTIONS}
        ).set_index("Date")
        st.bar_chart(
            chart,
            x_label="Date",
            y_label=ui.KPI_TRANSACTIONS,
            height=280,
        )

    ui.section("Activity by Data Source")
    breakdown = monitoring.source_breakdown(rows)
    if breakdown.empty:
        st.caption("No activity to break down yet.")
    else:
        table = breakdown.rename(
            columns={"source": "Data Source", "predictions": ui.KPI_TRANSACTIONS}
        )
        table["Data Source"] = table["Data Source"].map(ui.pretty_source)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Data Source": st.column_config.TextColumn("Data Source"),
                ui.KPI_TRANSACTIONS: st.column_config.NumberColumn(
                    ui.KPI_TRANSACTIONS, format="%,d"
                ),
            },
        )

    if summary.get("labels_available"):
        ui.section(
            "Model Performance",
            "These metrics require ground-truth labels (actual outcomes). Labels "
            "come from uploaded CSV files and are never used as model inputs.",
        )
        st.caption(
            f"Computed from {ui.fmt_count(summary.get('labelled_count', 0))} labeled "
            "transactions."
        )
        ui.metric_row([
            {"label": ui.PERF_ACCURACY, "value": ui.fmt_percent(summary["accuracy"]),
             "help": ui.HELP_ACCURACY},
            {"label": ui.PERF_PRECISION, "value": ui.fmt_percent(summary["precision"]),
             "help": ui.HELP_PRECISION},
            {"label": ui.PERF_RECALL, "value": ui.fmt_percent(summary["recall"]),
             "help": ui.HELP_RECALL},
            {"label": ui.PERF_F1, "value": ui.fmt_percent(summary["f1"]),
             "help": ui.HELP_F1},
        ])
        with st.expander("Detailed Evaluation Metrics", expanded=False):
            ui.metric_row([
                {"label": "True Positives", "value": ui.fmt_count(summary["tp"]),
                 "help": "Fraudulent transactions correctly detected as fraud."},
                {"label": "False Positives", "value": ui.fmt_count(summary["fp"]),
                 "help": "Legitimate transactions incorrectly flagged as fraud."},
                {"label": "True Negatives", "value": ui.fmt_count(summary["tn"]),
                 "help": "Legitimate transactions correctly identified as legitimate."},
                {"label": "False Negatives", "value": ui.fmt_count(summary["fn"]),
                 "help": "Fraudulent transactions that were not detected."},
            ])
    else:
        st.caption(
            "Not available yet. Actual outcomes come from uploaded CSV files that "
            "include an isFraud column with 0/1 values. Fraud Check predictions and "
            "unlabeled uploads carry no outcomes, so they are excluded."
        )

    st.divider()
    st.caption("Save a dated snapshot of the figures above for reporting.")
    if st.button("Save Monitoring Snapshot", type="secondary"):
        snapshot = monitoring.save_snapshot(summary)
        status = snapshot.get("status")
        if status == "mysql":
            st.success("Monitoring snapshot saved.")
        elif status == "failed":
            st.error("The monitoring snapshot could not be saved.")
        else:
            st.warning(
                "The database is unavailable, so the snapshot was written to a local "
                "fallback file instead."
            )
        ui.technical_details({
            "Snapshot destination": snapshot.get("destination"),
            "Raw response": snapshot.get("message"),
        })

    ui.technical_details({
        "Model version": summary.get("model_version"),
        "Monitoring scope": "Uploaded CSV + Fraud Check"
                            + (" + legacy records" if include_legacy else ""),
        "Prediction rows considered": summary.get("sample_count"),
        "Labeled transactions": summary.get("labelled_count"),
        "Snapshot timestamp": summary.get("timestamp"),
    }, note="Values exactly as recorded, before display formatting.")


# ---------------------------------------------------------------------------
# Page 5 — Data Drift Analysis
# ---------------------------------------------------------------------------
def _drift_value(value) -> str:
    """Reference/Recent cells: numeric means formatted, category shares kept as-is."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return ui.fmt_amount(value)
    return ui.NOT_AVAILABLE if value is None else str(value)


def page_data_drift() -> None:
    ui.page_header(
        "Data Drift Analysis",
        "Compare an inference batch with the reference dataset to identify "
        "distribution changes.",
    )

    if not database.is_available():
        st.warning(
            "Data drift analysis requires saved inference batches, which are stored "
            "in a database that cannot be reached right now."
        )
        return

    try:
        batches = database.fetch_inference_batches(
            sources=(config.BATCH_SOURCE_UPLOADED,)
        )
    except Exception as exc:
        ui.technical_error("Saved inference batches could not be loaded.", exc)
        return

    if not batches:
        ui.empty_state(
            "No inference batch available.",
            "Upload and analyze a CSV file to compare its data against the "
            "reference dataset.",
        )
        return

    options = {_batch_option_label(batch): batch for batch in batches}
    choice = st.selectbox("Recent Inference Data to Compare", list(options))
    batch = options[choice]

    meets_minimum, minimum_message = upload_inference.meets_min_rows(batch["row_count"])
    if not meets_minimum:
        ui.empty_state(
            f"At least {ui.fmt_count(config.MIN_DRIFT_ROWS)} transactions are required "
            "to calculate drift.",
            ui.humanise_fields(minimum_message),
        )
        return

    reference_df = load_reference_frame()
    current_df = upload_inference.load_batch_frame(batch["batch_id"])

    model_features = features.get_feature_order()
    table = drift.compare_distributions(
        reference_df,
        current_df,
        numeric_features=[
            feature for feature in model_features
            if feature not in ("type", "isDestMerchant")
        ],
        categorical_features=["type"],
    )

    ui.metric_row([
        {"label": "Reference Transactions", "value": ui.fmt_count(len(reference_df)),
         "help": "The original model-development dataset, used as the fixed reference."},
        {"label": "Transactions Compared", "value": ui.fmt_count(batch["row_count"])},
        {"label": "Features Compared", "value": ui.fmt_count(len(table))},
    ])

    st.caption(
        f"Reference Data: original model-development dataset "
        f"({ui.fmt_count(len(reference_df))} transactions).  ·  "
        f"Recent Inference Data: batch #{batch['batch_id']} — "
        f"{batch.get('label') or ui.pretty_source(batch.get('source'))}, "
        f"{ui.fmt_count(batch['row_count'])} transactions, "
        f"{ui.fmt_datetime(batch.get('created_at'))}."
    )
    st.caption(
        "Population Stability Index (PSI): measures how much the distribution of a "
        "feature has changed compared with the reference data. A higher score means "
        "the recent data behaves differently from the data the model was developed on. "
        "For Transaction Type, the drift score is the largest category-share change, "
        "not PSI. These are prototype indicators, not production alerting."
    )

    if table.empty:
        st.caption("No comparable features were found for this batch.")
        return

    display = table.copy()
    display["Feature"] = display["Feature"].map(ui.pretty_feature)
    display["Status"] = display["Status"].map(ui.pretty_status)
    display["Reference"] = display["Reference"].map(_drift_value)
    display["Current"] = display["Current"].map(_drift_value)
    display = display.rename(columns={
        "Reference": "Reference Data",
        "Current": "Recent Inference Data",
        "Drift Score": "Drift Score",
    })

    status_counts = display["Status"].value_counts().to_dict()
    st.caption(
        "Feature status summary — "
        + "  ·  ".join(
            f"{label}: {ui.fmt_count(count)}" for label, count in status_counts.items()
        )
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width="medium"),
            "Reference Data": st.column_config.TextColumn(
                "Reference Data", width="small",
                help="Mean value in the reference data (numeric features) or the "
                     "reference majority category and its share (categorical features).",
            ),
            "Recent Inference Data": st.column_config.TextColumn(
                "Recent Inference Data", width="small",
                help="Numeric mean or share of the reference majority category in the selected batch.",
            ),
            "Drift Score": st.column_config.NumberColumn(
                "Drift Score", format="%.4f", width="small",
                help="Numeric features: Population Stability Index (PSI). "
                     "Transaction Type: largest category-share change.",
            ),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
    )

    ui.technical_details({
        "Drift method — numeric features":
            "Population stability index (PSI) with quantile bins derived from the "
            "reference data",
        "Drift method — categorical features":
            "Largest absolute change in category share",
        "No significant drift": f"PSI below {config.PSI_OK}",
        "Possible drift": f"PSI from {config.PSI_OK} up to {config.PSI_WARNING}",
        "Significant drift": f"PSI at or above {config.PSI_WARNING}",
        "Categorical drift threshold": str(config.CATEGORICAL_DRIFT_THRESHOLD),
        "Minimum transactions for comparison": ui.fmt_count(config.MIN_DRIFT_ROWS),
        "Features compared (internal names)": ", ".join(table["Feature"].tolist()),
    }, note="Prototype heuristics documented for this project — they are not "
            "universal industry standards. Drift is reported only: the model is "
            "never retrained or modified.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.sidebar.markdown("### " + ui.APP_TITLE)
    st.sidebar.caption(ui.APP_SUBTITLE)

    # Startup initialization (safe): never crashes Streamlit when the database is down.
    _ensure_db_initialized()

    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    with st.sidebar:
        ui.technical_details({
            "Model version": config.MODEL_VERSION,
            "Database connection":
                "Available" if database.is_available() else "Unavailable",
            "Upload limits":
                f"{config.UPLOAD_MAX_BYTES // (1024 * 1024)} MB / "
                f"{ui.fmt_count(config.UPLOAD_MAX_ROWS)} transactions",
            "Minimum transactions for drift": ui.fmt_count(config.MIN_DRIFT_ROWS),
        }, note="Implementation details, provided for demonstration and review.")

    if page == "Fraud Check":
        page_fraud_check()
    elif page == "Upload Transaction Data":
        page_upload()
    elif page == "Transaction History":
        page_transaction_history()
    elif page == "Monitoring":
        page_monitoring()
    else:
        page_data_drift()


main()
