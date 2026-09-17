# Fraud Detection for Mobile Financial Transactions

## What this project does

This project detects potentially fraudulent mobile-money transactions using machine learning.

A transaction is passed to the model and the system returns:

- fraud probability
- fraud / legitimate decision
- prediction threshold

The application also stores transactions and predictions in MySQL.

## Why I built it

Fraud can cause serious financial losses for customers and providers. The goal is to detect suspicious transactions quickly while keeping false alarms manageable — flagging too many legitimate transactions creates extra costs and frustrates users.

## What I built

- Data analysis of a synthetic mobile-money transaction dataset
- Feature engineering
- A fraud detection model based on XGBoost
- Experiments with imbalanced data (most transactions are legitimate)
- Model evaluation
- Threshold analysis
- Financial impact analysis
- A Streamlit web application
- MySQL transaction storage
- Prediction logging for every scored transaction
- Uploaded Inference Data: score any uploaded PaySim-style CSV with the frozen saved model
- Inference history: every scored upload batch is persisted with a batch ID (Historical Data)
- Cleanup: delete a single inference batch, or the legacy prediction rows, from the Historical Data tab
- Monitoring of actual inference activity (batch-aware, with labeled performance when ground truth exists)
- Drift analysis: fixed training reference vs the latest scored inference batch
- Automated tests

## Model

I compared several models and selected XGBoost. The final model uses a saved preprocessing pipeline and a saved XGBoost model, both loaded from the `artifacts/` folder. The application does NOT retrain the model when making predictions — it only runs the saved pipeline.

## Results

These are the results from the finalized analysis notebook, evaluated on the synthetic test set:

| Metric | Value |
| --- | ---: |
| Accuracy | 99.28% |
| Precision | 94.54% |
| Recall | 98.68% |
| F1 | 96.57% |
| ROC-AUC | 99.91% |
| PR-AUC | 99.64% |

In simple terms:

- **Precision (94.54%)**: of all transactions the model flagged as fraud, 94.54% were actually fraud.
- **Recall (98.68%)**: of all real fraud transactions, the model caught 98.68%.

The dataset is synthetic and PaySim-like, so these results should NOT be treated as real-world production fraud performance. Real banking data may behave differently.

## Application flow

```
Single transaction (Fraud Check)
→ feature preparation
→ saved preprocessing pipeline
→ XGBoost model
→ fraud probability
→ fraud / legitimate decision
→ MySQL prediction log (or JSONL fallback when MySQL is unavailable)

Uploaded CSV (Batch Scoring → Uploaded Inference Data)
→ validate columns/rows/labels
→ exact-record overlap check vs the original dataset
→ saved preprocessing pipeline (transform only)
→ XGBoost model (predict only)
→ persisted as an inference batch (Historical Data, Monitoring, Drift)
→ optional evaluation when the CSV contains isFraud (used AFTER prediction only)
```

## Inference history, Monitoring, and Drift

- **MySQL Historical Data** now shows *previously scored inference activity*: every
  scored uploaded CSV becomes an inference batch (batch ID, source, filename, row
  count, fraud count, labeled status) with its predictions stored in
  `prediction_logs`. The original `transactions` table is never re-scored or modified.
- **Monitoring** shows what the model has actually been doing: prediction totals,
  fraud counts, average probability, the latest batch, volume over time, and a
  source/batch breakdown. Old pre-redesign test records are preserved but hidden
  behind an "Include legacy records" toggle.
- **Drift** compares the fixed training reference (the full original dataset) against
  the latest/selected scored inference batch — never another slice of the training
  data. If ground truth (`isFraud`) is available in a batch, it is stored per
  prediction (`actual_label`, after prediction only) and Monitoring shows
  accuracy/precision/recall/F1 and the confusion matrix.
- Old smoke-test/fallback records are **preserved by default**; they are classified
  as legacy (no inference batch) and hidden behind the "Include legacy records"
  toggle. They are removed only if you explicitly use **Delete legacy records**.

## Uploaded Inference Data

Batch Scoring has two clearly separated modes:

1. **MySQL Historical Data** — shows *previously scored inference activity*. Every
   successfully scored upload is persisted as an inference batch, and this tab
   lists those batches (newest first) together with their stored predictions.
   The original `transactions` table (the 11,142-row source/training dataset) is
   never re-scored and is never modified.
2. **Uploaded Inference Data** — you upload any PaySim-style CSV and every row
   is scored one at a time through the same frozen saved model and
   preprocessing pipeline. On success the batch is persisted automatically with
   a batch ID — there is no manual "Log Predictions" step for uploads. Each new
   upload replaces the previous active upload; re-uploading the same file (or
   the same records in a different row order) reuses the existing batch instead
   of creating a duplicate.

### Cleanup (deleting test data)

- **Delete this batch** (MySQL Historical Data tab) removes one inference batch
  together with its own predictions, in a single database transaction (commit
  on success, rollback on failure). Other batches, the model, and the source
  dataset are not affected. A confirmation is always required first.
- **Delete legacy records** removes only prediction rows that have no inference
  batch (`batch_id IS NULL` — pre-redesign/test data). A normal batch delete
  never touches legacy rows, and legacy deletion never touches inference batches.
- Deletes are always scoped to the requested `batch_id` using parameterized
  queries — there are no broad "delete everything" statements.

### Required CSV columns

```
type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, nameDest
```

### Optional source columns (allowed, never used as model inputs)

```
step, nameOrig, isFraud, transaction_id
```

`isFraud` (when present) is held completely out of the model. It is used only
**after** prediction to show confusion-matrix metrics (TP/TN/FP/FN, accuracy,
precision, recall, F1). Unlabeled CSVs show predictions only — labels are never
invented.

Unexpected columns are rejected, never silently dropped. One invalid row
rejects the whole file with a row-level error report.

### Exact-overlap note

Uploaded rows are compared (by a deterministic canonical hash of the seven raw
inference fields) against the original 11,142-row model-development dataset.
The UI reports "X of Y uploaded records exactly match records in the original
dataset" (or "No exact duplicate records found"). **This detects exact
duplicate records only — it does not prove that the data is "unseen" or
statistically/distributionally novel.** Uploaded data is therefore called
"Uploaded Inference Data", never automatically "unseen data".

### Limits and independence

- Demo-oriented upload limits: **10 MB** and **5,000 rows**.
- Inference uses only the saved artifacts (preprocessor + XGBoost + threshold
  **0.44**), so uploaded inference works **even when MySQL is unavailable**.
- MySQL is only involved if you click **Log Predictions** — and even then the
  existing JSONL fallback is used when MySQL is down.
- Scoring runs the canonical per-transaction `predict_raw()` path, so very
  large uploads take longer; a progress bar is shown while scoring.

## Tech used

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- MySQL
- SQLAlchemy
- PyMySQL
- Pytest

## Project structure

```
artifacts/                    saved model, preprocessor, metadata
src/                          application code (features, inference, database, upload inference)
scripts/                      CSV → MySQL loader
tests/                        automated tests
app.py                        Streamlit application
ui_presentation.py            Streamlit UI helpers (pages, metrics, formatting)
fraud_capstone_final.ipynb    analysis and model training notebook
Fraud_Analysis_Dataset.csv    synthetic dataset
requirements.txt              Python dependencies
```

## Run locally

1. Create and activate a Python environment.
2. Install the requirements:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`:

   ```bash
   copy .env.example .env        # Windows
   cp .env.example .env          # macOS / Linux
   ```

4. Open `.env` and enter your local MySQL password.
5. Make sure MySQL is running locally.
6. Load the dataset into MySQL:

   ```bash
   python scripts/load_csv_to_mysql.py
   ```

7. Start the application:

   ```bash
   streamlit run app.py
   ```

## MySQL

Database: `bia_fraud_detection`

Tables:

- `transactions` — the original 11,142-row source/training dataset. Read-only for the application; never used as inference history.
- `inference_batches` — one row per scored inference batch (source, filename, content/dataset hashes, row and fraud counts, labeled flag). Uploaded batches are persisted here automatically.
- `prediction_logs` — stores one row per prediction (probability, decision, features, latency), linked to its batch via `batch_id`, with `actual_label` when the uploaded CSV provided ground truth.
- `model_monitoring` — stores monitoring snapshots of the prediction logs.

If MySQL is unavailable, predictions are still logged locally to a JSONL fallback file, so no prediction is ever lost silently.

## Testing

The automated test suite passes (81 tests: the original suite plus tests for
inference batches, legacy scoping, labeled evaluation, drift, the snapshot fix,
and safe batch/legacy deletion).

```bash
python -m pytest -q
```

## Limitations

- The dataset is synthetic; real banking data may behave differently.
- This is a production-style prototype, not a live banking fraud system.
- No automatic retraining.
- Monitoring is basic.
- A scored upload is persisted as an inference batch (inference_batches +
  prediction_logs); the parsed file itself is session-scoped. Inference
  history therefore needs MySQL, and batches can be deleted from the
  Historical Data page.
- The exact-overlap check detects exact duplicate records only; it does not
  prove that uploaded data is statistically or distributionally novel.
- The financial impact analysis uses illustrative assumptions.
- The saved prediction artifact uses a different threshold (0.44) than the analytical notebook's selected threshold (0.01), so live predictions may differ slightly from the notebook's reported results.

## Future improvements

- Real transaction data
- API deployment
- Authentication
- Stronger monitoring and alerting
- Model retraining workflow
- Cloud deployment