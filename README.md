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
- Batch scoring of recent transactions
- Basic monitoring of prediction logs
- Basic drift analysis of incoming data
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
Transaction
→ feature preparation
→ saved preprocessing pipeline
→ XGBoost model
→ fraud probability
→ fraud / legitimate decision
→ MySQL prediction log
```

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
src/                          application code (features, inference, database)
scripts/                      CSV → MySQL loader
tests/                        automated tests
app.py                        Streamlit application
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

- `transactions` — stores the raw transaction rows loaded from the dataset.
- `prediction_logs` — stores one row per prediction made by the application (probability, decision, features, latency, and more).
- `model_monitoring` — stores monitoring snapshots of the prediction logs.

If MySQL is unavailable, predictions are still logged locally to a JSONL fallback file, so no prediction is ever lost silently.

## Testing

The automated test suite passes (18 tests passed during validation).

```bash
python -m pytest -q
```

## Limitations

- The dataset is synthetic; real banking data may behave differently.
- This is a production-style prototype, not a live banking fraud system.
- No automatic retraining.
- Monitoring is basic.
- The financial impact analysis uses illustrative assumptions.
- The saved prediction artifact uses a different threshold (0.44) than the analytical notebook's selected threshold (0.01), so live predictions may differ slightly from the notebook's reported results.

## Future improvements

- Real transaction data
- API deployment
- Authentication
- Stronger monitoring and alerting
- Model retraining workflow
- Cloud deployment