import numpy as np
import pandas as pd

from src import features


def test_engineer_features_transfer_non_merchant():
    df = pd.DataFrame(
        [
            {
                "type": "TRANSFER",
                "amount": 1000.0,
                "oldbalanceOrg": 5000.0,
                "newbalanceOrig": 3500.0,
                "oldbalanceDest": 200.0,
                "newbalanceDest": 900.0,
                "nameDest": "C123",
            }
        ]
    )

    engineered = features.engineer_features(df)

    assert engineered.loc[0, "log_amount"] == np.log1p(1000.0)
    assert engineered.loc[0, "errorBalanceOrig"] == 500.0
    assert engineered.loc[0, "errorBalanceDest"] == 300.0
    assert engineered.loc[0, "isDestMerchant"] == 0


def test_engineer_features_cash_out_merchant():
    df = pd.DataFrame(
        [
            {
                "type": "CASH_OUT",
                "amount": 250.0,
                "oldbalanceOrg": 300.0,
                "newbalanceOrig": 50.0,
                "oldbalanceDest": 1000.0,
                "newbalanceDest": 1250.0,
                "nameDest": "M987",
            }
        ]
    )

    engineered = features.engineer_features(df)

    assert engineered.loc[0, "errorBalanceOrig"] == 0.0
    assert engineered.loc[0, "errorBalanceDest"] == 0.0
    assert engineered.loc[0, "isDestMerchant"] == 1


def test_prepare_single_expected_columns():
    raw = {
        "type": "payment",
        "amount": 25,
        "oldbalanceOrg": 100,
        "newbalanceOrig": 75,
        "oldbalanceDest": 0,
        "newbalanceDest": 0,
        "nameDest": "M111",
    }

    prepared = features.prepare_single(raw)

    assert list(prepared.columns) == features.get_feature_order()
    assert prepared.loc[0, "type"] == "PAYMENT"
    assert prepared.loc[0, "isDestMerchant"] == 1


def test_validate_transaction_rejects_invalid_values():
    errors = features.validate_transaction(
        {
            "type": "WIRE",
            "amount": -1,
            "oldbalanceOrg": 100,
            "newbalanceOrig": 100,
            "oldbalanceDest": 0,
            "newbalanceDest": 0,
            "nameDest": "",
        }
    )

    assert any("amount" in error for error in errors)
    assert any("type" in error for error in errors)
    assert any("nameDest" in error for error in errors)
