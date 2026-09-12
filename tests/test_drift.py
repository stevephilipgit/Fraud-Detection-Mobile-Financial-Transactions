import pandas as pd

from src import drift


def test_identical_numeric_distribution_has_no_drift():
    series = pd.Series([1, 2, 3, 4, 5, 6])

    assert drift.psi_numeric(series, series) == 0.0


def test_changed_numeric_distribution_has_positive_drift():
    reference = pd.Series([1, 2, 3, 4, 5, 6])
    current = pd.Series([100, 101, 102, 103, 104, 105])

    assert drift.psi_numeric(reference, current) > 0.0


def test_categorical_drift_detects_share_change():
    reference = pd.Series(["PAYMENT", "PAYMENT", "TRANSFER", "CASH_OUT"])
    current = pd.Series(["TRANSFER", "TRANSFER", "TRANSFER", "CASH_OUT"])

    result = drift.categorical_drift(reference, current)

    assert result["score"] >= 0.25
    assert result["shares"]["PAYMENT"]["current"] == 0.0


def test_compare_distributions_returns_status_table():
    reference = pd.DataFrame({"amount": [10, 20, 30, 40], "type": ["PAYMENT", "PAYMENT", "TRANSFER", "CASH_OUT"]})
    current = pd.DataFrame({"amount": [10, 20, 300, 400], "type": ["TRANSFER", "TRANSFER", "TRANSFER", "CASH_OUT"]})

    table = drift.compare_distributions(reference, current, numeric_features=["amount"], categorical_features=["type"])

    assert list(table["Feature"]) == ["amount", "type"]
    assert set(table["Status"]).issubset({"OK", "WARNING", "DRIFT"})
