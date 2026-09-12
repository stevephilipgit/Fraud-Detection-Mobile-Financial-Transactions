import numpy as np

from src import features, inference


def test_inference_matches_direct_artifact_pipeline():
    raw = {
        "type": "TRANSFER",
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameDest": "C123456789",
    }
    detector = inference.FraudDetector().load()

    app_result = detector.predict_raw(raw)
    model_input = features.prepare_single(raw)
    encoded = detector.preprocessor.transform(model_input)
    direct_probability = float(detector.model.predict_proba(encoded)[:, 1][0])
    direct_prediction = int(direct_probability >= detector.threshold)

    assert np.isclose(app_result["fraud_probability"], direct_probability, rtol=1e-6, atol=1e-8)
    assert app_result["prediction"] == direct_prediction
