from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "deployment" / "tourism_model.joblib"
TRAIN_DATA_PATH = PROJECT_DIR / "artifacts" / "train.csv"
TARGET_COLUMN = "ProdTaken"

# Load model and feature names
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TRAIN_DATA_PATH.exists():
    raise FileNotFoundError(f"Training data not found: {TRAIN_DATA_PATH}")

model = joblib.load(MODEL_PATH)

train_sample = pd.read_csv(TRAIN_DATA_PATH, nrows=1)
FEATURES = [
    column for column in train_sample.columns
    if column != TARGET_COLUMN
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Tourism Prediction API is running",
        "endpoints": ["/health", "/predict"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "number_of_features": len(FEATURES)
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Send a valid JSON object."
        }), 400

    # Accept either a single record or multiple records
    if "instances" in payload:
        records = payload["instances"]
    elif "data" in payload:
        records = [payload["data"]]
    else:
        return jsonify({
            "error": "Use {'data': {...}} or {'instances': [{...}]}."
        }), 400

    if not isinstance(records, list) or not records:
        return jsonify({
            "error": "'instances' must be a non-empty list."
        }), 400

    if not all(isinstance(record, dict) for record in records):
        return jsonify({
            "error": "Each input record must be a JSON object."
        }), 400

    required_missing = [
        feature for feature in FEATURES
        if any(feature not in record for record in records)
    ]

    if required_missing:
        return jsonify({
            "error": "Some required features are missing.",
            "missing_features": required_missing
        }), 400

    try:
        input_df = pd.DataFrame(records, columns=FEATURES)

        predictions = model.predict(input_df).tolist()

        results = []
        for index, prediction in enumerate(predictions):
            result = {
                "prediction": int(prediction),
                "interpretation": (
                    "Customer is predicted to purchase the package"
                    if int(prediction) == 1
                    else "Customer is predicted not to purchase the package"
                )
            }

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    input_df.iloc[[index]]
                )[0]
                classes = model.classes_
                result["probabilities"] = {
                    str(label): round(float(probability), 4)
                    for label, probability in zip(classes, probabilities)
                }

            results.append(result)

        return jsonify({"predictions": results})

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": "Unable to process the supplied input.",
            "details": str(error)
        }), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
