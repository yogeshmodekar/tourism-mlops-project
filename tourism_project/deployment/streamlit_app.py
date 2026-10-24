import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Tourism Package Purchase Prediction")
st.write(
    "Enter customer details to estimate whether a customer "
    "is likely to purchase the tourism package."
)

# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "deployment" / "tourism_model.joblib"
TRAIN_PATH = PROJECT_DIR / "artifacts" / "train.csv"
TARGET = "ProdTaken"


@st.cache_resource
def load_model(path, modified_time):
    """Load the trained model."""
    return joblib.load(path)


@st.cache_data
def load_training_data(path, modified_time):
    """Load training data used to create the input fields."""
    return pd.read_csv(path)


# --------------------------------------------------
# 3. Load model and training data
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Run your training script to generate the model.")
    st.stop()

if not TRAIN_PATH.exists():
    st.error(f"Training data not found: {TRAIN_PATH}")
    st.info("Run your data preparation script first.")
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

    train_df = load_training_data(
        str(TRAIN_PATH),
        TRAIN_PATH.stat().st_mtime,
    )

except Exception as error:
    st.error(f"Unable to load the model or training data: {error}")
    st.stop()

if TARGET not in train_df.columns:
    st.error(f"Target column '{TARGET}' was not found in train.csv.")
    st.stop()

# Preserve the training feature order.
FEATURES = [column for column in train_df.columns if column != TARGET]

if not FEATURES:
    st.error("No model input features were found in train.csv.")
    st.stop()

# If the model pipeline exposes its expected feature names,
# verify that the training columns match them.
expected_features = getattr(model, "feature_names_in_", None)

if expected_features is not None:
    expected_features = list(expected_features)

    if set(FEATURES) != set(expected_features):
        st.error(
            "The training CSV columns do not match the model's "
            "expected input features. Please regenerate the training "
            "data and model using the same preprocessing pipeline."
        )
        st.write("CSV features:", FEATURES)
        st.write("Model features:", expected_features)
        st.stop()

# --------------------------------------------------
# 4. Create customer input form
# --------------------------------------------------
st.subheader("Customer Information")
st.caption("Choose realistic values based on your dataset.")

user_input = {}

with st.form("tourism_prediction_form"):
    columns = st.columns(2)

    for index, feature in enumerate(FEATURES):
        series = train_df[feature].dropna()
        input_column = columns[index % 2]

        with input_column:
            if series.empty:
                st.warning(f"No available training values for {feature}.")
                user_input[feature] = None
                continue

            # Boolean columns
            if pd.api.types.is_bool_dtype(series.dtype):
                options = list(pd.unique(series))
                user_input[feature] = st.selectbox(
                    feature,
                    options=options,
                    format_func=str,
                    key=f"input_{feature}",
                )

            # Numeric columns
            elif pd.api.types.is_numeric_dtype(series.dtype):
                numeric_series = pd.to_numeric(
                    series,
                    errors="coerce",
                ).dropna()

                if numeric_series.empty:
                    st.warning(
                        f"Cannot determine numeric values for {feature}."
                    )
                    user_input[feature] = None
                    continue

                minimum = float(numeric_series.min())
                maximum = float(numeric_series.max())
                default = float(numeric_series.median())

                if not all(
                    np.isfinite(value)
                    for value in (minimum, maximum, default)
                ):
                    st.warning(
                        f"Invalid numeric range for {feature}."
                    )
                    user_input[feature] = None
                    continue

                # Integer fields
                if pd.api.types.is_integer_dtype(numeric_series.dtype):
                    minimum_int = int(minimum)
                    maximum_int = int(maximum)
                    default_int = int(round(default))

                    user_input[feature] = st.number_input(
                        feature,
                        min_value=minimum_int,
                        max_value=maximum_int,
                        value=max(
                            minimum_int,
                            min(default_int, maximum_int),
                        ),
                        step=1,
                        key=f"input_{feature}",
                    )

                # Floating-point fields
                else:
                    user_input[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=max(minimum, min(default, maximum)),
                        key=f"input_{feature}",
                    )

            # Text, categorical, or mixed-type columns
            else:
                options = list(pd.unique(series))

                user_input[feature] = st.selectbox(
                    feature,
                    options=options,
                    format_func=str,
                    key=f"input_{feature}",
                )

    submitted = st.form_submit_button(
        "Predict Purchase",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# 5. Predict customer purchase outcome
# --------------------------------------------------
if submitted:
    input_df = pd.DataFrame([user_input], columns=FEATURES)

    # Convert input columns to match their training data types.
    for feature in FEATURES:
        if pd.api.types.is_numeric_dtype(train_df[feature].dtype):
            input_df[feature] = pd.to_numeric(
                input_df[feature],
                errors="coerce",
            )

    try:
        prediction = model.predict(input_df)[0]
        prediction_int = int(prediction)

        st.divider()
        st.subheader("Prediction Result")

        if prediction_int == 1:
            st.success(
                "Customer is predicted to purchase the package."
            )
        else:
            st.info(
                "Customer is predicted not to purchase the package."
            )

        st.metric("Predicted class", prediction_int)

        # Show class probabilities when supported by the model.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            probability_df = pd.DataFrame(
                {
                    "Class": [str(label) for label in classes],
                    "Probability": [
                        float(probability)
                        for probability in probabilities
                    ],
                }
            )

            st.subheader("Prediction Probabilities")

            st.bar_chart(
                probability_df.set_index("Class")["Probability"]
            )

            purchase_indices = [
                index
                for index, label in enumerate(classes)
                if str(label) == "1"
            ]

            if purchase_indices:
                purchase_probability = float(
                    probabilities[purchase_indices[0]]
                )

                st.metric(
                    "Estimated purchase probability",
                    f"{purchase_probability:.1%}",
                )

        with st.expander("View submitted customer details"):
            st.dataframe(input_df, use_container_width=True)

        st.caption(
            "Predictions are estimates from the trained model, "
            "not guarantees of customer behavior."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.caption(
            "Check that the model was trained with the same input "
            "features and preprocessing represented in train.csv."
        )
