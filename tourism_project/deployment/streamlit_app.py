from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Project paths and model loading
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "deployment" / "tourism_model.joblib"
TRAIN_PATH = PROJECT_DIR / "artifacts" / "train.csv"
TARGET = "ProdTaken"

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Tourism Package Purchase Prediction")
st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase the tourism package."
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_training_data(path):
    return pd.read_csv(path)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Run your model training script first.")
    st.stop()

if not TRAIN_PATH.exists():
    st.error(f"Training dataset not found: {TRAIN_PATH}")
    st.stop()

model = load_model(str(MODEL_PATH))
train_df = load_training_data(str(TRAIN_PATH))

if TARGET not in train_df.columns:
    st.error(f"Target column '{TARGET}' was not found in train.csv.")
    st.stop()

# Keep the exact feature order used by the training dataset.
FEATURES = [col for col in train_df.columns if col != TARGET]

st.subheader("Customer Information")

# --------------------------------------------------
# 2. Build input fields from training data
# --------------------------------------------------
user_input = {}

with st.form("tourism_prediction_form"):
    columns = st.columns(2)

    for index, feature in enumerate(FEATURES):
        series = train_df[feature].dropna()
        column = columns[index % 2]

        with column:
            if series.empty:
                st.warning(f"No training values available for {feature}.")
                user_input[feature] = None

            elif (
                pd.api.types.is_object_dtype(series.dtype)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(series.dtype)
            ):
                options = list(pd.unique(series))
                user_input[feature] = st.selectbox(
                    feature,
                    options=options,
                    format_func=str,
                    key=f"input_{feature}"
                )

            elif pd.api.types.is_integer_dtype(series.dtype):
                minimum = int(series.min())
                maximum = int(series.max())
                default = int(series.median())

                user_input[feature] = st.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=max(minimum, min(default, maximum)),
                    step=1,
                    key=f"input_{feature}"
                )

            else:
                minimum = float(series.min())
                maximum = float(series.max())
                default = float(series.median())

                user_input[feature] = st.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=max(minimum, min(default, maximum)),
                    key=f"input_{feature}"
                )

    submitted = st.form_submit_button(
        "Predict Purchase",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 3. Make prediction
# --------------------------------------------------
if submitted:
    input_df = pd.DataFrame([user_input], columns=FEATURES)

    try:
        prediction = int(model.predict(input_df)[0])

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Prediction: Customer is likely to purchase the package.")
        else:
            st.info("Prediction: Customer is unlikely to purchase the package.")

        st.metric("Predicted class", prediction)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            probability_df = pd.DataFrame({
                "Class": [str(c) for c in classes],
                "Probability": [float(p) for p in probabilities]
            })

            st.write("Prediction probabilities")
            st.bar_chart(
                probability_df.set_index("Class")["Probability"]
            )

            for class_label, probability in zip(classes, probabilities):
                if int(class_label) == 1:
                    st.metric(
                        "Estimated purchase probability",
                        f"{probability:.1%}"
                    )
                    break

        with st.expander("View submitted customer details"):
            st.dataframe(input_df, use_container_width=True)

        st.caption(
            "Predictions are estimates from the trained model, "
            "not guarantees of customer behavior."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
