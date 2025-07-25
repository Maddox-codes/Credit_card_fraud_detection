import streamlit as st
import joblib
import pandas as pd

MODEL_FILENAME = "best_fraud_detection_model.joblib"

@st.cache_resource
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found. Ensure '{filename}' is in the same folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model(MODEL_FILENAME)

ALL_EXPECTED_FEATURES = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'] + [f'dummy_V{i}' for i in range(29, 34)]

FEATURE_LABELS = {
    'Time': ("Time Since First Transaction", "Enter the seconds elapsed between this transaction and the first one in the dataset."),
    'Amount': ("Transaction Amount", "Enter the monetary value of the transaction (e.g., 129.99)."),
    'V1': ("Transaction Profile 1", "Anonymized feature representing a core aspect of the transaction's profile."),
    'V2': ("Transaction Profile 2", "Anonymized feature representing a core aspect of the transaction's profile."),
    'V3': ("Transaction Profile 3", "Anonymized feature representing a core aspect of the transaction's profile."),
    'V4': ("Behavioral Indicator 1", "Anonymized feature potentially indicating a deviation from normal spending behavior."),
    'V5': ("Transaction Profile 4", "Anonymized feature representing a core aspect of the transaction's profile."),
    'V6': ("Transaction Profile 5", "Anonymized feature representing a core aspect of the transaction's profile."),
    'V7': ("Transaction Profile 6", "Anonymized feature representing a core aspect of the transaction's profile."),
    'V8': ("Contextual Data 1", "Anonymized feature related to the transaction's context."),
    'V9': ("Transaction Profile 7", "Anonymized feature representing a core aspect of the transaction's profile."),
    'V10': ("Behavioral Indicator 2", "Anonymized feature potentially indicating a deviation from normal spending behavior."),
    'V11': ("Behavioral Indicator 3", "Anonymized feature potentially indicating a deviation from normal spending behavior."),
    'V12': ("Behavioral Indicator 4", "Anonymized feature potentially indicating a deviation from normal spending behavior."),
    'V13': ("Contextual Data 2", "Anonymized feature related to the transaction's context."),
    'V14': ("Behavioral Indicator 5", "Anonymized feature potentially indicating a deviation from normal spending behavior."),
    'V15': ("Contextual Data 3", "Anonymized feature related to the transaction's context."),
    'V16': ("Behavioral Indicator 6", "Anonymized feature potentially indicating a deviation from normal spending behavior."),
    'V17': ("Behavioral Indicator 7", "Anonymized feature potentially indicating a deviation from normal spending behavior."),
    'V18': ("Behavioral Indicator 8", "Anonymized feature potentially indicating a deviation from normal spending behavior."),
    'V19': ("Contextual Data 4", "Anonymized feature related to the transaction's context."),
    'V20': ("Contextual Data 5", "Anonymized feature related to the transaction's context."),
    'V21': ("Contextual Data 6", "Anonymized feature related to the transaction's context."),
    'V22': ("Internal Risk Factor 1", "Anonymized feature likely related to an internal risk assessment score."),
    'V23': ("Internal Risk Factor 2", "Anonymized feature likely related to an internal risk assessment score."),
    'V24': ("Internal Risk Factor 3", "Anonymized feature likely related to an internal risk assessment score."),
    'V25': ("Internal Risk Factor 4", "Anonymized feature likely related to an internal risk assessment score."),
    'V26': ("Internal Risk Factor 5", "Anonymized feature likely related to an internal risk assessment score."),
    'V27': ("Internal Risk Factor 6", "Anonymized feature likely related to an internal risk assessment score."),
    'V28': ("Internal Risk Factor 7", "Anonymized feature likely related to an internal risk assessment score."),
}

st.set_page_config(layout="wide")

st.title("💳 Credit Card Fraud Detection Simulator")
st.write(
    "This application uses a machine learning model to predict if a transaction is fraudulent. "
    "Fill in the details below to simulate a prediction."
)

st.header("Enter Transaction Details")
st.info(
    "**Note:** Most features below (e.g., 'Transaction Profile') are anonymized for privacy reasons. "
    "The labels are illustrative placeholders to make the form easier to use.",
    icon="ℹ️"
)

with st.form("transaction_form"):
    input_data = {}
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    feature_index = 0

    for feature in ALL_EXPECTED_FEATURES:
        if 'dummy' in feature:
            input_data[feature] = 0.0
            continue

        label, help_text = FEATURE_LABELS.get(feature, (feature, f"Anonymized feature: {feature}"))
        target_col = columns[feature_index % 3]

        input_data[feature] = target_col.number_input(
            label=label,
            value=0.0,
            format="%.4f",
            key=feature,
            help=help_text
        )
        feature_index += 1

    submitted = st.form_submit_button("Predict Fraud")

if submitted and model is not None:
    input_df = pd.DataFrame([input_data])
    try:
        input_df = input_df[ALL_EXPECTED_FEATURES]
        prediction = model.predict(input_df)
        fraud_confidence = None

        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(input_df)
            fraud_confidence = prediction_proba[0][1]

        st.header("Prediction Result")
        if prediction[0] == 1:
            st.error("🚨 Prediction: Fraudulent Transaction")
            if fraud_confidence is not None:
                st.metric(label="Confidence Score (Fraud)", value=f"{fraud_confidence:.2%}")
        else:
            st.success("✅ Prediction: Not a Fraudulent Transaction")
            if fraud_confidence is not None:
                st.metric(label="Confidence Score (Not Fraud)", value=f"{prediction_proba[0][0]:.2%}")

        with st.expander("View Raw Input Data"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
elif submitted and model is None:
    st.warning("Cannot perform prediction because the model is not loaded.")

st.subheader("Beyond Simulation: How Real-Time Fraud Detection Works")
st.markdown("""
In a real-world setting, fraud detection systems are fully automated and operate in milliseconds. Here's a simplified overview:
* **Real-time Processing:** Transactions are analyzed instantly as they occur.
* **Automated Feature Engineering:** The system automatically generates features (like spending frequency, location changes, etc.) from raw transaction data.
* **Low-Latency Models:** The machine learning models are highly optimized for speed to provide decisions in real-time.
* **Feedback Loop:** The system continuously learns. When a transaction is confirmed as fraud (or not), this feedback is used to retrain and improve the model.
""")
