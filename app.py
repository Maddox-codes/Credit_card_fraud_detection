
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
        st.error(f"Model file not found. Please ensure '{filename}' is in the same folder as the app script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model(MODEL_FILENAME)

ALL_EXPECTED_FEATURES = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'] + [f'dummy_V{i}' for i in range(29, 34)]

st.set_page_config(layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.write(
    "This application uses a pre-trained machine learning model to predict "
    "whether a credit card transaction is fraudulent. Fill in the transaction details below to get a prediction."
)

st.header("Enter Transaction Details")

with st.form("transaction_form"):
    input_data = {}
    
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    feature_index = 0
    for feature in ALL_EXPECTED_FEATURES:
        if 'dummy' in feature:
            input_data[feature] = 0.0
            continue

        target_col = columns[feature_index % 3]

        label = feature
        help_text = f"Anonymized feature: {feature}"
        if feature == 'Time':
            label = "Time"
            help_text = "Seconds elapsed between this transaction and the first transaction in the dataset."
        elif feature == 'Amount':
            label = "Transaction Amount"
            help_text = "The monetary value of the transaction."

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

        with st.expander("View Input Data"):
            st.dataframe(input_df)

    except KeyError as e:
        st.error(f"Error: A feature mismatch occurred. The model expected a feature that was not provided. Details: {e}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
elif submitted and model is None:
    st.warning("Cannot perform prediction because the model is not loaded.")

st.info(
    """
    **Disclaimer:** This is a simulator and should not be used for actual financial decisions.
    The accuracy of the prediction depends entirely on the quality and characteristics of the model it was trained on.
    """,
    icon="ℹ️"
)

st.subheader("Beyond Simulation: Real-Time Fraud Detection")
st.markdown("""
This application simulates a scenario where you manually input transaction details for a prediction. In a real-world setting, fraud detection systems are fully automated and operate in real-time. Here's how they typically work:

* **Real-time Processing:** Transactions are analyzed instantly as they occur, without manual data entry.
* **Message Queues:** High-volume systems use message queues (like Apache Kafka or RabbitMQ) to handle the massive stream of incoming transactions reliably.
* **Low-Latency Models:** The machine learning models are highly optimized for speed to provide decisions in milliseconds.
* **Feedback Loop:** The system continuously improves. When a transaction is flagged and a customer confirms it was legitimate (or vice-versa), this feedback is used to retrain and fine-tune the model, making it more accurate over time.
* **Automated Actions:** Based on the real-time prediction, the system can trigger alerts to a fraud analyst or take immediate action, such as temporarily blocking the transaction and notifying the customer.
""")
