
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Cyber Risk Prediction",
    page_icon="🔐",
    layout="centered"
)

# Load saved model and preprocessor
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_artifacts()

# Application title
st.title("🔐 Cybersecurity Awareness Prediction")

st.write(
    "Enter your digital behaviour details to predict "
    "your cybersecurity awareness level."
)

st.divider()

# User input form
with st.form("prediction_form"):

    st.subheader("User Behaviour Information")

    age_group = st.selectbox(
        "Age Group",
        ["18-25", "26-35", "36-45", "46-55", "56"]
    )

    social_media_hours = st.selectbox(
        "Daily Social Media Usage",
        [
            "Less than 1 hour",
            "1-2 hours",
            "2-4 hours",
            "4-6 hours",
            "More than 6 hours",
            "More than 5 hours"
        ]
    )

    social_media_scam_exposure = st.selectbox(
        "Social Media Scam Exposure",
        ["Never", "Once", "2-3 times", "4-5 times", "More than 5 times"]
    )

    personal_fraud_loss = st.selectbox(
        "Experienced Personal Social Media Fraud Loss?",
        ["No", "Yes"]
    )

    transaction_fraud_loss = st.selectbox(
        "Experienced Transaction Fraud Loss?",
        ["No", "Yes"]
    )

    otp_exposure = st.selectbox(
        "OTP / Password Request Exposure",
        ["Never", "Once", "2-3 times", "4-5 times", "More than 5 times"]
    )

    suspicious_link_clicks = st.selectbox(
        "Suspicious Links Clicked",
        ["Never", "Once", "2-3 times", "4-5 times", "More than 5 times"]
    )

    submitted = st.form_submit_button(
        "Predict Cybersecurity Awareness",
        use_container_width=True
    )


# Prediction
if submitted:

    input_data = pd.DataFrame({
        "Age_Group": [age_group],
        "Social_Media_Hours": [social_media_hours],
        "Social_Media_Scam_Exposure": [social_media_scam_exposure],
        "Personal_Social_Media_Fraud_Loss": [personal_fraud_loss],
        "Transaction_Fraud_Loss": [transaction_fraud_loss],
        "OTP_Password_Request_Exposure": [otp_exposure],
        "Suspicious_Link_Clicks": [suspicious_link_clicks]
    })

    try:
        # Apply manual categorical encoding
        encoded_data = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(encoded_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 0:
            st.warning("Predicted Level: Lower Cybersecurity Awareness")
        else:
            st.success("Predicted Level: Higher Cybersecurity Awareness")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
