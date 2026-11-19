
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cyber Risk Prediction",
    page_icon="🔐",
    layout="wide"
)

# Dashboard styling
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

h1 {
    font-size: 2rem !important;
}

h2 {
    font-size: 1.5rem !important;
}

h3 {
    font-size: 1.15rem !important;
}

[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    padding: 15px;
    border-radius: 10px;
}

[data-testid="stTabs"] button {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. CONSTANTS
# ============================================================

FEATURE_COLUMNS = [
    "Age_Group",
    "Social_Media_Hours",
    "Social_Media_Scam_Exposure",
    "Personal_Social_Media_Fraud_Loss",
    "Transaction_Fraud_Loss",
    "OTP_Password_Request_Exposure",
    "Suspicious_Link_Clicks"
]

TARGET_COLUMN = "Overall_Cybersecurity_Awareness"

CLASS_NAMES = {
    0: "Lower Cybersecurity Awareness",
    1: "Higher Cybersecurity Awareness"
}

TARGET_MAPPING = {
    1.0: 0,
    2.0: 0,
    3.0: 1,
    4.0: 1,
    5.0: 1
}

DATA_PATH = "data/cybersecurity_dataset.csv"


# ============================================================
# 3. LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load("models/model.pkl")

    preprocessor = joblib.load("models/preprocessor.pkl")

    return model, preprocessor


@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):
        return None

    data = pd.read_csv(DATA_PATH)

    # Remove completely empty rows
    data = data.dropna(how="all")

    return data


try:

    model, preprocessor = load_models()

except Exception as e:

    st.error(
        "Unable to load the trained model or preprocessor. "
        "Please check the models folder."
    )

    with st.expander("Technical details"):
        st.code(str(e))

    st.stop()


# ============================================================
# 4. REUSABLE COMPACT CHART FUNCTION
# ============================================================

def display_bar_chart(
    series,
    title,
    xlabel="",
    ylabel="Number of Users"
):

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    series.plot(
        kind="bar",
        ax=ax,
        width=0.65
    )

    ax.set_title(
        title,
        fontsize=12,
        pad=12
    )

    ax.set_xlabel(
        xlabel,
        fontsize=9
    )

    ax.set_ylabel(
        ylabel,
        fontsize=9
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # Center the chart
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.pyplot(fig, use_container_width=False)

    plt.close(fig)


# ============================================================
# 5. APPLICATION HEADER
# ============================================================

st.title("🔐 Cybersecurity Awareness Prediction")

st.write(
    """
    An interactive machine learning application that analyzes
    digital behaviour and predicts a user's cybersecurity
    awareness category.
    """
)

st.divider()


# ============================================================
# 6. APPLICATION TABS
# ============================================================

predict_tab, insights_tab, performance_tab, about_tab = st.tabs([
    "🔍 Predict",
    "📊 Data Insights",
    "📈 Model Performance",
    "ℹ️ About"
])


# ============================================================
# TAB 1: PREDICTION
# ============================================================

with predict_tab:

    st.subheader("User Behaviour Information")

    st.write(
        """
        Enter the user's digital behaviour details to estimate
        their cybersecurity awareness category.
        """
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

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
                [
                    "Never",
                    "Once",
                    "2-3 times",
                    "4-5 times",
                    "More than 5 times"
                ]
            )

            personal_fraud_loss = st.selectbox(
                "Experienced Personal Social Media Fraud Loss?",
                ["No", "Yes"]
            )

        with col2:

            transaction_fraud_loss = st.selectbox(
                "Experienced Transaction Fraud Loss?",
                ["No", "Yes"]
            )

            otp_exposure = st.selectbox(
                "OTP / Password Request Exposure",
                [
                    "Never",
                    "Once",
                    "2-3 times",
                    "4-5 times",
                    "More than 5 times"
                ]
            )

            suspicious_link_clicks = st.selectbox(
                "Suspicious Links Clicked",
                [
                    "Never",
                    "Once",
                    "2-3 times",
                    "4-5 times",
                    "More than 5 times"
                ]
            )

        submitted = st.form_submit_button(
            "Predict Cybersecurity Awareness",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION LOGIC
    # --------------------------------------------------------

    if submitted:

        input_data = pd.DataFrame({

            "Age_Group": [age_group],

            "Social_Media_Hours": [social_media_hours],

            "Social_Media_Scam_Exposure": [
                social_media_scam_exposure
            ],

            "Personal_Social_Media_Fraud_Loss": [
                personal_fraud_loss
            ],

            "Transaction_Fraud_Loss": [
                transaction_fraud_loss
            ],

            "OTP_Password_Request_Exposure": [
                otp_exposure
            ],

            "Suspicious_Link_Clicks": [
                suspicious_link_clicks
            ]

        })

        try:

            # Apply saved manual categorical preprocessing
            encoded_data = preprocessor.transform(input_data)

            # Convert to DataFrame if necessary
            if not isinstance(encoded_data, pd.DataFrame):

                encoded_data = pd.DataFrame(
                    encoded_data,
                    columns=FEATURE_COLUMNS
                )

            # Check feature count
            if encoded_data.shape[1] != model.n_features_in_:

                raise ValueError(
                    "Input feature count does not match the model."
                )

            # Ensure correct feature order
            if isinstance(encoded_data, pd.DataFrame):

                if set(encoded_data.columns) != set(FEATURE_COLUMNS):
                    raise ValueError(
                        "Preprocessor output features do not match "
                        "the expected feature names."
                    )

                encoded_data = encoded_data[FEATURE_COLUMNS]

            # Ensure numeric values
            encoded_data = encoded_data.apply(
                pd.to_numeric,
                errors="coerce"
            )

            # Check missing values
            if encoded_data.isna().any().any():

                raise ValueError(
                    "Preprocessing generated missing or "
                    "non-numeric values."
                )

            # Predict class
            prediction = model.predict(encoded_data)[0]

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 1:

                st.success(
                    "Predicted Level: Higher Cybersecurity Awareness"
                )

            else:

                st.warning(
                    "Predicted Level: Lower Cybersecurity Awareness"
                )

            # ------------------------------------------------
            # PREDICTION PROBABILITY
            # ------------------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    encoded_data
                )[0]

                class_probabilities = dict(
                    zip(model.classes_, probabilities)
                )

                lower_probability = class_probabilities[0]

                higher_probability = class_probabilities[1]

                st.subheader("Prediction Probability")

                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:

                    st.metric(
                        "Lower Awareness",
                        f"{lower_probability:.2%}"
                    )

                    st.progress(
                        float(lower_probability)
                    )

                with prob_col2:

                    st.metric(
                        "Higher Awareness",
                        f"{higher_probability:.2%}"
                    )

                    st.progress(
                        float(higher_probability)
                    )

                st.caption(
                    "These are SVM-estimated probabilities, "
                    "not guarantees of prediction correctness."
                )

            # ------------------------------------------------
            # PROCESSED INPUT
            # ------------------------------------------------

            with st.expander("View Processed Input"):

                st.write("Original Input")

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

                st.write("Encoded Input")

                st.dataframe(
                    encoded_data,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Prediction could not be completed. "
                "Please verify the input and saved model files."
            )

            with st.expander("Technical details"):
                st.code(str(e))


# ============================================================
# TAB 2: DATA INSIGHTS
# ============================================================

with insights_tab:

    st.subheader("Dataset Insights")

    data = load_dataset()

    if data is None:

        st.warning(
            "Dataset not found. Place your CSV file at "
            "`data/cybersecurity_dataset.csv`."
        )

    elif TARGET_COLUMN not in data.columns:

        st.error(
            f"Target column '{TARGET_COLUMN}' was not found "
            "in the dataset."
        )

    else:

        # Convert target to numeric
        data[TARGET_COLUMN] = pd.to_numeric(
            data[TARGET_COLUMN],
            errors="coerce"
        )

        # Create binary target
        data["Awareness_Category"] = (
            data[TARGET_COLUMN].map(TARGET_MAPPING)
        )

        st.write(
            """
            Explore the distribution of user behaviour
            and cybersecurity awareness in the dataset.
            """
        )

        # ----------------------------------------------------
        # DATASET OVERVIEW
        # ----------------------------------------------------

        st.markdown("### Dataset Overview")

        total_records = len(data)

        total_features = len(FEATURE_COLUMNS)

        target_values = data["Awareness_Category"].dropna()

        lower_count = int((target_values == 0).sum())

        higher_count = int((target_values == 1).sum())

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Records", f"{total_records:,}")

        col2.metric("Selected Features", total_features)

        col3.metric("Lower Awareness", lower_count)

        col4.metric("Higher Awareness", higher_count)

        st.divider()

        # ----------------------------------------------------
        # ORIGINAL AWARENESS SCORE DISTRIBUTION
        # ----------------------------------------------------

        st.markdown("### Cybersecurity Awareness Distribution")

        target_distribution = (
            data[TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        target_distribution.index = [
            f"Score {int(value)}"
            for value in target_distribution.index
        ]

        display_bar_chart(
            target_distribution,
            "Original Awareness Score Distribution",
            xlabel="Awareness Score"
        )

        # ----------------------------------------------------
        # BINARY TARGET DISTRIBUTION
        # ----------------------------------------------------

        st.markdown("### Binary Awareness Classification")

        binary_distribution = (
            data["Awareness_Category"]
            .map(CLASS_NAMES)
            .value_counts()
        )

        display_bar_chart(
            binary_distribution,
            "Lower vs Higher Awareness",
            xlabel="Awareness Category"
        )

        st.caption(
            "Scores 1 and 2 are grouped as lower awareness; "
            "scores 3, 4 and 5 are grouped as higher awareness. "
            "This is the project's chosen target grouping."
        )

        st.divider()

        # ----------------------------------------------------
        # AGE DISTRIBUTION
        # ----------------------------------------------------

        st.markdown("### Age Group Distribution")

        age_distribution = (
            data["Age_Group"].value_counts()
        )

        display_bar_chart(
            age_distribution,
            "Users by Age Group",
            xlabel="Age Group"
        )

        st.divider()

        # ----------------------------------------------------
        # SOCIAL MEDIA USAGE
        # ----------------------------------------------------

        st.markdown("### Daily Social Media Usage")

        hours_distribution = (
            data["Social_Media_Hours"].value_counts()
        )

        display_bar_chart(
            hours_distribution,
            "Daily Social Media Usage",
            xlabel="Usage Duration"
        )

        st.divider()

        # ----------------------------------------------------
        # SCAM EXPOSURE
        # ----------------------------------------------------

        st.markdown("### Social Media Scam Exposure")

        scam_distribution = (
            data["Social_Media_Scam_Exposure"].value_counts()
        )

        display_bar_chart(
            scam_distribution,
            "Social Media Scam Exposure",
            xlabel="Exposure Frequency"
        )

        st.divider()

        # ----------------------------------------------------
        # OTP EXPOSURE
        # ----------------------------------------------------

        st.markdown("### OTP / PASSWORD EXPOSURE")

        otp_distribution = (
            data["OTP_Password_Request_Exposure"].value_counts()
        )

        display_bar_chart(
            otp_distribution,
            "OTP / Password Request Exposure",
            xlabel="Exposure Frequency"
        )

        st.divider()

        # ----------------------------------------------------
        # SUSPICIOUS LINK CLICKS
        # ----------------------------------------------------

        st.markdown("### Suspicious Link Clicks")

        link_distribution = (
            data["Suspicious_Link_Clicks"].value_counts()
        )

        display_bar_chart(
            link_distribution,
            "Suspicious Link Clicks",
            xlabel="Click Frequency"
        )

        st.divider()

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        st.markdown("### Data Quality Overview")

        quality_col1, quality_col2 = st.columns(2)

        with quality_col1:

            st.metric(
                "Missing Values",
                int(data[FEATURE_COLUMNS].isna().sum().sum())
            )

        with quality_col2:

            st.metric(
                "Duplicate Rows",
                int(data.duplicated().sum())
            )

        with st.expander("View Dataset Sample"):

            st.dataframe(
                data.head(10),
                use_container_width=True
            )


# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================

with performance_tab:

    st.subheader("SVM Model Performance")

    st.write(
        """
        Evaluation results of the Support Vector Machine
        classifier using the held-out test dataset.
        """
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Test Accuracy", "63.58%")

    col2.metric("Test Samples", "302")

    col3.metric("Model", "SVM (RBF)")

    st.divider()

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.markdown("### Classification Report")

    performance_data = pd.DataFrame(
        {
            "Metric": [
                "Precision",
                "Recall",
                "F1-score"
            ],

            "Lower Awareness (0)": [
                0.35,
                0.12,
                0.18
            ],

            "Higher Awareness (1)": [
                0.67,
                0.89,
                0.77
            ]
        }
    )

    st.dataframe(
        performance_data.set_index("Metric").style.format(
            "{:.2f}"
        ),
        use_container_width=True
    )

    # --------------------------------------------------------
    # CLASS-WISE PERFORMANCE CHARTS
    # --------------------------------------------------------

    st.markdown("### Class-wise Performance")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        display_bar_chart(
            performance_data.set_index("Metric")[
                "Lower Awareness (0)"
            ],
            "Lower Awareness Performance",
            xlabel="Evaluation Metric",
            ylabel="Score"
        )

    with chart_col2:

        display_bar_chart(
            performance_data.set_index("Metric")[
                "Higher Awareness (1)"
            ],
            "Higher Awareness Performance",
            xlabel="Evaluation Metric",
            ylabel="Score"
        )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown("### Confusion Matrix")

    cm = np.array([
        [12, 88],
        [22, 180]
    ])

    fig, ax = plt.subplots(figsize=(4.8, 3.5))

    image = ax.imshow(
        cm,
        cmap="Blues",
        aspect="auto"
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="black"
            )

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Lower",
        "Higher"
    ])

    ax.set_yticklabels([
        "Lower",
        "Higher"
    ])

    ax.set_xlabel(
        "Predicted Label",
        fontsize=9
    )

    ax.set_ylabel(
        "Actual Label",
        fontsize=9
    )

    ax.set_title(
        "SVM Confusion Matrix",
        fontsize=12,
        pad=12
    )

    ax.tick_params(labelsize=9)

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    fig.tight_layout()

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.pyplot(
            fig,
            use_container_width=False
        )

    plt.close(fig)

    st.caption(
        "Rows represent actual classes; columns represent "
        "predicted classes."
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION SUMMARY
    # --------------------------------------------------------

    st.markdown("### Prediction Summary")

    col1, col2 = st.columns(2)

    col1.metric(
        "Correct Predictions",
        "192"
    )

    col2.metric(
        "Incorrect Predictions",
        "110"
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    st.markdown("### Model Configuration")

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.write("**Algorithm:** Support Vector Machine")

        st.write("**Kernel:** RBF")

        st.write("**C:** 1")

    with config_col2:

        st.write("**Gamma:** 3")

        st.write("**Probability:** Enabled")

        st.write("**Input Features:** 7")

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE INTERPRETATION
    # --------------------------------------------------------

    st.markdown("### Performance Interpretation")

    st.warning(
        """
        The model identifies users with higher cybersecurity
        awareness more effectively than users with lower
        cybersecurity awareness.

        The recall for the lower-awareness class is only 12%.
        This means that the model misses many users belonging
        to that category.

        Although the overall test accuracy is 63.58%,
        the model requires further improvement before
        practical deployment.
        """
    )


# ============================================================
# TAB 4: ABOUT
# ============================================================

with about_tab:

    st.subheader("About This Project")

    st.write(
        """
        User Behaviour Based Cyber Risk Prediction is a
        machine learning classification project designed
        to estimate cybersecurity awareness based on
        users' digital behaviour.
        """
    )

    st.divider()

    st.markdown("### Problem Statement")

    st.write(
        """
        Digital activities such as social media usage,
        scam exposure, suspicious link interactions and
        OTP/password request exposure may provide useful
        information about cybersecurity awareness.

        This project explores whether these behavioural
        characteristics can be used to classify users
        into lower and higher cybersecurity awareness
        categories.
        """
    )

    st.markdown("### Project Information")

    st.write("**Problem Type:** Binary Classification")

    st.write(
        "**Target Variable:** Overall Cybersecurity Awareness"
    )

    st.write("**Algorithm:** Support Vector Machine")

    st.write("**Kernel:** RBF")

    st.write("**Input Features:** Seven behavioural features")

    st.divider()

    st.markdown("### Dataset")

    st.write(
        """
        The project uses a cybersecurity survey dataset
        containing information about digital behaviour,
        scam exposure and cybersecurity awareness.
        """
    )

    st.markdown(
        "[View Dataset - Mendeley Data](https://data.mendeley.com/datasets/tb9nnp6c5f/1)"
    )

    st.divider()

    st.markdown("### Technology Stack")

    st.write(
        """
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Matplotlib
        - Joblib
        - Streamlit
        """
    )

    st.divider()

    st.markdown("### Limitations")

    st.warning(
        """
        This application provides experimental predictions
        based on the available survey dataset.

        The awareness categories are created by grouping
        the original survey scores into two classes.

        The model has limited recall for lower-awareness
        users, so its predictions should not be treated
        as definitive assessments of an individual's
        cybersecurity awareness.
        """
    )

    st.caption(
        "Machine Learning Portfolio Project | Cybersecurity Awareness"
    )
