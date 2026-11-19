# User Behaviour Based Cyber Risk Prediction

A machine-learning classification project that uses self-reported digital behaviour from a cybersecurity survey to estimate whether a respondent belongs to a **lower** or **higher cybersecurity-awareness group**.

## Live Demo

**[Open Cyber Risk Prediction](http://cyberriskprediction.duckdns.org:8080/)**

The application is hosted on an AWS EC2 instance and served using Streamlit. DuckDNS keeps the domain pointed at the instance's current public IP address when that address changes.

> **Availability note:** The demo is available only while the EC2 instance and Streamlit application are running. The current demo uses HTTP; do not enter sensitive or personal information.

## Application Screenshots

### Prediction

![Prediction form](images/Prediction%20form.png)

![Prediction result](images/Prediction%20result.png)

### Data Insights

![Data insights 1](images/Data%20Insights-1.png)

![Data insights 2](images/Data%20Insights-2.png)

![Data insights 3](images/Data%20Insights-3.png)

![Data insights 4](images/Data%20Insights-4.png)

![Data insights 5](images/Data%20Insights-5.png)

### Model Performance

![Model performance 1](images/Model%20Performance-1.png)

![Model performance 2](images/Model%20Performance-2.png)

![Model performance 3](images/Model%20Performace-3.png)

### About

![About project 1](images/About%20project-1.png)

![About project 2](images/About%20project-2.png)

![About project 3](images/About%20project-3.png)

## Problem Statement

People interact with social platforms, payment services and online messages every day. Scam exposure, suspicious links and requests for OTPs or passwords are among the behaviours explored in this survey.

This project investigates the following question:

**Can selected self-reported digital-behaviour attributes help estimate a respondent's cybersecurity-awareness category?**

The project is intended for learning and exploratory analysis. Its predictions should not be used to make consequential decisions about individuals or to assess someone's actual ability to stay safe online.

## Dataset

- **Source:** [Mendeley Data – Cybersecurity Survey Dataset](https://data.mendeley.com/datasets/tb9nnp6c5f/1)
- **Records:** 1,509, after removing one completely empty row.
- **Original columns:** 15.
- **Selected model inputs:** 7.
- **Target:** `Overall_Cybersecurity_Awareness`.

### Selected Features

1. `Age_Group`
2. `Social_Media_Hours`
3. `Social_Media_Scam_Exposure`
4. `Personal_Social_Media_Fraud_Loss`
5. `Transaction_Fraud_Loss`
6. `OTP_Password_Request_Exposure`
7. `Suspicious_Link_Clicks`

### Target Definition

For this project, the original awareness score was grouped into two classes:

| Original score | Project label |
|---|---|
| 1 or 2 | `0` – Lower Cybersecurity Awareness |
| 3, 4 or 5 | `1` – Higher Cybersecurity Awareness |

This is a **project-defined grouping**, not a universal or independently validated cybersecurity-awareness threshold.

The resulting class distribution is:

| Class | Records |
|---|---:|
| Lower awareness | 500 |
| Higher awareness | 1,009 |
| **Total** | **1,509** |

The class imbalance is important when interpreting the model's results.

## Project Workflow

1. Collected the survey dataset from Mendeley Data.
2. Explored the data, including column types, missing values and category distributions.
3. Selected seven behavioural input features and the awareness target.
4. Filled missing values in `Transaction_Fraud_Loss` using the mode.
5. Converted the original awareness score into the project's two target classes.
6. Split the data into training and test sets using a stratified split (`test_size=0.2`, `random_state=42`).
7. Applied manual category-to-number mappings while retaining the seven input columns.
8. Trained and evaluated classification models.
9. Saved the selected model and preprocessing object using Joblib.
10. Built a Streamlit application to accept user inputs, apply the saved preprocessing and generate predictions.
11. Deployed the application on AWS EC2 and configured a DuckDNS hostname for access.

### Selected Model

The deployed model is an RBF-kernel Support Vector Classifier:

```python
SVC(C=1, gamma=3, probability=True)
```

The saved preprocessor applies the manual category mappings using `pandas.DataFrame.replace`. The input feature names, order and category mappings must remain consistent between training and prediction.

## Model Results

### Reported Model Comparison

| Model | Reported test accuracy |
|---|---:|
| Logistic Regression | 49.67% |
| Random Forest | 55.30% |
| Selected SVM | **63.58%** |

The selected SVM accuracy is calculated from the supplied confusion matrix: 192 correct predictions out of 302 test records. Confirm that all model scores were generated using the same untouched test split and evaluation procedure before treating them as a strictly comparable model comparison.

### Selected SVM Confusion Matrix

| Actual class | Predicted lower (0) | Predicted higher (1) |
|---|---:|---:|
| Lower awareness (0) | 12 | 88 |
| Higher awareness (1) | 22 | 180 |

### Class-wise Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Lower awareness (0) | 0.35 | 0.12 | 0.18 | 100 |
| Higher awareness (1) | 0.67 | 0.89 | 0.77 | 202 |

**Interpretation:** The selected SVM identifies the higher-awareness class substantially better than the lower-awareness class. It correctly identifies only 12 of the 100 actual lower-awareness test cases. Accuracy alone therefore does not describe the model's performance adequately; class-wise precision, recall, F1-score and the confusion matrix are also important.

The application displays `predict_proba` outputs as model-estimated probabilities. These are not guarantees of correctness, and probability calibration has not been established.

## Streamlit Application

The application contains four sections:

- **Predict:** accepts the seven selected behavioural attributes and displays the predicted class and model-estimated probability.
- **Data Insights:** explores distributions and patterns in the survey dataset.
- **Model Performance:** displays the selected model's evaluation metrics and confusion matrix.
- **About:** explains the project, dataset, technology stack and limitations.

## Tech Stack

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- AWS EC2
- DuckDNS

## Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Check the required project files

Make sure the application can access its saved artifacts and dataset. The expected paths used by the deployed app are:

```text
app.py
data/
    cybersecurity_dataset.csv
models/
    model.pkl
    preprocessor.pkl
images/
    screenshots...
requirements.txt
```

If your repository uses different paths, update this example to match the actual project layout and the paths referenced in `app.py`.

### 5. Start Streamlit

```bash
streamlit run app.py
```

The local application normally opens at `http://localhost:8501`.

## Deployment

The live application is hosted on an AWS EC2 Linux instance. Streamlit is run on port `8080`, and DuckDNS maps the hostname to the instance's current public IP.

Live URL: **http://cyberriskprediction.duckdns.org:8080/**

The EC2 instance must be running and the Streamlit process must be available for the demo to load. The DuckDNS updater is configured on the instance to refresh the DNS record periodically.

## Challenges

- **Categorical input consistency:** inference must use the same category-to-number mappings and feature order as model training.
- **Preprocessor serialization:** a notebook-defined custom function caused a pickle loading issue. The preprocessor was changed to use the built-in `pandas.DataFrame.replace` method with the mapping dictionary.
- **Class imbalance:** the two target classes are not equally represented, and the selected model has low recall for the lower-awareness class.
- **Evaluation interpretation:** accuracy alone hides the model's weak performance on the lower-awareness class.
- **Deployment compatibility:** the Python and scikit-learn versions used to load saved model artifacts need to be compatible with the versions used to create them.
- **Cloud deployment:** the Streamlit app was deployed on EC2, and DuckDNS was configured to keep a hostname updated when the public IP changes.

## Limitations and Future Improvements

### Current Limitations

- The target classes are based on a project-defined grouping of the original survey score.
- The data is self-reported and may contain response and sampling biases.
- The class distribution is imbalanced.
- The selected SVM has low recall for the lower-awareness class (0.12).
- The displayed probabilities have not been shown to be calibrated.
- The application does not yet provide a validated feature-attribution explanation for an individual prediction.
- A systematic error analysis by demographic or behavioural subgroup has not yet been documented.

### Future Improvements

1. Recheck model comparison and cross-validation using only the training data, keeping the test set untouched until final evaluation.
2. Evaluate class-sensitive metrics and investigate imbalance-handling methods within cross-validation.
3. Perform subgroup error analysis to understand which cases are misclassified.
4. Add feature interpretation using permutation importance or SHAP, with care around correlated features and categorical encodings.
5. Investigate probability calibration and decision thresholds.
6. Validate the target definition with domain expertise and, if possible, a better-grounded awareness measure.
7. Configure HTTPS and a reverse proxy for a more secure public deployment.
8. Improve deployment monitoring and restart behaviour.

## Contact

- **LinkedIn:** www.linkedin.com/in/m-mohamed-yaseen
- **Email:** mohamedyaseenm0068@gmail.com

---

**Disclaimer:** This is an educational machine-learning portfolio project. It should not be used as a standalone cybersecurity assessment or to make decisions about individuals.
