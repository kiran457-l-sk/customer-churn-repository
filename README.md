# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave a service. The project uses multiple classification algorithms, hyperparameter tuning, class balancing, and a Streamlit web application for interactive predictions.

## Project Overview

Customer churn is a major challenge for telecom companies. Identifying customers who are likely to leave can help businesses take preventive retention measures.

This project builds a machine learning classification system to predict customer churn using customer demographics, service subscriptions, contract details, and billing information.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

## Machine Learning Models

The following models were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. Balanced Logistic Regression

Hyperparameter tuning was performed using GridSearchCV and RandomizedSearchCV with 5-fold cross-validation.

Class weighting (`class_weight='balanced'`) was also tested to address class imbalance.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.820 | 0.682 | 0.603 | 0.640 | 0.862 |
| Random Forest | 0.813 | 0.700 | 0.512 | 0.591 | 0.866 |
| XGBoost | 0.815 | 0.690 | 0.544 | 0.609 | 0.864 |
| Balanced Logistic Regression | 0.750 | 0.519 | 0.828 | 0.637 | 0.862 |

### Selected Model

Balanced Logistic Regression was selected for deployment because it achieved higher churn recall.

- Accuracy: 75%
- Churn Precision: 52%
- Churn Recall: 83%
- Churn F1-score: 0.64
- ROC-AUC: 0.862

The model identifies more potential churners, although it also produces more false-positive predictions.

## Streamlit Application

The project includes an interactive Streamlit application where users can:

- Enter customer information.
- Predict whether a customer is likely to churn.
- View estimated churn and retention probabilities.

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── customer_churn_model.joblib
├── requirements.txt
├── README.md
├── .gitignore
│
└── notebooks/
    └── customer_churn_analysis.ipynb
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/kiran457-l-sk/customer-churn-prediction.git
```

### 2. Navigate to the project directory

```bash
cd customer-churn-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## Future Improvements

- Improve model performance and probability calibration.
- Explore business-based retention strategies.
- Add more detailed customer risk analysis.
- Deploy the application publicly.

## Author

**KIRAN**

GitHub: https://github.com/kiran457-l-sk
