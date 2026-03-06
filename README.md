# Machine Learning Classification Project: Prediction of churn

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Project Overview

This project aims to build and evaluate machine learning models for a **classification task** using structured data.

The objective is to predict if a client of a telecom service will churn or not, based on a set of input features while comparing different machine learning algorithms and optimizing model performance through **hyperparameter tuning**.

The workflow includes:

* Data exploration and preprocessing
* Feature engineering
* Model training
* Hyperparameter tuning
* Model evaluation
* Model comparison

---

# Project Structure

```
project-name
│
├── data
│   └── dataset.csv
│
├── notebooks
│   └── exploration.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── training.py
│   └── evaluation.py
│
├── models
│   └── trained_models.pkl
│
├── results
│   └── figures
│
├── README.md
└── requirements.txt
```

---

# Dataset

The dataset contains multiple features describing observations used to predict the **target variable**.

Typical preprocessing steps:

* Handling missing values
* Encoding categorical variables
* Feature scaling (when necessary)
* Train/test split

Example:

```python
X = df.drop("target", axis=1)
y = df["target"]
```

Train/Test split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# Models Implemented

Three machine learning models were implemented and compared:

| Model               | Type              |
| ------------------- | ----------------- |
| Logistic Regression | Linear Model      |
| Random Forest       | Ensemble Learning |
| XGBoost             | Gradient Boosting |

### Logistic Regression

A baseline linear model used for classification.

Advantages:

* Fast to train
* Interpretable coefficients
* Good baseline model

Note: **requires feature standardization.**

---

### Random Forest

Random Forest is an **ensemble method based on decision trees**.

Advantages:

* Robust to overfitting
* Handles nonlinear relationships
* Works well with tabular datasets
* Does not require feature scaling

---

### XGBoost

XGBoost is a **gradient boosting algorithm** widely used in machine learning competitions.

Advantages:

* Very high predictive performance
* Handles complex feature interactions
* Efficient and scalable

Feature scaling is **not required but sometimes beneficial**.

---

# Hyperparameter Tuning

To improve performance, **GridSearchCV** was used for hyperparameter tuning.

Example for **Random Forest**:

```python
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
```

Grid search implementation:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
```

Best parameters:

```python
print(grid_search.best_params_)
```

---

# Model Evaluation

Models were evaluated using several classification metrics.

Metrics used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example:

```python
from sklearn.metrics import classification_report

y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

Confusion matrix:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
```

---

# Model Comparison

| Model               | Accuracy    | Precision | Recall     | F1-score |
| ------------------- | --------    | --------- | ------     | -------- |
| Logistic Regression | 0,7413      | 0,5098    | 0,6952     | 0,5882   |
| Random Forest       | 0,7647      | 0,5513    | 0,6176     | 0,5826   |
| XGBoost             | 0,7591      | 0,5457    | 0,5588     | 0,5522   |

The final model selected is a Random Forest Classifier, tuned via GridSearchCV with 5-fold cross-validation.

##Confusion Matrix

Predicted No    Predicted Yes
Actual No              845             188
Actual Yes             143             231

Interpretation
The model performs well at identifying customers who stay (845 true negatives).
However, in a churn context, recall is the most critical metric — a missed churner is more costly than a false alarm.

188 false negatives → churners predicted as staying (high business cost)
143 false positives → loyal customers flagged as churners (lower impact)

A recall of 61.5% means the model catches roughly 6 out of 10 customers at risk of churning.


# Feature Importance

Random Forest provides feature importance scores that help identify the most influential variables in the prediction.

Example:

```python
import pandas as pd

importance = best_model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print(feature_importance.head())
```

These insights help understand **which features contribute the most to the model's predictions**.

---

# Key Insights

* Ensemble models significantly outperform linear models on complex datasets.
* Hyperparameter tuning improves predictive performance.
* Random Forest provides useful feature importance for model interpretability.

---

# Future Improvements

Possible improvements include:

* Testing additional models such as:

  * Gradient Boosting
  * LightGBM
  * CatBoost
* Using **RandomizedSearchCV** for faster tuning
* Implementing **cross-validation pipelines**
* Deploying the model as an API

---

# Installation

Clone the repository:

```bash
https://github.com/KANOHA242/Machine-Learning-Project--Churn-Prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

Run the streamlit app to test with this command: 

```
python -m streamlit run "src/app.py"
```
---

# Requirements

Main libraries used:

```
kaggle
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
```

## 👤 Author

Project developed as part of a **Machine Learning / Data Science study project**.

**[KANOHA ELENGA Jihane]**  
📧 [jihanekanoha@gmail.com]

