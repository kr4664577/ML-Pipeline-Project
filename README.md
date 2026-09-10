# ⚙️ ML Pipeline with Hyperparameter Tuning

## 📌 Overview

This project builds an end-to-end **Scikit-learn machine learning pipeline** for Titanic survival prediction.

The pipeline handles numerical and categorical preprocessing automatically and trains a **Random Forest Classifier**. `GridSearchCV` is then used to search for better hyperparameter combinations.

## 🧠 Workflow

```text
Titanic Dataset
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
ColumnTransformer
   ↙             ↘
Numerical       Categorical
Imputation      Imputation
Scaling         One-Hot Encoding
   ↘             ↙
      Random Forest
            ↓
       GridSearchCV
            ↓
      Model Evaluation
```

## 🔧 Preprocessing Pipeline

### Numerical Features
- `Age`
- `Fare`
- `SibSp`
- `Parch`

Processing:
- Median imputation
- Standard scaling

### Categorical Features
- `Pclass`
- `Sex`
- `Embarked`

Processing:
- Most-frequent imputation
- One-hot encoding

## 🤖 Model

**Algorithm:** Random Forest Classifier

The pipeline uses:

- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `GridSearchCV`
- 5-fold cross-validation

## 🔍 Hyperparameter Tuning

The grid searches over:

- `n_estimators`: 50, 100, 200
- `max_depth`: None, 5, 10
- `min_samples_split`: 2, 5, 10

The executed notebook found these best parameters:

```text
max_depth = 5
min_samples_split = 2
n_estimators = 100
```

## 📊 Results

| Metric | Result |
|---|---:|
| Default model accuracy | 0.8268 |
| Best CV score from GridSearchCV | 0.8300 |
| Tuned model test accuracy | 0.8156 |
| Mean 5-fold CV accuracy | 0.8160 |

The tuned model's test accuracy was lower than the default model's accuracy for this split. This highlights why model selection should consider cross-validation rather than relying on a single test-set score.

## 🧰 Technologies

- Python
- Pandas
- Scikit-learn
- Google Colab / Jupyter Notebook

## ▶️ How to Run

1. Keep `train.csv` in the same directory as the notebook.
2. Open `Untitled17.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells.
4. Review the default model, GridSearchCV results, cross-validation scores, and classification report.

## 🎯 Learning Outcome

This project demonstrates how to create reproducible ML preprocessing pipelines and systematically tune model hyperparameters using cross-validation.
