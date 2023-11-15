# Customer Churn Prediction — AI/ML Project

An end-to-end machine learning project that predicts whether a telecom customer will churn.

Built with Python, pandas, and scikit-learn. It includes data generation/loading,
preprocessing, feature engineering, model training, evaluation, and inference — in a
clean, reproducible folder structure you can run, extend, and present.

## Problem
Churn costs businesses far more than acquisition. This model flags at-risk customers
so a retention team can intervene early.

## Project Structure
```
customer-churn-ml/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   └── README.md          # where data goes / how it's generated
├── notebooks/
│   └── 01_eda.ipynb       # exploratory data analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # generates / loads a realistic synthetic dataset
│   ├── features.py        # preprocessing pipeline
│   ├── train.py           # trains Logistic Regression + Random Forest, saves best
│   ├── evaluate.py        # metrics, confusion matrix, classification report
│   └── predict.py         # CLI inference for a single customer
├── tests/
│   └── test_train.py
├── models/                # saved model lands here (model.joblib)
└── reports/               # metrics.json lands here
```

## Quick Start
```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.predict --tenure 12 --monthly-charges 79.5 --total-charges 950 --contract One-year --internet-service Fiber optic --payment-method Electronic check --senior-citizen 0 --partner Yes --dependents No
```

## Results (typical run)
- Models compared: Logistic Regression, Random Forest
- Selection metric: ROC-AUC on a held-out test set
- Outputs: `models/model.joblib`, `reports/metrics.json`
- See `reports/metrics.json` after training for your exact run's numbers — do not
  copy claimed numbers from elsewhere, report what your run produces.

## Dataset
A realistic synthetic dataset is generated deterministically in `src/data_loader.py`
(seed = 42, 5,000 customers) so the project runs offline with no downloads and is
fully reproducible. Swap `load_data()` with a real CSV loader to use real data:
put your file at `data/churn.csv` with a `Churn` target column.

## Extend It
- Add XGBoost / Gradient Boosting to the model comparison
- Tune hyperparameters with GridSearchCV
- Deploy `predict.py` behind FastAPI
- Add SHAP explainability

## Tech Stack
Python 3.10+, pandas, numpy, scikit-learn, matplotlib, joblib, PyYAML
