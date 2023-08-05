"""Preprocessing pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Partner", "Dependents", "Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL + ["SeniorCitizen"]

def build_preprocessor():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL),
        ("passthrough", "passthrough", ["SeniorCitizen"]),
    ])
