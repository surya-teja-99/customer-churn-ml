"""Generate / load the churn dataset."""
import numpy as np
import pandas as pd

def generate_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n)
    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.normal(65, 30, n).clip(18, 120), 2)
    # Fiber + month-to-month costs more
    monthly += np.where(internet == "Fiber optic", 12, 0)
    monthly += np.where(contracts == "Month-to-month", 5, 0)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.1, n), 2)
    senior = rng.integers(0, 2, n)
    partner = rng.choice(["Yes", "No"], n)
    dependents = rng.choice(["Yes", "No"], n)
    # Churn probability: higher for month-to-month, fiber, electronic check, low tenure, high charges
    logit = (
        -1.2
        + np.where(contracts == "Month-to-month", 1.1, -0.4)
        + np.where(internet == "Fiber optic", 0.55, 0)
        + np.where(payment == "Electronic check", 0.45, 0)
        - tenure / 36.0
        + (monthly - 65) / 45.0
        + senior * 0.2
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "Contract": contracts, "InternetService": internet, "PaymentMethod": payment,
        "Churn": churn,
    })

def load_data(n=5000, seed=42):
    return generate_data(n=n, seed=seed)

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df["Churn"].value_counts(normalize=True))
