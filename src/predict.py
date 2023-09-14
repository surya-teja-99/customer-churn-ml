"""Predict churn for a single customer from the CLI."""
import argparse, pathlib, joblib, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tenure", type=int, required=True)
    p.add_argument("--monthly-charges", type=float, required=True)
    p.add_argument("--total-charges", type=float, required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--internet-service", required=True)
    p.add_argument("--payment-method", required=True)
    p.add_argument("--senior-citizen", type=int, default=0)
    p.add_argument("--partner", default="No")
    p.add_argument("--dependents", default="No")
    a = p.parse_args()
    model = joblib.load(ROOT / "models/model.joblib")
    row = pd.DataFrame([{"tenure": a.tenure, "MonthlyCharges": a.monthly_charges, "TotalCharges": a.total_charges,
        "SeniorCitizen": a.senior_citizen, "Partner": a.partner, "Dependents": a.dependents,
        "Contract": a.contract, "InternetService": a.internet_service, "PaymentMethod": a.payment_method}])
    proba = model.predict_proba(row)[0,1]
    print(f"Churn probability: {proba:.1%} -> {'LIKELY CHURN' if proba >= 0.5 else 'likely to stay'}")
if __name__ == "__main__":
    main()
