"""Evaluate the saved model on a fresh hold-out split."""
import pathlib, joblib, yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from .data_loader import load_data
from .features import FEATURES
ROOT = pathlib.Path(__file__).resolve().parents[1]
def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    model = joblib.load(ROOT / cfg["model_path"])
    df = load_data(n=cfg["n_customers"], seed=cfg["seed"])
    X, y = df[FEATURES], df["Churn"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["seed"], stratify=y)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 4))
if __name__ == "__main__":
    main()
