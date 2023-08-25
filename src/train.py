"""Train and compare models, save the best one."""
import json, pathlib, joblib, yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from .data_loader import load_data
from .features import build_preprocessor, FEATURES
from sklearn.pipeline import Pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    df = load_data(n=cfg["n_customers"], seed=cfg["seed"])
    X, y = df[FEATURES], df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["seed"], stratify=y)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=cfg["seed"], n_jobs=-1),
    }
    results, best_name, best_score, best_model = {}, None, -1, None
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
        metrics = {"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba)}
        results[name] = metrics
        print(f"{name}: {metrics}")
        if metrics["roc_auc"] > best_score:
            best_name, best_score, best_model = name, metrics["roc_auc"], pipe
    (ROOT / cfg["model_path"]).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(best_model, ROOT / cfg["model_path"])
    report = {"best_model": best_name, "results": results, "n_rows": len(df), "seed": cfg["seed"]}
    (ROOT / cfg["metrics_path"]).parent.mkdir(exist_ok=True, parents=True)
    (ROOT / cfg["metrics_path"]).write_text(json.dumps(report, indent=2))
    print(f"Best: {best_name} (ROC-AUC {best_score:.3f}) -> saved to {cfg['model_path']}")
    return report

if __name__ == "__main__":
    main()
