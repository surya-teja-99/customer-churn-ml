from src.data_loader import load_data
from src.features import build_preprocessor, FEATURES
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def test_data_shape_and_target():
    df = load_data(n=200)
    assert len(df) == 200
    assert set(df["Churn"].unique()) <= {0, 1}

def test_pipeline_fits():
    df = load_data(n=200)
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(df[FEATURES], df["Churn"])
    assert pipe.predict(df[FEATURES].head(1))[0] in (0, 1)
