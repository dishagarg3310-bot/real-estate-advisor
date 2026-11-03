"""Train classification and regression models and track them with MLflow.

Run from the project folder: python src/train_models.py
"""
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from features import (  # noqa: E402
    CLASSIFICATION_TARGET,
    FEATURES,
    RANDOM_STATE,
    REGRESSION_TARGET,
    build_preprocessor,
    load_data,
    split_data,
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("real-estate-advisor")


def make_pipeline(model) -> Pipeline:
    preprocessor = build_preprocessor()
    preprocessor.set_params(sparse_threshold=0)  # dense output, needed by gradient boosting
    return Pipeline([("preprocess", preprocessor), ("model", model)])


CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

REGRESSORS = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(
        n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}


def run_all(task, models, target, train_df, test_df, score_metric, higher_is_better):
    best_name, best_score, best_pipeline = None, None, None
    for name, model in models.items():
        print(f"[{task}] training {name} ...", flush=True)
        pipeline = make_pipeline(model)
        start = time.time()
        with mlflow.start_run(run_name=f"{task}_{name}"):
            mlflow.log_params({"task": task, "model": name, "n_features": len(FEATURES),
                               **{k: str(v) for k, v in model.get_params().items()
                                  if k in ("n_estimators", "max_depth", "min_samples_leaf", "alpha",
                                           "max_iter", "learning_rate")}})
            pipeline.fit(train_df[FEATURES], train_df[target])
            pred = pipeline.predict(test_df[FEATURES])

            if task == "classification":
                proba = pipeline.predict_proba(test_df[FEATURES])[:, 1]
                metrics = {
                    "accuracy": accuracy_score(test_df[target], pred),
                    "precision": precision_score(test_df[target], pred),
                    "recall": recall_score(test_df[target], pred),
                    "f1": f1_score(test_df[target], pred),
                    "roc_auc": roc_auc_score(test_df[target], proba),
                }
            else:
                metrics = {
                    "rmse": float(np.sqrt(mean_squared_error(test_df[target], pred))),
                    "mae": mean_absolute_error(test_df[target], pred),
                    "r2": r2_score(test_df[target], pred),
                }
            metrics["train_seconds"] = time.time() - start
            mlflow.log_metrics(metrics)

            path = MODELS_DIR / f"{task}_{name}.joblib"
            joblib.dump(pipeline, path)
            mlflow.log_artifact(str(path))

        score = metrics[score_metric]
        print(f"    {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
        better = best_score is None or (score > best_score if higher_is_better else score < best_score)
        if better:
            best_name, best_score, best_pipeline = name, score, pipeline

    joblib.dump(best_pipeline, MODELS_DIR / f"best_{task}.joblib")
    print(f"[{task}] best model: {best_name} ({score_metric}={best_score:.4f}) -> models/best_{task}.joblib\n")


if __name__ == "__main__":
    data = load_data()
    train_df, test_df = split_data(data)
    run_all("classification", CLASSIFIERS, CLASSIFICATION_TARGET, train_df, test_df, "roc_auc", True)
    run_all("regression", REGRESSORS, REGRESSION_TARGET, train_df, test_df, "rmse", False)
    print("Done. View the runs with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")