"""Tune XGBoost classifier and regressor with random search, tracked in MLflow.

Run from the project folder: python src/tune_models.py
"""
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

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

N_TRIALS = 12
TUNE_ROWS = 60000  # subsample of the training set used for the search
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SEARCH_SPACE = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 5, 10],
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("real-estate-advisor-tuning")


def classification_metrics(y_true, pred, proba):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def regression_metrics(y_true, pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
    }


def make_model(task, params):
    common = dict(tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE, **params)
    return XGBClassifier(eval_metric="logloss", **common) if task == "classification" else XGBRegressor(**common)


def tune(task, target, train_df, test_df):
    score_name = "roc_auc" if task == "classification" else "rmse"
    higher_is_better = task == "classification"

    # Validation split inside the training data (the test set is not touched during tuning)
    stratify = train_df[target] if task == "classification" else None
    tune_df = train_df.sample(min(TUNE_ROWS, len(train_df)), random_state=RANDOM_STATE)
    fit_df, val_df = train_test_split(
        tune_df, test_size=0.25, random_state=RANDOM_STATE,
        stratify=tune_df[target] if stratify is not None else None,
    )
    pre = build_preprocessor()
    x_fit = pre.fit_transform(fit_df[FEATURES])
    x_val = pre.transform(val_df[FEATURES])

    best_params, best_score = None, None
    trials = list(ParameterSampler(SEARCH_SPACE, n_iter=N_TRIALS, random_state=RANDOM_STATE))

    with mlflow.start_run(run_name=f"xgboost_{task}_tuning"):
        mlflow.log_params({"task": task, "n_trials": N_TRIALS, "tune_rows": len(tune_df)})

        for i, params in enumerate(trials, 1):
            start = time.time()
            with mlflow.start_run(run_name=f"trial_{i:02d}", nested=True):
                mlflow.log_params(params)
                model = make_model(task, params)
                model.fit(x_fit, fit_df[target])
                if task == "classification":
                    metrics = classification_metrics(val_df[target], model.predict(x_val),
                                                     model.predict_proba(x_val)[:, 1])
                else:
                    metrics = regression_metrics(val_df[target], model.predict(x_val))
                metrics = {f"val_{k}": v for k, v in metrics.items()}
                mlflow.log_metrics(metrics)
            score = metrics[f"val_{score_name}"]
            print(f"[{task}] trial {i:02d}/{N_TRIALS} {params} -> val_{score_name}={score:.5f} "
                  f"({time.time() - start:.0f}s)", flush=True)
            better = best_score is None or (score > best_score if higher_is_better else score < best_score)
            if better:
                best_params, best_score = params, score

        # Refit the best configuration on the full training set, evaluate once on the test set
        print(f"[{task}] best params: {best_params}\n[{task}] refitting on {len(train_df):,} rows ...", flush=True)
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", make_model(task, best_params))])
        pipeline.fit(train_df[FEATURES], train_df[target])
        pred = pipeline.predict(test_df[FEATURES])
        if task == "classification":
            metrics = classification_metrics(test_df[target], pred, pipeline.predict_proba(test_df[FEATURES])[:, 1])
        else:
            metrics = regression_metrics(test_df[target], pred)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        path = MODELS_DIR / f"tuned_xgboost_{task}.joblib"
        joblib.dump(pipeline, path)
        mlflow.log_artifact(str(path))
        mlflow.sklearn.log_model(
            pipeline, name=f"xgboost_{task}", serialization_format="cloudpickle"
        )
    print(f"[{task}] TEST: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"[{task}] model size: {path.stat().st_size / 1024:.0f} KB\n")


if __name__ == "__main__":
    data = load_data()
    train_df, test_df = split_data(data)
    tune("classification", CLASSIFICATION_TARGET, train_df, test_df)
    tune("regression", REGRESSION_TARGET, train_df, test_df)
    print("Baselines (test): classifier HistGB accuracy 0.9989, ROC AUC 1.0000 | "
          "regressor HistGB RMSE 1.2067, MAE 0.9668")
    print("Done. View runs: mlflow ui --backend-store-uri sqlite:///mlflow.db  "
          "(experiment: real-estate-advisor-tuning)")