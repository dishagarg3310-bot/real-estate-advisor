"""Confusion matrix, feature importance and MLflow model registry for the best models.

Run from the project folder: python src/evaluate_and_register.py
"""
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from features import (  # noqa: E402
    CLASSIFICATION_TARGET,
    FEATURES,
    RANDOM_STATE,
    REGRESSION_TARGET,
    load_data,
    split_data,
)

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("real-estate-advisor")

data = load_data()
_, test_df = split_data(data)
sample = test_df.sample(5000, random_state=RANDOM_STATE)

clf = joblib.load("models/best_classification.joblib")
reg = joblib.load("models/best_regression.joblib")

with mlflow.start_run(run_name="best_models_evaluation"):
    # Confusion matrix on the full test set
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        clf, test_df[FEATURES], test_df[CLASSIFICATION_TARGET],
        display_labels=["Not good", "Good"], cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(REPORTS / "confusion_matrix.png"))

    print(classification_report(test_df[CLASSIFICATION_TARGET], clf.predict(test_df[FEATURES]),
                                target_names=["Not good", "Good"], digits=4))

    # Permutation importance on the raw input columns (5,000 test rows)
    for task, model, target, scoring in [
        ("classification", clf, CLASSIFICATION_TARGET, "roc_auc"),
        ("regression", reg, REGRESSION_TARGET, "r2"),
    ]:
        print(f"Computing {task} feature importance ...", flush=True)
        result = permutation_importance(model, sample[FEATURES], sample[target], scoring=scoring,
                                        n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
        importance = (pd.DataFrame({"Feature": FEATURES, "Importance": result.importances_mean})
                      .sort_values("Importance", ascending=False).round(5))
        path = REPORTS / f"feature_importance_{task}.csv"
        importance.to_csv(path, index=False)
        mlflow.log_artifact(str(path))
        print(importance.head(8).to_string(index=False), "\n")

    # Register the best models in the MLflow model registry
    mlflow.sklearn.log_model(clf, name="classifier", registered_model_name="real-estate-good-investment")
    mlflow.sklearn.log_model(reg, name="regressor", registered_model_name="real-estate-future-price")

print("Done. Open MLflow UI -> Models tab to see the registered models.")