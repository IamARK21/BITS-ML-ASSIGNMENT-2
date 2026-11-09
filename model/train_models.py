"""Train, tune and evaluate the five classifiers on the Breast Cancer Wisconsin data.

Run from the project root:

    python -m model.train_models

Outputs:
    model/*.pkl          one fitted scikit-learn Pipeline per model
    model/metrics.csv    the six test-set metrics for every model
    model/metadata.json  split sizes, tuned hyper-parameters, library version
    data/test_data.csv   the held-out test split (upload it in the app)
    README.md            the metrics table between the METRICS markers is refreshed
"""

import json
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):  # also allow `python model/train_models.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model.common import (
    CLASS_NAMES,
    FEATURES,
    ID_COL,
    METADATA_PATH,
    METRIC_NAMES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    ROOT,
    TARGET,
    TEST_DATA_PATH,
    compute_metrics,
    encode_target,
    load_dataset,
    positive_scores,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_SCORING = "f1"  # F1 on the malignant class balances missed cancers against false alarms

README_PATH = ROOT / "README.md"
README_START = "<!-- METRICS:START -->"
README_END = "<!-- METRICS:END -->"


def model_specs() -> dict:
    """Return {name: (pipeline, param_grid)}.

    Logistic Regression, KNN and SVM depend on feature scale (the raw features
    range from ~0.001 to ~4000), so they get a StandardScaler. Decision Trees and
    Gaussian Naive Bayes are unaffected by per-feature scaling, so they don't.
    """
    return {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000))]),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [3, 4, 5, 6, 8, None],
                "clf__min_samples_leaf": [1, 2, 4, 8],
            },
        ),
        "K-Nearest Neighbors": (
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes (Gaussian)": (
            Pipeline([("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        ),
        "Support Vector Machine": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.1, 0.01, 0.001]},
        ),
    }


def train_and_evaluate(verbose: bool = False) -> dict:
    """Split the data, tune each model with stratified CV, score on the test set.

    Returns a dict with ``models`` (name -> fitted pipeline), ``metrics``
    (DataFrame), ``metadata`` (dict) and ``test_df`` (the held-out rows).
    Nothing is written to disk here, so the app can call it as a fallback.
    """
    df = load_dataset()
    y = encode_target(df[TARGET])

    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    train_idx, test_idx = sorted(train_idx), sorted(test_idx)
    X_train, y_train = df.loc[train_idx, FEATURES], y.loc[train_idx]
    X_test, y_test = df.loc[test_idx, FEATURES], y.loc[test_idx]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models, rows, tuned = {}, [], {}

    for name, (pipeline, grid) in model_specs().items():
        if verbose:
            print(f"Tuning {name} ...", flush=True)
        search = GridSearchCV(pipeline, grid, scoring=CV_SCORING, cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        metrics = compute_metrics(y_test, best.predict(X_test), positive_scores(best, X_test))
        rows.append({"Model": name, **metrics})
        models[name] = best
        tuned[name] = {
            "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
            f"cv_{CV_SCORING}": round(float(search.best_score_), 4),
        }

    class_counts = y.value_counts().sort_index()
    metadata = {
        "sklearn_version": sklearn.__version__,
        # Pickled models only load reliably with the versions that created them;
        # these are used to pin requirements.txt for deployment.
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "dataset": "Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository",
        "n_instances": int(len(df)),
        "n_features": len(FEATURES),
        "class_counts": {CLASS_NAMES[i]: int(n) for i, n in class_counts.items()},
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "cv_folds": CV_FOLDS,
        "cv_scoring": CV_SCORING,
        "models": tuned,
    }
    return {
        "models": models,
        "metrics": pd.DataFrame(rows, columns=["Model", *METRIC_NAMES]),
        "metadata": metadata,
        "test_df": df.loc[test_idx, [ID_COL, TARGET, *FEATURES]].reset_index(drop=True),
    }


def save_artifacts(results: dict) -> None:
    for name, fitted in results["models"].items():
        joblib.dump(fitted, MODEL_DIR / MODEL_FILES[name])
    results["metrics"].round(4).to_csv(METRICS_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(results["metadata"], indent=2) + "\n")
    results["test_df"].to_csv(TEST_DATA_PATH, index=False)


def metrics_markdown(metrics: pd.DataFrame) -> str:
    lines = [
        "| ML Model Name | " + " | ".join(METRIC_NAMES) + " |",
        "|---|" + "---:|" * len(METRIC_NAMES),
    ]
    for _, row in metrics.iterrows():
        lines.append(f"| {row['Model']} | " + " | ".join(f"{row[m]:.4f}" for m in METRIC_NAMES) + " |")
    return "\n".join(lines)


def update_readme(metrics: pd.DataFrame) -> bool:
    """Replace the table between the README markers. Returns False if markers are absent."""
    if not README_PATH.exists():
        return False
    text = README_PATH.read_text()
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1 or end < start:
        return False
    new = text[: start + len(README_START)] + "\n" + metrics_markdown(metrics) + "\n" + text[end:]
    README_PATH.write_text(new)
    return True


def main() -> None:
    results = train_and_evaluate(verbose=True)
    save_artifacts(results)

    meta = results["metadata"]
    print(f"\nTrain/test: {meta['n_train']}/{meta['n_test']} rows, {meta['n_features']} features")
    print("\nTuned hyper-parameters:")
    for name, info in meta["models"].items():
        print(f"  {name:<24} {info['best_params']}  (CV {CV_SCORING} = {info[f'cv_{CV_SCORING}']:.4f})")
    print("\nTest-set metrics:\n")
    print(metrics_markdown(results["metrics"]))
    print(f"\nSaved models and metrics to {MODEL_DIR.relative_to(ROOT)}/ "
          f"and the test split to {TEST_DATA_PATH.relative_to(ROOT)}")
    if update_readme(results["metrics"]):
        print("Updated the metrics table in README.md")


if __name__ == "__main__":
    main()
