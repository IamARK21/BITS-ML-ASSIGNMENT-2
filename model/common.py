"""Constants and helpers shared by the training script and the Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

DATASET_PATH = DATA_DIR / "breast_cancer.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"

ID_COL = "id"
TARGET = "diagnosis"
POSITIVE_LABEL = "M"  # malignant is the positive class (1)
CLASS_NAMES = ["Benign", "Malignant"]  # index = encoded label

# The 10 cell-nucleus measurements, each summarised three ways -> 30 features.
BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
FEATURE_GROUPS = {"mean": "Mean", "se": "Standard error", "worst": "Worst (mean of 3 largest)"}
FEATURES = [f"{base}_{group}" for group in FEATURE_GROUPS for base in BASE_FEATURES]

# Display name -> saved file. Order here is the order used everywhere in the UI.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

_LABEL_MAP = {"M": 1, "MALIGNANT": 1, "1": 1, "1.0": 1, "B": 0, "BENIGN": 0, "0": 0, "0.0": 0}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and turn spaces into underscores.

    This lets the Kaggle copy of the dataset (which uses names like
    ``concave points_mean``) be uploaded without any editing.
    """
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def encode_target(labels: pd.Series) -> pd.Series:
    """Map diagnosis labels (M/B, Malignant/Benign or 1/0) to 1/0."""
    encoded = labels.astype(str).str.strip().str.upper().map(_LABEL_MAP)
    if encoded.isna().any():
        bad = sorted(labels[encoded.isna()].astype(str).unique())[:5]
        raise ValueError(f"Unrecognised '{TARGET}' values {bad}; expected M/B or 1/0.")
    return encoded.astype(int)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Read the full dataset with normalised column names."""
    return normalise_columns(pd.read_csv(path))


def positive_scores(model, X) -> np.ndarray:
    """Score for the positive (malignant) class, used for AUC and ROC curves."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """The six evaluation metrics required by the assignment.

    AUC is undefined when only one class is present (possible with a small
    uploaded file), so it is reported as NaN in that case.
    """
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
