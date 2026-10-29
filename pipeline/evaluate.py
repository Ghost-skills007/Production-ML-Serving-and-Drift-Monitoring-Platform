"""Evaluation metrics for a trained classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    positive_class: str | None = None,
) -> dict[str, float | list[list[int]]]:
    """Return accuracy, macro F1, ROC-AUC and the confusion matrix.

    ``positive_class`` selects the column of ``predict_proba`` used for ROC-AUC.
    When omitted, the model's last detected class is treated as positive.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    classes = list(model.classes_)
    pos = positive_class if positive_class in classes else classes[-1]
    pos_index = classes.index(pos)

    roc_auc = float("nan")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, pos_index]
        # ROC-AUC needs both classes present in the held-out labels.
        if y_test.nunique() > 1:
            roc_auc = float(roc_auc_score((y_test == pos).astype(int), proba))

    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc),
        "positive_class": pos,
        "confusion_matrix": cm,
    }


def confusion_matrix_frame(metrics: dict, classes: list[str]) -> pd.DataFrame:
    """Nice table form of the confusion matrix (rows = actual, cols = predicted)."""
    return pd.DataFrame(np.array(metrics["confusion_matrix"]), index=classes, columns=classes)
