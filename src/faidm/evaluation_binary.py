import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    brier_score_loss
)
from sklearn.calibration import CalibrationDisplay

from .utils import save_fig

def evaluate_binary_classifier(name: str, model, X_test, y_test, figures_dir: Path) -> dict:
    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    save_fig(figures_dir / f"clf_{name}_confusion_matrix.png")

    rep = classification_report(y_test, y_pred, digits=4, output_dict=True)

    metrics = {
        "model": name,
        "accuracy": rep["accuracy"],
        "precision_pos": rep["1"]["precision"],
        "recall_pos": rep["1"]["recall"],
        "f1_pos": rep["1"]["f1-score"],
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        metrics["pr_auc"] = average_precision_score(y_test, y_proba)
        metrics["brier"] = brier_score_loss(y_test, y_proba)

        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{name} - ROC Curve")
        save_fig(figures_dir / f"clf_{name}_roc_curve.png")

        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{name} - PR Curve")
        save_fig(figures_dir / f"clf_{name}_pr_curve.png")

        CalibrationDisplay.from_predictions(y_test, y_proba, n_bins=10)
        plt.title(f"{name} - Calibration")
        save_fig(figures_dir / f"clf_{name}_calibration.png")

    return metrics

def save_predictions(name: str, model, X_test, y_test, out_path: Path) -> None:
    df = pd.DataFrame({"y_true": y_test.values, "y_pred": model.predict(X_test)})
    if hasattr(model, "predict_proba"):
        df["p_diabetes"] = model.predict_proba(X_test)[:, 1]
    df.to_csv(out_path, index=False)
