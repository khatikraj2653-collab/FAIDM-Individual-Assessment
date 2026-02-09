import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from .features import get_feature_groups, build_preprocessor
from .data import make_target_binary_merge12
from .evaluation_binary import evaluate_binary_classifier, save_predictions

def train_and_evaluate_binary_merge12(
    df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    models_dir: Path,
    predictions_dir: Path,
    random_state: int,
    test_size: float,
    rf_estimators: int,
    hgb_max_iter: int,
    enable_grid_search: bool
) -> pd.DataFrame:
    X = df.drop(columns=["Diabetes_012"]).copy()
    y = make_target_binary_merge12(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric, ordinal, binary = get_feature_groups(df)
    preprocessor = build_preprocessor(numeric, ordinal, binary)

    results = []

    # 1) Logistic Regression (baseline)
    logreg = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))
    ])
    logreg.fit(X_train, y_train)
    joblib.dump(logreg, models_dir / "logreg_binary_merge12.joblib")
    results.append(evaluate_binary_classifier("logreg_binary_merge12", logreg, X_test, y_test, figures_dir))
    save_predictions("logreg_binary_merge12", logreg, X_test, y_test, predictions_dir / "pred_logreg_binary_merge12.csv")

    # 2) Random Forest (smaller for speed)
    rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=rf_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])
    rf.fit(X_train, y_train)
    joblib.dump(rf, models_dir / "rf_binary_merge12.joblib")
    results.append(evaluate_binary_classifier("rf_binary_merge12", rf, X_test, y_test, figures_dir))
    save_predictions("rf_binary_merge12", rf, X_test, y_test, predictions_dir / "pred_rf_binary_merge12.csv")

    # 3) HistGradientBoosting
    hgb = Pipeline([
        ("prep", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            random_state=random_state,
            max_iter=hgb_max_iter,
            learning_rate=0.08
        ))
    ])
    hgb.fit(X_train, y_train)
    joblib.dump(hgb, models_dir / "hgb_binary_merge12.joblib")
    results.append(evaluate_binary_classifier("hgb_binary_merge12", hgb, X_test, y_test, figures_dir))
    save_predictions("hgb_binary_merge12", hgb, X_test, y_test, predictions_dir / "pred_hgb_binary_merge12.csv")

    # Optional: GridSearch (off by default)
    if enable_grid_search:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        tuned = Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=6000, class_weight="balanced"))
        ])
        grid = {"clf__C": [0.1, 1.0, 10.0]}  # smaller grid for speed
        gs = GridSearchCV(
            tuned,
            param_grid=grid,
            scoring="average_precision",  # PR-AUC
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        joblib.dump(best, models_dir / "logreg_binary_merge12_tuned.joblib")

        m = evaluate_binary_classifier("logreg_binary_merge12_tuned", best, X_test, y_test, figures_dir)
        m["best_params"] = str(gs.best_params_)
        m["best_cv_pr_auc"] = float(gs.best_score_)
        results.append(m)
        save_predictions("logreg_binary_merge12_tuned", best, X_test, y_test, predictions_dir / "pred_logreg_binary_merge12_tuned.csv")

    out = pd.DataFrame(results)
    out.to_csv(tables_dir / "classification_metrics_binary_merge12_summary.csv", index=False)
    return out
