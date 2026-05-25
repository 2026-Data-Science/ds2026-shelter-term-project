from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.classification.features import (
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    TARGET_COLUMN,
)
from src.classification.preprocessing import build_preprocessing_pipeline

_CV_MODELS: dict = {
    "DummyClassifier": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
}

_SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
}


def run_cross_validation(df: pd.DataFrame) -> None:
    X = df.drop(
        columns=[col for col in FORBIDDEN_MODEL_INPUT_COLUMNS if col in df.columns]
    )
    y = df[TARGET_COLUMN]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for model_name, classifier in _CV_MODELS.items():
        print(f"\nEvaluating {model_name}...")
        model = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", classifier),
        ])
        cv_result = cross_validate(
            model, X, y, cv=cv, scoring=_SCORING, n_jobs=-1
        )

        result = {
            "model": model_name,
            "accuracy_mean": cv_result["test_accuracy"].mean(),
            "accuracy_std": cv_result["test_accuracy"].std(),
            "macro_f1_mean": cv_result["test_macro_f1"].mean(),
            "macro_f1_std": cv_result["test_macro_f1"].std(),
            "weighted_f1_mean": cv_result["test_weighted_f1"].mean(),
            "weighted_f1_std": cv_result["test_weighted_f1"].std(),
        }
        results.append(result)

        acc = f"{result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}"
        mf1 = f"{result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}"
        wf1 = (
            f"{result['weighted_f1_mean']:.4f}"
            f" ± {result['weighted_f1_std']:.4f}"
        )
        print(f"  Accuracy   : {acc}")
        print(f"  Macro F1   : {mf1}")
        print(f"  Weighted F1: {wf1}")

    print()
    print("Cross Validation Summary")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    _df = pd.read_csv("data/train.csv")
    print("=" * 60)
    print("5-Fold Cross Validation  (pre-tuning exploration)")
    print("=" * 60)
    run_cross_validation(_df)
