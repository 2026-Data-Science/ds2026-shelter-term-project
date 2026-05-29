from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.classification.features import (
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    TARGET_COLUMN,
    load_merged_data,
)
from src.classification.preprocessing import build_preprocessing_pipeline

_FINAL_MODELS: dict = {
    "DummyClassifier": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(
        C=1,
        class_weight=None,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


def run_final_training(df: pd.DataFrame) -> None:
    drop_cols = [col for col in FORBIDDEN_MODEL_INPUT_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    results = []
    for model_name, classifier in _FINAL_MODELS.items():
        print(f"\nTraining {model_name}...")
        model = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", classifier),
        ])
        model.fit(X_train, y_train)

        print(f"Predicting with {model_name}...")
        pred = model.predict(X_test)

        results.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "weighted_f1": f1_score(y_test, pred, average="weighted"),
        })
        print(classification_report(y_test, pred))

    print("=" * 60)
    print("Final Model Comparison")
    pd.set_option("display.max_columns", None)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    _df = load_merged_data("data/train.csv", "data/wter-evkm.csv")
    print("=" * 60)
    print("Hold-out Training & Evaluation  (tuned hyperparameters)")
    print("=" * 60)
    run_final_training(_df)
