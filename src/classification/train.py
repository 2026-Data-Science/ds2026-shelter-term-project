"""Final training and reporting step using the tuned hyperparameters.

Fits Dummy / LogisticRegression / RandomForest on an 80/20 stratified split,
prints the model comparison and per-class report, and saves the three report
figures for the RandomForest model: per-class metrics, confusion matrix, and
top-15 feature importances.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.classification.features import (
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    TARGET_COLUMN,
    load_merged_data,
)
from src.classification.preprocessing import build_preprocessing_pipeline

# Final models with the best hyperparameters found by the grid search.
_FINAL_MODELS: dict = {
    "DummyClassifier": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(
        C=1,
        class_weight=None,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        max_features=0.3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


def run_final_training(df: pd.DataFrame) -> None:
    # Remove ID/target/leakage columns before modelling.
    drop_cols = [col for col in FORBIDDEN_MODEL_INPUT_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN]

    # Same 80/20 stratified split (seed 42) as the grid search for a comparable hold-out.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    class_names = ["Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"]
    results = []
    y_pred_rf = None
    rf_pipeline = None
    # Train each model and collect accuracy / macro F1 / weighted F1 on the hold-out.
    for model_name, classifier in _FINAL_MODELS.items():
        print(f"\nTraining {model_name}...")
        model = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", classifier),
        ])
        model.fit(X_train, y_train)

        print(f"Predicting with {model_name}...")
        pred = model.predict(X_test)

        # Keep the RandomForest predictions/pipeline aside for the report figures.
        if model_name == "RandomForest":
            y_pred_rf = pred
            rf_pipeline = model

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

    if y_pred_rf is None:
        return

    # Figure 1: per-class precision / recall / F1 bar chart for the RandomForest model.
    report = classification_report(
        y_test, y_pred_rf,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics_df = pd.DataFrame(report).T.loc[
        class_names, ["precision", "recall", "f1-score"]
    ]
    metrics_df.columns = ["Precision", "Recall", "F1"]

    _, ax = plt.subplots(figsize=(10, 5))
    metrics_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/figure1_per_class_metrics.png", dpi=300)
    plt.show()

    # Figure 2: confusion matrix showing where the misclassifications land.
    cm = confusion_matrix(y_test, y_pred_rf, labels=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=20)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/figure2_confusion_matrix.png", dpi=300)
    plt.show()

    # Figure 3: top-15 RandomForest feature importances.
    # Recover post-encoding feature names so importances can be labelled.
    feature_names = (
        rf_pipeline.named_steps["preprocessing"]
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )
    importances = rf_pipeline.named_steps["classifier"].feature_importances_
    fi_series = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(15)
    )

    _, ax = plt.subplots(figsize=(10, 6))
    fi_series.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Feature Importances (Random Forest)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("outputs/figure3_feature_importance.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    _df = load_merged_data("data/train.csv", "data/wter-evkm.csv")
    print("=" * 60)
    print("Hold-out Training & Evaluation  (tuned hyperparameters)")
    print("=" * 60)
    run_final_training(_df)

