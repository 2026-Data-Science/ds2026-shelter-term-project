from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Some macOS shells make joblib's physical-core detector noisy; keep runs readable by
# setting an explicit worker cap below the logical core count when the user has not set one.
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 2) - 1))
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

import numpy as np
import pandas as pd

from src.classification.features import (
    LABEL_ORDER,
    LEAKAGE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_ID_COLUMN,
    engineer_features,
)
from src.classification.preprocessing import (
    CATEGORICAL_COLUMNS_AFTER_TE,
    NUMERIC_COLUMNS_AFTER_TE,
    build_preprocessing_pipeline,
)
from src.clustering.analysis import (
    plot_cluster_scatter,
    run_clustering_analysis,
    write_clustering_report,
)

# Force UTF-8 on Windows consoles that default to cp949.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_ROOT / "data" / "train.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "preprocessing_report.md"
CLUSTERING_REPORT_PATH = PROJECT_ROOT / "outputs" / "clustering.md"
CLUSTERING_PLOT_PATH = PROJECT_ROOT / "outputs" / "clustering_pca_scatter.png"


def _load_train() -> tuple[pd.DataFrame, pd.Series]:
    if not TRAIN_CSV.exists():
        sys.stderr.write(
            f"[main.py] '{TRAIN_CSV}' is missing.\n"
            "  Download train.csv from Kaggle and place it under 'data/':\n"
            "    https://www.kaggle.com/c/shelter-animal-outcomes/data\n"
        )
        sys.exit(1)

    train = pd.read_csv(TRAIN_CSV)
    required = [*RAW_FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_ID_COLUMN, *LEAKAGE_COLUMNS]
    missing = [col for col in required if col not in train.columns]
    if missing:
        sys.stderr.write(f"[main.py] train.csv schema mismatch. Missing columns: {missing}\n")
        sys.exit(2)

    X = train[RAW_FEATURE_COLUMNS].copy()
    y = train[TARGET_COLUMN].copy()
    return X, y


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table without external deps."""
    headers = [str(col) for col in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    rows: list[str] = [header_line, separator]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _format_class_proportions(y: pd.Series) -> str:
    rows = []
    proportions = y.value_counts(normalize=True).reindex(LABEL_ORDER)
    for label in LABEL_ORDER:
        count = int((y == label).sum())
        ratio = float(proportions.loc[label]) * 100.0
        rows.append(f"- {label:<18} {count:>6}  ({ratio:5.2f}%)")
    return "\n".join(rows)


def _write_report(
    X: pd.DataFrame,
    y: pd.Series,
    raw_breed_unique: int,
    rule_breed_unique: int,
    transformed: np.ndarray,
    onehot_count: int,
    numeric_count: int,
    sample_preview: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: list[str] = []
    lines.append("# Preprocessing Verification Report")
    lines.append("")
    lines.append(f"Generated at (UTC): {now_iso}")
    lines.append(f"Input file: `data/train.csv`")
    lines.append("")
    lines.append("## Input shape")
    lines.append("")
    lines.append(f"- X.shape = {X.shape}")
    lines.append(f"- y.shape = {y.shape}")
    lines.append(f"- Target classes = {LABEL_ORDER}")
    lines.append("")
    lines.append("## Class proportions")
    lines.append("")
    lines.append("| Class | Count | Ratio |")
    lines.append("|-------|-------|-------|")
    proportions = y.value_counts(normalize=True).reindex(LABEL_ORDER)
    for label in LABEL_ORDER:
        count = int((y == label).sum())
        ratio = float(proportions.loc[label]) * 100.0
        lines.append(f"| {label} | {count} | {ratio:.2f}% |")
    lines.append("")
    lines.append("## Cardinality flow (primary_breed)")
    lines.append("")
    lines.append(f"- Raw `Breed` unique values: **{raw_breed_unique}**")
    lines.append(f"- After rule-based simplification (`primary_breed`): **{rule_breed_unique}**")
    lines.append("- After target encoding: **5 numeric columns** (`breed_te_*`)")
    lines.append("")
    lines.append("## Pipeline output")
    lines.append("")
    lines.append(f"- Transformed shape: {transformed.shape}")
    lines.append(f"- Dtype: {transformed.dtype}")
    lines.append(
        f"- Numeric columns (including target encoded): {numeric_count}"
    )
    lines.append(f"- OneHot expanded columns: {onehot_count}")
    lines.append(f"- Total columns: {numeric_count + onehot_count}")
    lines.append("")
    lines.append("## Categorical columns sent to OneHot")
    lines.append("")
    for col in CATEGORICAL_COLUMNS_AFTER_TE:
        lines.append(f"- `{col}`")
    lines.append("")
    lines.append("> `primary_breed` is intentionally absent — it is replaced by the five `breed_te_*` columns.")
    lines.append("")
    lines.append("## Transformed sample (first 5 rows, first 10 columns)")
    lines.append("")
    lines.append(_dataframe_to_markdown(sample_preview))
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("Classification preprocessing — end-to-end check")
    print("=" * 60)

    X, y = _load_train()
    print(f"[1/4] Loaded train.csv : X.shape={X.shape}, y.shape={y.shape}")
    print(f"      Target classes  : {sorted(y.unique())}")
    print("      Class proportions:")
    print(_format_class_proportions(y))

    engineered = engineer_features(X)
    raw_breed_unique = X["Breed"].nunique(dropna=False)
    rule_breed_unique = engineered["primary_breed"].nunique(dropna=False)
    print(
        f"[2/4] primary_breed cardinality: raw={raw_breed_unique} "
        f"-> rule-based={rule_breed_unique}  "
        "(compressed to 5 numeric columns via target encoding)"
    )

    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(X, y)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)

    onehot = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["onehot"]
    )
    onehot_count = sum(len(cats) for cats in onehot.categories_)
    numeric_count = len(NUMERIC_COLUMNS_AFTER_TE)

    print(
        f"[3/4] Preprocessing pipeline: fit_transform OK  "
        f"-> shape={transformed.shape}, dtype={transformed.dtype}"
    )
    print(
        f"      Numeric (incl. target encoding) = {numeric_count}, "
        f"OneHot expanded = {onehot_count}, "
        f"total expected = {numeric_count + onehot_count}"
    )

    assert transformed.shape[0] == len(X), "Row count was not preserved after transform."
    assert np.issubdtype(transformed.dtype, np.number), "Transformed matrix is not numeric."

    print(
        f"[4/4] Categorical columns into OneHot: {CATEGORICAL_COLUMNS_AFTER_TE} "
        f"(primary_breed is replaced by target encoding)"
    )

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    preview = pd.DataFrame(transformed[:5, :10], columns=list(feature_names[:10]))

    _write_report(
        X=X,
        y=y,
        raw_breed_unique=int(raw_breed_unique),
        rule_breed_unique=int(rule_breed_unique),
        transformed=transformed,
        onehot_count=int(onehot_count),
        numeric_count=int(numeric_count),
        sample_preview=preview,
    )

    print("=" * 60)
    print("Classification preprocessing checks passed.")
    print(f"Report written: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 60)

    print("=" * 60)
    print("Clustering analysis — PCA + K-Means")
    print("=" * 60)

    # The clustering workflow is unsupervised; y is passed only for post-hoc interpretation.
    clustering_result = run_clustering_analysis(X, y)
    print(
        f"[1/4] Clustering preprocessing OK -> "
        f"shape={clustering_result.preprocessed_shape}"
    )
    print(
        f"[2/4] PCA reduction OK -> shape={clustering_result.pca_shape}, "
        f"explained_variance={clustering_result.pca_explained_variance:.4f}"
    )
    print(
        f"[3/4] KMeans selected K={clustering_result.selected_k}, "
        f"silhouette={clustering_result.silhouette_sample:.4f}, "
        f"inertia={clustering_result.inertia:.4f}"
    )

    # Plot generation is optional at runtime, but the markdown report is always written.
    plot_path = None
    plot_error = None
    try:
        plot_path = plot_cluster_scatter(clustering_result, CLUSTERING_PLOT_PATH)
    except RuntimeError as exc:
        plot_error = str(exc)

    write_clustering_report(
        result=clustering_result,
        report_path=CLUSTERING_REPORT_PATH,
        plot_path=plot_path.relative_to(PROJECT_ROOT) if plot_path is not None else None,
        plot_error=plot_error,
    )

    if plot_path is not None:
        print(f"[4/4] Visualization written: {plot_path.relative_to(PROJECT_ROOT)}")
    else:
        print(f"[4/4] Visualization skipped: {plot_error}")

    print("=" * 60)
    print("Clustering analysis completed.")
    print(f"Report written: {CLUSTERING_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
