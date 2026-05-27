from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from src.classification.train import run_final_training
from src.classification.validation import run_cross_validation
from src.classification.preprocessing import (
    CATEGORICAL_COLUMNS_AFTER_TE,
    NUMERIC_COLUMNS_AFTER_TE,
    build_preprocessing_pipeline,
)
from src.clustering.analysis import (
    DURATION_BINS_FILENAME,
    DURATION_BOXPLOT_FILENAME,
    OUTPUT_DIR,
    PCA_SCATTER_FILENAME,
    REPORT_PATH as CLUSTERING_REPORT_PATH,
    run_long_stay_workflow,
)
from src.clustering.pca_kmeans_comparison import COMPARISON_CSV, COMPARISON_MD
from src.clustering.pca_silhouette_validation import VALIDATION_CSV, VALIDATION_MD
from src.clustering.pca_threshold_kmeans_comparison import THRESHOLD_CSV, THRESHOLD_SUMMARY_MD
from src.clustering.final_cluster_profile_report import PROFILE_CSV, PROFILE_MD
from src.clustering.preprocessing import ENRICHED_TRAIN_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_ROOT / "data" / "train.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "preprocessing_report.md"

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def _load_raw_df() -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        sys.stderr.write(
            f"[main.py] '{TRAIN_CSV}' is missing.\n"
            "  Download train.csv from Kaggle and place it under 'data/':\n"
            "    https://www.kaggle.com/c/shelter-animal-outcomes/data\n"
        )
        sys.exit(1)
    return pd.read_csv(TRAIN_CSV)


def _load_train() -> tuple[pd.DataFrame, pd.Series]:
    df = _load_raw_df()
    required = [*RAW_FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_ID_COLUMN, *LEAKAGE_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        sys.stderr.write(f"[main.py] train.csv schema mismatch. Missing columns: {missing}\n")
        sys.exit(2)
    X = df[RAW_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


# ---------------------------------------------------------------------------
# 리포트 헬퍼
# ---------------------------------------------------------------------------

def _dataframe_to_markdown(df: pd.DataFrame) -> str:
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
    lines.append(f"- Numeric columns (including target encoded): {numeric_count}")
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


# ---------------------------------------------------------------------------
# Classification 단계별 함수
# ---------------------------------------------------------------------------

def _run_data_inspection(df: pd.DataFrame, step: str = "1/4") -> None:
    print("=" * 60)
    print(f"STEP {step}  Dataset inspection")
    print("=" * 60)

    print(f"Shape: {df.shape}  ({df.shape[0]} rows, {df.shape[1]} columns)")
    print()

    print("--- dtypes ---")
    print(df.dtypes.to_string())
    print()

    print("--- head(5) ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df.head().to_string())
    print()

    print("--- Missing values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct,
    })
    print(missing_df[missing_df["missing_count"] > 0].to_string())
    if missing_df["missing_count"].sum() == 0:
        print("  (no missing values)")
    print()

    print("--- describe() ---")
    print(df.describe(include="all").to_string())


def _run_preprocessing_check(X: pd.DataFrame, y: pd.Series) -> None:
    print()
    print("=" * 60)
    print("STEP 2/4  Preprocessing verification")
    print("=" * 60)

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
    print(f"      Report written: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 모드별 진입점
# ---------------------------------------------------------------------------

def run_classification() -> None:
    print("=" * 60)
    print("Classification — end-to-end pipeline")
    print("=" * 60)

    df = _load_raw_df()
    _run_data_inspection(df, step="1/4")

    X, y = _load_train()
    _run_preprocessing_check(X, y)

    print()
    print("=" * 60)
    print("STEP 3/4  5-Fold Cross Validation  (pre-tuning exploration)")
    print("=" * 60)
    run_cross_validation(df)

    print()
    print("=" * 60)
    print("STEP 4/4  Hold-out Training & Evaluation  (tuned hyperparameters)")
    print("=" * 60)
    run_final_training(df)

    print()
    print("=" * 60)
    print("Classification pipeline complete.")
    print("=" * 60)


def _run_clustering() -> None:
    print()
    print("=" * 60)
    print("STEP 2/2  Long-stay risk clustering")
    print("=" * 60)
    print("Long-stay risk clustering — intake join + K-Means profiling")

    results, enriched, clustered = run_long_stay_workflow()

    print(
        f"[1/4] Enriched train CSV written: "
        f"{ENRICHED_TRAIN_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"[2/4] Rows usable for clustering/profile: "
        f"{len(clustered):,} / {len(enriched):,}"
    )
    for species, result in results.items():
        print(
            f"[3/4] {species}: selected K={result.selected_k}, "
            f"silhouette={result.silhouette_sample:.4f}, "
            f"encoded={result.encoded_shape}, pca={result.pca_shape}"
        )
    print(
        f"[4/4] Report written: "
        f"{CLUSTERING_REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      PCA scatter    : "
        f"{(OUTPUT_DIR / PCA_SCATTER_FILENAME).relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      Duration boxplot: "
        f"{(OUTPUT_DIR / DURATION_BOXPLOT_FILENAME).relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      Duration bins  : "
        f"{(OUTPUT_DIR / DURATION_BINS_FILENAME).relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      PCA comparison : "
        f"{COMPARISON_MD.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      PCA comparison : "
        f"{COMPARISON_CSV.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      PCA validation : "
        f"{VALIDATION_MD.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      PCA validation : "
        f"{VALIDATION_CSV.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      PCA threshold  : "
        f"{THRESHOLD_SUMMARY_MD.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      PCA threshold  : "
        f"{THRESHOLD_CSV.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      Report 5.5     : "
        f"{PROFILE_MD.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"      Report 5.5     : "
        f"{PROFILE_CSV.relative_to(PROJECT_ROOT)}"
    )


def run_clustering() -> None:
    print("=" * 60)
    print("Clustering — end-to-end pipeline")
    print("=" * 60)

    df = _load_raw_df()
    _run_data_inspection(df, step="1/2")

    _run_clustering()

    print()
    print("=" * 60)
    print("Clustering pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data-science end-to-end pipeline runner"
    )
    parser.add_argument(
        "mode",
        choices=["classification", "clustering"],
        help="Pipeline mode to run",
    )
    args = parser.parse_args()

    if args.mode == "classification":
        run_classification()
    elif args.mode == "clustering":
        run_clustering()


if __name__ == "__main__":
    main()
