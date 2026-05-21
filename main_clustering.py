from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification.features import (
    RAW_FEATURE_COLUMNS,
    TRAIN_ID_COLUMN,
    engineer_features,
)
from src.clustering.preprocessing import (
    CLUSTERING_CATEGORICAL_COLUMNS,
    CLUSTERING_NUMERIC_COLUMNS,
    DEFAULT_TOP_K_BREEDS,
    build_clustering_pipeline,
)

# Force UTF-8 on Windows consoles that default to cp949.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_ROOT / "data" / "train.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "clustering_preprocessing_report.md"

_FORBIDDEN_FEATURE_SUBSTRINGS = ("Outcome", "breed_te")


def _load_train_features() -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        sys.stderr.write(
            f"[main_clustering.py] '{TRAIN_CSV}' is missing.\n"
            "  Download train.csv from Kaggle and place it under 'data/':\n"
            "    https://www.kaggle.com/c/shelter-animal-outcomes/data\n"
        )
        sys.exit(1)

    train = pd.read_csv(TRAIN_CSV)
    required = [*RAW_FEATURE_COLUMNS, TRAIN_ID_COLUMN]
    missing = [col for col in required if col not in train.columns]
    if missing:
        sys.stderr.write(
            f"[main_clustering.py] train.csv schema mismatch. Missing columns: {missing}\n"
        )
        sys.exit(2)

    return train[RAW_FEATURE_COLUMNS].copy()


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


def _feature_names_contain_forbidden(names: np.ndarray | list[str]) -> list[str]:
    hits: list[str] = []
    for name in names:
        text = str(name)
        if any(token in text for token in _FORBIDDEN_FEATURE_SUBSTRINGS):
            hits.append(text)
    return hits


def _write_report(
    X: pd.DataFrame,
    raw_breed_unique: int,
    rule_breed_unique: int,
    top_k_breed_count: int,
    top_k_param: int,
    transformed: np.ndarray,
    onehot_count: int,
    numeric_count: int,
    forbidden_hits: list[str],
    sample_preview: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: list[str] = []
    lines.append("# Clustering Preprocessing Verification Report")
    lines.append("")
    lines.append(f"Generated at (UTC): {now_iso}")
    lines.append("Input file: `data/train.csv`")
    lines.append("Entrypoint: `main_clustering.py`")
    lines.append("")
    lines.append("## Input shape")
    lines.append("")
    lines.append(f"- X.shape = {X.shape}")
    lines.append("- OutcomeType / OutcomeSubtype: **not used** (unsupervised pipeline)")
    lines.append("")
    lines.append("## Cardinality flow (primary_breed)")
    lines.append("")
    lines.append(f"- Raw `Breed` unique values: **{raw_breed_unique}**")
    lines.append(f"- After rule-based simplification (`primary_breed`): **{rule_breed_unique}**")
    lines.append(
        f"- After Top-K encoding (`top_k={top_k_param}`): **{top_k_breed_count}** kept + `Other`"
    )
    lines.append("")
    lines.append("## Pipeline output")
    lines.append("")
    lines.append(f"- Transformed shape: {transformed.shape}")
    lines.append(f"- Dtype: {transformed.dtype}")
    lines.append(f"- Numeric columns: {numeric_count}")
    lines.append(f"- OneHot expanded columns: {onehot_count}")
    lines.append(f"- Total columns: {numeric_count + onehot_count}")
    lines.append("")
    lines.append("## Categorical columns sent to OneHot")
    lines.append("")
    for col in CLUSTERING_CATEGORICAL_COLUMNS:
        lines.append(f"- `{col}`")
    lines.append("")
    lines.append(
        "> `primary_breed` is collapsed to Top-K + Other before OneHot "
        "(no `breed_te_*` target encoding)."
    )
    lines.append("")
    lines.append("## Leakage guard (feature names)")
    lines.append("")
    if forbidden_hits:
        lines.append("**FAILED** — forbidden substrings found:")
        for name in forbidden_hits:
            lines.append(f"- `{name}`")
    else:
        lines.append(
            "- No feature name contains `Outcome` or `breed_te` "
            f"(checked substrings: {_FORBIDDEN_FEATURE_SUBSTRINGS})."
        )
    lines.append("")
    lines.append("## Transformed sample (first 5 rows, first 10 columns)")
    lines.append("")
    lines.append(_dataframe_to_markdown(sample_preview))
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("Clustering preprocessing — end-to-end check")
    print("=" * 60)

    X = _load_train_features()
    print(f"[1/4] Loaded train.csv : X.shape={X.shape} (RAW_FEATURE_COLUMNS only)")

    engineered = engineer_features(X)
    raw_breed_unique = X["Breed"].nunique(dropna=False)
    rule_breed_unique = engineered["primary_breed"].nunique(dropna=False)
    print(
        f"[2/4] primary_breed cardinality: raw={raw_breed_unique} "
        f"-> rule-based={rule_breed_unique}  "
        f"(Top-K={DEFAULT_TOP_K_BREEDS} + Other before OneHot)"
    )

    pipeline = build_clustering_pipeline()
    transformed = pipeline.fit_transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)

    onehot = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["onehot"]
    )
    onehot_count = sum(len(cats) for cats in onehot.categories_)
    numeric_count = len(CLUSTERING_NUMERIC_COLUMNS)
    top_k_breed_count = len(pipeline.named_steps["breed_top_k"].top_breeds_)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    forbidden_hits = _feature_names_contain_forbidden(feature_names)

    print(
        f"[3/4] Preprocessing pipeline: fit_transform OK (no y)  "
        f"-> shape={transformed.shape}, dtype={transformed.dtype}"
    )
    print(f"      Numeric columns       = {numeric_count}")
    print(f"      OneHot expanded         = {onehot_count}")
    print(f"      Total expected          = {numeric_count + onehot_count}")
    print(f"      Top-K breeds kept (fit) = {top_k_breed_count}")

    assert transformed.shape[0] == len(X), "Row count was not preserved after transform."
    assert np.issubdtype(transformed.dtype, np.number), "Transformed matrix is not numeric."
    assert not forbidden_hits, (
        "Forbidden substrings in feature names: "
        f"{forbidden_hits} (checked: {_FORBIDDEN_FEATURE_SUBSTRINGS})"
    )

    print(
        f"[4/4] Leakage check: no Outcome/breed_te in {len(feature_names)} feature names"
    )
    print(f"      Categorical columns into OneHot: {CLUSTERING_CATEGORICAL_COLUMNS}")

    preview = pd.DataFrame(transformed[:5, :10], columns=list(feature_names[:10]))

    _write_report(
        X=X,
        raw_breed_unique=int(raw_breed_unique),
        rule_breed_unique=int(rule_breed_unique),
        top_k_breed_count=int(top_k_breed_count),
        top_k_param=DEFAULT_TOP_K_BREEDS,
        transformed=transformed,
        onehot_count=int(onehot_count),
        numeric_count=int(numeric_count),
        forbidden_hits=forbidden_hits,
        sample_preview=preview,
    )

    print("=" * 60)
    print("All checks passed.")
    print(f"Report written: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
