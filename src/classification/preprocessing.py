"""Leakage-safe preprocessing pipeline for the classification model.

Assembles three stages into one sklearn Pipeline: (1) feature engineering,
(2) breed target encoding, and (3) numeric/categorical imputation + scaling +
one-hot encoding. Because every fit-requiring step lives inside the Pipeline,
cross-validation refits them on the training fold only, so no target information
leaks from validation folds.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    ENGINEERED_CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    LABEL_ORDER,
    engineer_features,
)


TARGET_ENCODING_COLUMNS = [f"breed_te_{label}" for label in LABEL_ORDER]
DEFAULT_SMOOTHING_ALPHA = 50.0

# primary_breed is replaced by 5 numeric target-encoded columns and therefore dropped from categoricals.
NUMERIC_COLUMNS_AFTER_TE = ENGINEERED_NUMERIC_COLUMNS + TARGET_ENCODING_COLUMNS
CATEGORICAL_COLUMNS_AFTER_TE = [c for c in ENGINEERED_CATEGORICAL_COLUMNS if c != "primary_breed"]


class _FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """State-free wrapper that lets feature engineering live inside an sklearn Pipeline."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "_FeatureEngineeringTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X)


class BreedTargetEncoder(BaseEstimator, TransformerMixin):
    """Compress `primary_breed` (~220 levels) into 5 numeric P(class|breed) columns.

    Bayesian smoothing pulls rare breeds toward the global prior so that a breed
    with a single sample does not get assigned an extreme probability. The input
    DataFrame must follow the schema produced by `engineer_features`, which
    includes `primary_breed`.
    """

    def __init__(self, smoothing: float = DEFAULT_SMOOTHING_ALPHA) -> None:
        self.smoothing = float(smoothing)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BreedTargetEncoder":
        if "primary_breed" not in X.columns:
            raise ValueError("BreedTargetEncoder expects 'primary_breed' in input frame.")
        if y is None:
            raise ValueError("y is required to fit target encoding.")

        # Use only the training fold's prior — fit is repeated per fold to avoid leakage.
        prior = y.value_counts(normalize=True)
        self.prior_class_rates_ = {label: float(prior.get(label, 0.0)) for label in LABEL_ORDER}

        # Count breed x class co-occurrences on the training fold.
        breed = X["primary_breed"].astype(str)
        crosstab = pd.crosstab(breed, y).reindex(columns=LABEL_ORDER, fill_value=0)
        row_totals = crosstab.sum(axis=1)

        # Smoothed P(class|breed): blends each breed's counts with the global prior.
        encoding: dict[str, dict[str, float]] = {}
        for b, row in crosstab.iterrows():
            n = int(row_totals.loc[b])
            encoding[str(b)] = {
                label: float(
                    (row[label] + self.smoothing * self.prior_class_rates_[label])
                    / (n + self.smoothing)
                )
                for label in LABEL_ORDER
            }
        self.encoding_ = encoding
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "encoding_"):
            raise RuntimeError("BreedTargetEncoder must be fit before transform.")
        if "primary_breed" not in X.columns:
            raise ValueError("BreedTargetEncoder expects 'primary_breed' in input frame.")

        breed = X["primary_breed"].astype(str)
        # Drop the raw breed column; it is replaced by the 5 numeric encoded columns.
        result = X.drop(columns=["primary_breed"]).copy()

        # Breeds unseen in the training fold fall back to the training-fold prior.
        for label in LABEL_ORDER:
            col = f"breed_te_{label}"
            result[col] = breed.map(
                lambda b, label=label: self.encoding_.get(b, {}).get(
                    label, self.prior_class_rates_[label]
                )
            ).astype("float64")
        return result


def build_column_preprocessor() -> ColumnTransformer:
    # Numeric branch: median-impute (robust to skew), then standardize.
    # Median is robust against right-skewed columns such as age_days where the mean is dragged by outliers.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Categorical branch: mode-impute, then one-hot encode.
    # Ignore unseen categories during transform so a validation fold cannot break the pipeline.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS_AFTER_TE),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS_AFTER_TE),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessing_pipeline() -> Pipeline:
    """Chain feature engineering, target encoding, imputation, scaling and one-hot encoding.

    Wrapping the whole flow in a single Pipeline guarantees that, during cross
    validation, `.fit()` only sees the training portion of each fold; leakage is
    therefore prevented automatically.
    """
    return Pipeline(
        steps=[
            ("feature_engineering", _FeatureEngineeringTransformer()),
            ("target_encoding", BreedTargetEncoder()),
            ("preprocessor", build_column_preprocessor()),
        ]
    )
