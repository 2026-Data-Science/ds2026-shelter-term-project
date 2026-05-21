from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.classification.features import (
    ENGINEERED_CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    RAW_FEATURE_COLUMNS,
    engineer_features,
)

# Classification uses target encoding (220 breeds -> 5 breed_te_* columns, needs y).
# Clustering caps cardinality without labels: top 30 breeds by frequency, rest -> "Other".
DEFAULT_TOP_K_BREEDS = 30
BREED_OTHER_LABEL = "Other"

# All engineered numerics; no breed_te_* (those require OutcomeType).
CLUSTERING_NUMERIC_COLUMNS = list(ENGINEERED_NUMERIC_COLUMNS)
# primary_breed stays categorical after Top-K; it is not dropped like in classification TE.
CLUSTERING_CATEGORICAL_COLUMNS = list(ENGINEERED_CATEGORICAL_COLUMNS)
SPECIES_SPECIFIC_CATEGORICAL_COLUMNS = [
    column for column in CLUSTERING_CATEGORICAL_COLUMNS if column != "animal_type"
]


class _FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Same state-free wrapper as classification — reuses engineer_features()."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "_FeatureEngineeringTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X)


class NeuterStatusGrouper(BaseEstimator, TransformerMixin):
    """Group sex-specific sterilization labels for clustering only.

    `sex` is already represented as a separate feature, so clustering should not
    split otherwise similar animals only because females are labelled "Spayed"
    and males are labelled "Neutered".
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "NeuterStatusGrouper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if "neuter_status" not in X.columns:
            raise ValueError("NeuterStatusGrouper expects 'neuter_status' in input frame.")

        result = X.copy()
        result["neuter_status"] = result["neuter_status"].replace(
            {
                "Spayed": "Neutered",
            }
        )
        return result


class TopKBreedEncoder(BaseEstimator, TransformerMixin):
    """Unsupervised replacement for classification's BreedTargetEncoder.

    Classification: primary_breed -> 5 x P(OutcomeType | breed) using y (leakage risk for clustering).
    Clustering: keep one categorical column but collapse rare breeds to Other using only counts.
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K_BREEDS,
        other_label: str = BREED_OTHER_LABEL,
    ) -> None:
        self.top_k = int(top_k)
        self.other_label = str(other_label)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TopKBreedEncoder":
        if "primary_breed" not in X.columns:
            raise ValueError("TopKBreedEncoder expects 'primary_breed' in input frame.")
        breed = X["primary_breed"].astype(str)
        top = breed.value_counts().head(self.top_k).index.tolist()
        self.top_breeds_ = {str(b) for b in top}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "top_breeds_"):
            raise RuntimeError("TopKBreedEncoder must be fit before transform.")
        if "primary_breed" not in X.columns:
            raise ValueError("TopKBreedEncoder expects 'primary_breed' in input frame.")

        result = X.copy()
        breed = result["primary_breed"].astype(str)
        result["primary_breed"] = breed.where(
            breed.isin(self.top_breeds_),
            self.other_label,
        )
        return result


def build_clustering_preprocessor(include_animal_type: bool = True) -> ColumnTransformer:
    """Impute, scale numerics, and one-hot categoricals — no target encoding step.

    Differences from classification build_column_preprocessor():
    - Numeric columns: 8 engineered only (classification adds 5 breed_te_* -> 13).
    - Categorical columns: all 6 including primary_breed (classification drops it for TE).
    - Scaler: RobustScaler (classification uses StandardScaler).
    - Imputation / OneHot settings are the same (median, most_frequent, ignore unknown).
    """
    categorical_columns = (
        CLUSTERING_CATEGORICAL_COLUMNS
        if include_animal_type
        else SPECIES_SPECIFIC_CATEGORICAL_COLUMNS
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # Robust to outliers in age_days and outcome_hour — important for distance-based clustering.
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, CLUSTERING_NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_clustering_pipeline(
    top_k_breeds: int = DEFAULT_TOP_K_BREEDS,
    include_animal_type: bool = True,
) -> Pipeline:
    """Unsupervised preprocessing: feature engineering -> Top-K breed -> impute/scale/encode.

    Differences from classification build_preprocessing_pipeline():
    - No BreedTargetEncoder and no OutcomeType (y) at fit time.
    - Spayed and Neutered are grouped only for clustering, after shared feature engineering.
    - Extra TopKBreedEncoder step instead of supervised breed compression.
    - PCA is applied in src.clustering.analysis, after this reusable preprocessing step.
    - For Cat/Dog-specific runs, animal_type is used as the filter and excluded as a feature.
    """
    return Pipeline(
        steps=[
            ("feature_engineering", _FeatureEngineeringTransformer()),
            ("neuter_status_grouper", NeuterStatusGrouper()),
            ("breed_top_k", TopKBreedEncoder(top_k=top_k_breeds)),
            ("preprocessor", build_clustering_preprocessor(include_animal_type)),
        ]
    )
