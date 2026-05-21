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
# Clustering caps cardinality without labels: breeds below 1% frequency -> "Other".
DEFAULT_MIN_BREED_FREQUENCY = 0.01
BREED_OTHER_LABEL = "Other"
DEFAULT_TOP_K_COLORS = 15
COLOR_OTHER_LABEL = "Other"
TIME_FEATURE_COLUMNS = [
    "outcome_year",
    "outcome_month",
    "outcome_dayofweek",
    "outcome_hour",
    "outcome_is_weekend",
    "outcome_season",
]

# Keep animal-state numerics only; calendar/time fields are excluded from clustering.
CLUSTERING_NUMERIC_COLUMNS = [
    column for column in ENGINEERED_NUMERIC_COLUMNS if column not in TIME_FEATURE_COLUMNS
]
# Keep animal-descriptor categoricals only; calendar/time fields are excluded from clustering.
CLUSTERING_CATEGORICAL_COLUMNS = [
    column for column in ENGINEERED_CATEGORICAL_COLUMNS if column not in TIME_FEATURE_COLUMNS
]
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


class BreedFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Unsupervised replacement for classification's BreedTargetEncoder.

    Classification: primary_breed -> 5 x P(OutcomeType | breed) using y (leakage risk for clustering).
    Clustering: keep one categorical column but collapse breeds below a frequency threshold to Other.
    """

    def __init__(
        self,
        min_frequency: float = DEFAULT_MIN_BREED_FREQUENCY,
        other_label: str = BREED_OTHER_LABEL,
    ) -> None:
        self.min_frequency = float(min_frequency)
        self.other_label = str(other_label)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BreedFrequencyEncoder":
        if "primary_breed" not in X.columns:
            raise ValueError("BreedFrequencyEncoder expects 'primary_breed' in input frame.")
        if not 0.0 <= self.min_frequency <= 1.0:
            raise ValueError("min_frequency must be between 0.0 and 1.0.")

        breed = X["primary_breed"].astype(str)
        rates = breed.value_counts(normalize=True)
        retained = rates[rates >= self.min_frequency].index.tolist()

        # Keep at least one level for very small future subsets, instead of mapping everything to Other.
        if not retained and not rates.empty:
            retained = [rates.index[0]]

        self.retained_breeds_ = {str(b) for b in retained}
        self.breed_frequencies_ = {str(breed): float(rate) for breed, rate in rates.items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "retained_breeds_"):
            raise RuntimeError("BreedFrequencyEncoder must be fit before transform.")
        if "primary_breed" not in X.columns:
            raise ValueError("BreedFrequencyEncoder expects 'primary_breed' in input frame.")

        result = X.copy()
        breed = result["primary_breed"].astype(str)
        result["primary_breed"] = breed.where(
            breed.isin(self.retained_breeds_),
            self.other_label,
        )
        return result


class TopKColorEncoder(BaseEstimator, TransformerMixin):
    """Keep the most common primary colors and group all remaining colors as Other."""

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K_COLORS,
        other_label: str = COLOR_OTHER_LABEL,
    ) -> None:
        self.top_k = int(top_k)
        self.other_label = str(other_label)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TopKColorEncoder":
        if "primary_color" not in X.columns:
            raise ValueError("TopKColorEncoder expects 'primary_color' in input frame.")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        color = X["primary_color"].astype(str)
        retained = color.value_counts().head(self.top_k).index.tolist()
        self.retained_colors_ = {str(c) for c in retained}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "retained_colors_"):
            raise RuntimeError("TopKColorEncoder must be fit before transform.")
        if "primary_color" not in X.columns:
            raise ValueError("TopKColorEncoder expects 'primary_color' in input frame.")

        result = X.copy()
        color = result["primary_color"].astype(str)
        result["primary_color"] = color.where(
            color.isin(self.retained_colors_),
            self.other_label,
        )
        return result


def build_clustering_preprocessor(include_animal_type: bool = True) -> ColumnTransformer:
    """Impute, scale numerics, and one-hot categoricals — no target encoding step.

    Differences from classification build_column_preprocessor():
    - Numeric columns: animal-state numerics only; DateTime-derived fields are excluded.
    - Categorical columns: animal descriptors only, including primary_breed.
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
            # Robust to outliers in age_days — important for distance-based clustering.
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
    min_breed_frequency: float = DEFAULT_MIN_BREED_FREQUENCY,
    top_k_colors: int = DEFAULT_TOP_K_COLORS,
    include_animal_type: bool = True,
) -> Pipeline:
    """Unsupervised preprocessing: feature engineering -> rare-breed grouping -> encode.

    Differences from classification build_preprocessing_pipeline():
    - No BreedTargetEncoder and no OutcomeType (y) at fit time.
    - Spayed and Neutered are grouped only for clustering, after shared feature engineering.
    - Breeds below the configured frequency threshold are grouped as Other.
    - Only the most frequent primary colors are retained; the rest are grouped as Other.
    - DateTime-derived fields are excluded so clusters focus on animal profiles.
    - PCA is applied in src.clustering.analysis, after this reusable preprocessing step.
    - For Cat/Dog-specific runs, animal_type is used as the filter and excluded as a feature.
    """
    return Pipeline(
        steps=[
            ("feature_engineering", _FeatureEngineeringTransformer()),
            ("neuter_status_grouper", NeuterStatusGrouper()),
            ("breed_frequency", BreedFrequencyEncoder(min_frequency=min_breed_frequency)),
            ("color_top_k", TopKColorEncoder(top_k=top_k_colors)),
            ("preprocessor", build_clustering_preprocessor(include_animal_type)),
        ]
    )
