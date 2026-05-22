from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

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
AGE_GROUP_COLUMN = "age_group"
AGE_GROUP_LABELS = ("baby", "young", "adult", "senior")
TIME_FEATURE_COLUMNS = [
    "outcome_year",
    "outcome_month",
    "outcome_dayofweek",
    "outcome_hour",
    "outcome_is_weekend",
    "outcome_season",
]
CLUSTERING_VARIANT_ORDER = ("v1", "v2", "v3")


@dataclass(frozen=True)
class ClusteringVariant:
    """Feature-set switch for comparable clustering experiments."""

    name: str
    title: str
    description: str
    use_age_days: bool
    use_age_group: bool
    use_color: bool


CLUSTERING_VARIANTS = {
    "v1": ClusteringVariant(
        name="v1",
        title="Clustering V1",
        description="Uses numeric age_days and top primary_color levels.",
        use_age_days=True,
        use_age_group=False,
        use_color=True,
    ),
    "v2": ClusteringVariant(
        name="v2",
        title="Clustering V2",
        description="Removes color and replaces numeric age_days with age_group.",
        use_age_days=False,
        use_age_group=True,
        use_color=False,
    ),
    "v3": ClusteringVariant(
        name="v3",
        title="Clustering V3",
        description="Removes both age and color from clustering inputs.",
        use_age_days=False,
        use_age_group=False,
        use_color=False,
    ),
}


def get_clustering_variant(variant: Union[str, ClusteringVariant]) -> ClusteringVariant:
    """Resolve a variant name to a feature-set configuration."""
    if isinstance(variant, ClusteringVariant):
        return variant
    key = str(variant).lower()
    if key not in CLUSTERING_VARIANTS:
        valid = ", ".join(CLUSTERING_VARIANT_ORDER)
        raise ValueError(f"Unknown clustering variant '{variant}'. Expected one of: {valid}.")
    return CLUSTERING_VARIANTS[key]


def get_clustering_columns(
    variant: Union[str, ClusteringVariant] = "v3",
    include_animal_type: bool = True,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical columns for a clustering feature-set variant."""
    spec = get_clustering_variant(variant)

    numeric_columns = [
        column for column in ENGINEERED_NUMERIC_COLUMNS if column not in TIME_FEATURE_COLUMNS
    ]
    if not spec.use_age_days:
        numeric_columns = [column for column in numeric_columns if column != "age_days"]

    categorical_columns = [
        column for column in ENGINEERED_CATEGORICAL_COLUMNS if column not in TIME_FEATURE_COLUMNS
    ]
    if not spec.use_color:
        categorical_columns = [
            column for column in categorical_columns if column != "primary_color"
        ]
    if spec.use_age_group:
        categorical_columns = [*categorical_columns, AGE_GROUP_COLUMN]
    if not include_animal_type:
        categorical_columns = [
            column for column in categorical_columns if column != "animal_type"
        ]

    return numeric_columns, categorical_columns


# Backward-compatible defaults point at the final v3 feature set.
CLUSTERING_NUMERIC_COLUMNS, CLUSTERING_CATEGORICAL_COLUMNS = get_clustering_columns("v3")
SPECIES_SPECIFIC_NUMERIC_COLUMNS, SPECIES_SPECIFIC_CATEGORICAL_COLUMNS = (
    get_clustering_columns("v3", include_animal_type=False)
)


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
        return group_neuter_status_feature(X)


def group_neuter_status_feature(X: pd.DataFrame) -> pd.DataFrame:
    """Map sex-specific sterilization labels to one clustering status column."""
    if "neuter_status" not in X.columns:
        raise ValueError("group_neuter_status_feature expects 'neuter_status' in input frame.")

    result = X.copy()
    result["neuter_status"] = result["neuter_status"].replace(
        {
            "Spayed": "Neutered",
        }
    )
    return result


def add_age_group_feature(X: pd.DataFrame) -> pd.DataFrame:
    """Convert continuous age_days into interpretable shelter-life-stage groups."""
    if "age_days" not in X.columns:
        raise ValueError("add_age_group_feature expects 'age_days' in input frame.")

    result = X.copy()
    age_days = pd.to_numeric(result["age_days"], errors="coerce")

    # Simple fixed bins keep cluster profiles explainable across both species.
    result[AGE_GROUP_COLUMN] = pd.cut(
        age_days,
        bins=[float("-inf"), 180.0, 730.0, 2555.0, float("inf")],
        labels=AGE_GROUP_LABELS,
    ).astype("object")
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


class AgeGroupTransformer(BaseEstimator, TransformerMixin):
    """Add baby/young/adult/senior age groups when a variant uses categorical age."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AgeGroupTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_age_group_feature(X)


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


def build_clustering_preprocessor(
    include_animal_type: bool = True,
    variant: Union[str, ClusteringVariant] = "v3",
) -> ColumnTransformer:
    """Impute, scale numerics, and one-hot categoricals — no target encoding step.

    Differences from classification build_column_preprocessor():
    - Numeric/categorical columns are selected by v1/v2/v3 feature-set variant.
    - DateTime-derived fields are excluded from all clustering variants.
    - Scaler: RobustScaler (classification uses StandardScaler).
    - Imputation / OneHot settings are the same (median, most_frequent, ignore unknown).
    """
    numeric_columns, categorical_columns = get_clustering_columns(
        variant=variant,
        include_animal_type=include_animal_type,
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # Robust to age outliers in v1 and harmless for binary numeric flags in v2/v3.
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
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_clustering_pipeline(
    min_breed_frequency: float = DEFAULT_MIN_BREED_FREQUENCY,
    top_k_colors: int = DEFAULT_TOP_K_COLORS,
    include_animal_type: bool = True,
    variant: Union[str, ClusteringVariant] = "v3",
) -> Pipeline:
    """Unsupervised preprocessing: feature engineering -> rare-breed grouping -> encode.

    Differences from classification build_preprocessing_pipeline():
    - No BreedTargetEncoder and no OutcomeType (y) at fit time.
    - Spayed and Neutered are grouped only for clustering, after shared feature engineering.
    - Age and color handling are selected by the requested v1/v2/v3 variant.
    - Breeds below the configured frequency threshold are grouped as Other.
    - DateTime-derived fields are excluded so clusters focus on animal profiles.
    - PCA is applied in src.clustering.analysis, after this reusable preprocessing step.
    - For Cat/Dog-specific runs, animal_type is used as the filter and excluded as a feature.
    """
    spec = get_clustering_variant(variant)
    steps: list[tuple[str, object]] = [
        ("feature_engineering", _FeatureEngineeringTransformer()),
        ("neuter_status_grouper", NeuterStatusGrouper()),
    ]
    if spec.use_age_group:
        steps.append(("age_group", AgeGroupTransformer()))
    steps.append(("breed_frequency", BreedFrequencyEncoder(min_frequency=min_breed_frequency)))
    if spec.use_color:
        steps.append(("color_top_k", TopKColorEncoder(top_k=top_k_colors)))
    steps.append(
        (
            "preprocessor",
            build_clustering_preprocessor(
                include_animal_type=include_animal_type,
                variant=spec,
            ),
        )
    )
    return Pipeline(
        steps=steps
    )
