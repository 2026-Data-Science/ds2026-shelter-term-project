from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
ENRICHED_TRAIN_PATH = DATA_DIR / "train_with_latest_intake.csv"
INTAKE_GLOB = "wter-evkm*.csv"

BREED_MIN_SPECIES_RATIO = 0.01

LOS_BINS = [-np.inf, 1, 7, 30, 90, 180, np.inf]
LOS_LABELS = ["<=1 day", "1-7 days", "7-30 days", "30-90 days", "90-180 days", ">180 days"]

# Duration and outcome are intentionally excluded from clustering inputs.
NUMERIC_FEATURES = ["has_name"]
CATEGORICAL_FEATURES = [
    "primary_breed",
    "intake_type",
    "intake_condition",
    "neuter_status",
    "intake_age_group",
]
PROFILE_ONLY_FEATURES = [
    "sex",
    "primary_color",
    "is_mixed_breed",
    "length_of_stay_days",
    "duration_category",
]

_AGE_PATTERN = re.compile(
    r"^\s*(\d+)\s+(day|days|week|weeks|month|months|year|years)\s*$",
    re.IGNORECASE,
)


def find_intake_csv(data_dir: Path = DATA_DIR) -> Path:
    """Find the downloaded Austin Animal Center intake CSV under data/."""
    matches = sorted(data_dir.glob(INTAKE_GLOB))
    if not matches:
        raise FileNotFoundError(f"No intake CSV matching data/{INTAKE_GLOB} was found.")
    return max(matches, key=lambda path: path.stat().st_mtime)


def validate_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    """Fail early when an input CSV does not have the expected schema."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _age_to_days(value: object) -> float:
    """Convert Austin age text, such as '2 years', to approximate days."""
    if pd.isna(value):
        return float("nan")
    match = _AGE_PATTERN.match(str(value))
    if not match:
        return float("nan")
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("day"):
        return float(amount)
    if unit.startswith("week"):
        return float(amount * 7)
    if unit.startswith("month"):
        return float(amount * 30)
    if unit.startswith("year"):
        return float(amount * 365)
    return float("nan")


def _age_group(age_days: object) -> str:
    """Bucket intake age into interpretable life-stage categories."""
    if pd.isna(age_days):
        return "Unknown"
    value = float(age_days)
    if value <= 180:
        return "baby"
    if value <= 730:
        return "young"
    if value <= 2555:
        return "adult"
    return "senior"


def _split_sex_status(value: object) -> tuple[str, str]:
    """Split `sex_upon_intake` into sex and sterilization status."""
    if pd.isna(value):
        return "Unknown", "Unknown"
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return "Unknown", "Unknown"
    lowered = text.lower()

    if "female" in lowered:
        sex = "Female"
    elif "male" in lowered:
        sex = "Male"
    else:
        sex = "Unknown"

    if "neutered" in lowered or "spayed" in lowered:
        neuter_status = "Neutered"
    elif "intact" in lowered:
        neuter_status = "Intact"
    else:
        neuter_status = "Unknown"
    return sex, neuter_status


def _primary_breed(value: object) -> str:
    """Reduce breed cardinality by keeping the first slash token and removing trailing Mix."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    first = text.split("/")[0].strip()
    first = re.sub(r"\s+Mix$", "", first, flags=re.IGNORECASE).strip()
    return first or "Unknown"


def _is_mixed_breed(value: object) -> int:
    """Flag mixed-breed records from slash or Mix markers."""
    if pd.isna(value):
        return 0
    text = str(value).lower()
    return int(("mix" in text) or ("/" in text))


def _primary_color(value: object) -> str:
    """Keep the first color token for post-hoc profiling."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    return text.split("/")[0].strip() or "Unknown"


def _bucket_rare_categories_by_species(
    frame: pd.DataFrame,
    column: str,
    min_ratio: float = BREED_MIN_SPECIES_RATIO,
) -> pd.Series:
    """Group rare categorical values within each species to reduce one-hot noise."""
    bucketed = frame[column].fillna("Unknown").astype(str).copy()
    for _, species_index in frame.groupby("AnimalType").groups.items():
        species_values = bucketed.loc[species_index]
        ratios = species_values.value_counts(normalize=True)
        keep_values = set(ratios[ratios.ge(min_ratio)].index)
        keep_values.add("Unknown")
        bucketed.loc[species_index] = species_values.where(
            species_values.isin(keep_values),
            "Other",
        )
    return bucketed


def match_latest_previous_intake(train: pd.DataFrame, intakes: pd.DataFrame) -> pd.DataFrame:
    """Match each outcome row to the same-animal intake at or before outcome time."""
    train = train.copy()
    intakes = intakes.copy().reset_index(drop=True)

    train["_animal_id"] = train["AnimalID"].astype(str).str.strip()
    train["_outcome_dt"] = pd.to_datetime(train["DateTime"], errors="coerce")
    intakes["_animal_id"] = intakes["animal_id"].astype(str).str.strip()
    intakes["_intake_dt"] = pd.to_datetime(intakes["datetime"], errors="coerce")
    intakes["_intake_row"] = np.arange(len(intakes), dtype=int)

    intake_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for animal_id, group in intakes.dropna(subset=["_intake_dt"]).groupby("_animal_id"):
        ordered = group.sort_values("_intake_dt")
        intake_lookup[animal_id] = (
            ordered["_intake_dt"].values.astype("datetime64[ns]"),
            ordered["_intake_row"].to_numpy(dtype=int),
        )

    matched_rows: list[float] = []
    for animal_id, outcome_dt in zip(train["_animal_id"], train["_outcome_dt"]):
        intake_values = intake_lookup.get(animal_id)
        if intake_values is None or pd.isna(outcome_dt):
            matched_rows.append(np.nan)
            continue
        intake_dates, intake_rows = intake_values
        position = np.searchsorted(
            intake_dates,
            np.datetime64(outcome_dt.to_datetime64()),
            side="right",
        ) - 1
        matched_rows.append(float(intake_rows[position]) if position >= 0 else np.nan)

    train["_matched_intake_row"] = matched_rows
    matched = train[train["_matched_intake_row"].notna()].copy()
    matched["_matched_intake_row"] = matched["_matched_intake_row"].astype(int)

    intake_columns = [
        "_intake_row",
        "_intake_dt",
        "intake_type",
        "intake_condition",
        "found_location",
        "sex_upon_intake",
        "age_upon_intake",
        "animal_type",
        "breed",
        "color",
    ]
    intake_subset = intakes[[column for column in intake_columns if column in intakes.columns]]
    intake_subset = intake_subset.add_prefix("intake_")
    matched = matched.merge(
        intake_subset,
        left_on="_matched_intake_row",
        right_on="intake__intake_row",
        how="left",
    )
    return matched.rename(
        columns={
            "intake__intake_row": "intake_row",
            "intake__intake_dt": "intake_datetime",
        }
    )


def build_train_with_latest_intake(train: pd.DataFrame, intakes: pd.DataFrame) -> pd.DataFrame:
    """Add latest intake fields and length-of-stay duration to train.csv rows."""
    matched = match_latest_previous_intake(train, intakes)
    original_columns = list(train.columns)
    extra_columns = [
        "AnimalID",
        "intake_datetime",
        "intake_intake_type",
        "intake_intake_condition",
        "intake_sex_upon_intake",
        "intake_age_upon_intake",
        "intake_animal_type",
        "intake_breed",
        "intake_color",
        "intake_found_location",
    ]
    available_extra = [column for column in extra_columns if column in matched.columns]
    enriched = train.merge(
        matched[available_extra],
        on="AnimalID",
        how="left",
    )
    outcome_dt = pd.to_datetime(enriched["DateTime"], errors="coerce")
    intake_dt = pd.to_datetime(enriched["intake_datetime"], errors="coerce")
    enriched["length_of_stay_days"] = (outcome_dt - intake_dt).dt.total_seconds() / 86400.0
    enriched["length_of_stay_hours"] = enriched["length_of_stay_days"] * 24.0
    enriched["length_of_stay_bin"] = pd.cut(
        enriched["length_of_stay_days"],
        bins=LOS_BINS,
        labels=LOS_LABELS,
    )
    added_columns = [column for column in available_extra if column != "AnimalID"]
    added_columns += ["length_of_stay_days", "length_of_stay_hours", "length_of_stay_bin"]
    return enriched[original_columns + added_columns]


def build_long_stay_clustering_frame(enriched: pd.DataFrame) -> pd.DataFrame:
    """Create clustering/profile features from train rows enriched with intake data."""
    frame = enriched.copy()
    frame = frame[frame["length_of_stay_days"].notna()].copy()
    frame = frame[frame["length_of_stay_days"].ge(0)].copy()

    name_clean = frame["Name"].fillna("").astype(str).str.strip()
    frame["has_name"] = (name_clean != "").astype(int)

    breed_source = frame["intake_breed"].combine_first(frame["Breed"])
    color_source = frame["intake_color"].combine_first(frame["Color"])
    frame["primary_breed"] = breed_source.map(_primary_breed)
    frame["primary_breed"] = _bucket_rare_categories_by_species(frame, "primary_breed")
    frame["is_mixed_breed"] = breed_source.map(_is_mixed_breed)
    frame["primary_color"] = color_source.map(_primary_color)

    sex_status = frame["intake_sex_upon_intake"].map(_split_sex_status)
    frame["sex"] = sex_status.map(lambda item: item[0])
    frame["neuter_status"] = sex_status.map(lambda item: item[1])

    frame["intake_age_days"] = frame["intake_age_upon_intake"].map(_age_to_days)
    frame["intake_age_group"] = frame["intake_age_days"].map(_age_group)
    frame["intake_type"] = (
        frame["intake_intake_type"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )
    frame["intake_condition"] = (
        frame["intake_intake_condition"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )
    frame["duration_category"] = pd.cut(
        frame["length_of_stay_days"],
        bins=LOS_BINS,
        labels=LOS_LABELS,
    ).astype("object")
    frame["long_stay_30"] = frame["length_of_stay_days"].gt(30).astype(int)
    frame["long_stay_60"] = frame["length_of_stay_days"].gt(60).astype(int)
    frame["long_stay_90"] = frame["length_of_stay_days"].gt(90).astype(int)
    return frame


def build_long_stay_preprocessor() -> ColumnTransformer:
    """Build preprocessing for duration-free long-stay clustering inputs."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
