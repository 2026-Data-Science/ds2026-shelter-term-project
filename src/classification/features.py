from __future__ import annotations

import re

import pandas as pd


TARGET_COLUMN = "OutcomeType"
TRAIN_ID_COLUMN = "AnimalID"
TEST_ID_COLUMN = "ID"

# 15 of 16 OutcomeSubtype values map to a single OutcomeType with 100% probability,
# so including this column would leak the target.
LEAKAGE_COLUMNS = ["OutcomeSubtype"]
FORBIDDEN_MODEL_INPUT_COLUMNS = [
    TRAIN_ID_COLUMN,
    TEST_ID_COLUMN,
    TARGET_COLUMN,
    *LEAKAGE_COLUMNS,
]

RAW_FEATURE_COLUMNS = [
    "Name",
    "AnimalType",
    "SexuponOutcome",
    "AgeuponOutcome",
    "Breed",
    "Color",
]

# Adoption 40.29% vs Died 0.74% (54.66:1). Macro F1 is the headline metric, not accuracy.
LABEL_ORDER = ["Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"]

ENGINEERED_NUMERIC_COLUMNS = [
    "has_name",
    "age_days",
    "is_mixed_breed",
]
ENGINEERED_CATEGORICAL_COLUMNS = [
    "animal_type",
    "sex",
    "neuter_status",
    "primary_breed",
    "primary_color",
]
ENGINEERED_FEATURE_COLUMNS = ENGINEERED_NUMERIC_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS

_AGE_PATTERN = re.compile(
    r"^\s*(\d+)\s+(day|days|week|weeks|month|months|year|years)\s*$",
    re.IGNORECASE,
)


def assert_no_forbidden_model_inputs(frame: pd.DataFrame) -> None:
    forbidden = [col for col in FORBIDDEN_MODEL_INPUT_COLUMNS if col in frame.columns]
    if forbidden:
        raise ValueError(f"Forbidden model input columns detected: {forbidden}")


def _age_to_days(value: object) -> float:
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



def _split_sexuponoutcome(value: object) -> tuple[str, str]:
    """Split the combined `sex` and `neuter_status` signals stored in one column."""
    if pd.isna(value):
        return "Unknown", "Unknown"
    text = str(value).strip()
    if not text or text == "Unknown":
        return "Unknown", "Unknown"
    lowered = text.lower()

    # "female" contains "male" as a substring, so the female check must come first.
    if "female" in lowered:
        sex = "Female"
    elif "male" in lowered:
        sex = "Male"
    else:
        sex = "Unknown"

    if "neutered" in lowered:
        neuter_status = "Neutered"
    elif "spayed" in lowered:
        neuter_status = "Spayed"
    elif "intact" in lowered:
        neuter_status = "Intact"
    else:
        neuter_status = "Unknown"

    return sex, neuter_status


def _primary_breed(value: object) -> str:
    """Reduce 1,380 -> 220 cardinality by keeping the token before any slash and dropping a trailing ' Mix'."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    first = text.split("/")[0].strip()
    first = re.sub(r"\s+Mix$", "", first, flags=re.IGNORECASE).strip()
    return first or "Unknown"


def _is_mixed_breed(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).lower()
    return int(("mix" in text) or ("/" in text))


def _primary_color(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    return text.split("/")[0].strip() or "Unknown"


def engineer_features(raw_features: pd.DataFrame) -> pd.DataFrame:
    assert_no_forbidden_model_inputs(raw_features)
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in raw_features.columns]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")

    raw = raw_features[RAW_FEATURE_COLUMNS].copy()
    out = pd.DataFrame(index=raw.index)

    # The raw name text is too high cardinality to be a useful signal; only its presence is encoded.
    name_clean = raw["Name"].fillna("").astype(str).str.strip()
    out["has_name"] = (name_clean != "").astype(int)

    out["animal_type"] = (
        raw["AnimalType"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )

    out["age_days"] = raw["AgeuponOutcome"].map(_age_to_days)

    sex_status = raw["SexuponOutcome"].map(_split_sexuponoutcome)
    out["sex"] = sex_status.map(lambda item: item[0])
    out["neuter_status"] = sex_status.map(lambda item: item[1])

    out["primary_breed"] = raw["Breed"].map(_primary_breed)
    out["is_mixed_breed"] = raw["Breed"].map(_is_mixed_breed)
    out["primary_color"] = raw["Color"].map(_primary_color)

    out = out[ENGINEERED_FEATURE_COLUMNS]
    _validate_engineered(out)
    return out


def _validate_engineered(frame: pd.DataFrame) -> None:
    assert_no_forbidden_model_inputs(frame)
    extra = [col for col in frame.columns if col not in ENGINEERED_FEATURE_COLUMNS]
    if extra:
        raise ValueError(f"Unexpected engineered columns: {extra}")
    missing = [col for col in ENGINEERED_FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing engineered columns: {missing}")
