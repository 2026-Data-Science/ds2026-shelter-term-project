# Classification — Preprocessing Logic Reference

Technical detail and design rationale for the preprocessing pipeline under `src/classification/`. The output produced by `python main.py` is reproduced in the verification section at the end.

---

## 1. Dataset

### 1.1 Columns (`train.csv`, 26,729 rows × 10 columns)

| Column | Role |
|---|---|
| AnimalID | Identifier (excluded from learning) |
| Name | Feature (transformed) |
| DateTime | Feature (transformed) |
| AnimalType | Feature |
| SexuponOutcome | Feature (transformed) |
| AgeuponOutcome | Feature (transformed) |
| Breed | Feature (transformed) |
| Color | Feature (transformed) |
| OutcomeSubtype | Excluded (leakage) |
| OutcomeType | Target |

### 1.2 Target distribution

| OutcomeType | Count | Ratio |
|---|---|---|
| Adoption | 10,769 | 40.29% |
| Transfer | 9,422 | 35.25% |
| Return_to_owner | 4,786 | 17.91% |
| Euthanasia | 1,555 | 5.82% |
| Died | 197 | 0.74% |

Adoption : Died = **54.66 : 1**. Accuracy alone is not reliable, and Macro F1 is treated as the headline metric.

### 1.3 Missing values

| Column | Missing ratio |
|---|---|
| OutcomeSubtype | 50.93% |
| Name | 28.77% |
| Remaining 7 columns | 0% |

### 1.4 Cardinality

| Column | Unique values | Treatment |
|---|---|---|
| Breed | 1,380 | `primary_breed` rule-based reduction (1,380 -> 220) and target encoding (220 -> 5) |
| Color | 366 | `primary_color` (token before slash, 366 -> 57) |
| AnimalType | 2 | Kept as-is |
| SexuponOutcome | 5 | Split into `sex` and `neuter_status` |

---

## 2. Feature Selection

| Column | Used? | Rationale |
|---|---|---|
| AnimalID | No | Pure identifier, no predictive signal |
| OutcomeSubtype | No | Leakage (see 2.1) |
| Name | Yes (transformed) | The raw text is too high cardinality; presence is encoded as `has_name` (0/1) |
| DateTime | Yes (decomposed) | The single string contains multiple signals; decomposed into six features |
| AnimalType | Yes | Outcome distribution differs by species |
| SexuponOutcome | Yes (split) | Two independent signals combined into one column; split into `sex` and `neuter_status` |
| AgeuponOutcome | Yes (transformed) | Free-text age converted to integer days |
| Breed | Yes (decomposed + target encoded) | Cardinality reduction (1,380 -> 220 -> 5) |
| Color | Yes (cleaned) | Token before slash retained |

### 2.1 OutcomeSubtype leakage evidence

Among 16 distinct values of `OutcomeSubtype`, 15 map to a specific `OutcomeType` with 100% probability. Including this column would let the model recover the target directly from a single feature.

| OutcomeSubtype | Mapped OutcomeType | Match |
|---|---|---|
| Suffering | Euthanasia | 100% |
| Aggressive | Euthanasia | 100% |
| Foster | Adoption | 100% |
| SCRP | Transfer | 100% |
| ... (15 of 16) | ... | 100% |

---

## 3. Feature Engineering — seven transformation rules

| Source | Result | Intent |
|---|---|---|
| `Name` | `has_name` (0/1) | Avoid name-text cardinality |
| `AgeuponOutcome` | `age_days` (int) | Convert `"X years/months/weeks/days"` strings to integer days |
| `DateTime` | `outcome_year`, `outcome_month`, `outcome_dayofweek`, `outcome_hour`, `outcome_is_weekend`, `outcome_season` | Decompose the timestamp string into six independent signals |
| `SexuponOutcome` | `sex`, `neuter_status` | Separate the two signals combined into one column |
| `Breed` | `primary_breed`, `is_mixed_breed` | Token before slash, drop trailing " Mix" (1,380 -> 220) |
| `Color` | `primary_color` | Token before slash (366 -> 57) |
| `AnimalType` | `animal_type` | Kept as-is (2 values) |

Result: 7 raw columns -> **14 engineered columns** (8 numeric + 6 categorical).

### 3.1 `primary_breed` cardinality reduction

A tree-based, axis-aligned split model cannot effectively use 1,380 sparse one-hot columns. Two stages are applied.

**Stage 1: 1,380 -> 220** (rule-based simplification in `features.py`)

```
"Pit Bull/Labrador"                  -> "Pit Bull"
"Labrador Retriever Mix"             -> "Labrador Retriever"
"Labrador Retriever Mix/Pit Bull"    -> "Labrador Retriever"
```

- Token before the slash kept
- Trailing " Mix" suffix removed
- The target column is not used, so there is no leakage risk

**Stage 2: 220 -> 5** (`BreedTargetEncoder` in `preprocessing.py`)

Each breed is replaced by **five conditional probability columns** `P(class | breed)`:

```
breed_te_Adoption
breed_te_Died
breed_te_Euthanasia
breed_te_Return_to_owner
breed_te_Transfer
```

Bayesian smoothing with `alpha = 50`:

```
P(class | breed) = (count_class_in_breed + alpha * prior_class) / (count_breed + alpha)
```

This stops a rare breed (for example a single sample) from being assigned an extreme 100%/0% probability; the prior pulls underpopulated breeds back toward the overall distribution. A breed seen 200 times barely receives any prior pull.

**Fold-safe property**: `BreedTargetEncoder.fit()` consumes `y`, so leakage is a theoretical risk. Wrapping it inside `sklearn.pipeline.Pipeline` ensures that, during cross validation, `.fit()` only sees the training rows of each fold and `.transform()` is applied to validation rows. Unseen breeds at transform time fall back to the prior of the training fold.

Compression ratio: 1,380 -> 5 ≈ **276 : 1**. The one-hot blow-up is avoided while the class-distribution information is preserved.

---

## 4. Preprocessing Pipeline

```
raw frame
   -> engineer_features                              # 7 raw -> 14 engineered
   -> BreedTargetEncoder                             # drop primary_breed, add 5 breed_te_* columns
   -> ColumnTransformer
       |- numeric:     SimpleImputer(median)       + StandardScaler
       |- categorical: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")
```

### 4.1 Design rationale per step

| Setting | Reason |
|---|---|
| `SimpleImputer(strategy="median")` (numeric) | Median is robust to skew. The mean of `age_days` is dragged by very old animals |
| `SimpleImputer(strategy="most_frequent")` (categorical) | Mean/median is not defined for categorical values |
| `StandardScaler` | Normalises scale gaps that would otherwise bias scale-sensitive models |
| `OneHotEncoder(handle_unknown="ignore")` | Lets a validation fold contain unseen categories without raising an error |
| `verbose_feature_names_out=False` | Keeps the output column names short |

### 4.2 Pipeline integration

The entire flow is a single `sklearn.pipeline.Pipeline`. During cross validation only the training portion of each fold sees `.fit()`, which automatically prevents leakage. Downstream model code can chain its estimator at the end:

```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.classification.preprocessing import build_preprocessing_pipeline

pipeline = Pipeline(
    steps=[
        *build_preprocessing_pipeline().steps,
        ("model", DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, class_weight="balanced")),
    ]
)
```

---

## 5. Verification (`python main.py` output)

```
============================================================
Classification preprocessing — end-to-end check
============================================================
[1/4] Loaded train.csv : X.shape=(26729, 7), y.shape=(26729,)
      Target classes  : ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
      Class proportions:
        - Adoption            10769  (40.29%)
        - Died                  197  ( 0.74%)
        - Euthanasia           1555  ( 5.82%)
        - Return_to_owner      4786  (17.91%)
        - Transfer             9422  (35.25%)
[2/4] primary_breed cardinality: raw=1380 -> rule-based=220
      (compressed to 5 numeric columns via target encoding)
[3/4] Preprocessing pipeline: fit_transform OK
      -> shape=(26729, 83), dtype=float64
      Numeric (incl. target encoding) = 13, OneHot expanded = 70, total expected = 83
[4/4] Categorical columns into OneHot: ['animal_type', 'outcome_season', 'sex',
      'neuter_status', 'primary_color']
      (primary_breed is replaced by target encoding)
============================================================
All checks passed.
============================================================
```

`main.py` also writes `outputs/preprocessing_report.md` containing the same content plus the transformed-sample preview, so the run is fully auditable from the report alone.
