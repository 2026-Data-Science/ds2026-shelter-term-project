# Preprocessing Verification Report

Generated at (UTC): 2026-05-29T16:07:11+00:00
Input file: `data/train.csv`

## Input shape

- X.shape = (26729, 6)
- y.shape = (26729,)
- Target classes = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']

## Class proportions

| Class | Count | Ratio |
|-------|-------|-------|
| Adoption | 10769 | 40.29% |
| Died | 197 | 0.74% |
| Euthanasia | 1555 | 5.82% |
| Return_to_owner | 4786 | 17.91% |
| Transfer | 9422 | 35.25% |

## Cardinality flow (primary_breed)

- Raw `Breed` unique values: **1380**
- After rule-based simplification (`primary_breed`): **220**
- After target encoding: **5 numeric columns** (`breed_te_*`)

## Pipeline output

- Transformed shape: (26729, 74)
- Dtype: float64
- Numeric columns (including target encoded): 8
- OneHot expanded columns: 66
- Total columns: 74

## Categorical columns sent to OneHot

- `animal_type`
- `sex`
- `neuter_status`
- `primary_color`

> `primary_breed` is intentionally absent — it is replaced by the five `breed_te_*` columns.

## Transformed sample (first 5 rows, first 10 columns)

| has_name | age_days | is_mixed_breed | breed_te_Adoption | breed_te_Died | breed_te_Euthanasia | breed_te_Return_to_owner | breed_te_Transfer | animal_type_Cat | animal_type_Dog |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6356 | -0.3962 | 0.2343 | -0.7400 | -0.7017 | 0.2962 | 0.8651 | -0.4875 | 0.0000 | 1.0000 |
| 0.6356 | -0.3962 | 0.2343 | -0.4970 | 1.0053 | 0.0316 | -1.1564 | 1.2584 | 1.0000 | 0.0000 |
| 0.6356 | -0.0590 | 0.2343 | -1.7992 | -0.7738 | 2.7625 | 1.3538 | -0.9915 | 0.0000 | 1.0000 |
| -1.5733 | -0.7139 | 0.2343 | -0.4970 | 1.0053 | 0.0316 | -1.1564 | 1.2584 | 1.0000 | 0.0000 |
| -1.5733 | -0.0590 | 0.2343 | -3.2251 | 0.7076 | -1.0223 | 0.5844 | 1.1913 | 0.0000 | 1.0000 |
