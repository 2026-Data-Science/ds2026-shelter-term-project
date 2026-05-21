# Preprocessing Verification Report

Generated at (UTC): 2026-05-21T12:23:12+00:00
Input file: `data/train.csv`

## Input shape

- X.shape = (26729, 7)
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

- Transformed shape: (26729, 83)
- Dtype: float64
- Numeric columns (including target encoded): 13
- OneHot expanded columns: 70
- Total columns: 83

## Categorical columns sent to OneHot

- `animal_type`
- `outcome_season`
- `sex`
- `neuter_status`
- `primary_color`

> `primary_breed` is intentionally absent — it is replaced by the five `breed_te_*` columns.

## Transformed sample (first 5 rows, first 10 columns)

| has_name | age_days | outcome_year | outcome_month | outcome_dayofweek | outcome_hour | outcome_is_weekend | is_mixed_breed | breed_te_Adoption | breed_te_Died |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6356 | -0.3962 | -0.5810 | -1.4090 | -0.5408 | 1.0651 | -0.6926 | 0.2343 | -0.7400 | -0.7017 |
| 0.6356 | -0.3962 | -1.9298 | 0.8794 | 1.4143 | -0.7328 | 1.4439 | 0.2343 | -0.4970 | 1.0053 |
| 0.6356 | -0.0590 | 0.7678 | -1.6951 | 0.9255 | -0.7328 | 1.4439 | 0.2343 | -1.7992 | -0.7738 |
| -1.5733 | -0.7139 | -0.5810 | 0.0212 | 0.4368 | 1.3648 | -0.6926 | 0.2343 | -0.4970 | 1.0053 |
| -1.5733 | -0.0590 | -1.9298 | 1.1655 | 0.4368 | -0.7328 | -0.6926 | 0.2343 | -3.2251 | 0.7076 |
